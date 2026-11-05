"""MLX port of boltz.model.layers.pairformer (inference only; v2 attention, no kernels/chunking/dropout)."""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from boltz_mlx import kernels as _K
from boltz_mlx.layers.norm import LayerNorm as _FastLN

from boltz_mlx.layers.attentionv2 import AttentionPairBias as AttentionPairBiasV2
from boltz_mlx.layers.transition import Transition
from boltz_mlx.layers.triangular_attention.attention import (
    TriangleAttentionEndingNode,
    TriangleAttentionStartingNode,
)
from boltz_mlx.layers.triangular_mult import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)


class PairformerLayer(nn.Module):
    """Pairformer layer (s and z tracks)."""

    def __init__(
        self,
        token_s: int,
        token_z: int,
        num_heads: int = 16,
        dropout: float = 0.25,
        pairwise_head_width: int = 32,
        pairwise_num_heads: int = 4,
        post_layer_norm: bool = False,
        v2: bool = True,
    ) -> None:
        super().__init__()
        if not v2:
            raise NotImplementedError("only the v2 (attentionv2) pairformer layer is ported")
        self.token_z = token_z
        self.dropout = dropout  # unused at inference
        self.num_heads = num_heads
        self.post_layer_norm = post_layer_norm

        self.pre_norm_s = _FastLN(token_s)
        self.attention = AttentionPairBiasV2(token_s, token_z, num_heads)

        self.tri_mul_out = TriangleMultiplicationOutgoing(token_z)
        self.tri_mul_in = TriangleMultiplicationIncoming(token_z)

        self.tri_att_start = TriangleAttentionStartingNode(token_z, pairwise_head_width, pairwise_num_heads, inf=1e9)
        self.tri_att_end = TriangleAttentionEndingNode(token_z, pairwise_head_width, pairwise_num_heads, inf=1e9)

        self.transition_s = Transition(token_s, token_s * 4)
        self.transition_z = Transition(token_z, token_z * 4)

        self.s_post_norm = _FastLN(token_s) if post_layer_norm else nn.Identity()

    def __call__(
        self,
        s: mx.array,
        z: mx.array,
        mask: mx.array,
        pair_mask: mx.array,
        chunk_size_tri_attn: Optional[int] = None,
        use_kernels: bool = False,
        use_cuequiv_mul: bool = False,
        use_cuequiv_attn: bool = False,
        xn_in: Optional[mx.array] = None,
        next_ln=None,
    ) -> tuple[mx.array, mx.array]:
        # chunk_size_tri_attn / use_* are accepted for call-compat and ignored.
        # xn_in / next_ln (f16, R3-pair PAIRLNX; passed by PairformerModule): this layer's tri_mul_out.norm_in(z) as
        # written by the previous layer's transition, and the next layer's norm_in for this layer's transition.
        pair_bias = None
        if _K.use_pair_block_f16(z):
            if _K.pair_lnx_ok(z):
                z, xn_next, pair_bias = _K.pair_block_f16(self, z, pair_mask, xn_in=xn_in, lnx=(next_ln, self.attention))
                if next_ln is not None:
                    self.__dict__["_pb_carry"] = (z, xn_next)  # read (and dropped) by the module loop
            else:
                z = _K.pair_block_f16(self, z, pair_mask, xn_in=xn_in)
        else:
            z = _K.radd(z, self.tri_mul_out(z, mask=pair_mask))
            z = _K.radd(z, self.tri_mul_in(z, mask=pair_mask))
            z = _K.radd(z, self.tri_att_start(z, mask=pair_mask))
            z = _K.radd(z, self.tri_att_end(z, mask=pair_mask))
            z = _K.radd(z, self.transition_z(z))

        s_normed = self.pre_norm_s(s)
        s = _K.radd(s, self.attention(s=s_normed, z=z, mask=mask, k_in=s_normed, pair_bias=pair_bias))
        s = _K.radd(s, self.transition_s(s))
        s = self.s_post_norm(s)
        return s, z


class PairformerModule(nn.Module):
    """Stack of PairformerLayers."""

    def __init__(
        self,
        token_s: int,
        token_z: int,
        num_blocks: int,
        num_heads: int = 16,
        dropout: float = 0.25,
        pairwise_head_width: int = 32,
        pairwise_num_heads: int = 4,
        post_layer_norm: bool = False,
        activation_checkpointing: bool = False,
        v2: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.token_z = token_z
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.num_heads = num_heads
        self.post_layer_norm = post_layer_norm

        self.layers = [
            PairformerLayer(
                token_s,
                token_z,
                num_heads,
                dropout,
                pairwise_head_width,
                pairwise_num_heads,
                post_layer_norm,
                v2,
            )
            for _ in range(num_blocks)
        ]

    def __call__(
        self,
        s: mx.array,
        z: mx.array,
        mask: mx.array,
        pair_mask: mx.array,
        use_kernels: bool = False,
    ) -> tuple[mx.array, mx.array]:
        big = _K.chunking(z.shape[1])
        n, xn = len(self.layers), None
        for i, layer in enumerate(self.layers):
            nxt = self.layers[i + 1].tri_mul_out.norm_in if i + 1 < n else None
            s, z = layer(s, z, mask, pair_mask, xn_in=xn, next_ln=nxt)
            xn = _carry(layer, z)
            if big:  # bound memory: don't let the lazy graph span many layers at large N
                mx.eval(*_K.guard_first(s, z))
            else:  # pipelined submission: the GPU runs this layer while Python builds the next (kernels.submit)
                _K.submit(s, z)
        return s, z


class PairformerNoSeqLayer(nn.Module):
    """Pairformer layer without the sequence track."""

    def __init__(
        self,
        token_z: int,
        dropout: float = 0.25,
        pairwise_head_width: int = 32,
        pairwise_num_heads: int = 4,
        post_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.token_z = token_z
        self.dropout = dropout  # unused at inference
        self.post_layer_norm = post_layer_norm

        self.tri_mul_out = TriangleMultiplicationOutgoing(token_z)
        self.tri_mul_in = TriangleMultiplicationIncoming(token_z)

        self.tri_att_start = TriangleAttentionStartingNode(token_z, pairwise_head_width, pairwise_num_heads, inf=1e9)
        self.tri_att_end = TriangleAttentionEndingNode(token_z, pairwise_head_width, pairwise_num_heads, inf=1e9)

        self.transition_z = Transition(token_z, token_z * 4)

    def __call__(
        self,
        z: mx.array,
        pair_mask: mx.array,
        chunk_size_tri_attn: Optional[int] = None,
        use_kernels: bool = False,
        use_cuequiv_mul: bool = False,
        use_cuequiv_attn: bool = False,
        xn_in: Optional[mx.array] = None,
        next_ln=None,
    ) -> mx.array:
        if _K.use_pair_block_f16(z):
            if next_ln is not None and _K.pair_lnx_ok(z):  # R3-pair: carry the next layer's norm_in output
                z, xn_next, _ = _K.pair_block_f16(self, z, pair_mask, xn_in=xn_in, lnx=(next_ln, None))
                self.__dict__["_pb_carry"] = (z, xn_next)
                return z
            return _K.pair_block_f16(self, z, pair_mask, xn_in=xn_in)
        z = _K.radd(z, self.tri_mul_out(z, mask=pair_mask))
        z = _K.radd(z, self.tri_mul_in(z, mask=pair_mask))
        z = _K.radd(z, self.tri_att_start(z, mask=pair_mask))
        z = _K.radd(z, self.tri_att_end(z, mask=pair_mask))
        z = _K.radd(z, self.transition_z(z))
        return z


class PairformerNoSeqModule(nn.Module):
    """Stack of PairformerNoSeqLayers."""

    def __init__(
        self,
        token_z: int,
        num_blocks: int,
        dropout: float = 0.25,
        pairwise_head_width: int = 32,
        pairwise_num_heads: int = 4,
        post_layer_norm: bool = False,
        activation_checkpointing: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.token_z = token_z
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.post_layer_norm = post_layer_norm

        self.layers = [
            PairformerNoSeqLayer(token_z, dropout, pairwise_head_width, pairwise_num_heads, post_layer_norm)
            for _ in range(num_blocks)
        ]

    def __call__(self, z: mx.array, pair_mask: mx.array, use_kernels: bool = False) -> mx.array:
        n, xn = len(self.layers), None
        for i, layer in enumerate(self.layers):
            nxt = self.layers[i + 1].tri_mul_out.norm_in if i + 1 < n else None
            z = layer(z, pair_mask, xn_in=xn, next_ln=nxt)
            xn = _carry(layer, z)
            if _K.chunking(z.shape[1]):
                mx.eval(*_K.guard_first(z))
            else:
                _K.submit(z)
        return z


def _carry(layer, z):
    """The next layer's norm_in output written by `layer`'s transition (R3-pair PAIRLNX), if it belongs to z."""
    c = layer.__dict__.pop("_pb_carry", None)
    return c[1] if (c is not None and c[0] is z) else None
