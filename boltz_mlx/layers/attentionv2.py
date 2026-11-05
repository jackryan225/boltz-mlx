"""MLX port of boltz.model.layers.attentionv2.AttentionPairBias (inference only)."""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from boltz_mlx import kernels as _K
from boltz_mlx.layers.norm import LayerNorm as _FastLN


class AttentionPairBias(nn.Module):
    """Attention with pair bias (v2: separate query and key/value inputs)."""

    def __init__(
        self,
        c_s: int,
        c_z: Optional[int] = None,
        num_heads: Optional[int] = None,
        inf: float = 1e6,
        compute_pair_bias: bool = True,
    ) -> None:
        super().__init__()
        assert c_s % num_heads == 0

        self.c_s = c_s
        self.num_heads = num_heads
        self.head_dim = c_s // num_heads
        self.inf = inf

        self.proj_q = nn.Linear(c_s, c_s)
        self.proj_k = nn.Linear(c_s, c_s, bias=False)
        self.proj_v = nn.Linear(c_s, c_s, bias=False)
        self.proj_g = nn.Linear(c_s, c_s, bias=False)

        self.compute_pair_bias = compute_pair_bias
        if compute_pair_bias:
            # torch: Sequential(LayerNorm, Linear, Rearrange("b ... h -> b h ..."))
            self.proj_z = [_FastLN(c_z), nn.Linear(c_z, num_heads, bias=False)]

        self.proj_o = nn.Linear(c_s, c_s, bias=False)

    def __call__(self, s: mx.array, z: mx.array, mask: mx.array, k_in: mx.array, multiplicity: int = 1,
                 pair_bias: Optional[mx.array] = None) -> mx.array:
        """s: (B, N, D); z: (B, N, N, c_z) (or bias (B, N, N, H)); mask: (B, N); k_in: (B, N, D).
        pair_bias: proj_z(z) already computed as (B, H, N, N) (f16, R3-pair: by the pair transition's epilogue)."""
        B = s.shape[0]
        H, hd = self.num_heads, self.head_dim

        q = self.proj_q(s).reshape(B, -1, H, hd)
        k = self.proj_k(k_in).reshape(B, -1, H, hd)
        v = self.proj_v(k_in).reshape(B, -1, H, hd)
        _K.record(f"attn{self.c_s}.q", q)
        _K.record(f"attn{self.c_s}.k", k)
        _K.record(f"attn{self.c_s}.v", v)

        bias = z
        if pair_bias is not None:
            bias = pair_bias
        elif self.compute_pair_bias and _K.f16():
            bias = _K.guard("attn.pair_bias", _K.lin16(self.proj_z[0].f16(z), self.proj_z[1]))
        elif self.compute_pair_bias:
            for layer in self.proj_z:
                bias = layer(bias)
        if pair_bias is None:
            bias = bias.transpose(0, 3, 1, 2)  # b ... h -> b h ...  (B, H, N, N)
        if multiplicity > 1:
            bias = mx.repeat(bias, multiplicity, axis=0)

        g = mx.sigmoid(self.proj_g(s))

        q = q.transpose(0, 2, 1, 3)  # (B, H, N, hd)
        k = k.transpose(0, 2, 3, 1)  # (B, H, hd, N)
        v = v.transpose(0, 2, 1, 3)  # (B, H, N, hd)

        attn = mx.matmul(q, k)  # (B, H, N, N)
        _K.record(f"attn{self.c_s}.qk_logits", attn)
        _K.record(f"attn{self.c_s}.pair_bias", bias)
        attn = attn / (hd**0.5) + bias
        attn = attn + (1 - mask[:, None, None, :].astype(mx.float32)) * -self.inf
        attn = mx.softmax(attn, axis=-1)

        o = mx.matmul(attn, v).transpose(0, 2, 1, 3).reshape(B, -1, self.c_s)
        y = self.proj_o(g * o)
        _K.record(f"attn{self.c_s}.out_proj", y)
        return y
