"""MLX port of boltz.model.modules.trunkv2 (inference only).

This file currently holds: TemplateV2Module, MSAModule, MSALayer, BFactorModule,
DistogramModule.  (InputEmbedder / ContactConditioning are ported separately and are
merged into this file by the integrator.)
"""

import mlx.core as mx
import mlx.nn as nn

from boltz_mlx import kernels as _K
import numpy as np

from boltz_mlx.layers.outer_product_mean import OuterProductMean
from boltz_mlx.layers.pair_averaging import PairWeightedAveraging
from boltz_mlx.layers.pairformer import PairformerNoSeqLayer, PairformerNoSeqModule
from boltz_mlx.layers.transition import Transition
from boltz_mlx.layers.norm import LayerNorm as _FastLN

# boltz.data.const.num_tokens (len(const.tokens))
NUM_TOKENS = 33


def _torch_linspace_f32(start: float, end: float, steps: int) -> np.ndarray:
    """float32 linspace computed exactly like torch.linspace (symmetric two-sided formula)."""
    step = np.float32((np.float32(end) - np.float32(start)) / np.float32(steps - 1))
    i = np.arange(steps, dtype=np.float32)
    lo = np.float32(start) + step * i
    hi = np.float32(end) - step * (np.float32(steps - 1) - i)
    return np.where(np.arange(steps) < steps // 2, lo, hi).astype(np.float32)


# --------------------------------------------------------------------------------------
# Template module
# --------------------------------------------------------------------------------------
class TemplateV2Module(nn.Module):
    """Template module (v2)."""

    def __init__(
        self,
        token_z: int,
        template_dim: int,
        template_blocks: int,
        dropout: float = 0.25,
        pairwise_head_width: int = 32,
        pairwise_num_heads: int = 4,
        post_layer_norm: bool = False,
        activation_checkpointing: bool = False,
        min_dist: float = 3.25,
        max_dist: float = 50.75,
        num_bins: int = 38,
        **kwargs,
    ) -> None:
        super().__init__()
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.num_bins = num_bins
        self.relu = nn.ReLU()
        self.z_norm = _FastLN(token_z)
        self.v_norm = _FastLN(template_dim)
        self.z_proj = nn.Linear(token_z, template_dim, bias=False)
        self.a_proj = nn.Linear(NUM_TOKENS * 2 + num_bins + 5, template_dim, bias=False)
        self.u_proj = nn.Linear(template_dim, token_z, bias=False)
        self.pairformer = PairformerNoSeqModule(
            template_dim,
            num_blocks=template_blocks,
            dropout=dropout,
            pairwise_head_width=pairwise_head_width,
            pairwise_num_heads=pairwise_num_heads,
            post_layer_norm=post_layer_norm,
            activation_checkpointing=activation_checkpointing,
        )

    def __call__(self, z: mx.array, feats: dict, pair_mask: mx.array, use_kernels: bool = False) -> mx.array:
        """z: (B, N, N, token_z); pair_mask: (B, N, N) -> (B, N, N, token_z)."""
        res_type = feats["template_restype"].astype(mx.float32)  # (B, T, N, 33)
        frame_rot = feats["template_frame_rot"]  # (B, T, N, 3, 3)
        frame_t = feats["template_frame_t"]  # (B, T, N, 3)
        frame_mask = feats["template_mask_frame"]  # (B, T, N)
        cb_coords = feats["template_cb"]  # (B, T, N, 3)
        ca_coords = feats["template_ca"]  # (B, T, N, 3)
        cb_mask = feats["template_mask_cb"]  # (B, T, N)
        visibility_ids = feats["visibility_ids"]  # (B, T, N)
        template_mask = mx.any(feats["template_mask"] != 0, axis=2).astype(mx.float32)  # (B, T)
        num_templates = mx.maximum(template_mask.sum(axis=1), 1)  # (B,)

        # Pairwise masks
        b_cb_mask = (cb_mask[:, :, :, None] * cb_mask[:, :, None, :])[..., None]
        b_frame_mask = (frame_mask[:, :, :, None] * frame_mask[:, :, None, :])[..., None]

        # Asym mask: template features only attend within the same chain
        B, T, N = res_type.shape[:3]
        tmlp_pair_mask = (visibility_ids[:, :, :, None] == visibility_ids[:, :, None, :]).astype(mx.float32)

        # Distogram of CB distances (direct differences; torch.cdist uses a matmul expansion
        # whose rounding differs by ~1e-6 relative, only relevant for bin-edge ties).
        diff = cb_coords[:, :, :, None, :] - cb_coords[:, :, None, :, :]
        cb_dists = mx.sqrt((diff * diff).sum(axis=-1))  # (B, T, N, N)
        boundaries = mx.array(_torch_linspace_f32(self.min_dist, self.max_dist, self.num_bins - 1))
        bins = (cb_dists[..., None] > boundaries).sum(axis=-1)  # (B, T, N, N)
        distogram = (bins[..., None] == mx.arange(self.num_bins)).astype(mx.float32)

        # Unit vector of CA_i in frame j.  NB: upstream takes the norm over the trailing
        # singleton dim (shape [..., 3, 1]), i.e. |vector| elementwise, so unit_vector is the
        # elementwise sign.  Reproduced literally.
        frame_rot_t = mx.expand_dims(frame_rot, 2).swapaxes(-1, -2)  # (B, T, 1, N, 3, 3)
        frame_t_ = mx.expand_dims(mx.expand_dims(frame_t, 2), -1)  # (B, T, 1, N, 3, 1)
        ca = mx.expand_dims(mx.expand_dims(ca_coords, 3), -1)  # (B, T, N, 1, 3, 1)
        vector = mx.matmul(frame_rot_t, ca - frame_t_)  # (B, T, N, N, 3, 1)
        norm = mx.sqrt((vector * vector).sum(axis=-1, keepdims=True))
        unit_vector = mx.where(norm > 0, vector / mx.where(norm > 0, norm, 1.0), mx.zeros_like(vector))
        unit_vector = unit_vector.squeeze(-1)  # (B, T, N, N, 3)

        a_tij = mx.concatenate([distogram, b_cb_mask, unit_vector, b_frame_mask], axis=-1)
        a_tij = a_tij * tmlp_pair_mask[..., None]

        res_type_i = mx.broadcast_to(res_type[:, :, :, None], (B, T, N, N, res_type.shape[-1]))
        res_type_j = mx.broadcast_to(res_type[:, :, None, :], (B, T, N, N, res_type.shape[-1]))
        a_tij = mx.concatenate([a_tij, res_type_i, res_type_j], axis=-1)
        a_tij = self.a_proj(a_tij)

        # Expand mask over templates
        pair_mask = mx.broadcast_to(pair_mask[:, None], (B, T, N, N)).reshape(B * T, N, N)

        # Input projections
        v = self.z_proj(self.z_norm(z[:, None])) + a_tij  # (B, T, N, N, d)
        v = v.reshape(B * T, N, N, -1)
        v = v + self.pairformer(v, pair_mask)
        v = self.v_norm(v)
        v = v.reshape(B, T, N, N, -1)

        # Aggregate templates
        tm = template_mask[:, :, None, None, None]
        nt = num_templates[:, None, None, None]
        u = (v * tm).sum(axis=1) / nt

        return self.u_proj(self.relu(u))


# --------------------------------------------------------------------------------------
# MSA module
# --------------------------------------------------------------------------------------
class MSAModule(nn.Module):
    """MSA module."""

    def __init__(
        self,
        msa_s: int,
        token_z: int,
        token_s: int,
        msa_blocks: int,
        msa_dropout: float,
        z_dropout: float,
        pairwise_head_width: int = 32,
        pairwise_num_heads: int = 4,
        activation_checkpointing: bool = False,
        use_paired_feature: bool = True,
        subsample_msa: bool = False,
        num_subsampled_msa: int = 1024,
        **kwargs,
    ) -> None:
        super().__init__()
        if subsample_msa:
            raise NotImplementedError("random MSA subsampling is training-only and not ported")
        self.msa_blocks = msa_blocks
        self.msa_dropout = msa_dropout
        self.z_dropout = z_dropout
        self.use_paired_feature = use_paired_feature

        self.s_proj = nn.Linear(token_s, msa_s, bias=False)
        self.msa_proj = nn.Linear(NUM_TOKENS + 2 + int(use_paired_feature), msa_s, bias=False)
        self.layers = [
            MSALayer(msa_s, token_z, msa_dropout, z_dropout, pairwise_head_width, pairwise_num_heads)
            for _ in range(msa_blocks)
        ]

    def __call__(self, z: mx.array, emb: mx.array, feats: dict, use_kernels: bool = False) -> mx.array:
        """z: (B, N, N, token_z); emb: (B, N, token_s) -> (B, N, N, token_z)."""
        msa = feats["msa"]  # (B, S, N) int
        msa = (msa[..., None] == mx.arange(NUM_TOKENS)).astype(mx.float32)
        has_deletion = feats["has_deletion"].astype(mx.float32)[..., None]
        deletion_value = feats["deletion_value"].astype(mx.float32)[..., None]
        is_paired = feats["msa_paired"].astype(mx.float32)[..., None]
        msa_mask = feats["msa_mask"]
        token_mask = feats["token_pad_mask"].astype(mx.float32)
        token_mask = token_mask[:, :, None] * token_mask[:, None, :]

        if self.use_paired_feature:
            m = mx.concatenate([msa, has_deletion, deletion_value, is_paired], axis=-1)
        else:
            m = mx.concatenate([msa, has_deletion, deletion_value], axis=-1)

        m = self.msa_proj(m)
        m = m + mx.expand_dims(self.s_proj(emb), 1)

        for layer in self.layers:
            z, m = layer(z, m, token_mask, msa_mask)
            if _K.chunking(z.shape[1]):
                mx.eval(*_K.guard_first(z, m))
            else:  # pipelined submission: the GPU runs this layer while Python builds the next (kernels.submit)
                _K.submit(z, m)
        return z


class MSALayer(nn.Module):
    """One MSA-module block."""

    def __init__(
        self,
        msa_s: int,
        token_z: int,
        msa_dropout: float,
        z_dropout: float,
        pairwise_head_width: int = 32,
        pairwise_num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.msa_dropout = msa_dropout
        self.msa_transition = Transition(dim=msa_s, hidden=msa_s * 4)
        self.pair_weighted_averaging = PairWeightedAveraging(c_m=msa_s, c_z=token_z, c_h=32, num_heads=8)
        self.pairformer_layer = PairformerNoSeqLayer(
            token_z=token_z,
            dropout=z_dropout,
            pairwise_head_width=pairwise_head_width,
            pairwise_num_heads=pairwise_num_heads,
        )
        self.outer_product_mean = OuterProductMean(c_in=msa_s, c_hidden=32, c_out=token_z)

    def __call__(
        self,
        z: mx.array,
        m: mx.array,
        token_mask: mx.array,
        msa_mask: mx.array,
        chunk_heads_pwa: bool = False,
        chunk_size_transition_z=None,
        chunk_size_transition_msa=None,
        chunk_size_outer_product=None,
        chunk_size_tri_attn=None,
        use_kernels: bool = False,
    ) -> tuple[mx.array, mx.array]:
        # chunk_* / use_kernels accepted for call-compat; ignored.
        m = _K.radd(m, self.pair_weighted_averaging(m, z, token_mask))
        m = _K.radd(m, self.msa_transition(m))
        z = _K.radd(z, self.outer_product_mean(m, msa_mask))
        z = self.pairformer_layer(z, token_mask)
        return z, m


# --------------------------------------------------------------------------------------
# Output heads
# --------------------------------------------------------------------------------------
class BFactorModule(nn.Module):
    """B-factor histogram head."""

    def __init__(self, token_s: int, num_bins: int) -> None:
        super().__init__()
        self.bfactor = nn.Linear(token_s, num_bins)
        self.num_bins = num_bins

    def __call__(self, s: mx.array) -> mx.array:
        return self.bfactor(s)


class DistogramModule(nn.Module):
    """Distogram head."""

    def __init__(self, token_z: int, num_bins: int, num_distograms: int = 1) -> None:
        super().__init__()
        self.distogram = nn.Linear(token_z, num_distograms * num_bins)
        self.num_distograms = num_distograms
        self.num_bins = num_bins

    def __call__(self, z: mx.array) -> mx.array:
        z = z + z.swapaxes(1, 2)
        return self.distogram(z).reshape(z.shape[0], z.shape[1], z.shape[2], self.num_distograms, self.num_bins)
