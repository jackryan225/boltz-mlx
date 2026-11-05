"""MLX port of boltz.model.layers.pair_averaging (inference, non-chunked path)."""

import mlx.core as mx
import mlx.nn as nn

from boltz_mlx import kernels as _K
from boltz_mlx.layers.norm import LayerNorm as _FastLN


class PairWeightedAveraging(nn.Module):
    """Pair weighted averaging layer."""

    def __init__(self, c_m: int, c_z: int, c_h: int, num_heads: int, inf: float = 1e6) -> None:
        super().__init__()
        self.c_m = c_m
        self.c_z = c_z
        self.c_h = c_h
        self.num_heads = num_heads
        self.inf = inf

        self.norm_m = _FastLN(c_m)
        self.norm_z = _FastLN(c_z)

        self.proj_m = nn.Linear(c_m, c_h * num_heads, bias=False)
        self.proj_g = nn.Linear(c_m, c_h * num_heads, bias=False)
        self.proj_z = nn.Linear(c_z, num_heads, bias=False)
        self.proj_o = nn.Linear(c_h * num_heads, c_m, bias=False)

    def __call__(self, m: mx.array, z: mx.array, mask: mx.array, chunk_heads: bool = False) -> mx.array:
        """m: (B, S, N, c_m); z: (B, N, N, c_z); mask: (B, N, N) -> (B, S, N, c_m).

        `chunk_heads` is accepted for call-compatibility and ignored (memory-only option).
        """
        B, S, N, _ = m.shape
        if _K.f16() and B == 1 and self.c_h == 32 and m.dtype == mx.float32 and z.dtype == mx.float32 \
                and _K.msa_kernel("PWA"):
            return self._f16_fused(m, z, mask)
        m = self.norm_m(m)

        v = self.proj_m(m).reshape(B, S, N, self.num_heads, self.c_h)
        _K.record("pwa.proj_m", v)
        v = v.transpose(0, 3, 1, 2, 4)  # (B, H, S, N, c_h)

        if _K.f16():
            b = _K.guard("pwa.proj_z", _K.lin16(self.norm_z.f16(z), self.proj_z)).transpose(0, 3, 1, 2)
        else:
            b = self.proj_z(self.norm_z(z)).transpose(0, 3, 1, 2)  # (B, H, N, N)
        b = b.astype(mx.float32) + (1 - mask[:, None]) * -self.inf
        w = mx.softmax(b, axis=-1)

        g = mx.sigmoid(self.proj_g(m))  # (B, S, N, H*c_h)

        # o[b,h,s,i,d] = sum_j w[b,h,i,j] * v[b,h,s,j,d]
        o = mx.matmul(w[:, :, None], v)  # (B, H, S, N, c_h)
        _K.record("pwa.weighted_avg", o)
        o = o.transpose(0, 2, 3, 1, 4).reshape(B, S, N, self.num_heads * self.c_h)
        y = self.proj_o(g * o)
        _K.record("pwa.proj_o", y)
        return y

    def _f16_fused(self, m: mx.array, z: mx.array, mask: mx.array) -> mx.array:
        """f16 mode: kernels_msa.pwa_weights computes softmax_j(LN(z) proj_z + mask) straight from z (fp32 LN and
        logits; the fp16 proj_z weight copy is the one the MLX f16 path uses, so the fp16 weight set is unchanged), and
        kernels_msa.pwa_avg the weighted average over tokens j (fp32 operands and accumulation, numerics policy) with the
        sigmoid gate in its epilogue, reading v and g in their GEMM layout (no transposed copies). proj_m / proj_g /
        proj_o stay the fp32 MLX GEMMs."""
        from boltz_mlx import kernels_msa as _KM

        mn = self.norm_m(m)
        v = self.proj_m(mn)  # (1, S, N, H * c_h) fp32
        g = self.proj_g(mn)  # pre-sigmoid gate, fp32
        wz = _K.w16(self.proj_z, "_w16", self.proj_z.weight)
        w = _KM.pwa_weights(z, mask, self.norm_z.weight, self.norm_z.bias, self.norm_z.eps, wz, self.inf)
        og = _KM.pwa_avg(w, v, g, **_K.MSA_PWA_CFG)
        y = self.proj_o(og)
        _K.record("pwa.proj_o", y)
        return y
