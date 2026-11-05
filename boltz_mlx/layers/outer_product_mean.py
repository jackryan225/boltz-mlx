"""MLX port of boltz.model.layers.outer_product_mean (inference, non-chunked path)."""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from boltz_mlx import kernels as _K
from boltz_mlx.layers.norm import LayerNorm as _FastLN


class OuterProductMean(nn.Module):
    """Outer product mean layer."""

    def __init__(self, c_in: int, c_hidden: int, c_out: int) -> None:
        super().__init__()
        self.c_hidden = c_hidden
        self.norm = _FastLN(c_in)
        self.proj_a = nn.Linear(c_in, c_hidden, bias=False)
        self.proj_b = nn.Linear(c_in, c_hidden, bias=False)
        self.proj_o = nn.Linear(c_hidden * c_hidden, c_out)

    def __call__(self, m: mx.array, mask: mx.array, chunk_size: Optional[int] = None) -> mx.array:
        """m: (B, S, N, c_in); mask: (B, S, N) -> (B, N, N, c_out)."""
        B, S, N, _ = m.shape
        c = self.c_hidden
        if _K.f16() and S * c > _K.OPM_REASSOC_MAX_K and B == 1 and c == 32 and m.dtype == mx.float32 \
                and _K.msa_kernel("OPM"):
            return self._f16_fused(m, mask)
        mask = mask.astype(m.dtype)[..., None]  # (B, S, N, 1)

        m = self.norm(m)
        a = self.proj_a(m) * mask  # (B, S, N, c)
        b = self.proj_b(m) * mask

        # pair mask: sum_s mask_si * mask_sj, clamp(min=1)
        pm = mask[:, :, None, :, :] * mask[:, :, :, None, :]  # (B, S, N, N, 1)
        num_mask = mx.maximum(pm.sum(axis=1), 1)  # (B, N, N, 1)

        if _K.f16() and S * c <= _K.OPM_REASSOC_MAX_K:
            return self._f16_reassoc(a, b, num_mask, B, S, N, c)

        # z[b,i,j,c,d] = sum_s a[b,s,i,c] * b[b,s,j,d]
        a2 = a.transpose(0, 2, 3, 1).reshape(B, N * c, S)  # (B, N*c, S)
        b2 = b.reshape(B, S, N * c)  # (B, S, N*c)
        step = _K.CHUNK_ROWS if _K.chunking(N) else N  # rows i are independent
        outs = []
        for i0 in range(0, N, step):
            i1 = min(N, i0 + step)
            z = mx.matmul(a2[:, i0 * c : i1 * c], b2).reshape(B, i1 - i0, c, N, c)  # (B, i, c, j, d)
            z = z.transpose(0, 1, 3, 2, 4).reshape(B, i1 - i0, N, c * c)
            _K.record("opm.outer_sum", z)
            if _K.f16():
                z = _K.guard("opm.proj_o", _K.lin16(_div16(z, num_mask[:, i0:i1]), self.proj_o))
            else:
                z = self.proj_o(z / num_mask[:, i0:i1])
            _K.record("opm.proj_o", z)
            if step != N:
                mx.eval(z)
            outs.append(z)
        return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=1)

    def _f16_fused(self, m, mask):
        """f16 mode, many MSA rows: the outer sum over MSA rows runs in one kernel (kernels_msa.opm_outer: fp32
        operands, fp32 simdgroup-MMA accumulation, numerics policy) whose epilogue divides by num_mask in fp32 and
        writes z / num in fp16, pair-major — the operand of the fp16 proj_o GEMM, rounded exactly where the MLX f16
        path rounds it (_div16). a and b come from one fp32 GEMM (concatenated proj_a / proj_b weights); the MSA mask
        is applied in the kernel's loads; num_mask = max(mask^T mask, 1) is one small fp32 GEMM (exact integer
        counts) instead of the (S, N, N) broadcast product."""
        from boltz_mlx import kernels_msa as _KM

        _, S, N, _ = m.shape
        c = self.c_hidden
        maskf = mask.astype(mx.float32).reshape(S, N)
        wab = self.__dict__.get("_wab")
        if wab is None:  # fp32 [W_a; W_b] (fp32 parameters: the OPM operands stay fp32)
            wab = mx.concatenate([self.proj_a.weight, self.proj_b.weight], 0).astype(mx.float32)
            mx.eval(wab)
            self.__dict__["_wab"] = wab
        ab = self.norm(m) @ wab.T  # (1, S, N, 2c) fp32: a = ab[..., :c], b = ab[..., c:] (unmasked)
        num = mx.maximum(maskf.T @ maskf, 1.0)  # (N, N) = clamp(sum_s mask_si mask_sj, min=1)
        step = _K.CHUNK_ROWS if _K.chunking(N) else N  # rows i are independent
        outs = []
        for i0 in range(0, N, step):
            ni = min(N, i0 + step) - i0
            if _K.msa_opm_nax() and "bias" in self.proj_o:  # proj_o in the kernel epilogue (NAX hardware only)
                wo = _K.w16(self.proj_o, "_w16", self.proj_o.weight)
                z = _KM.opm_outer_proj(ab, maskf, num, wo, self.proj_o.bias, i0, ni, **_K.MSA_OPM_NAX_CFG)
                z = _K.guard("opm.proj_o", z.reshape(1, ni, N, -1))
            else:
                z16 = _KM.opm_outer(ab, maskf, num, i0, ni, **_K.MSA_OPM_CFG)  # (ni * N, c * c) fp16
                z = _K.guard("opm.proj_o", _K.lin16(z16.reshape(1, ni, N, c * c), self.proj_o))
            _K.record("opm.proj_o", z)
            if step != N:
                mx.eval(z)
            outs.append(z)
        return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=1)

    def _f16_reassoc(self, a, b, num_mask, B, S, N, c):
        """f16 mode, few MSA rows (S*c <= OPM_REASSOC_MAX_K): proj_o(sum_s a_i (x) b_j / num) computed as
        (sum_{s,c} a[s,i,c] * B'[s,j,c,:]) / num + bias with B'[s,j,c,o] = sum_d b[s,j,d] W[o,c,d] — the same linear
        map re-associated, so the N^2 x c^2 outer-product tensor is never formed (S=1: ~30x fewer FLOPs). All fp32
        (the MSA reduction stays fp32, numerics policy). W_o is the fp16 copy w16 gives the lin16 path, upcast: proj_o
        stays in the fp16 weight set (committed manifest) and the mixed checkpoint stays bit-identical to the runtime
        cast on this path too."""
        w = _K.w16(self.proj_o, "_w16", self.proj_o.weight).astype(mx.float32)
        co = w.shape[0]
        bp = mx.einsum("bsjd,ocd->bscjo", b.astype(mx.float32), w.reshape(co, c, c))  # (B, S, c, N, co)
        b2 = bp.reshape(B, S * c, N * co)
        a2 = a.astype(mx.float32).transpose(0, 2, 1, 3).reshape(B, N, S * c)  # (B, i, (s, c))
        bias = self.proj_o.get("bias") if "bias" in self.proj_o else None
        step = _K.CHUNK_ROWS if _K.chunking(N) else N
        outs = []
        for i0 in range(0, N, step):
            i1 = min(N, i0 + step)
            z = (a2[:, i0:i1] @ b2).reshape(B, i1 - i0, N, co) / num_mask[:, i0:i1]
            if bias is not None:
                z = z + bias.astype(mx.float32)
            _K.record("opm.proj_o", z)
            if step != N:
                mx.eval(z)
            outs.append(z)
        return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=1)


@mx.compile
def _div16(z, n):
    return (z / n).astype(mx.float16)
