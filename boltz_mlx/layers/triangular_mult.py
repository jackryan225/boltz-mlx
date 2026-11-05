"""MLX port of boltz.model.layers.triangular_mult (non-kernel path, inference only)."""

import mlx.core as mx
import mlx.nn as nn

from boltz_mlx import kernels as _K
from boltz_mlx.layers.norm import LayerNorm as _FastLN


class _TriangleMultiplication(nn.Module):
    outgoing: bool = True

    def __init__(self, dim: int = 128) -> None:
        super().__init__()
        self.norm_in = _FastLN(dim, eps=1e-5)
        self.p_in = nn.Linear(dim, 2 * dim, bias=False)
        self.g_in = nn.Linear(dim, 2 * dim, bias=False)

        self.norm_out = _FastLN(dim)
        self.p_out = nn.Linear(dim, dim, bias=False)
        self.g_out = nn.Linear(dim, dim, bias=False)

    def __call__(self, x: mx.array, mask: mx.array, use_kernels: bool = False, xn: mx.array | None = None) -> mx.array:
        """x: (B, N, N, D); mask: (B, N, N). `use_kernels` is ignored (no cuequivariance)."""
        D = self.p_out.weight.shape[0]
        B, N, M, _ = x.shape
        if B == 1 and _K.f16():
            return self._f16_b1(self.norm_in.f16(x) if xn is None else xn, mask, D, N, M)
        xn = self.norm_in(x)
        if B == 1:
            return self._fast_b1(xn, mask, D, N, M)
        w = getattr(self, "_w_in", None)
        if w is None:  # p_in | g_in | g_out fused into one projection of norm_in(x)
            w = self._w_in = mx.concatenate([self.p_in.weight, self.g_in.weight, self.g_out.weight], 0)
        h = xn @ w.T
        x = h[..., : 2 * D] * mx.sigmoid(h[..., 2 * D : 4 * D])
        g_out = h[..., 4 * D :]

        x = x * mask[..., None]

        a, b = x[..., :D], x[..., D:]  # each (B, N, N, D)
        if self.outgoing:
            x = mx.matmul(a.transpose(0, 3, 1, 2), b.transpose(0, 3, 2, 1))
        else:
            x = mx.matmul(a.transpose(0, 3, 2, 1), b.transpose(0, 3, 1, 2))
        x = x.transpose(0, 2, 3, 1)  # (B, i, j, D)

        return self.p_out(self.norm_out(x)) * mx.sigmoid(g_out)

    def _fast_b1(self, xn, mask, D, N, M):
        """Channel-major layout: projections computed as W @ x^T give (2D, N*M) directly, so the
        per-channel triangle matmuls need no transposes; gate*mask fused by mx.compile."""
        w = getattr(self, "_w_t", None)
        if w is None:
            self._w_t = w = mx.concatenate([self.p_in.weight, self.g_in.weight], 0)
        xf = xn.reshape(N * M, -1)
        mf = mask.reshape(1, N * M)
        cb = _K.CHUNK_CH if _K.chunking(N) else D
        outs = []
        for c0 in range(0, D, cb):  # channel blocks are independent (channel-wise triangle products)
            c1 = min(D, c0 + cb)
            if cb == D:
                wc = w
            else:
                rows = [slice(c0 + k * D, c1 + k * D) for k in range(4)]
                wc = mx.concatenate([w[r] for r in rows], 0)
            n = c1 - c0
            h_t = wc @ xf.T  # (4n, N*M): rows [p_a | p_b | g_a | g_b]
            _K.record("trimult.proj_pg", h_t)
            ab = _gate_mask(h_t[: 2 * n], h_t[2 * n :], mf).reshape(2 * n, N, M)
            _K.record("trimult.ab_gated", ab)
            a, b = ab[:n], ab[n:]
            if self.outgoing:  # x[i,j,d] = sum_k a[d,i,k] b[d,j,k]
                o = mx.matmul(a, b.transpose(0, 2, 1))
            else:  # x[i,j,d] = sum_k a[d,k,i] b[d,k,j]
                o = mx.matmul(a.transpose(0, 2, 1), b)
            _K.record("trimult.contraction_" + ("out" if self.outgoing else "in"), o)
            if cb != D:
                mx.eval(o)
            outs.append(o)
        o = outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=0)
        o = o.reshape(D, N * M).T  # (N*M, D)
        g_out = xf @ self.g_out.weight.T
        _K.record("trimult.g_out", g_out)
        p_o = self.p_out(self.norm_out(o))
        _K.record("trimult.p_out", p_o)
        out = p_o * mx.sigmoid(g_out)
        return out.reshape(1, N, M, D)

    def _f16_b1(self, xn, mask, D, N, M):
        """--precision f16 (B=1): fp16 projections; gate*mask*2^-K in fp32 with fp16 output; fp16 contraction
        (fp32 accumulate) of the pre-scaled operands; norm_out undoes the 2^-2K scale exactly via eps*2^-4K;
        p_out / g_out fp16; output gate in fp32 with fp16 output."""
        kexp = _K.TRIMULT_K
        o, xf = self._f16_contract(xn, mask, D, N, M)
        o = o.reshape(D, N * M).T
        g_out = _K.guard("trimult.g_out", xf @ _K.w16(self, "_w_gouth", self.g_out.weight).T)
        _K.record("trimult.g_out", g_out)
        p_o = _K.guard("trimult.p_out", _K.lin16(self.norm_out.f16(o, eps_scale=2.0 ** (-4 * kexp)), self.p_out))
        _K.record("trimult.p_out", p_o)
        return _K._gate16(p_o, g_out).reshape(1, N, M, D)

    def f16_addln(self, z, mask, xn, ln_next, w_tb=None):
        """(z + self(z), ln_next(z + self(z)) fp16) for --precision f16, B = 1, from xn = norm_in(z) in fp16: the
        projection and contraction as _f16_b1, then norm_out, p_out, g_out, the gate, the residual add and the next
        LayerNorm in one NAX kernel (kernels_nax3.trimul_tail_addln; pair_block_f16_epi). w_tb: also return the next
        triangle attention's tri bias from the LN output (R3-pair PAIRTB)."""
        from boltz_mlx import kernels_nax3 as _KN3

        _, N, M, D = z.shape
        o, xf = self._f16_contract(xn, mask, D, N, M)
        if _K.nax("TAIL3") and _KN3.PAIR_SITE_CFG.get("tail3") is not None:  # R5: low-register fused tail (bitwise)
            return _KN3.trimul_tail3_addln(
                o.reshape(D, N * M), self.norm_out, 2.0 ** (-4 * _K.TRIMULT_K),
                _K.w16(self.p_out, "_w16", self.p_out.weight), xf, _K.w16(self, "_w_gouth", self.g_out.weight), z,
                ln_next, w_tb=w_tb, **_KN3.site_cfg("tail3", N))
        if _K.nax("TAIL2") and _KN3.PAIR_SITE_CFG.get("tail2") is not None:  # R4-pair: transpose+LN, then 2-GEMM tail
            return _KN3.trimul_tail2_addln(
                o.reshape(D, N * M), self.norm_out, 2.0 ** (-4 * _K.TRIMULT_K),
                _K.w16(self.p_out, "_w16", self.p_out.weight), xf, _K.w16(self, "_w_gouth", self.g_out.weight), z,
                ln_next, w_tb=w_tb)
        return _KN3.trimul_tail_addln(
            o.reshape(D, N * M), self.norm_out, 2.0 ** (-4 * _K.TRIMULT_K), _K.w16(self.p_out, "_w16", self.p_out.weight),
            xf, _K.w16(self, "_w_gouth", self.g_out.weight), z, ln_next, w_tb=w_tb, **_KN3.PAIR_SITE_CFG["tail"])

    def _f16_contract(self, xn, mask, D, N, M):
        """f16 projection (+ gate, mask, 2^-K) and per-channel contraction: (o [D, N, M] fp16 scaled by 2^-2K, xf)."""
        wh = _K.w16(self, "_w_th", self.p_in.weight, self.g_in.weight)
        kexp = _K.TRIMULT_K
        sc = mx.array(2.0 ** -kexp, mx.float32)
        xf = xn.reshape(N * M, -1)
        mf = mask.reshape(1, N * M).astype(mx.float32)
        cb = _K.CHUNK_CH if _K.chunking(N) else D
        use_nax = _K.nax("TRIMUL") and N * M >= _K.NAX_TRIMUL_CFG["tp"] and xf.dtype == mx.float16 \
            and (2 * min(cb, D)) % _K.NAX_TRIMUL_CFG["tc"] == 0
        if use_nax:
            from boltz_mlx import kernels_nax2 as _KN2
        outs = []
        for c0 in range(0, D, cb):
            c1 = min(D, c0 + cb)
            if cb == D:
                wc = wh
            else:
                wc = mx.concatenate([wh[slice(c0 + k * D, c1 + k * D)] for k in range(4)], 0)
            n = c1 - c0
            if use_nax:  # projection + gate + mask + 2^-K + channel-major fp16 store in one NAX kernel
                ab = _K.guard("trimult.proj", _KN2.trimul_proj_nax(xf, wc, mf.reshape(-1), 2.0 ** -kexp,
                                                                   **_K.NAX_TRIMUL_CFG)).reshape(2 * n, N, M)
            else:
                h_t = _K.guard("trimult.proj", wc @ xf.T)  # (4n, N*M) fp16
                _K.record("trimult.proj_pg", h_t)
                ab = _K._gate_mask_scale16(h_t[: 2 * n], h_t[2 * n :], mf, sc).reshape(2 * n, N, M)
            _K.record("trimult.ab_gated", ab)
            a, b = ab[:n], ab[n:]
            if self.outgoing:
                o = mx.matmul(a, b.transpose(0, 2, 1))
            else:
                o = mx.matmul(a.transpose(0, 2, 1), b)
            o = _K.guard("trimult.contraction", o)
            _K.record("trimult.contraction_" + ("out" if self.outgoing else "in"), o)
            if cb != D:
                mx.eval(o)
            outs.append(o)
        o = outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=0)
        return o, xf


@mx.compile
def _gate_mask(p, g, m):
    return p * mx.sigmoid(g) * m


class TriangleMultiplicationOutgoing(_TriangleMultiplication):
    outgoing = True


class TriangleMultiplicationIncoming(_TriangleMultiplication):
    outgoing = False
