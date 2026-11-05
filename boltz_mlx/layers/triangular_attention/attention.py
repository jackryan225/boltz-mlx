"""MLX port of boltz.model.layers.triangular_attention.attention (non-kernel, unchunked)."""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from boltz_mlx import kernels as _K
from boltz_mlx.layers.triangular_attention.primitives import Attention, LayerNorm


class TriangleAttention(nn.Module):
    """Triangle attention (AF2 Algorithms 12/13/14)."""

    def __init__(self, c_in: int, c_hidden: int, no_heads: int, starting: bool = True, inf: float = 1e9) -> None:
        super().__init__()
        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.starting = starting
        self.inf = inf

        self.layer_norm = LayerNorm(c_in)
        self.linear = nn.Linear(c_in, no_heads, bias=False)
        self.mha = Attention(c_in, c_in, c_in, c_hidden, no_heads)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        chunk_size: Optional[int] = None,
        use_kernels: bool = False,
        xn: Optional[mx.array] = None,
        keep_t: bool = False,
    ) -> mx.array:
        """x: [B, I, J, C_in]; mask: [B, I, J]. `chunk_size`/`use_kernels` are ignored."""
        if mask is None:
            mask = mx.ones(x.shape[:-1], dtype=x.dtype)

        if not self.starting:
            x = x.swapaxes(-2, -3)
            mask = mask.swapaxes(-1, -2)

        mha = self.mha
        if _K.f16():
            return self._f16(x, mask, mha, xn, keep_t)
        x = self.layer_norm(x)
        H, C = self.no_heads, mha.c_hidden
        w = getattr(self, "_w", None)
        if w is None:  # q | k | v | g | tri-bias projections of the normed pair rep
            w = self._w = mx.concatenate(
                [mha.linear_q.weight, mha.linear_k.weight, mha.linear_v.weight, mha.linear_g.weight, self.linear.weight], 0
            )
        HC = H * C
        if _K.USE_KERNELS and _K.chunking(x.shape[1]):
            # rows i attend only within their own row: chunk over i; tri bias from all rows (small)
            bias = (x @ w[4 * HC :].T).transpose(0, 3, 2, 1)  # [B, H, K, Q]
            wq = w[: 4 * HC]
            outs = []
            for i0 in range(0, x.shape[1], _K.CHUNK_ROWS):
                xc = x[:, i0 : i0 + _K.CHUNK_ROWS]
                o = mha.linear_o(_K.tri_attention_fused(xc @ wq.T, H, C, mask[:, i0 : i0 + _K.CHUNK_ROWS], self.inf, bias=bias))
                mx.eval(o)
                outs.append(o)
            x = mx.concatenate(outs, axis=1)
            if not self.starting:
                x = x.swapaxes(-2, -3)
            return x
        h = x @ w.T  # [B, I, J, 4*H*C + H]
        _K.record("triatt.proj_qkvg_bias", h)
        if _K.USE_KERNELS:
            core = _K.tri_attention_fused(h, H, C, mask, self.inf)
            _K.record("triatt.core_out", core)
            x = mha.linear_o(core)
            _K.record("triatt.out_proj", x)
            if not self.starting:
                x = x.swapaxes(-2, -3)
            return x
        B, I, J = x.shape[:3]
        q = h[..., :HC].reshape(B * I, J, H, C).transpose(0, 2, 1, 3)
        k = h[..., HC : 2 * HC].reshape(B * I, J, H, C).transpose(0, 2, 1, 3)
        v = h[..., 2 * HC : 3 * HC].reshape(B * I, J, H, C).transpose(0, 2, 1, 3)
        g = h[..., 3 * HC : 4 * HC]
        tri_bias = h[..., 4 * HC :].transpose(0, 3, 1, 2)[:, None]  # [B, 1, H, J, K]
        mask_bias = (self.inf * (mask - 1))[:, :, None, None, :]  # [B, I, 1, 1, K]
        bias = (tri_bias + mask_bias).reshape(B * I, H, J, J)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=C**-0.5, mask=bias)
        o = o.transpose(0, 2, 1, 3).reshape(B, I, J, HC)
        x = mha.linear_o(o * mx.sigmoid(g))

        if not self.starting:
            x = x.swapaxes(-2, -3)
        return x

    def _f16(self, x, mask, mha, xn=None, keep_t=False):
        """--precision f16: LN writes fp16; fused projection fp16; the MMA kernel reads fp16 h (softmax and
        accumulation fp32 in registers, tri bias fp32) and writes fp16; linear_o fp16.
        xn: the LayerNorm output already computed (fused add+LN; for the ending node already transposed).
        keep_t: return the ending node's output in the transposed layout (consumer reads it transposed)."""
        x = self.layer_norm.f16(x) if xn is None else xn
        H, C = self.no_heads, mha.c_hidden
        wh = _K.w16(self, "_wh", mha.linear_q.weight, mha.linear_k.weight, mha.linear_v.weight, mha.linear_g.weight,
                    self.linear.weight)
        HC = H * C
        J = x.shape[2]
        use_nax = _K.triatt_nax() and J >= 32
        if use_nax:
            from boltz_mlx import kernels_nax as _KN
        if _K.chunking(x.shape[1]):
            tb = x @ wh[4 * HC :].T  # [B, Q, K, H] (all rows: the tri bias is shared by every row chunk)
            bias = tb.transpose(0, 3, 1, 2) if use_nax else tb.transpose(0, 3, 2, 1)
            wq = wh[: 4 * HC]
            outs = []
            for i0 in range(0, x.shape[1], _K.CHUNK_ROWS):
                xc = x[:, i0 : i0 + _K.CHUNK_ROWS]
                hc = _K.guard("triatt.proj", xc @ wq.T)
                mc = mask[:, i0 : i0 + _K.CHUNK_ROWS]
                if use_nax:
                    core = _KN.tri_attention_nax(hc, H, C, mc, self.inf, bias=bias, **_K.triatt_cfg(J))
                else:
                    core = _K.tri_attention_fused(hc, H, C, mc, self.inf, bias=bias, out_dtype=mx.float16)
                o = _K.guard("triatt.out_proj", _K.lin16(core, mha.linear_o))
                mx.eval(o)
                outs.append(o)
            x = mx.concatenate(outs, axis=1)
        else:
            core = self._f16_core(x, mask, wh, use_nax)
            x = _K.guard("triatt.out_proj", _K.lin16(core, mha.linear_o))
            _K.record("triatt.out_proj", x)
        if not self.starting and not keep_t:
            x = x.swapaxes(-2, -3)
        return x

    def _f16_core(self, x, mask, wh, use_nax, tb=None):
        """Unchunked f16 attention core (before linear_o) on the fp16 LN output x: [B, I, J, H*C] fp16.
        tb: the tri bias already computed as [B, H, J, J] fp16 (R3-pair PAIRTB, NAX path only)."""
        H, C = self.no_heads, self.mha.c_hidden
        HC = H * C
        if use_nax:  # NAX flash core; projection split into q|k|v|g (512 cols) + tri bias (H cols)
            from boltz_mlx import kernels_nax as _KN

            if _K.nax("PAIRPROJ") and x.dtype == mx.float16 and x.shape[-1] % 32 == 0:
                from boltz_mlx import kernels_nax3 as _KN3

                h = _K.guard("triatt.proj", _KN3.pair_proj16(x, wh[: 4 * HC]))
            else:
                h = _K.guard("triatt.proj", x @ wh[: 4 * HC].T)
            if tb is None:
                tb = _K.guard("triatt.proj_bias", x @ wh[4 * HC :].T).transpose(0, 3, 1, 2)
            _K.record("triatt.proj_qkvg_bias", h)
            core = _KN.tri_attention_nax(h, H, C, mask, self.inf, bias=tb, **_K.triatt_cfg(h.shape[2]))
        else:
            h = _K.guard("triatt.proj", x @ wh.T)
            _K.record("triatt.proj_qkvg_bias", h)
            core = _K.tri_attention_fused(h, H, C, mask, self.inf, out_dtype=mx.float16)
        _K.record("triatt.core_out", core)
        return core

    def tb_weight16(self):
        """The tri-bias projection's fp16 weight [H, C_in] (rows of the f16 path's fused projection weight)."""
        mha = self.mha
        wh = _K.w16(self, "_wh", mha.linear_q.weight, mha.linear_k.weight, mha.linear_v.weight, mha.linear_g.weight,
                    self.linear.weight)
        return wh[4 * self.no_heads * mha.c_hidden:]

    def tb_ok(self, xn) -> bool:
        """A precomputed tri bias is consumed only by the NAX core (J >= 32)."""
        return _K.triatt_nax() and xn.shape[2] >= 32

    def f16_addln(self, z, mask, xn, ln_next, trans_out: bool = False, tb=None, w_tb=None):
        """(z + self(z), ln_next(z + self(z)) fp16 [transposed with trans_out]) for --precision f16, unchunked, from
        xn = this module's fp16 LayerNorm output (for the ending node already in the transposed layout): the core as
        _f16, then linear_o, the residual add (read back transposed for the ending node) and the next LayerNorm in one
        NAX kernel (kernels_nax3.linear_addln; pair_block_f16_epi). tb: this module's tri bias precomputed by the
        producer of xn; w_tb: also return the next module's tri bias from the LN output (R3-pair PAIRTB)."""
        from boltz_mlx import kernels_nax3 as _KN3

        if mask is None:
            mask = mx.ones(z.shape[:-1], dtype=z.dtype)
        if not self.starting:
            mask = mask.swapaxes(-1, -2)
        mha = self.mha
        wh = _K.w16(self, "_wh", mha.linear_q.weight, mha.linear_k.weight, mha.linear_v.weight, mha.linear_g.weight,
                    self.linear.weight)
        use_nax = _K.triatt_nax() and xn.shape[2] >= 32
        core = self._f16_core(xn, mask, wh, use_nax, tb=tb if use_nax else None)
        w = _K.w16(mha.linear_o, "_w16", mha.linear_o.weight)
        b = mha.linear_o.get("bias") if "bias" in mha.linear_o else None
        return _KN3.linear_addln(core, w, z, ln_next, bias=b, trans_y=not self.starting, trans_out=trans_out,
                                 w_tb=w_tb, **_KN3.site_cfg("lin" if self.starting else "lin_end", z.shape[1]))


TriangleAttentionStartingNode = TriangleAttention


class TriangleAttentionEndingNode(TriangleAttention):
    """Triangle attention around the ending node."""

    def __init__(self, c_in: int, c_hidden: int, no_heads: int, inf: float = 1e9) -> None:
        super().__init__(c_in, c_hidden, no_heads, starting=False, inf=inf)
