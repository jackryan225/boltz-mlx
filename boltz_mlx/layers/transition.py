"""MLX port of boltz.model.layers.transition (inference only)."""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from boltz_mlx import kernels as _K
from boltz_mlx.layers.norm import LayerNorm as _FastLN


class Transition(nn.Module):
    """Perform a two-layer (SwiGLU-style) MLP."""

    def __init__(self, dim: int = 128, hidden: int = 512, out_dim: Optional[int] = None) -> None:
        super().__init__()
        if out_dim is None:
            out_dim = dim

        self.norm = _FastLN(dim, eps=1e-5)
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.fc2 = nn.Linear(dim, hidden, bias=False)
        self.fc3 = nn.Linear(hidden, out_dim, bias=False)
        self.silu = nn.SiLU()
        self.hidden = hidden

    def __call__(self, x: mx.array, chunk_size: Optional[int] = None, xn: Optional[mx.array] = None) -> mx.array:
        # `chunk_size` is accepted for call-compatibility with upstream; chunking is a
        # memory optimisation only and is not implemented (results are identical).
        if _K.f16():
            return self._f16(x, xn)
        w = getattr(self, "_w12", None)
        if w is None:
            w = self._w12 = mx.concatenate([self.fc1.weight, self.fc2.weight], 0)
        if _K.TRANSITION_IMPL == "fused" and _K.transition_supported(x, w, self.fc3.weight):
            # One kernel for LN + both GEMMs + SwiGLU: no [rows, 2H]/[rows, H] intermediates, so no row
            # chunking is needed at large N either (peak extra memory = the output only).
            wp = getattr(self, "_wp", None)
            if wp is None:
                wp = self._wp = _K.pack_transition_weights(w, self.fc3.weight)
            return _K.transition_fused(x, self.norm.weight, self.norm.bias, w, self.fc3.weight,
                                       eps=self.norm.eps, packed=wp)
        if x.ndim == 4 and _K.chunking(x.shape[1]):
            outs = []
            for i0 in range(0, x.shape[1], _K.CHUNK_ROWS):
                o = self.fc3(_swiglu(self.norm(x[:, i0 : i0 + _K.CHUNK_ROWS]) @ w.T, self.hidden))
                mx.eval(o)
                outs.append(o)
            return mx.concatenate(outs, axis=1)
        h = self.norm(x) @ w.T
        _K.record(f"transition{x.shape[-1]}.up", h)
        y = self.fc3(_swiglu(h, self.hidden))
        _K.record(f"transition{x.shape[-1]}.down", y)
        return y

    def _f16(self, x, xn=None):
        """--precision f16: LN writes fp16, both GEMMs fp16 in/out, SwiGLU in fp32 with fp16 output."""
        wh = _K.w16(self, "_w12h", self.fc1.weight, self.fc2.weight)
        w3 = _K.w16(self, "_w3h", self.fc3.weight)

        k = _K.TRANSITION_DOWN_K.get(x.shape[-1], 0)

        cfg = self._nax_cfg(x)

        def body(xx, xxn=None):
            xx16 = self.norm.f16(xx) if xxn is None else xxn
            if cfg is not None and xx16.size // xx16.shape[-1] >= cfg["tm"]:
                # NAX: up GEMM, SwiGLU (x 2^-k) and down GEMM in one kernel, hidden in registers
                from boltz_mlx import kernels_nax2 as _KN2

                y = _K.guard("transition.down", _KN2.transition_nax(xx16, wh, w3, hidden_scale=2.0 ** -k, **cfg))
                _K.record(f"transition{x.shape[-1]}.down", y)
                return y
            h = _K.guard("transition.up", xx16 @ wh.T)
            _K.record(f"transition{x.shape[-1]}.up", h)
            if k:
                a = _K._swiglu16_scale(h[..., : self.hidden], h[..., self.hidden :], mx.array(2.0 ** -k, mx.float32))
            else:
                a = _K.swiglu16(h, self.hidden)
            y = _K.guard("transition.down", a @ w3.T)
            _K.record(f"transition{x.shape[-1]}.down", y)
            return y

        if x.ndim == 4 and _K.chunking(x.shape[1]):
            outs = []
            for i0 in range(0, x.shape[1], _K.CHUNK_ROWS):
                o = body(x[:, i0 : i0 + _K.CHUNK_ROWS])
                mx.eval(o)
                outs.append(o)
            y = mx.concatenate(outs, axis=1)
        else:
            y = body(x, xn)
        return _K.Scaled(y, 2.0 ** k) if k else y

    def _nax_cfg(self, x):
        """NAX transition config for this module / input (f16 path), or None."""
        D = x.shape[-1]
        if not _K.nax("TRANSITION") or D not in _K.NAX_TRANSITION_CFG:
            return None
        cfg = _K.NAX_TRANSITION_CFG[D]
        if self.hidden % cfg["hb"] or self.fc3.weight.shape[0] != D:
            return None
        return cfg

    def f16_resid(self, z, xn):
        """z + transition(z) in fp32 from the fp16 LN output xn (pair_block_f16), with the residual add fused into
        the NAX kernel (bitwise equal to radd(z, transition(z, xn=xn))). None if the NAX path does not apply (or in
        per-op guard mode, which needs the fp16 output)."""
        cfg = self._nax_cfg(z)
        if cfg is None or _K.GUARD_MODE == "op" or _K.TRANSITION_DOWN_K.get(z.shape[-1], 0) \
                or z.dtype != mx.float32 or z.size // z.shape[-1] < cfg["tm"]:
            return None
        from boltz_mlx import kernels_nax2 as _KN2

        wh = _K.w16(self, "_w12h", self.fc1.weight, self.fc2.weight)
        w3 = _K.w16(self, "_w3h", self.fc3.weight)
        if self._trans3(z):  # R6-trans
            from boltz_mlx import kernels_nax3 as _KN3

            return _KN3.transition3(xn, wh, w3, z)[0]
        return _KN2.transition_nax(xn, wh, w3, resid=z, **cfg)

    def _trans3(self, z) -> bool:
        if z.shape[-1] != 128 or not _K.nax("TRANS3"):
            return False
        from boltz_mlx import kernels_nax3 as _KN3

        return _KN3.PAIR_SITE_CFG.get("trans3") is not None

    def f16_resid_lnx(self, z, xn, next_ln=None, attention=None):
        """f16_resid plus, from the new rows (kernels_nax3.transition_lnx, R3-pair PAIRLNX): next_ln(z_new) in fp16
        (the next layer's tri_mul_out.norm_in output) and attention's pair bias half(proj_z(z_new)) as [1, H, N, N].
        Returns (z_new, xn_next | None, pair_bias | None); z_new bitwise == f16_resid."""
        from boltz_mlx import kernels_nax3 as _KN3

        cfg = self._nax_cfg(z)
        wh = _K.w16(self, "_w12h", self.fc1.weight, self.fc2.weight)
        w3 = _K.w16(self, "_w3h", self.fc3.weight)
        ln_s = w_s = None
        if attention is not None and attention.compute_pair_bias:
            ln_s, lin = attention.proj_z
            w_s = _K.w16(lin, "_w16", lin.weight)  # lin16's fp16 weight (same mixed-checkpoint entry)
        if self._trans3(z):  # R6-trans
            return _KN3.transition3(xn, wh, w3, z, next_ln=next_ln, ln_s=ln_s, w_s=w_s)
        return _KN3.transition_lnx(xn, wh, w3, z, next_ln=next_ln, ln_s=ln_s, w_s=w_s, **cfg)


@mx.compile
def _swiglu_impl(a, b):
    return nn.silu(a) * b


def _swiglu(h, hidden):
    return _swiglu_impl(h[..., :hidden], h[..., hidden:])
