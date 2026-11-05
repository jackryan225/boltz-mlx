"""MLX port of boltz/model/modules/transformersv2.py (inference only)."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from boltz_mlx import kernels as _K
from boltz_mlx import kernels_difftx as _DT
from boltz_mlx import kernels_nax3 as _N3
from boltz_mlx.kernels import layer_norm as _ln_kernel
from boltz_mlx.layers.attentionv2 import AttentionPairBias
from boltz_mlx.layers.norm import LayerNorm as _FastLN


class SwiGLU(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        x, gates = mx.split(x, 2, axis=-1)
        return nn.silu(gates) * x


class Sigmoid(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return mx.sigmoid(x)


class AdaLN(nn.Module):
    """Algorithm 26"""

    def __init__(self, dim: int, dim_single_cond: int) -> None:
        super().__init__()
        self.a_norm = _FastLN(dim, affine=False)
        self.s_norm = _FastLN(dim_single_cond, affine=True, bias=False)
        self.s_scale = nn.Linear(dim_single_cond, dim)
        self.s_bias = nn.Linear(dim_single_cond, dim, bias=False)

    def __call__(self, a: mx.array, s: mx.array) -> mx.array:
        a = self.a_norm(a)
        s = self.s_norm(s)
        return mx.sigmoid(self.s_scale(s)) * a + self.s_bias(s)


class ConditionedTransitionBlock(nn.Module):
    """Algorithm 25"""

    def __init__(self, dim_single: int, dim_single_cond: int, expansion_factor: int = 2) -> None:
        super().__init__()
        self.adaln = AdaLN(dim_single, dim_single_cond)
        dim_inner = int(dim_single * expansion_factor)
        self.swish_gate = [nn.Linear(dim_single, dim_inner * 2, bias=False), SwiGLU()]
        self.a_to_b = nn.Linear(dim_single, dim_inner, bias=False)
        self.b_to_a = nn.Linear(dim_inner, dim_single, bias=False)
        self.output_projection = [nn.Linear(dim_single_cond, dim_single), Sigmoid()]

    def __call__(self, a: mx.array, s: mx.array) -> mx.array:
        a = self.adaln(a, s)
        h = a
        for f in self.swish_gate:
            h = f(h)
        b = h * self.a_to_b(a)
        o = s
        for f in self.output_projection:
            o = f(o)
        return o * self.b_to_a(b)


class DiffusionTransformerLayer(nn.Module):
    """Algorithm 23 (one layer)"""

    def __init__(
        self, heads: int, dim: int = 384, dim_single_cond: int | None = None, post_layer_norm: bool = False
    ) -> None:
        super().__init__()
        dim_single_cond = dim if dim_single_cond is None else dim_single_cond
        self.adaln = AdaLN(dim, dim_single_cond)
        self.pair_bias_attn = AttentionPairBias(c_s=dim, num_heads=heads, compute_pair_bias=False)
        # torch registers the same Linear as both `output_projection_linear` and `output_projection.0`;
        # share the object so both flattened parameter names exist (and load identically).
        self.output_projection_linear = nn.Linear(dim_single_cond, dim)
        self.output_projection = [self.output_projection_linear, Sigmoid()]
        self.transition = ConditionedTransitionBlock(dim_single=dim, dim_single_cond=dim_single_cond)
        self.post_layer_norm = post_layer_norm
        if post_layer_norm:
            self.post_lnorm = _FastLN(dim)

    def __call__(self, a, s, bias=None, mask=None, to_keys=None, multiplicity=1, cond=None, attn_bias=None):
        if cond is not None:
            return self.fast(a, cond, bias, mask, to_keys, multiplicity, attn_bias)
        b = self.adaln(a, s)

        k_in = b
        if to_keys is not None:
            k_in = to_keys(b)
            mask = mx.squeeze(to_keys(mask[..., None]), -1)

        b = self.pair_bias_attn(s=b, z=bias, mask=mask, multiplicity=multiplicity, k_in=k_in)

        b = mx.sigmoid(self.output_projection_linear(s)) * b
        a = a + b
        a = a + self.transition(a, s)
        if self.post_layer_norm:
            a = self.post_lnorm(a)
        return a

    def full_attn_bias(self, bias, mask, to_keys, multiplicity):
        """Pair bias (B,N,M,H) -> additive (B*mult, H, N, M) incl. key mask; step-invariant."""
        if to_keys is not None:
            mask = mx.squeeze(to_keys(mask[..., None]), -1)
        zb = bias.transpose(0, 3, 1, 2)
        if multiplicity > 1:
            zb = mx.repeat(zb, multiplicity, axis=0)
        return zb + (1 - mask[:, None, None, :].astype(mx.float32)) * -self.pair_bias_attn.inf

    # ---- fast path: fused projections + precomputed s-conditioning (same math) ----
    def _fused(self):
        dt = _K.DIFF_DTYPE
        f = getattr(self, "_fw", None)
        if f is None or f[-1] != dt:
            at = self.pair_bias_attn
            tr = self.transition
            if dt == mx.float16:  # fp16 diffusion GEMMs: built from the parameters by _K.w16 (mixed checkpoint)
                f = self._fw = (
                    _K.w16(self, "_fwh_qkvg", at.proj_q.weight, at.proj_k.weight, at.proj_v.weight, at.proj_g.weight),
                    _K.w16(self, "_fwh_ff", tr.swish_gate[0].weight, tr.a_to_b.weight),
                    _K.w16(self, "_fwh_o", at.proj_o.weight),
                    _K.w16(self, "_fwh_ba", tr.b_to_a.weight),
                    at.proj_q.bias,
                    dt,
                )
                return f
            w_qkvg = mx.concatenate([at.proj_q.weight, at.proj_k.weight, at.proj_v.weight, at.proj_g.weight], 0)
            b_q = at.proj_q.bias
            w_ff = mx.concatenate([tr.swish_gate[0].weight, tr.a_to_b.weight], 0)
            w_o = at.proj_o.weight
            w_ba = tr.b_to_a.weight
            f = self._fw = tuple(x.astype(dt) for x in (w_qkvg, w_ff, w_o, w_ba)) + (b_q, dt)
            mx.eval(*f[:5])
        return f

    def fast(self, a, cond, bias, mask, to_keys, multiplicity, attn_bias=None):
        """cond: (attn_scale, attn_shift, out_gate, tr_scale, tr_shift, tr_gate), sigmoids applied."""
        if attn_bias is None:
            attn_bias = self.full_attn_bias(bias, mask, to_keys, multiplicity)
        a = self.fast_attn(a, cond, to_keys, attn_bias)
        a = self.fast_transition(a, cond)
        if self.post_layer_norm:
            a = self.post_lnorm(a)
        return a

    def fast_attn(self, a, cond, to_keys, attn_bias):
        """a + out_gate * (gated pair-bias attention of AdaLN(a)); returns the new residual stream."""
        if _DT.supported_attn(self, a, to_keys, attn_bias, cond):
            return _DT.attn_fused(self, a, cond, attn_bias)
        if isinstance(cond, _DT.RawCondLayer):
            cond = cond.arrays()
        attn_scale, attn_shift, out_gate = cond[0], cond[1], cond[2]
        at = self.pair_bias_attn
        D, H = at.c_s, at.num_heads
        hd = D // H
        w_qkvg, w_ff, w_o, w_ba, b_q, _ = self._fused()

        b = attn_scale * _ln_kernel(a, None, None, 1e-5) + attn_shift
        if (to_keys is None and attn_bias is not None and H == 16 and D == 768 and a.ndim == 3
                and _K.DIFF_DTYPE == mx.float16 and _K.nax("DIFFATTN")):
            # NAX: fp16 projection -> flash attention (q bias, pair bias, softmax and sigmoid(g) gate in-kernel,
            # fp16 out) -> fp16 output projection. Same rounding points as mm_t / SDPA up to fp16 Q / P.
            from boltz_mlx import kernels_nax2 as _KN2

            B, N = a.shape[0], a.shape[1]
            h = b.astype(mx.float16) @ w_qkvg.T
            _K.record(f"difftx{D}.qkvg", h)
            o = _KN2.attention_nax(h.reshape(B * N, 4 * D), attn_bias, B=B, N=N, heads=H, c=hd, qbias=b_q,
                                   **_K.diffattn_cfg(B * N))
            o = (o @ w_o.T).astype(mx.float32).reshape(B, N, D)
            _K.record(f"difftx{D}.out_proj", o)
            return a + out_gate * o
        h = _K.mm_t(b, w_qkvg)
        _K.record(f"difftx{D}.qkvg", h)
        q, k, v, g = h[..., :D] + b_q, h[..., D : 2 * D], h[..., 2 * D : 3 * D], h[..., 3 * D :]
        if to_keys is not None:
            kv = to_keys(mx.concatenate([k, v], axis=-1))
            k, v = kv[..., :D], kv[..., D:]
        B = q.shape[0]
        q = q.reshape(B, -1, H, hd).transpose(0, 2, 1, 3)
        k = k.reshape(B, -1, H, hd).transpose(0, 2, 1, 3)
        v = v.reshape(B, -1, H, hd).transpose(0, 2, 1, 3)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=hd**-0.5, mask=attn_bias)
        o = o.transpose(0, 2, 1, 3).reshape(B, -1, D)
        _K.record(f"difftx{D}.attn_core", o)
        o = _K.mm_t(mx.sigmoid(g) * o, w_o)
        _K.record(f"difftx{D}.out_proj", o)
        return a + out_gate * o

    def fast_transition(self, a, cond):
        """a + tr_gate * ConditionedTransition(a); returns the new residual stream."""
        if _DT.supported_transition(self, a, cond):
            return _DT.transition_fused(self, a, cond)
        if isinstance(cond, _DT.RawCondLayer):
            cond = cond.arrays()
        tr_scale, tr_shift, tr_gate = cond[3], cond[4], cond[5]
        _, w_ff, _, w_ba, _, _ = self._fused()
        t = tr_scale * _ln_kernel(a, None, None, 1e-5) + tr_shift
        inner = self.transition.b_to_a.weight.shape[1]
        hf = _K.mm_t(t, w_ff)
        _K.record(f"difftx{a.shape[-1]}.ff_up", hf)
        x, gates, ab = hf[..., :inner], hf[..., inner : 2 * inner], hf[..., 2 * inner :]
        y = _K.mm_t(nn.silu(gates) * x * ab, w_ba)
        _K.record(f"difftx{a.shape[-1]}.ff_down", y)
        return a + tr_gate * y


def _cond_layer(cond, i):
    """Layer i's conditioning: a RawCondLayer view (fused kernels) or the 6 sliced arrays."""
    if isinstance(cond, _DT.RawCond):
        return cond.layer(i)
    return [x[..., i, :] for x in cond]


class DiffusionTransformer(nn.Module):
    """Algorithm 23"""

    def __init__(
        self,
        depth: int,
        heads: int,
        dim: int = 384,
        dim_single_cond: int | None = None,
        pair_bias_attn: bool = True,
        activation_checkpointing: bool = False,  # ignored (inference only)
        post_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        dim_single_cond = dim if dim_single_cond is None else dim_single_cond
        self.pair_bias_attn = pair_bias_attn
        self.layers = [DiffusionTransformerLayer(heads, dim, dim_single_cond, post_layer_norm) for _ in range(depth)]

    def __call__(self, a, s, bias=None, mask=None, to_keys=None, multiplicity=1, cond=None, fast=True, prepared=None):
        if prepared is not None:
            cond = prepared["cond"]
            brep = prepared.get("attn_brep", 1)
            if isinstance(cond, _N3.RawCond16) or (cond is None and a.ndim == 3 and self.nax3(to_keys)):
                # --precision f16 + BOLTZ_MLX_NAX_GEMM: fused-kernel token transformer (kernels_nax3)
                if cond is None:
                    cond = self.precompute_cond_raw16(s)
                return _N3.transformer_forward(self, a, cond, prepared["attn_bias"], brep)
            if cond is None:
                cond = self.precompute_cond(s, raw=_DT.transformer_supported(self, to_keys))
            for i, layer in enumerate(self.layers):
                ab = prepared["attn_bias"][i]
                if brep != 1:  # bias prepared once per group of samples (nax3 path), expanded for this fallback
                    ab = mx.repeat(ab, brep, axis=0)
                a = layer(a, s, None, None, to_keys, multiplicity, cond=_cond_layer(cond, i), attn_bias=ab)
            return a
        if self.pair_bias_attn:
            B, N, M, D = bias.shape
            L = len(self.layers)
            bias = bias.reshape(B, N, M, L, D // L)
        if fast and cond is None:
            cond = self.precompute_cond(s, raw=_DT.transformer_supported(self, to_keys))

        for i, layer in enumerate(self.layers):
            bias_l = bias[:, :, :, i] if self.pair_bias_attn else None
            c_i = None if cond is None else _cond_layer(cond, i)
            a = layer(a, s, bias_l, mask, to_keys, multiplicity, cond=c_i)
        return a

    def prepare(self, s, bias, mask, to_keys, multiplicity, with_cond: bool):
        """Step-invariant tensors for the diffusion loop: per-layer additive attention biases and,
        when the conditioning s is constant across steps (atom transformers), its projections."""
        B, N, M, D = bias.shape
        L = len(self.layers)
        bias = bias.reshape(B, N, M, L, D // L)
        if multiplicity > 1 and not with_cond and self.nax3(to_keys) and mask.shape[0] == B * multiplicity:
            # nax3: the samples of one input share pair bias and key mask -> one bias per input, read by the
            # attention kernel for `multiplicity` consecutive samples (lig_aff: 348 -> 70 MB, 5x less bias traffic)
            ab = [layer.full_attn_bias(bias[:, :, :, i], mask[::multiplicity], to_keys, 1)
                  for i, layer in enumerate(self.layers)]
            return {"attn_bias": ab, "cond": None, "attn_brep": multiplicity}
        ab = [layer.full_attn_bias(bias[:, :, :, i], mask, to_keys, multiplicity) for i, layer in enumerate(self.layers)]
        return {"attn_bias": ab, "cond": self.precompute_cond(s) if with_cond else None}

    def nax3(self, to_keys=None) -> bool:
        """The kernels_nax3 token-transformer path applies (BOLTZ_MLX_NAX_GEMM, fp16 diffusion matmuls, GPU, D 768 /
        16 heads / hidden 1536, full attention)."""
        if to_keys is not None or not _K.nax("GEMM") or _K.DIFF_DTYPE != mx.float16 or not _K.USE_KERNELS:
            return False
        L0 = self.layers[0]
        at = L0.pair_bias_attn
        return (at.c_s == 768 and at.num_heads == 16 and L0.transition.b_to_a.weight.shape == (768, 1536)
                and not L0.post_layer_norm and self.pair_bias_attn)

    def precompute_cond_raw16(self, s):
        """The conditioning of all layers as kernels_nax3.RawCond16: the two fp16 conditioning GEMM outputs, kept
        in fp16 (no fp32 copy, no sigmoid / slice passes; consumers add the fp32 bias and apply sigmoid in-kernel).
        Same GEMMs and operands as precompute_cond in fp16 (bitwise the same values)."""
        w_ln, b_ln, w_raw, b_raw, _ = self._cond_weights()
        B, N, D = s.shape
        h1 = _N3.cond_gemm(_ln_kernel(s, None, None, 1e-5, out_dtype=mx.float16), w_ln)
        h2 = _N3.cond_gemm(s.astype(mx.float16), w_raw)
        return _N3.RawCond16(h1.reshape(B * N, -1), b_ln, h2.reshape(B * N, -1), b_raw, ntok=N, L=len(self.layers), D=D)

    def _cond_weights(self):
        dt = _K.DIFF_DTYPE
        cw = getattr(self, "_cw", None)
        if cw is None or cw[-1] != dt:
            w_ln, b_ln, w_raw, b_raw = [], [], [], []
            for L in self.layers:
                for ad in (L.adaln, L.transition.adaln):
                    g = ad.s_norm.weight[None, :]  # fold LN affine weight into the linears
                    w_ln += [ad.s_scale.weight * g, ad.s_bias.weight * g]
                    b_ln += [ad.s_scale.bias, mx.zeros((ad.s_bias.weight.shape[0],))]
                op = L.transition.output_projection[0]
                w_raw += [L.output_projection_linear.weight, op.weight]
                b_raw += [L.output_projection_linear.bias, op.bias]
            if dt == mx.float16:
                # w_ln is a product (LN weight folded in, in fp32): cast at runtime from the fp32 parameters;
                # w_raw is a plain concatenation of parameters: fp16 parameters in the mixed checkpoint
                cw = self._cw = (
                    _K.w16(self, "_cwh_ln", mx.concatenate(w_ln, 0)),
                    mx.concatenate(b_ln, 0),
                    _K.w16(self, "_cwh_raw", *w_raw),
                    mx.concatenate(b_raw, 0),
                    dt,
                )
                mx.eval(*cw[:4])
                return cw
            cw = self._cw = (
                mx.concatenate(w_ln, 0).astype(dt),
                mx.concatenate(b_ln, 0),
                mx.concatenate(w_raw, 0).astype(dt),
                mx.concatenate(b_raw, 0),
                dt,
            )
            mx.eval(*cw[:4])
        return cw

    def precompute_cond_shared(self, s1, multiplicity):
        """precompute_cond(s1) for `multiplicity` samples that share the conditioning s1 (batch 1)."""
        if self.nax3(None):
            rc = self.precompute_cond_raw16(s1)
            rc.rep = multiplicity
            return rc
        if _DT.transformer_supported(self, None):
            rc = self.precompute_cond(s1, raw=True)
            rc.rep = multiplicity
            return rc
        return [mx.repeat(x, multiplicity, axis=0) for x in self.precompute_cond(s1)]

    def precompute_cond(self, s, raw=False):
        """All layers' s-only projections in two matmuls. Returns 6 arrays (..., L, D); raw=True returns the
        un-sliced pre-sigmoid projections as a kernels_difftx.RawCond (consumed in place by the fused kernels)."""
        if raw and _K.DIFF_DTYPE == mx.float32 and s.ndim == 3 and s.shape[-1] == 768:
            return _DT.cond_raw(self, s)
        w_ln, b_ln, w_raw, b_raw, _ = self._cond_weights()
        L = len(self.layers)
        h1 = _K.mm_t(_ln_kernel(s, None, None, 1e-5), w_ln) + b_ln
        h2 = _K.mm_t(s, w_raw) + b_raw
        if raw:
            return _DT.RawCond(h1.reshape(*s.shape[:-1], L, 4, -1), h2.reshape(*s.shape[:-1], L, 2, -1))
        h1 = h1.reshape(*s.shape[:-1], L, 4, -1)
        h2 = mx.sigmoid(h2.reshape(*s.shape[:-1], L, 2, -1))
        return (
            mx.sigmoid(h1[..., 0, :]),
            h1[..., 1, :],
            h2[..., 0, :],
            mx.sigmoid(h1[..., 2, :]),
            h1[..., 3, :],
            h2[..., 1, :],
        )


class AtomTransformer(nn.Module):
    """Algorithm 7 (windowed local attention over atoms)"""

    def __init__(self, attn_window_queries: int, attn_window_keys: int, **diffusion_transformer_kwargs) -> None:
        super().__init__()
        self.attn_window_queries = attn_window_queries
        self.attn_window_keys = attn_window_keys
        self.diffusion_transformer = DiffusionTransformer(**diffusion_transformer_kwargs)

    def _shape(self, q, c, bias, to_keys, mask, multiplicity):
        W, H = self.attn_window_queries, self.attn_window_keys
        B, N, D = q.shape
        NW = N // W
        c = c.reshape(B * NW, W, -1)
        mask = mask.reshape(B * NW, W)
        if multiplicity != 1:
            bias = mx.repeat(bias, multiplicity, axis=0)
        bias = bias.reshape(bias.shape[0] * NW, W, H, -1)
        tk = to_keys.gather if (_K.TOKEYS_GATHER and _K.f16() and hasattr(to_keys, "gather")) else to_keys

        def to_keys_new(x):
            return tk(x.reshape(B, NW * W, -1)).reshape(B * NW, H, -1)

        return c, bias, mask.astype(mx.float32), to_keys_new

    def prepare(self, q_shape, c, bias, to_keys, mask, multiplicity):
        c, bias, mask, tk = self._shape(mx.zeros(q_shape), c, bias, to_keys, mask, multiplicity)
        return self.diffusion_transformer.prepare(c, bias, mask, tk, 1, with_cond=True)

    def nax_ok(self, n_atoms: int) -> bool:
        """--precision f16 + BOLTZ_MLX_NAX_ATOM: the kernels_atom path applies (fp16 diffusion matmuls, GPU, Boltz-2
        atom-transformer shapes, atom count a multiple of the query window)."""
        if not _K.nax("ATOM") or _K.DIFF_DTYPE != mx.float16 or not _K.USE_KERNELS:
            return False
        from boltz_mlx import kernels_atom as _KA

        W, H = self.attn_window_queries, self.attn_window_keys
        return n_atoms % W == 0 and _KA.supported(self.diffusion_transformer, W, H)

    def prepare_nax(self, c, bias, to_keys, mask, multiplicity):
        """Step-invariant tensors of the kernels_atom path, computed once per input (not per sample): c [B, N, D] atom
        conditioning, bias [B, N / W, W, H, L * heads] pair bias, mask [B, N] atom mask (inputs before the multiplicity
        repeat). The conditioning becomes a kernels_nax3.RawCond16 shared by `multiplicity` consecutive samples (rep);
        the per-layer additive attention biases (pair bias + key mask, the same values as the MLX path's per-sample
        ones) are read for `multiplicity` samples (attn_brep). Evaluated here (outside the compiled step)."""
        W, H = self.attn_window_queries, self.attn_window_keys
        B, N, _ = c.shape
        NW = N // W
        dt = self.diffusion_transformer
        L = len(dt.layers)
        bias = bias.reshape(B * NW, W, H, L, -1)
        tk = to_keys.gather if (_K.TOKEYS_GATHER and hasattr(to_keys, "gather")) else to_keys

        def to_keys_new(x):
            return tk(x.reshape(B, NW * W, -1)).reshape(B * NW, H, -1)

        mask_r = mask.reshape(B * NW, W).astype(mx.float32)
        ab = [layer.full_attn_bias(bias[:, :, :, i], mask_r, to_keys_new, 1) for i, layer in enumerate(dt.layers)]
        rc = dt.precompute_cond_raw16(c)
        rc.rep = multiplicity
        mx.eval(ab, rc.h1, rc.b1, rc.h2, rc.b2)
        return {"nax_atom": True, "attn_bias": ab, "cond": rc, "attn_brep": multiplicity, "NA": N, "W": W, "H": H}

    def __call__(self, q, c, bias, to_keys, mask, multiplicity=1, prepared=None):
        B, N, D = q.shape
        if prepared is not None and prepared.get("nax_atom"):
            from boltz_mlx import kernels_atom as _KA

            return _KA.transformer_forward(self.diffusion_transformer, q.reshape(B * N, D), prepared).reshape(B, N, D)
        W = self.attn_window_queries
        NW = N // W
        c_r, bias_r, mask_r, tk = self._shape(q, c, bias, to_keys, mask, multiplicity)
        q = self.diffusion_transformer(
            a=q.reshape(B * NW, W, -1),
            s=c_r,
            bias=bias_r,
            mask=mask_r,
            multiplicity=1,  # bias already expanded with multiplicity
            to_keys=tk,
            prepared=prepared,
        )
        return q.reshape(B, NW * W, D)
