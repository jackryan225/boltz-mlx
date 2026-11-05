"""kernels_nax3 (round-3 NAX GEMM epilogues), CPU-side checks (the GPU tests are tests/test_nax3_kernels.py):

1. Lane-level emulation (tests/nax3_emul.py) of every kernel's index arithmetic == the MLX-op reference with the same
   rounding points: fragment rows / columns, B^T fragment rows, f / f + 1 pairing, xor-1 / xor-8 row groups, transposed
   reads / writes, the last tile moved back, clamped rows, conditioning rows of shared samples.
2. Wiring: the integrated paths (pair_block_f16_epi sites, the token-transformer forward) with each Metal kernel
   replaced by its MLX reference agree with the existing paths (slot / gate offsets, SwiGLU group order, shared-bias
   expansion, transposition flags).
Tiny shapes only (CPU, < 1 GB).
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

import nax3_emul as E

pytestmark = pytest.mark.skipif(mx.default_device() != mx.cpu, reason="CPU logic checks")


def _N3():
    from boltz_mlx import kernels_nax3 as N3

    return N3


def _close(a, b, tol, what=""):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    assert np.isfinite(a).all(), f"{what}: non-finite / unwritten elements"
    err = np.abs(a - b).max() / max(np.abs(b).max(), 1e-30)
    assert err <= tol, (what, err)


class _LN:
    def __init__(self, D, rng, bias=True):
        self.w = mx.array(rng.normal(1.0, 0.1, D).astype(np.float32))
        self.b = mx.array(rng.normal(0.0, 0.1, D).astype(np.float32)) if bias else None
        self.eps = 1e-5

    def get(self, k, d=None):
        return {"weight": self.w, "bias": self.b}.get(k, d)


def _h(rng, shape, s=1.0):
    return (rng.normal(size=shape) * s).astype(np.float16)


@pytest.mark.parametrize("N,tm,ty,to,bias", [(7, 16, False, True, False), (7, 16, True, False, False),
                                             (9, 32, True, True, True), (6, 16, False, False, False)])
def test_linear_addln_emulation(N, tm, ty, to, bias):
    N3 = _N3()
    rng = np.random.default_rng(N + tm)
    D, R = 128, N * N
    x, w = _h(rng, (R, D)), _h(rng, (D, D), D ** -0.5)
    z = rng.normal(size=(R, D)).astype(np.float32)
    b = rng.normal(size=D).astype(np.float32) * 0.1 if bias else None
    ln = _LN(D, rng)
    zo, lo = E.linear_addln(x, w, z, np.array(ln.w), np.array(ln.b), ln.eps, bias=b, trans_y=ty, trans_out=to, tm=tm,
                            nsg=2)
    zr, lr = N3.linear_addln_ref(mx.array(x).reshape(1, N, N, D), mx.array(w), mx.array(z).reshape(1, N, N, D), ln,
                                 bias=None if b is None else mx.array(b), trans_y=ty, trans_out=to)
    _close(zo, np.array(zr).reshape(R, D), 2e-3, "z_out")  # fp16 y may differ by an ulp (fp32 sum order)
    _close(lo, np.array(lr.astype(mx.float32)).reshape(R, D), 2e-3, "ln_out")


@pytest.mark.parametrize("N,tm", [(7, 16), (5, 16)])
def test_trimul_tail_emulation(N, tm):
    N3 = _N3()
    rng = np.random.default_rng(N)
    D, R = 128, N * N
    o = _h(rng, (D, R), 0.05)  # 2^-2K-scaled contraction output
    xn, wpo, wgo = _h(rng, (R, D)), _h(rng, (D, D), D ** -0.5), _h(rng, (D, D), D ** -0.5)
    z = rng.normal(size=(R, D)).astype(np.float32)
    lo_, ln = _LN(D, rng), _LN(D, rng)
    es = 2.0 ** -16
    zo, lno = E.trimul_tail_addln(o, np.array(lo_.w), np.array(lo_.b), lo_.eps * es, wpo, xn, wgo, z, np.array(ln.w),
                                  np.array(ln.b), ln.eps, tm=tm, nsg=2)
    zr, lr = N3.trimul_tail_addln_ref(mx.array(o), lo_, es, mx.array(wpo), mx.array(xn), mx.array(wgo),
                                      mx.array(z).reshape(1, N, N, D), ln)
    _close(zo, np.array(zr).reshape(R, D), 2e-3, "z_out")  # fp16 rounding points may flip by one ulp (sum order)
    _close(lno, np.array(lr.astype(mx.float32)).reshape(R, D), 5e-3, "ln_out")


@pytest.mark.parametrize("cfg_i", [0, 1, 3, 4, 6, 7, 8, 9, 10, 11])
@pytest.mark.parametrize("epi", [0, 1, 2])
def test_gemm16_emulation(cfg_i, epi):
    N3 = _N3()
    cfg = (N3.SWIGLU_CFGS[cfg_i % len(N3.SWIGLU_CFGS)] if epi == 2 else N3.GEMM_CFGS[cfg_i])
    rng = np.random.default_rng(cfg_i * 3 + epi)
    M, K, NOUT, ntok, rep = 37, 64, 128, 13, 3          # 37 rows: ragged row tail; cond rows shared by 3 samples
    ngrp = 3 if epi == 2 else 1
    x, w = _h(rng, (M, K)), _h(rng, (ngrp * NOUT, K), K ** -0.5)
    kw = {}
    if epi == 1:
        gl = 4 * NOUT
        kw = dict(gate=_h(rng, (-(-M // (ntok * rep)) * ntok, gl)), gate_bias=rng.normal(size=gl).astype(np.float32),
                  gate_off=2 * NOUT, resid=rng.normal(size=(M, NOUT)).astype(np.float32), ntok=ntok, rep=rep)
    if epi == 0 and cfg_i == 0:
        kw = dict(bias=rng.normal(size=NOUT).astype(np.float32), out_dtype=np.float32)
    got = E.gemm16(x, w, epi=epi, ngrp=ngrp, **kw, **cfg)
    mkw = {k: (mx.array(v) if isinstance(v, np.ndarray) else v) for k, v in kw.items()}
    if "out_dtype" in mkw:
        mkw["out_dtype"] = mx.float32
    ref = N3.gemm16_ref(mx.array(x), mx.array(w), epi=epi, ngrp=ngrp, **mkw)
    _close(got, np.array(ref.astype(mx.float32)), 2e-3 if epi != 1 else 1e-5, f"epi {epi}")


def test_adaln16_emulation_and_ref():
    N3 = _N3()
    rng = np.random.default_rng(0)
    L, D, ntok, rep = 3, 128, 5, 2
    M = 2 * rep * ntok
    a2 = rng.normal(size=(M, D)).astype(np.float32)
    h1 = _h(rng, (2 * ntok, L * 4 * D))
    b1 = rng.normal(size=L * 4 * D).astype(np.float32)
    rc = N3.RawCond16(mx.array(h1), mx.array(b1), mx.zeros((2 * ntok, L * 2 * D), mx.float16), mx.zeros((L * 2 * D,)),
                      ntok=ntok, L=L, D=D, rep=rep)
    for layer, slot in ((0, 0), (2, 2)):
        got = E.adaln16(a2, h1, b1, ntok=ntok, rep=rep, L=L, layer=layer, slot=slot)
        ref = N3.adaln16_ref(mx.array(a2), rc, layer, slot)
        _close(got, np.array(ref.astype(mx.float32)), 2e-3, "adaln16")
        # RawCond16.arrays (the MLX-path 6-tuple) agrees with the slot layout
        arr = rc.arrays(layer)
        sl = 0 if slot == 0 else 3
        ln = (a2 - a2.mean(-1, keepdims=True)) / np.sqrt(a2.var(-1, keepdims=True) + 1e-5)
        t = np.array(arr[sl]).reshape(M, D) * ln + np.array(arr[sl + 1]).reshape(M, D)
        _close(got, t, 2e-3, "arrays")


# ------------------------------------------------------------------------------------------------- wiring
def _attention_ref_brep(h, bias, *, B, N, heads, c, qbias=None, brep=1, **_):
    from boltz_mlx import kernels_nax2 as KN2

    if brep != 1:
        bias = mx.repeat(bias, brep, axis=0)
    return KN2.attention_ref(h, bias, B=B, N=N, heads=heads, c=c, qbias=qbias)


def test_token_transformer_wiring(monkeypatch):
    """transformer_forward with the kernels replaced by their references == the existing fp16-diffusion MLX path
    (6-array conditioning, SDPA) within fp16 rounding, incl. shared conditioning (rep) and the shared pair bias."""
    from boltz_mlx import kernels as K
    from boltz_mlx import kernels_nax2 as KN2
    from boltz_mlx.modules.transformersv2 import DiffusionTransformer

    N3 = _N3()
    mx.random.seed(0)
    depth, D, N, mult = 2, 768, 8, 3
    tr = DiffusionTransformer(depth=depth, heads=16, dim=D, dim_single_cond=D)
    rng = np.random.default_rng(1)
    for L in tr.layers:  # non-trivial AdaLN / gate conditioning
        for lin in (L.adaln.s_scale, L.transition.adaln.s_scale, L.output_projection_linear,
                    L.transition.output_projection[0]):
            lin.weight = mx.array(rng.normal(size=lin.weight.shape).astype(np.float32) * 0.05)
    prev = K.DIFF_DTYPE
    K.set_diff_dtype("fp16")
    try:
        s1 = mx.array(rng.normal(size=(1, N, D)).astype(np.float32))
        a = mx.array(rng.normal(size=(mult, N, D)).astype(np.float32))
        bias = mx.array(rng.normal(size=(1, N, N, 16 * depth)).astype(np.float32))
        mask = mx.array(np.tile(np.r_[np.ones(N - 2), np.zeros(2)].astype(np.float32), (mult, 1)))
        # existing path: per-layer biases for all samples, 6-array conditioning repeated over samples
        prep = tr.prepare(None, bias, mask, None, mult, with_cond=False)
        cond = [mx.repeat(x, mult, axis=0) for x in tr.precompute_cond(s1)]
        ref = tr(a, mx.repeat(s1, mult, 0), prepared={**prep, "cond": cond})
        # nax3 path with references (bias prepared once, brep = mult)
        rc = tr.precompute_cond_raw16(s1)
        rc.rep = mult
        ab1 = [L.full_attn_bias(bias.reshape(1, N, N, depth, 16)[:, :, :, i], mask[:1], None, 1)
               for i, L in enumerate(tr.layers)]
        monkeypatch.setattr(N3, "adaln16", lambda a2, rc_, l, s: N3.adaln16_ref(a2, rc_, l, s))
        monkeypatch.setattr(N3, "gemm16", lambda x, w, **kw: N3.gemm16_ref(x, w, **kw))
        monkeypatch.setattr(KN2, "attention_nax", _attention_ref_brep)
        for site in ("qkvg", "o", "ff", "ba"):
            monkeypatch.setitem(N3.DIFF_GEMM_CFG, site, dict(sm=32, sn=32, wm=2, wn=2, ku=2))
        got = N3.transformer_forward(tr, a, rc, ab1, brep=mult)
        mx.eval(got, ref)
        g, r = np.array(got), np.array(ref)
        rel = np.linalg.norm(g - r) / np.linalg.norm(r - np.array(a))  # relative to the update
        assert np.isfinite(g).all() and rel < 3e-3, rel
        # MLX-GEMM site fallback ('mlx') gives the same wiring
        for site in ("qkvg", "o", "ff", "ba"):
            monkeypatch.setitem(N3.DIFF_GEMM_CFG, site, "mlx")
        got2 = N3.transformer_forward(tr, a, rc, ab1, brep=mult)
        rel2 = np.linalg.norm(np.array(got2) - r) / np.linalg.norm(r - np.array(a))
        assert rel2 < 3e-3, rel2
        # RawCond16 slots == precompute_cond's 6 arrays up to fp16 rounding: the conditioning GEMM may run on a different
        # kernel (nax3_cfg.json "cond" site), and slots are stored pre-bias in fp16, so a value can differ by a few fp16 ULPs (<= 5e-4 abs for |x| < 1)
        c6 = tr.precompute_cond(s1)
        for i in range(depth):
            for x6, xr in zip(c6, rc.arrays(i)):
                np.testing.assert_allclose(np.array(xr[:1]), np.array(x6[..., i, :]), rtol=2.0 ** -10, atol=5e-4)
    finally:
        K.DIFF_DTYPE = prev


def test_prepare_shared_bias(monkeypatch):
    """prepare() with the nax3 gate returns one bias per input + attn_brep; the fallback loop expands it back to
    exactly the per-sample biases of the existing path."""
    from boltz_mlx.modules.transformersv2 import DiffusionTransformer

    depth, D, N, mult = 2, 768, 6, 4
    tr = DiffusionTransformer(depth=depth, heads=16, dim=D, dim_single_cond=D)
    rng = np.random.default_rng(2)
    bias = mx.array(rng.normal(size=(1, N, N, 16 * depth)).astype(np.float32))
    mask = mx.array(np.tile(np.r_[np.ones(N - 1), 0].astype(np.float32), (mult, 1)))
    full = tr.prepare(None, bias, mask, None, mult, with_cond=False)
    monkeypatch.setattr(DiffusionTransformer, "nax3", lambda self, to_keys=None: to_keys is None)
    shared = tr.prepare(None, bias, mask, None, mult, with_cond=False)
    assert shared["attn_brep"] == mult and "attn_brep" not in full
    for f, s in zip(full["attn_bias"], shared["attn_bias"]):
        assert s.shape[0] == 1
        assert bool(mx.array_equal(mx.repeat(s, mult, axis=0), f).item())


def _pair_modules(D=128, seed=0):
    from boltz_mlx.layers.triangular_attention.attention import TriangleAttention, TriangleAttentionEndingNode
    from boltz_mlx.layers.triangular_mult import TriangleMultiplicationOutgoing

    mx.random.seed(seed)
    return TriangleMultiplicationOutgoing(D), TriangleAttention(D, 32, 4), TriangleAttentionEndingNode(D, 32, 4)


def test_trimul_f16_addln_wiring(monkeypatch):
    """f16_addln (projection + contraction of _f16_b1, tail kernel -> reference) == z + _f16_b1(.) and its LN."""
    from boltz_mlx import kernels as K
    from boltz_mlx.layers.norm import LayerNorm

    N3 = _N3()
    tm, _, _ = _pair_modules()
    N, D = 9, 128
    rng = np.random.default_rng(3)
    z = mx.array(rng.normal(size=(1, N, N, D)).astype(np.float32))
    mask = mx.array((rng.random((1, N, N)) > 0.1).astype(np.float32))
    ln = LayerNorm(D)
    ln.weight = mx.array(rng.normal(1, 0.1, D).astype(np.float32))
    xn = tm.norm_in.f16(z)
    monkeypatch.setattr(N3, "trimul_tail_addln", lambda *a, **kw: N3.trimul_tail_addln_ref(*a, **kw))
    zo, lo = tm.f16_addln(z, mask, xn, ln)
    y = tm._f16_b1(xn, mask, D, N, N)
    zr = z + y.astype(mx.float32)
    lr = ln.f16(zr)
    np.testing.assert_array_equal(np.array(zo), np.array(zr))
    _close(np.array(lo.astype(mx.float32)), np.array(lr.astype(mx.float32)), 2e-3, "ln")
    assert K.TRIMULT_K == 4


@pytest.mark.parametrize("ending", [False, True])
def test_triatt_f16_addln_wiring(monkeypatch, ending):
    """linear_addln flags at the two tri-att sites: start node (LN output transposed for the ending node), ending node
    (core in the transposed layout, added back transposed)."""
    from boltz_mlx import kernels as K
    from boltz_mlx.layers.norm import LayerNorm

    N3 = _N3()
    _, ts, te = _pair_modules()
    mod = te if ending else ts
    N, D = 7, 128
    rng = np.random.default_rng(4)
    z = mx.array(rng.normal(size=(1, N, N, D)).astype(np.float32))
    core = mx.array(_h(rng, (1, N, N, D)))  # in the module's own layout ([j, i] rows for the ending node)
    ln = LayerNorm(D)
    monkeypatch.setattr(type(mod), "_f16_core", lambda self, x, mask, wh, use_nax, tb=None: core)
    monkeypatch.setattr(N3, "linear_addln", lambda *a, **kw: N3.linear_addln_ref(*a, **kw))
    zo, lo = mod.f16_addln(z, None, core, ln, trans_out=not ending)  # xn: only its shape is used here
    y = K.lin16(core, mod.mha.linear_o)          # module output in the core layout
    if ending:
        y = y.swapaxes(1, 2)                     # what the unfused ending node returns (keep_t = False)
    zr = z + y.astype(mx.float32)
    lr = ln.f16(zr)
    if not ending:
        lr = lr.swapaxes(1, 2)                   # add_ln(trans_out=True) for the ending node's input
    np.testing.assert_array_equal(np.array(zo), np.array(zr))
    _close(np.array(lo.astype(mx.float32)), np.array(lr.astype(mx.float32)), 2e-3, "ln")


def test_flags_never_on_outside_f16_gpu():
    """Round-3 groups: never on the CPU device, never outside --precision f16 unless forced by env."""
    from boltz_mlx import kernels as K

    for g in ("GEMM", "PAIREPI"):
        if K.NAX_FLAGS.get(g) is None:
            assert K.nax(g) == (K.f16() and g not in K.NAX_DEFAULT_OFF and K.mpp_enabled())
            assert not K.nax(g)  # CPU device -> f16() is False
    assert not K.pair_epi_ok(mx.zeros((1, 8, 8, 128)))
    assert isinstance(nn.Module(), nn.Module)


@pytest.mark.parametrize("N,D,B", [(64, 3, 1), (1536, 256, 5), (3072, 256, 1), (928, 128, 2)])
def test_to_keys_gather_equals_contraction(N, D, B):
    """encodersv2.ToKeys.gather (f16 atom-attention keys) == the one-hot contraction, value for value (the only
    possible difference, the sign of zeros in key slots past the sequence ends, compares equal)."""
    from boltz_mlx.modules.encodersv2 import make_to_keys

    tk = make_to_keys(N, 32, 128)
    rng = np.random.default_rng(N)
    x = mx.array(rng.normal(size=(B, N, D)).astype(np.float32))
    a, g = tk(x), tk.gather(x)
    assert a.shape == g.shape == (B, N // 32, 128, D)
    np.testing.assert_array_equal(np.array(a), np.array(g))
    m = mx.array((rng.random((B, N, 1)) > 0.3).astype(np.float32))  # masks go through the same map
    np.testing.assert_array_equal(np.array(tk(m)), np.array(tk.gather(m)))


# ------------------------------------------------------------------------------------------------- R3-pair
@pytest.mark.parametrize("N,tm,ty,to", [(7, 16, False, True), (9, 32, True, False), (6, 16, False, False)])
def test_linear_addln_tb_emulation(N, tm, ty, to):
    """_LN_EPI_TB (tri bias from the LN output rows, head-major in the LN output's layout) == reference."""
    N3 = _N3()
    rng = np.random.default_rng(N + 100)
    D, R = 128, N * N
    x, w, wtb = _h(rng, (R, D)), _h(rng, (D, D), D ** -0.5), _h(rng, (4, D), D ** -0.5)
    z = rng.normal(size=(R, D)).astype(np.float32)
    ln = _LN(D, rng)
    zo, lo, tb = E.linear_addln(x, w, z, np.array(ln.w), np.array(ln.b), ln.eps, trans_y=ty, trans_out=to, tm=tm,
                                nsg=2, wtb=wtb)
    zr, lr, tr = N3.linear_addln_ref(mx.array(x).reshape(1, N, N, D), mx.array(w), mx.array(z).reshape(1, N, N, D), ln,
                                     trans_y=ty, trans_out=to, w_tb=mx.array(wtb))
    _close(zo, np.array(zr).reshape(R, D), 2e-3, "z_out")
    _close(lo, np.array(lr.astype(mx.float32)).reshape(R, D), 2e-3, "ln_out")
    _close(tb, np.array(tr.astype(mx.float32)).reshape(4, R), 5e-3, "tb")


def test_trimul_tail_tb_emulation():
    N3 = _N3()
    rng = np.random.default_rng(77)
    N, D = 7, 128
    R = N * N
    o = _h(rng, (D, R), 0.05)
    xn, wpo, wgo, wtb = _h(rng, (R, D)), _h(rng, (D, D), D ** -0.5), _h(rng, (D, D), D ** -0.5), _h(rng, (4, D), 0.1)
    z = rng.normal(size=(R, D)).astype(np.float32)
    lo_, ln = _LN(D, rng), _LN(D, rng)
    es = 2.0 ** -16
    zo, lno, tb = E.trimul_tail_addln(o, np.array(lo_.w), np.array(lo_.b), lo_.eps * es, wpo, xn, wgo, z,
                                      np.array(ln.w), np.array(ln.b), ln.eps, tm=16, nsg=2, wtb=wtb)
    zr, lr, tr = N3.trimul_tail_addln_ref(mx.array(o), lo_, es, mx.array(wpo), mx.array(xn), mx.array(wgo),
                                          mx.array(z).reshape(1, N, N, D), ln, w_tb=mx.array(wtb))
    _close(zo, np.array(zr).reshape(R, D), 2e-3, "z_out")
    _close(lno, np.array(lr.astype(mx.float32)).reshape(R, D), 5e-3, "ln_out")
    _close(tb, np.array(tr.astype(mx.float32)).reshape(4, R), 1e-2, "tb")


@pytest.mark.parametrize("N,tm,nsg,want_xn,nhb", [(7, 16, 8, True, 16), (9, 16, 4, False, 16), (11, 16, 8, True, 0)])
def test_transition_lnx_epilogue_emulation(N, tm, nsg, want_xn, nhb):
    """_LNX_EPI row pass (clamped duplicate tiles, last tile moved back, lane + 32 i columns, head-major bias)
    == the references: next-layer LN of z_new and half(half(LN_s(z_new)) @ Wz^T) transposed."""
    N3 = _N3()
    rng = np.random.default_rng(N * 3 + nhb)
    D, R = 128, N * N
    zn = (rng.normal(size=(R, D)) * 2 + 0.5).astype(np.float32)
    l1, l2 = _LN(D, rng), _LN(D, rng)
    wz = _h(rng, (nhb, D), D ** -0.5) if nhb else None
    xo, pb = E.transition_lnx_epilogue(zn, (np.array(l1.w), np.array(l1.b), l1.eps) if want_xn else None,
                                       (np.array(l2.w), np.array(l2.b), l2.eps) if nhb else None, wz, tm=tm, nsg=nsg)
    zm = mx.array(zn).reshape(1, N, N, D)
    if want_xn:
        ref = N3._ln_ref(zm, l1.w, l1.b, l1.eps).astype(mx.float16)
        _close(xo, np.array(ref.astype(mx.float32)).reshape(R, D), 2e-3, "xn")
    if nhb:
        y = N3._ln_ref(zm, l2.w, l2.b, l2.eps).astype(mx.float16).astype(mx.float32)
        ref = (y @ mx.array(wz).astype(mx.float32).T).astype(mx.float16).transpose(0, 3, 1, 2)
        _close(pb, np.array(ref.astype(mx.float32)).reshape(nhb, R), 5e-3, "pair bias")


def test_r3pair_flags_f16_gpu_only():
    from boltz_mlx import kernels as K

    for g in ("PAIRLNX", "PAIRPROJ", "PAIRTB"):
        assert g in K.NAX_FLAGS
        if K.NAX_FLAGS[g] is None:
            assert K.nax(g) == (K.f16() and g not in K.NAX_DEFAULT_OFF and K.mpp_enabled())
            assert not K.nax(g)  # CPU device
    assert not K.pair_lnx_ok(mx.zeros((1, 8, 8, 128)))


def test_carry_helper():
    """_carry returns the next layer's LN output only for the z it was computed from, and drops it."""
    from boltz_mlx.layers.pairformer import PairformerNoSeqLayer, _carry

    layer = PairformerNoSeqLayer(128)
    z, xn = mx.zeros((1, 2, 2, 128)), mx.ones((1, 2, 2, 128), mx.float16)
    layer.__dict__["_pb_carry"] = (z, xn)
    assert _carry(layer, z) is xn and "_pb_carry" not in layer.__dict__
    layer.__dict__["_pb_carry"] = (z, xn)
    assert _carry(layer, mx.zeros((1, 2, 2, 128))) is None
    assert _carry(layer, z) is None
    assert "_pb_carry" not in layer.parameters()  # never a registered child / parameter


def _triatt_ref(h, n_heads, c, mask, inf, bias=None, **_):
    """NAX tri-attention semantics on CPU: h [B, I, J, 4HC] (q|k|v|g), bias [B, H, J, J] (query, key), mask [B, I, J]."""
    f = mx.float32
    B, I, J, _ = h.shape
    HC = n_heads * c
    hh = h[..., : 4 * HC].astype(f).reshape(B, I, J, 4, n_heads, c)
    q, k, v, g = hh[..., 0, :, :], hh[..., 1, :, :], hh[..., 2, :, :], hh[..., 3, :, :]
    s = mx.einsum("biqhc,bikhc->bihqk", q, k) * c ** -0.5
    s = s + (inf * (mask.astype(f) - 1))[:, :, None, None, :] + bias.astype(f)[:, None]
    p = mx.softmax(s, axis=-1)
    o = mx.einsum("bihqk,bikhc->biqhc", p, v) * mx.sigmoid(g)
    return o.reshape(B, I, J, HC).astype(mx.float16)


@pytest.mark.parametrize("which", [("PAIRTB",), ("PAIRLNX",), ("PAIRPROJ",), ("PAIRLNX", "PAIRPROJ", "PAIRTB"),
                                   ("TAIL2",), ("TAIL3",), ("TRANS3",), ("PAIRLNX", "PAIRPROJ", "PAIRTB", "TAIL3", "TRANS3")])
def test_r3pair_module_wiring(monkeypatch, which):
    """PairformerModule (3 random layers, N 40) through the f16 epilogue path with every Metal kernel replaced by its
    reference: R3-pair pieces on == off within fp16 noise (catches tri-bias / pair-bias layouts, the ending-node
    transposition and the carry of the next layer's norm_in output)."""
    from types import SimpleNamespace

    from boltz_mlx import kernels as K
    from boltz_mlx import kernels_nax as KN
    from boltz_mlx import kernels_nax2 as KN2
    from boltz_mlx.layers.pairformer import PairformerLayer, PairformerModule

    N3 = _N3()
    active = {"TRANSITION", "PAIREPI"}
    monkeypatch.setattr(K, "f16", lambda: True)
    monkeypatch.setattr(K, "nax", lambda kind: kind in active)
    monkeypatch.setattr(KN, "tri_attention_nax", _triatt_ref)
    monkeypatch.setattr(KN2, "transition_nax", KN2.transition_ref)
    monkeypatch.setattr(N3, "linear_addln", N3.linear_addln_ref)
    monkeypatch.setattr(N3, "trimul_tail_addln", N3.trimul_tail_addln_ref)
    monkeypatch.setattr(N3, "transition_lnx", N3.transition_lnx_ref)
    monkeypatch.setattr(N3, "transition_lnx_ok", lambda z, cfg: True)
    monkeypatch.setattr(N3, "pair_proj16", lambda x, w, cfg=None: x @ w.T)
    monkeypatch.setattr(N3, "trimul_tail2_addln", N3.trimul_tail_addln_ref)
    monkeypatch.setattr(N3, "trimul_tail3_addln", N3.trimul_tail_addln_ref)
    monkeypatch.setattr(N3, "transition3", N3.transition3_ref)
    mx.random.seed(5)
    layers = [PairformerLayer(384, 128, 16) for _ in range(3)]
    chain = SimpleNamespace(layers=layers)
    rng = np.random.default_rng(5)
    N = 40
    s = mx.array(rng.normal(size=(1, N, 384)).astype(np.float32))
    z = mx.array(rng.normal(size=(1, N, N, 128)).astype(np.float32))
    mask = mx.ones((1, N))
    pm = mx.array((rng.random((1, N, N)) > 0.05).astype(np.float32))
    so, zo = PairformerModule.__call__(chain, s, z, mask, pm)
    active |= set(which)
    s3, z3 = PairformerModule.__call__(chain, s, z, mask, pm)
    mx.eval(so, zo, s3, z3)
    rz = np.linalg.norm(np.array(z3 - zo)) / np.linalg.norm(np.array(zo - z))
    rs = np.linalg.norm(np.array(s3 - so)) / np.linalg.norm(np.array(so - s))
    print(which, "rel z", rz, "rel s", rs)
    assert rz < 3e-3 and rs < 3e-3, (which, rz, rs)
    if which == ("PAIRPROJ",):
        assert rz == 0  # PAIRPROJ alone: the reference computes the same projection
    assert all("_pb_carry" not in L.__dict__ for L in layers)


# ------------------------------------------------------------------------------------------------- R4-pair
@pytest.mark.parametrize("N,tp,nsg", [(7, 32, 4), (9, 16, 4), (5, 64, 8)])
def test_transpose_ln16_emulation(N, tp, nsg):
    """_TLN_SRC tile / lane indexing (ragged last tile) == LN of the transposed view."""
    N3 = _N3()
    rng = np.random.default_rng(N + tp)
    D, R = 128, N * N
    o = _h(rng, (D, R), 0.05)
    ln = _LN(D, rng)
    es = 2.0 ** -16
    got = E.transpose_ln16(o, np.array(ln.w), np.array(ln.b), ln.eps * es, tp=tp, nsg=nsg)
    ref = N3._ln_ref(mx.array(o).astype(mx.float32).T, ln.w, ln.b, ln.eps * es).astype(mx.float16)
    _close(got, np.array(ref.astype(mx.float32)), 2e-3, "on")


@pytest.mark.parametrize("N,tm,tb", [(7, 16, False), (9, 32, True), (6, 16, True)])
def test_gemm2_gate_addln_emulation(N, tm, tb):
    """_TAIL2_SRC == trimul_tail_addln_ref given the same norm_out output."""
    N3 = _N3()
    rng = np.random.default_rng(N * 7 + tm)
    D, R = 128, N * N
    o = _h(rng, (D, R), 0.05)
    xn, wpo, wgo = _h(rng, (R, D)), _h(rng, (D, D), D ** -0.5), _h(rng, (D, D), D ** -0.5)
    wtb = _h(rng, (4, D), 0.1) if tb else None
    z = rng.normal(size=(R, D)).astype(np.float32)
    lo_, ln = _LN(D, rng), _LN(D, rng)
    es = 2.0 ** -16
    on = np.array(N3._ln_ref(mx.array(o).astype(mx.float32).T, lo_.w, lo_.b, lo_.eps * es).astype(mx.float16))
    got = E.gemm2_gate_addln(on, wpo, xn, wgo, z, np.array(ln.w), np.array(ln.b), ln.eps, tm=tm, nsg=2, wtb=wtb)
    ref = N3.trimul_tail_addln_ref(mx.array(o), lo_, es, mx.array(wpo), mx.array(xn), mx.array(wgo),
                                   mx.array(z).reshape(1, N, N, D), ln, w_tb=None if wtb is None else mx.array(wtb))
    _close(got[0], np.array(ref[0]).reshape(R, D), 2e-3, "z_out")
    _close(got[1], np.array(ref[1].astype(mx.float32)).reshape(R, D), 5e-3, "ln_out")
    if tb:
        _close(got[2], np.array(ref[2].astype(mx.float32)).reshape(4, R), 1e-2, "tb")




# ------------------------------------------------------------------------------------------------- R5-tail
@pytest.mark.parametrize("ch", [2, 4])
@pytest.mark.parametrize("N,tb", [(7, False), (6, True)])
def test_tail3_chunked_emulation(ch, N, tb):
    """R5 tail3 order (p, then g, in column chunks; fp16 p overwritten by y16): bitwise the shipped order's outputs
    (each accumulator sees the same k sequence) and == the reference."""
    N3 = _N3()
    rng = np.random.default_rng(N * 5 + ch)
    D, R = 128, N * N
    o = _h(rng, (D, R), 0.05)
    xn, wpo, wgo = _h(rng, (R, D)), _h(rng, (D, D), D ** -0.5), _h(rng, (D, D), D ** -0.5)
    wtb = _h(rng, (4, D), 0.1) if tb else None
    z = rng.normal(size=(R, D)).astype(np.float32)
    lo_, ln = _LN(D, rng), _LN(D, rng)
    es = 2.0 ** -16
    args = (o, np.array(lo_.w), np.array(lo_.b), lo_.eps * es, wpo, xn, wgo, z, np.array(ln.w), np.array(ln.b), ln.eps)
    a = E.trimul_tail_addln(*args, tm=16, nsg=2, wtb=wtb)
    b = E.trimul_tail_addln(*args, tm=16, nsg=2, wtb=wtb, ch=ch)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)
    ref = N3.trimul_tail_addln_ref(mx.array(o), lo_, es, mx.array(wpo), mx.array(xn), mx.array(wgo),
                                   mx.array(z).reshape(1, N, N, D), ln, w_tb=None if wtb is None else mx.array(wtb))
    _close(b[0], np.array(ref[0]).reshape(R, D), 2e-3, "z_out")


def test_tail3_source_variants():
    """tail3 / stage-cut / diagnostic sources are built from the shipped tail source by exact anchors."""
    N3 = _N3()
    for unrk in (True, False):
        for st in (3, 4, 5):
            s = N3._tail3_src(unrk, st)
            assert "__KLOOP__" not in s and "__STAGE" not in s and s.count("#define YVAL") == 1
            assert ("return; }" in s) == (st != 5)
    for k in ("g_only", "p_dead"):
        assert "sk += sk_p;" in N3._tail_diag_src(k)
    assert N3._maxt_header(128).rstrip().endswith("#define kernel kernel, max_total_threads_per_threadgroup(128)")


# ------------------------------------------------------------------------------------------------- R6-trans
@pytest.mark.parametrize("N,nsg,lnx", [(7, 4, True), (5, 2, False), (6, 8, True)])
def test_rtrans_emulation(N, nsg, lnx):
    """_RTRANS_SRC index logic (hidden blocks, W3 block columns, Sh reused as A fragments, moved-back last tile) and the
    in-register epilogue (next LN, s-track LN -> NAX pair bias with a zero second fragment, head-major store) ==
    the transition_lnx reference."""
    N3 = _N3()
    rng = np.random.default_rng(N * 11 + nsg)
    D, H, R = 128, 128, N * N            # H 128 (4 hidden blocks of 32) keeps the emulation small
    xn = _h(rng, (R, D))
    w12, w3 = _h(rng, (2 * H, D), D ** -0.5), _h(rng, (D, H), H ** -0.5)
    z = rng.normal(size=(R, D)).astype(np.float32)
    l1, l2 = _LN(D, rng), _LN(D, rng)
    wz = _h(rng, (16, D), D ** -0.5)
    zo, xo, pb = E.transition_reg(xn, w12, w3, z, (np.array(l1.w), np.array(l1.b), l1.eps) if lnx else None,
                                  (np.array(l2.w), np.array(l2.b), l2.eps) if lnx else None, wz if lnx else None,
                                  nsg=nsg)
    zr, xr, br = N3.transition_lnx_ref(mx.array(xn).reshape(1, N, N, D), mx.array(w12), mx.array(w3),
                                       mx.array(z).reshape(1, N, N, D), next_ln=l1 if lnx else None,
                                       ln_s=l2 if lnx else None, w_s=mx.array(wz) if lnx else None)
    _close(zo, np.array(zr).reshape(R, D), 2e-3, "z_new")
    if lnx:
        _close(xo, np.array(xr.astype(mx.float32)).reshape(R, D), 5e-3, "next LN")
        _close(pb, np.array(br.astype(mx.float32)).reshape(16, R), 1e-2, "pair bias")
    else:
        assert xo is None and pb is None


def test_trans3_dispatch_mlx_ref(monkeypatch):
    """transition3 'mlx' impl on CPU (MLX ops only; lnx_rows -> reference) == the transition reference."""
    N3 = _N3()
    rng = np.random.default_rng(3)
    N, D, H = 9, 128, 512
    xn = mx.array(_h(rng, (1, N, N, D)))
    w12, w3 = mx.array(_h(rng, (2 * H, D), D ** -0.5)), mx.array(_h(rng, (D, H), H ** -0.5))
    z = mx.array(rng.normal(size=(1, N, N, D)).astype(np.float32))
    l1, l2 = _LN(D, rng), _LN(D, rng)
    wz = mx.array(_h(rng, (16, D), D ** -0.5))

    def lnx_ref(zn, next_ln=None, ln_s=None, w_s=None, **_):
        r = N3.transition_lnx_ref(xn, w12, w3, z, next_ln=next_ln, ln_s=ln_s, w_s=w_s)
        return r[1], r[2]

    monkeypatch.setattr(N3, "lnx_rows", lnx_ref)
    for chunk in (0, 30):
        zo, xo, bo = N3.transition3(xn, w12, w3, z, next_ln=l1, ln_s=l2, w_s=wz, cfg=dict(impl="mlx", chunk=chunk))
        zr = N3.transition_lnx_ref(xn, w12, w3, z)[0]
        _close(np.array(zo), np.array(zr), 2e-3, f"mlx chunk {chunk}")
        assert xo is not None and bo is not None
