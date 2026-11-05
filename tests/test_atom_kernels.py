"""kernels_atom (NAX atom encoder / decoder) on the GPU vs the MLX-op references and vs the shipped f16 path:

  BOLTZ_MLX_DEVICE=gpu python -m pytest -q -s tests/test_atom_kernels.py

The index logic was checked lane by lane on the CPU (tests/test_atom_cpu.py); these tests check the Metal semantics
(NAX fragment layout of the relaxed 16x32x16 descriptor, loads / stores, shuffles), the kernels at the production
shapes (lig_aff affinity pass: 5 samples x 1536 atoms, shared bias; prot400: 3072 atoms), bitwise reproducibility,
compiled == eager, and the integrated modules on the recorded prot117 inputs with real weights.
"""

import mlx.core as mx
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(mx.default_device() != mx.gpu, reason="custom kernels need the GPU device")


def _KA():
    from boltz_mlx import kernels_atom as KA

    return KA


def _cmp(out, ref, fro=2e-3, mx_rel=1e-2):
    o = np.array(out.astype(mx.float32), np.float64)
    r = np.array(ref.astype(mx.float32), np.float64)
    assert np.isfinite(o).all()
    rf = np.linalg.norm(o - r) / np.linalg.norm(r)
    mr = np.abs(o - r).max() / np.abs(r).max()
    assert rf <= fro and mr <= mx_rel, (rf, mr)
    return rf


def _inputs(rng, NA, S, brep, n_pad=5, W=32, H=128, heads=4, c=32):
    KA = _KA()
    HC = heads * c
    h = mx.array((rng.normal(size=(S * NA, 4 * HC)) * 0.7).astype(np.float16))
    Bin = S // brep
    am = np.ones((Bin, NA), np.float32)
    am[:, NA - n_pad:] = 0
    km = KA.window_keys(mx.array(am.reshape(Bin * NA, 1)), NA, W, H).reshape(Bin * NA // W, 1, 1, H)
    pb = mx.array(rng.normal(size=(Bin * NA // W, heads, W, H)).astype(np.float32))
    bias = pb + (1 - km) * -1e6
    qb = mx.array((rng.normal(size=HC) * 0.3).astype(np.float32))
    return h, bias, qb


@pytest.mark.parametrize("NA,S,brep", [(96, 2, 2), (1536, 5, 5), (3072, 1, 1), (160, 3, 1)])
@pytest.mark.parametrize("cfg_i", range(10))
def test_atom_attention(NA, S, brep, cfg_i):
    KA = _KA()
    rng = np.random.default_rng(NA + cfg_i)
    h, bias, qb = _inputs(rng, NA, S, brep)
    o = KA.atom_attention(h, bias, NA=NA, qbias=qb, brep=brep, **KA.ATT_CFGS[cfg_i])
    rf = _cmp(o, KA.atom_attention_ref(h, bias, NA=NA, qbias=qb, brep=brep))
    if cfg_i == 0:
        print(f"atom attention NA={NA} S={S}: rel_fro vs ref {rf:.2e}")


@pytest.mark.parametrize("NA,S", [(1536, 5), (96, 4)])
@pytest.mark.parametrize("cfg_i", [3, 8])
def test_atom_attention_shared_bias_bitwise_and_deterministic(NA, S, cfg_i):
    """brep = S (one bias block per input) == the same bias in the MLX path's per-sample layout (sample-major: row =
    sample * NW + window) with brep = 1, bitwise; two runs bitwise equal. (Fixed 2026-10-28: the first version built
    the per-sample bias with mx.repeat(bias, S, axis=0) on the flattened [NW, ...] block, which interleaves windows —
    w0 w0 w0 .. w1 .. — not the production layout; max |d| 1.08 / 1.16 on the GPU, reproduced by the lane emulation.)"""
    from atom_legacy import per_sample

    KA = _KA()
    cfg = KA.ATT_CFGS[cfg_i]
    rng = np.random.default_rng(1)
    h, bias, qb = _inputs(rng, NA, S, S)
    o1 = KA.atom_attention(h, bias, NA=NA, qbias=qb, brep=S, **cfg)
    o2 = KA.atom_attention(h, per_sample(bias, NA, S), NA=NA, qbias=qb, brep=1, **cfg)
    o3 = KA.atom_attention(h, bias, NA=NA, qbias=qb, brep=S, **cfg)
    assert bool(mx.array_equal(o1, o2).item()) and bool(mx.array_equal(o1, o3).item())


@pytest.mark.parametrize("NA,S", [(1536, 5), (3072, 1), (96, 4)])
def test_production_layouts_vs_mlx_path(f16_atom, NA, S):
    """The kernel on the biases production builds — prepare_nax's block per input with brep = multiplicity, and the MLX
    path's per-sample biases (AtomTransformer.prepare) with brep = 1 — vs the shipped f16 attention core (to_keys + MLX
    SDPA fp32, tests/atom_legacy.py) on the same q|k|v|g; then the whole 3-layer AtomTransformer, kernels_atom vs the
    shipped f16 path, checked for EVERY sample (lig_aff affinity pass: 5 samples sharing one input; prot400: 1)."""
    import atom_legacy as AL

    KA = _KA()
    d = AL.production_inputs(NA, S)
    P, P0 = d["P"], d["P0"]
    assert P["attn_brep"] == S
    for l in range(3):
        assert P["attn_bias"][l].shape == (NA // 32, 4, 32, 128) and P0["attn_bias"][l].shape == (S * NA // 32, 4, 32, 128)
        assert bool(mx.array_equal(P0["attn_bias"][l], AL.per_sample(P["attn_bias"][l], NA, S)).item())
    ref = AL.legacy_core(d["h"], P0["attn_bias"][0], d["tkn"], d["b_q"])
    for cfg in (KA.ATT_CFG, KA.ATT_CFGS[8]):
        for bias, brep in ((P["attn_bias"][0], S), (P0["attn_bias"][0], 1)):
            o = KA.atom_attention(d["h"], bias, NA=NA, qbias=d["b_q"], brep=brep, **cfg)
            rf = _cmp(o, ref, 2e-3, 2e-2)
    print(f"NA={NA} S={S}: kernel vs shipped f16 attention core rel_fro {rf:.2e}")
    at, q = d["at"], d["q"]
    out_n = at(q, None, None, None, None, S, prepared=P)
    out_s = at(q, d["c_rep"], d["bias"], d["tk"], d["m_rep"], S, prepared=P0)
    for s in range(S):
        rs = _cmp(out_n[s] - q[s], out_s[s] - q[s], 3e-3, 3e-2)
    print(f"NA={NA} S={S}: AtomTransformer update kernels_atom vs shipped f16, last sample rel_fro {rs:.2e}")


def test_segment_mean_relu():
    KA = _KA()
    rng = np.random.default_rng(2)
    from test_atom_cpu import _att_matrix

    NA, T, S, D = 1536, 213, 5, 768
    att = _att_matrix(rng, NA, T, 2)
    seg_np = KA.token_segments(att)
    y = mx.array(rng.normal(size=(S, NA, D)).astype(np.float32))
    got = KA.segment_mean_relu(y, tuple(mx.array(x) for x in seg_np))
    ref = KA.segment_mean_relu_ref(np.array(y), seg_np)
    np.testing.assert_allclose(np.array(got), ref, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("site", ["qkvg", "o", "ff", "ba"])
def test_gemm_sites_at_atom_shapes(site):
    """kernels_nax3.gemm16 at the atom-transformer shapes (K 128 / 256, NOUT 128 / 256 / 512, M = 5 x 1536 rows,
    conditioning rows shared by 5 samples), every config of the site's list."""
    from boltz_mlx import kernels_nax3 as N3

    rng = np.random.default_rng(3)
    NA, S, D = 1536, 5, 128
    M = S * NA
    h = lambda *s, sc=1.0: mx.array((rng.normal(size=s) * sc).astype(np.float16))  # noqa: E731
    K, NOUT, epi, ngrp, lst = {"qkvg": (D, 512, N3.EPI_STORE, 1, N3.GEMM_CFGS), "o": (D, D, N3.EPI_GRESID, 1, N3.GEMM_CFGS),
                               "ff": (D, 256, N3.EPI_SWIGLU, 3, N3.SWIGLU_CFGS),
                               "ba": (256, D, N3.EPI_GRESID, 1, N3.GEMM_CFGS)}[site]
    x, w = h(M, K), h(ngrp * NOUT, K, sc=K ** -0.5)
    kw = {}
    if epi == N3.EPI_GRESID:
        kw = dict(gate=h(NA, 6 * D), gate_bias=mx.array(rng.normal(size=6 * D).astype(np.float32)), gate_off=3 * D,
                  resid=mx.array(rng.normal(size=(M, D)).astype(np.float32)), ntok=NA, rep=S)
    ref = N3.gemm16_ref(x, w, epi=epi, ngrp=ngrp, **kw)
    for cfg in lst:
        if K % (16 * cfg["ku"]) or NOUT % (cfg["wn"] * cfg["sn"]):
            continue
        _cmp(N3.gemm16(x, w, epi=epi, ngrp=ngrp, **kw, **cfg), ref, 2e-3 if epi != N3.EPI_GRESID else 1e-5, 1e-2)


@pytest.mark.parametrize("cfg_i", range(4))
@pytest.mark.parametrize("M,ntok,rep", [(7680, 1536, 5), (100, 50, 2)])
def test_atom_transition_fused(cfg_i, M, ntok, rep):
    """Fused conditioned transition vs the split references and vs the split NAX kernels (adaln16 -> gemm16 SWIGLU ->
    gemm16 GRESID), incl. a ragged row count (moved-back last tile)."""
    from boltz_mlx import kernels_nax3 as N3

    KA = _KA()
    rng = np.random.default_rng(cfg_i + M)
    L, D, HID = 3, 128, 256
    a2 = mx.array((rng.normal(size=(M, D)) * 2 + 0.3).astype(np.float32))
    rc = N3.RawCond16(mx.array((rng.normal(size=(ntok, L * 4 * D)) * 0.5).astype(np.float16)),
                      mx.array(rng.normal(size=L * 4 * D).astype(np.float32) * 0.3),
                      mx.array((rng.normal(size=(ntok, L * 2 * D)) * 0.5).astype(np.float16)),
                      mx.array(rng.normal(size=L * 2 * D).astype(np.float32) * 0.3), ntok=ntok, L=L, D=D, rep=rep)
    wff = mx.array((rng.normal(size=(3 * HID, D)) * D ** -0.5).astype(np.float16))
    wba = mx.array((rng.normal(size=(D, HID)) * HID ** -0.5).astype(np.float16))
    for layer in (0, 2):
        got = KA.atom_transition(a2, rc, layer, wff, wba, **KA.TRANS_CFGS[cfg_i])
        _cmp(got - a2, KA.atom_transition_ref(a2, rc, layer, wff, wba) - a2, 2e-3, 2e-2)
        t = N3.adaln16(a2, rc, layer, 2)
        hs = N3.gemm16(t, wff, epi=N3.EPI_SWIGLU, ngrp=3, **N3.SWIGLU_CFGS[2])
        split = N3.gemm16(hs, wba, epi=N3.EPI_GRESID, gate=rc.h2, gate_bias=rc.b2, gate_off=(layer * 2 + 1) * D,
                          resid=a2, ntok=ntok, rep=rep, **N3.GEMM_CFGS[0])
        _cmp(got - a2, split - a2, 2e-3, 2e-2)
        assert bool(mx.array_equal(got, KA.atom_transition(a2, rc, layer, wff, wba, **KA.TRANS_CFGS[cfg_i])).item())


@pytest.mark.parametrize("cfg_i", range(3))
@pytest.mark.parametrize("M,ntok,rep", [(7680, 1536, 5), (100, 50, 2)])
def test_adaln_gemm(cfg_i, M, ntok, rep):
    """AdaLN-prologue GEMM vs adaln16 -> gemm16 (kernels and references), ragged rows, bitwise reproducible."""
    from boltz_mlx import kernels_nax3 as N3

    KA = _KA()
    rng = np.random.default_rng(cfg_i + 3 * M)
    L, D = 3, 128
    a2 = mx.array((rng.normal(size=(M, D)) * 2 - 0.4).astype(np.float32))
    rc = N3.RawCond16(mx.array((rng.normal(size=(ntok, L * 4 * D)) * 0.5).astype(np.float16)),
                      mx.array(rng.normal(size=L * 4 * D).astype(np.float32) * 0.3),
                      mx.zeros((ntok, L * 2 * D), mx.float16), mx.zeros((L * 2 * D,)), ntok=ntok, L=L, D=D, rep=rep)
    w = mx.array((rng.normal(size=(512, D)) * D ** -0.5).astype(np.float16))
    for layer in (0, 2):
        got = KA.adaln_gemm(a2, rc, layer, 0, w, **KA.ADALN_GEMM_CFGS[cfg_i])
        _cmp(got, KA.adaln_gemm_ref(a2, rc, layer, 0, w), 2e-3, 1e-2)
        _cmp(got, N3.gemm16(N3.adaln16(a2, rc, layer, 0), w, **N3.GEMM_CFGS[0]), 2e-3, 1e-2)
        assert bool(mx.array_equal(got, KA.adaln_gemm(a2, rc, layer, 0, w, **KA.ADALN_GEMM_CFGS[cfg_i])).item())


def test_adaln16_d128_shared_rows():
    from boltz_mlx import kernels_nax3 as N3

    rng = np.random.default_rng(4)
    L, D, NA, S = 3, 128, 1536, 5
    a2 = mx.array(rng.normal(size=(S * NA, D)).astype(np.float32) * 2 + 0.5)
    rc = N3.RawCond16(mx.array(rng.normal(size=(NA, L * 4 * D)).astype(np.float16)),
                      mx.array(rng.normal(size=L * 4 * D).astype(np.float32)),
                      mx.array(rng.normal(size=(NA, L * 2 * D)).astype(np.float16)),
                      mx.array(rng.normal(size=L * 2 * D).astype(np.float32)), ntok=NA, L=L, D=D, rep=S)
    for layer, slot in ((0, 0), (2, 2)):
        _cmp(N3.adaln16(a2, rc, layer, slot), N3.adaln16_ref(a2, rc, layer, slot), 2e-3, 1e-2)


# ------------------------------------------------------------------------------------------------- integration
@pytest.fixture
def f16_atom(monkeypatch):
    from boltz_mlx import kernels as K

    monkeypatch.setattr(K, "PRECISION", "f16")
    monkeypatch.setitem(K.NAX_FLAGS, "ATOM", "1")
    prev = K.DIFF_DTYPE
    K.set_diff_dtype("fp16")
    yield K
    K.DIFF_DTYPE = prev


def _rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def test_encoder_decoder_vs_shipped_and_torch(f16_atom, monkeypatch):
    """Real weights, recorded prot117 encoder / decoder inputs, 2 samples: NAX path vs shipped f16 path vs the torch
    fp32 dump; compiled == eager bitwise."""
    from boltz_cfg import H, W
    from conftest import load_dump, load_feats, mxa
    from test_atom_cpu import _load
    from boltz_mlx.modules.encodersv2 import make_to_keys

    K = f16_atom
    feats = load_feats("prot117")
    d = load_dump("structure_module.score_model.atom_attention_encoder")
    enc = _load("enc")
    n = d["kw_q"].shape[1]
    tk = make_to_keys(n, W, H)
    q, c, eb, r1 = mxa(d["kw_q"]), mxa(d["kw_c"]), mxa(d["kw_atom_enc_bias"]), mxa(d["kw_r"])
    r = mx.concatenate([r1, r1 * 0.9 + 0.3])
    res = {}
    for flag in ("1", "0"):
        monkeypatch.setitem(K.NAX_FLAGS, "ATOM", flag)
        P = enc.prepare(feats, c, eb, tk, 2)
        assert bool(P.get("nax_atom")) == (flag == "1")

        def f(r_, P=P):
            return enc(feats=feats, q=q, c=c, atom_enc_bias=eb, to_keys=tk, r=r_, multiplicity=2, prepared=P)[:2]

        a_, q_ = f(r)
        if flag == "1":
            a_c, q_c = mx.compile(f)(r)
            assert bool(mx.array_equal(a_c, a_).item()) and bool(mx.array_equal(q_c, q_).item())
        res[flag] = (a_, q_)
    ta = (_rel(np.array(res["1"][0][:1]), d["out_0"]), _rel(np.array(res["0"][0][:1]), d["out_0"]))
    print(f"encoder a vs torch fp32: nax {ta[0]:.2e}, shipped f16 {ta[1]:.2e}; nax vs shipped "
          f"{_rel(np.array(res['1'][0]), np.array(res['0'][0])):.2e}")
    assert ta[0] <= max(1.5 * ta[1], ta[1] + 1e-4) and ta[0] < 3e-3
    q2 = mx.repeat(q, 2, 0)
    for s in range(2):  # every sample (shared conditioning / bias, brep = 2) vs the shipped f16 path
        assert _rel(np.array(res["1"][0][s]), np.array(res["0"][0][s])) < 2e-3
        assert _rel(np.array(res["1"][1][s] - q2[s]), np.array(res["0"][1][s] - q2[s])) < 3e-3

    d = load_dump("structure_module.score_model.atom_attention_decoder")
    dec = _load("dec")
    a1, q1, c, db = mxa(d["kw_a"]), mxa(d["kw_q"]), mxa(d["kw_c"]), mxa(d["kw_atom_dec_bias"])
    a, qq = mx.concatenate([a1, a1 * 1.1]), mx.concatenate([q1, q1 * 0.95])
    out = {}
    for flag in ("1", "0"):
        monkeypatch.setitem(K.NAX_FLAGS, "ATOM", flag)
        P = dec.prepare(feats, c, db, tk, 2)
        cc = c if flag == "1" else mx.repeat(c, 2, 0)

        def g(q_, P=P, cc=cc):
            return dec(a=a, q=q_, c=cc, atom_dec_bias=db, feats=feats, to_keys=tk, multiplicity=2, prepared=P)

        o = g(qq)
        if flag == "1":
            assert bool(mx.array_equal(mx.compile(g)(qq), o).item())
        out[flag] = o
    to = (_rel(np.array(out["1"][:1]), d["out"]), _rel(np.array(out["0"][:1]), d["out"]))
    print(f"decoder vs torch fp32: nax {to[0]:.2e}, shipped f16 {to[1]:.2e}")
    assert to[0] <= max(1.5 * to[1], to[1] + 1e-4) and to[0] < 3e-3
    for s in range(2):
        assert _rel(np.array(out["1"][s]), np.array(out["0"][s])) < 3e-3


def test_transformer_fused_sites_compiled(f16_atom, monkeypatch):
    """Random-init AtomTransformer at the lig_aff shape: fused sites (aqkvg, trans) vs split sites within fp16 noise;
    each variant compiled == eager bitwise."""
    from boltz_mlx.modules.encodersv2 import make_to_keys
    from boltz_mlx.modules.transformersv2 import AtomTransformer

    KA = _KA()
    mx.random.seed(0)
    at = AtomTransformer(attn_window_queries=32, attn_window_keys=128, dim=128, dim_single_cond=128, depth=3, heads=4)
    rng = np.random.default_rng(6)
    for L in at.diffusion_transformer.layers:
        for lin in (L.adaln.s_scale, L.transition.adaln.s_scale, L.output_projection_linear,
                    L.transition.output_projection[0]):
            lin.weight = mx.array(rng.normal(size=lin.weight.shape).astype(np.float32) * 0.05)
    NA, S = 1536, 5
    c = mx.array(rng.normal(size=(1, NA, 128)).astype(np.float32))
    bias = mx.array(rng.normal(size=(1, NA // 32, 32, 128, 12)).astype(np.float32))
    mask = mx.array(np.r_[np.ones(NA - 2), np.zeros(2)].astype(np.float32)[None])
    q = mx.array(rng.normal(size=(S, NA, 128)).astype(np.float32))
    assert at.nax_ok(NA)
    P = at.prepare_nax(c, bias, make_to_keys(NA, 32, 128), mask, S)
    outs = {}
    for name, sites in (("split", {"trans": "split", "aqkvg": "split"}),
                        ("fused", {"trans": KA.TRANS_CFGS[0], "aqkvg": KA.ADALN_GEMM_CFGS[0]})):
        for k, v in sites.items():
            monkeypatch.setitem(KA.GEMM_CFG, k, v)
        f = lambda q_: at(q_, None, None, None, None, S, prepared=P)  # noqa: E731
        e = f(q)
        assert bool(mx.array_equal(mx.compile(f)(q), e).item()), name
        outs[name] = e
    rf = _cmp(outs["fused"] - q, outs["split"] - q, 2e-3, 2e-2)
    print(f"atom transformer fused sites vs split: update rel_fro {rf:.2e}")


def test_flags_off_outside_f16():
    """Without --precision f16 the ATOM group is off unless forced by env (and then still needs fp16 diffusion)."""
    from boltz_mlx import kernels as K

    if K.NAX_FLAGS.get("ATOM") is None and K.PRECISION != "f16":
        assert not K.nax("ATOM")
