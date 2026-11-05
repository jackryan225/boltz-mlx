"""kernels_atom (NAX atom encoder / decoder, --precision f16 + BOLTZ_MLX_NAX_ATOM), CPU-side checks; the GPU tests are
tests/test_atom_kernels.py.

1. Lane-level emulation (tests/atom_emul.py) of the windowed NAX attention and the segment kernel == the MLX-op
   references (fragment rows / columns, key-window atom rows incl. clamping at the sequence ends, shared-bias windows,
   online softmax over key blocks, gated store), for every listed attention config.
2. The key-window map of the kernel (atom k * W - (H - W) / 2 + j) == encodersv2.ToKeys (upstream's indexing matrix).
3. Glue: segment mean / gather == the dense one-hot matmuls of the MLX path.
4. Wiring with the Metal kernels replaced by their references: the AtomAttentionEncoder / Decoder and the whole score
   model (multiplicity 2, shared times) on the recorded prot117 inputs and real weights == the shipped f16 MLX path
   within fp16 noise; eager ~= mx.compile (CPU compile fuses the reference ops differently; bitwise on the GPU test).
Small shapes only (CPU).
"""

import mlx.core as mx
import numpy as np
import pytest

import atom_emul as E

pytestmark = pytest.mark.skipif(mx.default_device() != mx.cpu, reason="CPU logic checks")


@pytest.fixture(autouse=True)
def _no_metal_dispatch(monkeypatch):
    """mx.fast.metal_kernel runs on the GPU stream whatever the default device is: a CPU check must never call one."""

    def refuse(*a, **kw):
        raise AssertionError(f"CPU test tried to build a Metal kernel: {kw.get('name', a[:1])}")

    monkeypatch.setattr(mx.fast, "metal_kernel", refuse)
    # the wiring checks swap the split kernels for references; the fused AdaLN+q|k|v|g site (atom_cfg.json may select it
    # after the GPU A/B) is checked by its own lane emulation here and by tests/test_atom_kernels.py on the GPU
    from boltz_mlx import kernels_atom as KA

    monkeypatch.setitem(KA.GEMM_CFG, "aqkvg", "split")


def _KA():
    from boltz_mlx import kernels_atom as KA

    return KA


def _rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    assert np.isfinite(a).all(), "non-finite / unwritten elements"
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def _attn_inputs(rng, NA, S, brep, W=32, H=128, heads=4, c=32, n_pad=5):
    """h [S * NA, 4 * heads * c] fp16, bias [S / brep * NA / W, heads, W, H] = pair bias + -1e6 key mask (keys outside
    the sequence and the last n_pad padded atoms), as DiffusionTransformer.full_attn_bias builds it."""
    KA = _KA()
    HC = heads * c
    h = (rng.normal(size=(S * NA, 4 * HC)) * 0.7).astype(np.float16)
    Bin = S // brep
    am = np.ones((Bin, NA), np.float32)
    am[:, NA - n_pad:] = 0
    km = np.array(KA.window_keys(mx.array(am.reshape(Bin * NA, 1)), NA, W, H)).reshape(Bin * NA // W, 1, 1, H)
    pb = rng.normal(size=(Bin * NA // W, heads, W, H)).astype(np.float32)
    bias = pb + (1 - km) * -1e6
    qb = (rng.normal(size=HC) * 0.3).astype(np.float32)
    return h, bias.astype(np.float32), qb


@pytest.mark.parametrize("NA,S,brep", [(96, 2, 2), (64, 3, 1), (160, 1, 1)])
@pytest.mark.parametrize("cfg_i", range(10))
def test_attention_emulation(NA, S, brep, cfg_i):
    KA = _KA()
    cfg = KA.ATT_CFGS[cfg_i]
    rng = np.random.default_rng(NA + S + cfg_i)
    h, bias, qb = _attn_inputs(rng, NA, S, brep)
    got = E.atom_attention(h, bias, NA=NA, qbias=qb, brep=brep, **cfg)
    ref = KA.atom_attention_ref(mx.array(h), mx.array(bias), NA=NA, qbias=mx.array(qb), brep=brep)
    r = _rel(got.astype(np.float32), np.array(ref.astype(mx.float32)))
    assert r < 2e-3, r


def test_attention_ref_vs_dense_softmax():
    """atom_attention_ref (windows via window_keys) == a dense per-sample softmax over all atoms with the bias scattered
    to the key atoms (keys outside the window at -inf): checks the window/key/bias bookkeeping of the reference."""
    KA = _KA()
    rng = np.random.default_rng(7)
    NA, S, W, H, heads, c = 96, 2, 32, 128, 4, 32
    HC = heads * c
    h, bias, qb = _attn_inputs(rng, NA, S, 2)
    ref = np.array(KA.atom_attention_ref(mx.array(h), mx.array(bias), NA=NA, qbias=mx.array(qb), brep=2)
                   .astype(mx.float32))
    hf = h.astype(np.float32)
    out = np.zeros((S * NA, HC), np.float32)
    for s in range(S):
        rows = hf[s * NA:(s + 1) * NA]
        q = (rows[:, :HC] + qb).astype(np.float16).astype(np.float32)
        for i in range(NA):
            k_win = i // W
            dense = np.full((heads, NA), -np.inf, np.float32)
            for j in range(H):
                a = k_win * W - (H - W) // 2 + j
                if 0 <= a < NA:
                    dense[:, a] = bias[k_win, :, i % W, j]  # Bin = 1: bias window = k_win
            for hh in range(heads):
                sl = slice(hh * c, (hh + 1) * c)
                lg = rows[:, HC:2 * HC][:, sl] @ q[i, sl] * c ** -0.5 + dense[hh]
                p = np.exp(lg - lg.max())
                p /= p.sum()
                g = 1 / (1 + np.exp(-rows[i, 3 * HC:][sl]))
                out[s * NA + i, sl] = (p @ rows[:, 2 * HC:3 * HC][:, sl]) * g
    assert _rel(ref, out) < 2e-3


@pytest.mark.parametrize("cfg_i", range(4))
@pytest.mark.parametrize("layer", [0, 2])
def test_transition_emulation(cfg_i, layer):
    """Fused conditioned transition (AdaLN prologue, up GEMM 3 groups + SwiGLU, down GEMM, gated residual) lane by lane
    == the split composition adaln16 -> gemm16 SWIGLU -> gemm16 GRESID (references), incl. the moved-back last tile
    (M = 80 rows, not a multiple of 32) and conditioning rows shared by 2 samples."""
    from boltz_mlx import kernels_nax3 as N3

    KA = _KA()
    cfg = KA.TRANS_CFGS[cfg_i]
    rng = np.random.default_rng(cfg_i * 7 + layer)
    L, D, HID, ntok, rep = 3, 128, 256, 40, 2
    M = ntok * rep
    a2 = (rng.normal(size=(M, D)) * 2 + 0.3).astype(np.float32)
    h1 = (rng.normal(size=(ntok, L * 4 * D)) * 0.5).astype(np.float16)
    b1 = rng.normal(size=L * 4 * D).astype(np.float32) * 0.3
    h2 = (rng.normal(size=(ntok, L * 2 * D)) * 0.5).astype(np.float16)
    b2 = rng.normal(size=L * 2 * D).astype(np.float32) * 0.3
    wff = (rng.normal(size=(3 * HID, D)) * D ** -0.5).astype(np.float16)
    wba = (rng.normal(size=(D, HID)) * HID ** -0.5).astype(np.float16)
    got = E.atom_transition(a2, h1, b1, h2, b2, wff, wba, ntok=ntok, rep=rep, L=L, layer=layer, **cfg)
    rc = N3.RawCond16(mx.array(h1), mx.array(b1), mx.array(h2), mx.array(b2), ntok=ntok, L=L, D=D, rep=rep)
    ref = np.array(KA.atom_transition_ref(mx.array(a2), rc, layer, mx.array(wff), mx.array(wba)))
    r = _rel(got - a2, ref - a2)   # relative to the update
    assert r < 2e-3, r


@pytest.mark.parametrize("cfg_i", range(3))
def test_adaln_gemm_emulation(cfg_i):
    """AdaLN prologue + GEMM (site aqkvg) lane by lane == adaln16 -> gemm16 EPI_STORE references (ragged rows, shared
    conditioning rows, slot offsets)."""
    from boltz_mlx import kernels_nax3 as N3

    KA = _KA()
    cfg = KA.ADALN_GEMM_CFGS[cfg_i]
    rng = np.random.default_rng(11 + cfg_i)
    L, D, NOUT, ntok, rep = 3, 128, 512, 45, 2
    M = ntok * rep
    a2 = (rng.normal(size=(M, D)) * 2 - 0.4).astype(np.float32)
    h1 = (rng.normal(size=(ntok, L * 4 * D)) * 0.5).astype(np.float16)
    b1 = rng.normal(size=L * 4 * D).astype(np.float32) * 0.3
    w = (rng.normal(size=(NOUT, D)) * D ** -0.5).astype(np.float16)
    rc = N3.RawCond16(mx.array(h1), mx.array(b1), mx.zeros((ntok, L * 2 * D), mx.float16), mx.zeros((L * 2 * D,)),
                      ntok=ntok, L=L, D=D, rep=rep)
    for layer, slot in ((1, 0), (2, 2)):
        got = E.adaln_gemm(a2, h1, b1, w, ntok=ntok, rep=rep, L=L, layer=layer, slot=slot, **cfg)
        ref = np.array(KA.adaln_gemm_ref(mx.array(a2), rc, layer, slot, mx.array(w)).astype(mx.float32))
        assert _rel(got.astype(np.float32), ref) < 2e-3


@pytest.mark.parametrize("N,D,B", [(64, 3, 1), (96, 8, 2), (928, 16, 1)])
def test_window_keys_equals_to_keys(N, D, B):
    from boltz_mlx.modules.encodersv2 import make_to_keys

    KA = _KA()
    rng = np.random.default_rng(N)
    x = mx.array(rng.normal(size=(B, N, D)).astype(np.float32))
    a = make_to_keys(N, 32, 128)(x)
    g = KA.window_keys(x.reshape(B * N, D), N, 32, 128)
    np.testing.assert_array_equal(np.array(a).reshape(-1, 128, D), np.array(g))


def _att_matrix(rng, NA, T, n_pad):
    """One-hot atom -> token [NA, T] with contiguous variable-size tokens and n_pad padded (all-zero) atoms."""
    sizes = rng.integers(1, 15, size=T)
    while sizes.sum() > NA - n_pad:
        sizes[rng.integers(T)] = max(1, sizes[rng.integers(T)] - 1)
        if sizes.sum() > NA - n_pad and (sizes == 1).all():
            break
    att = np.zeros((NA, T), np.float32)
    i = 0
    for t, s in enumerate(sizes):
        att[i:i + s, t] = 1
        i += s
    return att


def test_segments_and_gather_vs_dense():
    KA = _KA()
    rng = np.random.default_rng(3)
    NA, T, S, D = 96, 12, 3, 24
    att = _att_matrix(rng, NA, T, 9)
    y = rng.normal(size=(S, NA, D)).astype(np.float32)
    off, idx, wt = KA.token_segments(att)
    got = E.segment_mean_relu(y, off, idx, wt)
    ref = KA.segment_mean_relu_ref(y, (off, idx, wt))
    np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-7)
    # the MLX path: (att / (att.sum(atoms) + 1e-6))^T @ relu(y)
    a2t = mx.array(att)[None]
    mean = a2t / (a2t.sum(axis=1, keepdims=True) + 1e-6)
    dense = np.array(mx.swapaxes(mx.repeat(mean, S, axis=0), 1, 2) @ mx.maximum(mx.array(y), 0))
    np.testing.assert_allclose(got, dense, rtol=1e-5, atol=1e-6)
    # decoder: gather == one-hot matmul, exactly
    tok, valid = KA.atom_tokens(att)
    x = mx.array(rng.normal(size=(S, T, D)).astype(np.float32))
    g = mx.where(mx.array(valid)[None, :, None], mx.take(x, mx.array(tok), axis=1), 0.0)
    np.testing.assert_array_equal(np.array(g), np.array(mx.repeat(a2t, S, axis=0) @ x))


def test_segments_real_feats():
    """token_segments on the recorded prot117 atom_to_token: every valid atom in exactly one segment, ascending."""
    from conftest import load_feats

    KA = _KA()
    att = np.array(load_feats("prot117")["atom_to_token"][0]).astype(np.float32)
    off, idx, wt = KA.token_segments(att)
    assert off[-1] == int((att.sum(1) > 0).sum()) and (np.diff(off) >= 1).all()
    for t in range(att.shape[1]):
        seg = idx[off[t]:off[t + 1]]
        assert (np.diff(seg) > 0).all() and (att[seg, t] == 1).all()


# ------------------------------------------------------------------------------------------------- wiring
def _seg_mlx(y, seg):
    """MLX-op segment mean (dense matrix from the CSR arrays) — compile-safe stand-in for the Metal kernel."""
    off, idx, wt = (np.array(x) for x in seg)
    T, NA = wt.shape[0], y.shape[1]
    m = np.zeros((T, NA), np.float32)
    for t in range(T):
        m[t, idx[off[t]:off[t + 1]]] = wt[t]
    return mx.array(m)[None] @ mx.maximum(y, 0)


@pytest.fixture
def refs(monkeypatch):
    """Route the kernels_atom path on the CPU: flag check passes, every Metal kernel -> its MLX reference."""
    from boltz_mlx import kernels as K
    from boltz_mlx import kernels_nax3 as N3
    from boltz_mlx.modules.transformersv2 import AtomTransformer

    KA = _KA()
    orig_ok = AtomTransformer.nax_ok
    monkeypatch.setattr(AtomTransformer, "nax_ok",
                        lambda self, n: n % self.attn_window_queries == 0 and K.DIFF_DTYPE == mx.float16
                        and KA.supported(self.diffusion_transformer, self.attn_window_queries, self.attn_window_keys))
    monkeypatch.setattr(N3, "adaln16", lambda a2, rc, l, s: N3.adaln16_ref(a2, rc, l, s))
    monkeypatch.setattr(N3, "gemm16", lambda x, w, **kw: N3.gemm16_ref(x, w, **kw))
    monkeypatch.setattr(KA, "atom_attention", KA.atom_attention_ref)
    monkeypatch.setattr(KA, "segment_mean_relu", _seg_mlx)
    prev = K.DIFF_DTYPE
    K.set_diff_dtype("fp16")
    yield orig_ok
    K.DIFF_DTYPE = prev


def _load(kind):
    from boltz_cfg import ATOM_S, H, TOKEN_S, W
    from boltz_mlx.modules.encodersv2 import AtomAttentionDecoder, AtomAttentionEncoder
    from boltz_mlx.weights import load_module

    sm = "structure_module.score_model"
    if kind == "enc":
        mod = AtomAttentionEncoder(atom_s=ATOM_S, token_s=TOKEN_S, atoms_per_window_queries=W, atoms_per_window_keys=H,
                                   atom_encoder_depth=3, atom_encoder_heads=4, structure_prediction=True)
        return load_module(mod, f"{sm}.atom_attention_encoder")
    mod = AtomAttentionDecoder(atom_s=ATOM_S, token_s=TOKEN_S, attn_window_queries=W, attn_window_keys=H,
                               atom_decoder_depth=3, atom_decoder_heads=4)
    return load_module(mod, f"{sm}.atom_attention_decoder")


def _no_nax(monkeypatch):
    from boltz_mlx.modules.transformersv2 import AtomTransformer

    monkeypatch.setattr(AtomTransformer, "nax_ok", lambda self, n: False)


def test_encoder_wiring(refs, monkeypatch):
    from boltz_cfg import H, W
    from conftest import load_dump, load_feats, mxa
    from boltz_mlx.modules.encodersv2 import make_to_keys

    d = load_dump("structure_module.score_model.atom_attention_encoder")
    feats = load_feats("prot117")
    mod = _load("enc")
    n = d["kw_q"].shape[1]
    tk = make_to_keys(n, W, H)
    q, c, eb = mxa(d["kw_q"]), mxa(d["kw_c"]), mxa(d["kw_atom_enc_bias"])
    r1 = mxa(d["kw_r"])
    r = mx.concatenate([r1, r1 * 0.9 + 0.3])          # 2 samples sharing conditioning / bias
    P = mod.prepare(feats, c, eb, tk, 2)
    assert P.get("nax_atom") and P["attn_brep"] == 2 and P["attn_bias"][0].shape[0] == n // W
    a_n, q_n, _, _ = mod(feats=feats, q=q, c=c, atom_enc_bias=eb, to_keys=tk, r=r, multiplicity=2, prepared=P)
    f = mx.compile(lambda r_: mod(feats=feats, q=q, c=c, atom_enc_bias=eb, to_keys=tk, r=r_, multiplicity=2,
                                  prepared=P)[0])
    assert _rel(np.array(f(r)), np.array(a_n)) < 1e-3  # CPU compile fuses the reference ops differently (GPU: bitwise)
    _no_nax(monkeypatch)
    P0 = mod.prepare(feats, c, eb, tk, 2)
    assert not P0.get("nax_atom")
    a_s, q_s, _, _ = mod(feats=feats, q=q, c=c, atom_enc_bias=eb, to_keys=tk, r=r, multiplicity=2, prepared=P0)
    ra, rq = _rel(np.array(a_n), np.array(a_s)), _rel(np.array(q_n - mx.repeat(q, 2, 0)), np.array(q_s - mx.repeat(q, 2, 0)))
    # vs the torch fp32 dump (sample 0 = the recorded r)
    ta = _rel(np.array(a_n[:1]), d["out_0"])
    print(f"encoder nax(refs) vs shipped f16 (CPU): a {ra:.2e}, q update {rq:.2e}; a vs torch fp32 {ta:.2e} "
          f"(shipped {_rel(np.array(a_s[:1]), d['out_0']):.2e})")
    assert ra < 3e-3 and rq < 3e-3 and ta < 5e-3


def test_decoder_wiring(refs, monkeypatch):
    from boltz_cfg import H, W
    from conftest import load_dump, load_feats, mxa
    from boltz_mlx.modules.encodersv2 import make_to_keys

    d = load_dump("structure_module.score_model.atom_attention_decoder")
    feats = load_feats("prot117")
    mod = _load("dec")
    n = d["kw_q"].shape[1]
    tk = make_to_keys(n, W, H)
    a1, q1, c = mxa(d["kw_a"]), mxa(d["kw_q"]), mxa(d["kw_c"])
    db = mxa(d["kw_atom_dec_bias"])
    a = mx.concatenate([a1, a1 * 1.1])
    q = mx.concatenate([q1, q1 * 0.95])
    P = mod.prepare(feats, c, db, tk, 2)
    assert P.get("nax_atom")
    o_n = mod(a=a, q=q, c=c, atom_dec_bias=db, feats=feats, to_keys=tk, multiplicity=2, prepared=P)
    f = mx.compile(lambda q_: mod(a=a, q=q_, c=c, atom_dec_bias=db, feats=feats, to_keys=tk, multiplicity=2,
                                  prepared=P))
    assert _rel(np.array(f(q)), np.array(o_n)) < 1e-3
    _no_nax(monkeypatch)
    P0 = mod.prepare(feats, c, db, tk, 2)
    o_s = mod(a=a, q=q, c=mx.repeat(c, 2, 0), atom_dec_bias=db, feats=feats, to_keys=tk, multiplicity=2, prepared=P0)
    ro = _rel(np.array(o_n), np.array(o_s))
    to = _rel(np.array(o_n[:1]), d["out"])
    print(f"decoder nax(refs) vs shipped f16 (CPU): {ro:.2e}; vs torch fp32 {to:.2e} "
          f"(shipped {_rel(np.array(o_s[:1]), d['out']):.2e})")
    assert ro < 3e-3 and to < 5e-3


def test_score_model_wiring(refs, monkeypatch):
    """Whole DiffusionModule step (multiplicity 2, shared noise level) on the recorded prot117 conditioning: the
    kernels_atom path (refs) incl. s_to_a on the shared rows == the shipped f16 path within fp16 noise."""
    from boltz_cfg import H, SCORE_MODEL_ARGS, W
    from conftest import load_dump, load_feats, mxa
    from boltz_mlx.modules.diffusionv2 import DiffusionModule
    from boltz_mlx.modules.encodersv2 import make_to_keys
    from boltz_mlx.weights import load_module

    sm = "structure_module.score_model"
    d = load_dump(sm)
    feats = load_feats("prot117")
    mod = load_module(DiffusionModule(**SCORE_MODEL_ARGS), sm)
    n = d["kw_r_noisy"].shape[1]
    dc = dict(q=mxa(d["kw_diffusion_conditioning_q"]), c=mxa(d["kw_diffusion_conditioning_c"]),
              atom_enc_bias=mxa(d["kw_diffusion_conditioning_atom_enc_bias"]),
              atom_dec_bias=mxa(d["kw_diffusion_conditioning_atom_dec_bias"]),
              token_trans_bias=mxa(d["kw_diffusion_conditioning_token_trans_bias"]), to_keys=make_to_keys(n, W, H))
    r1 = mxa(d["kw_r_noisy"])[:1]
    r = mx.concatenate([r1, r1 * 0.8])
    times = mx.repeat(mxa(d["kw_times"]).reshape(-1)[:1], 2)
    kw = dict(s_inputs=mxa(d["kw_s_inputs"]), s_trunk=mxa(d["kw_s_trunk"]), feats=feats, multiplicity=2)

    def run():
        cond = {**dc, "shared_times": True}
        cond["prepared"] = mod.prepare(feats, cond, 2)
        return mod(r_noisy=r, times=times, diffusion_conditioning=cond, **kw), cond["prepared"]

    out_n, Pn = run()
    assert Pn["enc"].get("nax_atom") and Pn["dec"].get("nax_atom")
    _no_nax(monkeypatch)
    out_s, Ps = run()
    assert not Ps["enc"].get("nax_atom")
    ro = _rel(np.array(out_n), np.array(out_s))
    print(f"score model nax(refs) vs shipped f16 (CPU): {ro:.2e}")
    assert ro < 3e-3


@pytest.mark.parametrize("NA,S", [(1536, 5), (3072, 1), (96, 4)])
def test_production_layouts_vs_mlx_path(refs, NA, S):
    """Independent of atom_attention_ref's own bookkeeping: on the biases production builds (prepare_nax: one block per
    input, brep = multiplicity; AtomTransformer.prepare: the MLX path's per-sample biases, sample-major, brep = 1), the
    reference and the kernel's lane emulation == the shipped f16 attention core (to_keys + MLX SDPA, atom_legacy.py);
    the per-sample biases == the shared block repeated sample-major (bitwise); the whole AtomTransformer (refs) == the
    shipped f16 path for every sample."""
    import atom_legacy as AL

    KA = _KA()
    d = AL.production_inputs(NA, S)
    P, P0 = d["P"], d["P0"]
    assert P["attn_brep"] == S
    for l in range(3):
        assert P["attn_bias"][l].shape == (NA // 32, 4, 32, 128)
        np.testing.assert_array_equal(np.array(P0["attn_bias"][l]), np.array(AL.per_sample(P["attn_bias"][l], NA, S)))
    ref = np.array(AL.legacy_core(d["h"], P0["attn_bias"][0], d["tkn"], d["b_q"]))
    for bias, brep in ((P["attn_bias"][0], S), (P0["attn_bias"][0], 1)):
        o = KA.atom_attention_ref(d["h"], bias, NA=NA, qbias=d["b_q"], brep=brep)
        assert _rel(np.array(o.astype(mx.float32)), ref) < 2e-3
    if NA == 96:  # the kernel's index math on both production layouts
        hb = (np.array(d["h"]), np.array(d["b_q"]))
        e1 = E.atom_attention(hb[0], np.array(P["attn_bias"][0]), NA=NA, qbias=hb[1], brep=S, **KA.ATT_CFG)
        e2 = E.atom_attention(hb[0], np.array(P0["attn_bias"][0]), NA=NA, qbias=hb[1], brep=1, **KA.ATT_CFG)
        e3 = E.atom_attention(hb[0], np.array(P["attn_bias"][0]), NA=NA, qbias=hb[1], brep=S, order=1, hpt=4, nkf=4)
        assert np.array_equal(e1, e2) and np.array_equal(e1, e3)
        assert _rel(e1.astype(np.float32), ref) < 2e-3
        # the layout the first GPU test used by mistake (windows interleaved) is NOT equivalent
        wrong = np.array(mx.repeat(P["attn_bias"][0], S, axis=0))
        e4 = E.atom_attention(hb[0], wrong, NA=NA, qbias=hb[1], brep=1, **KA.ATT_CFG)
        assert _rel(e4.astype(np.float32), ref) > 0.05
    at, q = d["at"], d["q"]
    out_n = np.array(at(q, None, None, None, None, S, prepared=P))
    out_s = np.array(at(q, d["c_rep"], d["bias"], d["tk"], d["m_rep"], S, prepared=P0))
    qn = np.array(q)
    for s in range(S):
        assert _rel(out_n[s] - qn[s], out_s[s] - qn[s]) < 3e-3


def test_fused_transition_site_wiring(refs, monkeypatch):
    """transformer_forward with the fused-transition site (kernel -> its lane emulation) == the split sites (references)
    on a random-init AtomTransformer: slot / gate offsets per layer, weights, shared conditioning rows."""
    from boltz_mlx import kernels_nax3 as N3
    from boltz_mlx.modules.encodersv2 import make_to_keys
    from boltz_mlx.modules.transformersv2 import AtomTransformer

    KA = _KA()
    mx.random.seed(0)
    at = AtomTransformer(attn_window_queries=32, attn_window_keys=128, dim=128, dim_single_cond=128, depth=3, heads=4)
    rng = np.random.default_rng(5)
    for L in at.diffusion_transformer.layers:
        for lin in (L.adaln.s_scale, L.transition.adaln.s_scale, L.output_projection_linear,
                    L.transition.output_projection[0]):
            lin.weight = mx.array(rng.normal(size=lin.weight.shape).astype(np.float32) * 0.05)
    NA, S = 64, 2
    c = mx.array(rng.normal(size=(1, NA, 128)).astype(np.float32))
    bias = mx.array(rng.normal(size=(1, NA // 32, 32, 128, 12)).astype(np.float32))
    mask = mx.array(np.r_[np.ones(NA - 3), np.zeros(3)].astype(np.float32)[None])
    q = mx.array(rng.normal(size=(S, NA, 128)).astype(np.float32))
    P = at.prepare_nax(c, bias, make_to_keys(NA, 32, 128), mask, S)
    ref = at(q, None, None, None, None, S, prepared=P)

    def emul(a2, rc, layer, w_ff, w_ba, **cfg):
        return mx.array(E.atom_transition(np.array(a2), np.array(rc.h1), np.array(rc.b1), np.array(rc.h2),
                                          np.array(rc.b2), np.array(w_ff), np.array(w_ba), ntok=rc.ntok, rep=rc.rep,
                                          L=rc.L, layer=layer, **cfg))

    monkeypatch.setattr(KA, "atom_transition", emul)
    monkeypatch.setitem(KA.GEMM_CFG, "trans", KA.TRANS_CFGS[0])
    got = at(q, None, None, None, None, S, prepared=P)
    assert _rel(np.array(got - q), np.array(ref - q)) < 2e-3

    def emul_ag(a2, rc, layer, slot, w, **cfg):
        return mx.array(E.adaln_gemm(np.array(a2), np.array(rc.h1), np.array(rc.b1), np.array(w), ntok=rc.ntok,
                                     rep=rc.rep, L=rc.L, layer=layer, slot=slot, **cfg))

    monkeypatch.setattr(KA, "adaln_gemm", emul_ag)
    monkeypatch.setitem(KA.GEMM_CFG, "aqkvg", KA.ADALN_GEMM_CFGS[0])
    got2 = at(q, None, None, None, None, S, prepared=P)
    assert _rel(np.array(got2 - q), np.array(ref - q)) < 2e-3
    assert N3.EPI_GRESID == 1


def test_flag_default_off_and_never_outside_f16():
    from boltz_mlx import kernels as K

    if K.NAX_FLAGS.get("ATOM") is None:
        assert not K.nax("ATOM")  # CPU device -> f16() False; and default-off until validated
