"""R3-pair kernels and plumbing on the GPU (index logic: tests/test_nax3_cpu.py emulation):

  BOLTZ_MLX_DEVICE=gpu python -m pytest -q -s tests/test_nax3_pair.py

* transition_lnx: z_new bitwise == kernels_nax2.transition_nax(resid); the next-layer LN output bitwise == the LN
  kernel on z_new (same lane layout / arithmetic) — reported, asserted within fp16 ulps; pair bias vs lin16.
* tri-bias epilogue variants: z / LN outputs bitwise == the shipped variants; tb vs reference.
* pair_proj16 vs the MLX fp16 GEMM.
* PairformerLayer / PairformerModule / PairformerNoSeqModule (random init) with the R3-pair flags on vs off.
"""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(mx.default_device() != mx.gpu, reason="custom kernels need the GPU device")


def _N3():
    from boltz_mlx import kernels_nax3 as N3

    return N3


def _cmp(out, ref, fro=2e-3, mx_rel=1e-2):
    o = np.array(out.astype(mx.float32), np.float64)
    r = np.array(ref.astype(mx.float32), np.float64)
    assert np.isfinite(o).all()
    rf = np.linalg.norm(o - r) / max(np.linalg.norm(r), 1e-30)
    mr = np.abs(o - r).max() / max(np.abs(r).max(), 1e-30)
    assert rf <= fro and mr <= mx_rel, (rf, mr)
    return rf


class _LN:
    def __init__(self, D, rng, bias=True):
        self.w = mx.array(rng.normal(1.0, 0.1, D).astype(np.float32))
        self.b = mx.array(rng.normal(0.0, 0.1, D).astype(np.float32)) if bias else None
        self.eps = 1e-5

    def get(self, k, d=None):
        return {"weight": self.w, "bias": self.b}.get(k, d)


def _h(rng, shape, s=1.0):
    return mx.array((rng.normal(size=shape) * s).astype(np.float16))


@pytest.mark.parametrize("cfg", [dict(tm=16, hb=32, nsg=8, tgw=1), dict(tm=16, hb=32, nsg=4, tgw=0)])
@pytest.mark.parametrize("N", [6, 37, 117])
@pytest.mark.parametrize("want", ["both", "xn", "bias"])
def test_transition_lnx(cfg, N, want):
    from boltz_mlx import kernels as K
    from boltz_mlx import kernels_nax2 as KN2

    N3 = _N3()
    rng = np.random.default_rng(N + len(want))
    D, H = 128, 512
    xn = _h(rng, (1, N, N, D))
    w12, w3 = _h(rng, (2 * H, D), D ** -0.5), _h(rng, (D, H), H ** -0.5)
    z = mx.array(rng.normal(size=(1, N, N, D)).astype(np.float32))
    l1, l2 = _LN(D, rng), _LN(D, rng)
    wz = _h(rng, (16, D), D ** -0.5)
    kw = {}
    if want in ("both", "xn"):
        kw["next_ln"] = l1
    if want in ("both", "bias"):
        kw.update(ln_s=l2, w_s=wz)
    zo, xo, pb = N3.transition_lnx(xn, w12, w3, z, **kw, **cfg)
    zr = KN2.transition_nax(xn, w12, w3, resid=z, **cfg)
    assert bool(mx.array_equal(zo, zr).item()), "z_new must be bitwise the shipped fused transition's"
    if "next_ln" in kw:
        xr = K.layer_norm(zo, l1.w, l1.b, l1.eps, out_dtype=mx.float16)
        print(f"N={N} {cfg}: next LN bitwise == LN kernel: {bool(mx.array_equal(xo, xr).item())}")
        _cmp(xo, xr, 1e-3, 2e-3)
    else:
        assert xo is None
    if "w_s" in kw:
        # what the s-track computes today: lin16(proj_z[0].f16(z), proj_z[1]) (no bias) -> b h i j
        br = (K.layer_norm(zo, l2.w, l2.b, l2.eps, out_dtype=mx.float16) @ wz.T).transpose(0, 3, 1, 2)
        assert pb.shape == (1, 16, N, N)
        _cmp(pb, br, 3e-3, 2e-2)
    else:
        assert pb is None


@pytest.mark.parametrize("N", [7, 40, 117])
def test_tb_variants(N):
    N3 = _N3()
    rng = np.random.default_rng(N)
    D = 128
    x, w, wtb = _h(rng, (1, N, N, D)), _h(rng, (D, D), D ** -0.5), _h(rng, (4, D), D ** -0.5)
    z = mx.array(rng.normal(size=(1, N, N, D)).astype(np.float32))
    ln = _LN(D, rng)
    for ty, to in ((False, True), (True, False)):
        a = N3.linear_addln(x, w, z, ln, trans_y=ty, trans_out=to)
        b = N3.linear_addln(x, w, z, ln, trans_y=ty, trans_out=to, w_tb=wtb)
        assert bool(mx.array_equal(a[0], b[0]).item()) and bool(mx.array_equal(a[1], b[1]).item())
        _cmp(b[2], N3.linear_addln_ref(x, w, z, ln, trans_y=ty, trans_out=to, w_tb=wtb)[2], 3e-3, 2e-2)
        _cmp(b[2], N3._tb_ref(b[1], wtb), 2e-3, 1e-2)  # from the kernel's own LN output
    o = _h(rng, (D, N * N), 0.05)
    wpo, wgo = _h(rng, (D, D), D ** -0.5), _h(rng, (D, D), D ** -0.5)
    lo = _LN(D, rng)
    a = N3.trimul_tail_addln(o, lo, 2.0 ** -16, wpo, x, wgo, z, ln)
    b = N3.trimul_tail_addln(o, lo, 2.0 ** -16, wpo, x, wgo, z, ln, w_tb=wtb)
    assert bool(mx.array_equal(a[0], b[0]).item()) and bool(mx.array_equal(a[1], b[1]).item())
    _cmp(b[2], N3._tb_ref(b[1], wtb), 2e-3, 1e-2)


def test_pair_proj16_vs_mlx():
    N3 = _N3()
    rng = np.random.default_rng(3)
    x, w = _h(rng, (1, 117, 117, 128)), _h(rng, (512, 128), 128 ** -0.5)
    y = N3.pair_proj16(x, w)
    r = x @ w.T
    print("pair_proj16 bitwise == MLX fp16 GEMM:", bool(mx.array_equal(y, r).item()))
    _cmp(y, r, 1e-3, 2e-3)


def _f16(monkeypatch, r3: bool):
    from boltz_mlx import kernels as K

    monkeypatch.setattr(K, "PRECISION", "f16")
    for g in ("PAIRLNX", "PAIRPROJ", "PAIRTB"):
        monkeypatch.setitem(K.NAX_FLAGS, g, "1" if r3 else "0")
    return K


@pytest.mark.parametrize("N", [40, 117])
def test_pairformer_module_r3_vs_shipped(monkeypatch, N):
    """3-layer PairformerModule (carry of the next layer's norm_in output + s-track pair bias) and a standalone layer:
    R3-pair on vs off within fp16 noise; the carried LN output equals the recomputed one."""
    from boltz_mlx.layers.pairformer import PairformerLayer, PairformerModule

    mx.random.seed(N)
    layers = [PairformerLayer(384, 128, 16) for _ in range(3)]
    chain = SimpleNamespace(layers=layers)
    rng = np.random.default_rng(N)
    s = mx.array(rng.normal(size=(1, N, 384)).astype(np.float32))
    z = mx.array(rng.normal(size=(1, N, N, 128)).astype(np.float32))
    mask = mx.ones((1, N))
    pm = mx.array((rng.random((1, N, N)) > 0.05).astype(np.float32))
    K = _f16(monkeypatch, False)
    so, zo = PairformerModule.__call__(chain, s, z, mask, pm)
    K = _f16(monkeypatch, True)
    assert K.pair_lnx_ok(z)
    s3, z3 = PairformerModule.__call__(chain, s, z, mask, pm)
    rz = _cmp(z3 - z, zo - z, 3e-3, 3e-2)
    rs = _cmp(s3 - s, so - s, 3e-3, 3e-2)
    print(f"N={N}: chain update rel_fro R3 vs shipped z {rz:.2e} s {rs:.2e}")
    assert all("_pb_carry" not in L.__dict__ for L in layers)
    zt, xn_next, bias = K.pair_block_f16(layers[0], z, pm, lnx=(layers[1].tri_mul_out.norm_in, layers[0].attention))
    xr = layers[1].tri_mul_out.norm_in.f16(zt)
    print("carried next-layer LN bitwise == recomputed:", bool(mx.array_equal(xn_next, xr).item()))
    _cmp(xn_next, xr, 1e-3, 2e-3)
    _cmp(bias, K.lin16(layers[0].attention.proj_z[0].f16(zt), layers[0].attention.proj_z[1]).transpose(0, 3, 1, 2),
         3e-3, 2e-2)


def test_noseq_module_r3_vs_shipped(monkeypatch):
    from boltz_mlx.layers.pairformer import PairformerNoSeqLayer, PairformerNoSeqModule

    N = 50
    mx.random.seed(1)
    chain = SimpleNamespace(layers=[PairformerNoSeqLayer(128) for _ in range(2)])
    rng = np.random.default_rng(1)
    z = mx.array(rng.normal(size=(1, N, N, 128)).astype(np.float32))
    pm = mx.ones((1, N, N))
    _f16(monkeypatch, False)
    zo = PairformerNoSeqModule.__call__(chain, z, pm)
    _f16(monkeypatch, True)
    z3 = PairformerNoSeqModule.__call__(chain, z, pm)
    _cmp(z3 - z, zo - z, 3e-3, 3e-2)


# ------------------------------------------------------------------------------------------------- R4-pair
@pytest.mark.parametrize("cfg_i", range(3))
@pytest.mark.parametrize("N", [5, 37, 117])
def test_transpose_ln16(cfg_i, N):
    """o^T LayerNorm through the threadgroup tile == the LN kernel on the transposed view (bitwise expected)."""
    from boltz_mlx import kernels as K

    N3 = _N3()
    rng = np.random.default_rng(N + cfg_i)
    D = 128
    o = _h(rng, (D, N * N), 0.05)
    ln = _LN(D, rng)
    es = 2.0 ** -16
    got = N3.transpose_ln16(o, ln, es, **N3.TLN_CFGS[cfg_i])
    ref = K.layer_norm(o.T, ln.w, ln.b, ln.eps * es, out_dtype=mx.float16)
    print(f"N={N} cfg {cfg_i}: transpose_ln16 bitwise == layer_norm(o^T): {bool(mx.array_equal(got, ref).item())}")
    _cmp(got, ref, 1e-3, 2e-3)


@pytest.mark.parametrize("cfg_i", range(4))
@pytest.mark.parametrize("N", [6, 37, 117])
def test_tail2_vs_fused(cfg_i, N):
    N3 = _N3()
    cfg = N3.TAIL2_CFGS[cfg_i]
    if N * N < cfg["tm"]:
        pytest.skip("rows < tile")
    rng = np.random.default_rng(N * 3 + cfg_i)
    D = 128
    o = _h(rng, (D, N * N), 0.05)
    xn, wpo, wgo = _h(rng, (N * N, D)), _h(rng, (D, D), D ** -0.5), _h(rng, (D, D), D ** -0.5)
    wtb = _h(rng, (4, D), D ** -0.5)
    z = mx.array(rng.normal(size=(1, N, N, D)).astype(np.float32))
    lo, ln = _LN(D, rng), _LN(D, rng)
    es = 2.0 ** -16
    ref = N3.trimul_tail_addln_ref(o, lo, es, wpo, xn, wgo, z, ln, w_tb=wtb)
    got = N3.trimul_tail2_addln(o, lo, es, wpo, xn, wgo, z, ln, w_tb=wtb, cfg=cfg)
    _cmp(got[0], ref[0], 1e-3, 2e-3)
    _cmp(got[1], ref[1], 2e-3, 1e-2)
    _cmp(got[2], ref[2], 3e-3, 2e-2)
    fused = N3.trimul_tail_addln(o, lo, es, wpo, xn, wgo, z, ln)
    _cmp(got[0] - z, fused[0] - z, 2e-3, 2e-2)
    for st in (1, 2, 3, 4):  # ablation kernels run (timing only)
        mx.eval(N3.trimul_tail_stage(o, lo, es, wpo, xn, wgo, z, ln, st))


@pytest.mark.parametrize("N", [40, 117])
def test_pairformer_module_r4_vs_r3(monkeypatch, N):
    from boltz_mlx import kernels as K
    from boltz_mlx.layers.pairformer import PairformerLayer, PairformerModule

    mx.random.seed(N + 1)
    layers = [PairformerLayer(384, 128, 16) for _ in range(3)]
    chain = SimpleNamespace(layers=layers)
    rng = np.random.default_rng(N + 1)
    s = mx.array(rng.normal(size=(1, N, 384)).astype(np.float32))
    z = mx.array(rng.normal(size=(1, N, N, 128)).astype(np.float32))
    mask = mx.ones((1, N))
    pm = mx.array((rng.random((1, N, N)) > 0.05).astype(np.float32))
    _f16(monkeypatch, True)
    for g in ("TAIL2",):
        monkeypatch.setitem(K.NAX_FLAGS, g, "0")
    so, zo = PairformerModule.__call__(chain, s, z, mask, pm)
    for g in ("TAIL2",):
        monkeypatch.setitem(K.NAX_FLAGS, g, "1")
    s4, z4 = PairformerModule.__call__(chain, s, z, mask, pm)
    rz = _cmp(z4 - z, zo - z, 3e-3, 3e-2)
    rs = _cmp(s4 - s, so - s, 3e-3, 3e-2)
    print(f"N={N}: chain update rel_fro R4 vs R3 z {rz:.2e} s {rs:.2e}")


# ------------------------------------------------------------------------------------------------- R5-tail
def _tail_case(rng, N, D=128):
    o = _h(rng, (D, N * N), 0.05)
    xn, wpo, wgo = _h(rng, (N * N, D)), _h(rng, (D, D), D ** -0.5), _h(rng, (D, D), D ** -0.5)
    z = mx.array(rng.normal(size=(1, N, N, D)).astype(np.float32))
    return o, xn, wpo, wgo, z, _LN(D, rng), _LN(D, rng)


@pytest.mark.parametrize("cfg_i", range(10))
@pytest.mark.parametrize("N", [6, 37, 117])
def test_tail3_vs_shipped(cfg_i, N):
    """R5 tail3 (chunked p / g, thread bound): same arithmetic and accumulation order as the shipped fused tail ->
    bitwise equal outputs expected (reported); asserted within 1e-6."""
    N3 = _N3()
    cfg = N3.TAIL3_CFGS[cfg_i]
    rng = np.random.default_rng(N + 13 * cfg_i)
    o, xn, wpo, wgo, z, lo, ln = _tail_case(rng, N)
    es = 2.0 ** -16
    ref = N3.trimul_tail_addln(o, lo, es, wpo, xn, wgo, z, ln)
    got = N3.trimul_tail3_addln(o, lo, es, wpo, xn, wgo, z, ln, **cfg)
    bw = bool(mx.array_equal(got[0], ref[0]).item()) and bool(mx.array_equal(got[1], ref[1]).item())
    print(f"N={N} {cfg}: tail3 bitwise == shipped fused tail: {bw}")
    _cmp(got[0] - z, ref[0] - z, 1e-6, 1e-5)
    _cmp(got[1], ref[1], 1e-6, 1e-3)
    _cmp(got[0], N3.trimul_tail_addln_ref(o, lo, es, wpo, xn, wgo, z, ln)[0], 1e-3, 2e-3)


def test_tail3_tb_and_cuts():
    N3 = _N3()
    rng = np.random.default_rng(21)
    N = 50
    o, xn, wpo, wgo, z, lo, ln = _tail_case(rng, N)
    wtb = _h(rng, (4, 128), 0.1)
    es = 2.0 ** -16
    ref = N3.trimul_tail_addln(o, lo, es, wpo, xn, wgo, z, ln, w_tb=wtb)
    got = N3.trimul_tail3_addln(o, lo, es, wpo, xn, wgo, z, ln, w_tb=wtb)
    for a, b in zip(got, ref):
        _cmp(a, b, 1e-6, 1e-3)
    for st in (3, 4):  # profile cuts run
        mx.eval(N3.trimul_tail3_addln(o, lo, es, wpo, xn, wgo, z, ln, stage=st)[0])
    for k in ("g_only", "p_dead"):
        mx.eval(N3.trimul_tail_diag(o, lo, es, wpo, xn, wgo, z, ln, k))


def test_maxt_diagnostics_same_result():
    """The thread bound changes register allocation only: linear_addln / the pair transition give the same outputs."""
    from boltz_mlx import kernels_nax2 as KN2

    N3 = _N3()
    rng = np.random.default_rng(22)
    N, D = 40, 128
    x = _h(rng, (1, N, N, D))
    w = _h(rng, (D, D), D ** -0.5)
    z = mx.array(rng.normal(size=(1, N, N, D)).astype(np.float32))
    ln = _LN(D, rng)
    a = N3.linear_addln(x, w, z, ln, trans_out=True)
    b = N3.linear_addln_maxt(x, w, z, ln, trans_out=True)
    print("linear_addln maxt bitwise:", bool(mx.array_equal(a[0], b[0]).item()) and bool(mx.array_equal(a[1], b[1]).item()))
    _cmp(b[0], a[0], 1e-6, 1e-5)
    w12, w3 = _h(rng, (1024, D), D ** -0.5), _h(rng, (D, 512), 512 ** -0.5)
    t0 = KN2.transition_nax(x, w12, w3, resid=z, tm=16, hb=32, nsg=8, tgw=1)
    t1 = N3.transition_maxt(x, w12, w3, z)
    print("transition maxt bitwise:", bool(mx.array_equal(t0, t1).item()))
    _cmp(t1 - z, t0 - z, 1e-6, 1e-5)


@pytest.mark.parametrize("N", [40, 117])
def test_pairformer_module_tail3(monkeypatch, N):
    from boltz_mlx import kernels as K
    from boltz_mlx.layers.pairformer import PairformerLayer, PairformerModule

    mx.random.seed(N + 2)
    chain = SimpleNamespace(layers=[PairformerLayer(384, 128, 16) for _ in range(2)])
    rng = np.random.default_rng(N + 2)
    s = mx.array(rng.normal(size=(1, N, 384)).astype(np.float32))
    z = mx.array(rng.normal(size=(1, N, N, 128)).astype(np.float32))
    mask = mx.ones((1, N))
    pm = mx.array((rng.random((1, N, N)) > 0.05).astype(np.float32))
    _f16(monkeypatch, True)
    monkeypatch.setitem(K.NAX_FLAGS, "TAIL3", "0")
    so, zo = PairformerModule.__call__(chain, s, z, mask, pm)
    monkeypatch.setitem(K.NAX_FLAGS, "TAIL3", "1")
    s3, z3 = PairformerModule.__call__(chain, s, z, mask, pm)
    print(f"N={N}: module with TAIL3 bitwise == shipped: z {bool(mx.array_equal(z3, zo).item())} "
          f"s {bool(mx.array_equal(s3, so).item())}")
    _cmp(z3 - z, zo - z, 1e-5, 1e-3)
    _cmp(s3 - s, so - s, 1e-5, 1e-3)


# ------------------------------------------------------------------------------------------------- R6-trans
@pytest.mark.parametrize("cfg_i", range(12))
@pytest.mark.parametrize("N", [6, 37, 117])
def test_trans3_vs_shipped(cfg_i, N):
    """Every R6 transition implementation vs the shipped transition_lnx (z, next LN, pair bias): fp16-level agreement
    (not bitwise: other accumulation / reduction orders); rel errors printed."""
    N3 = _N3()
    cfg = N3.TRANS3_CFGS[cfg_i]
    rng = np.random.default_rng(N + 17 * cfg_i)
    D, H = 128, 512
    xn = _h(rng, (1, N, N, D))
    w12, w3 = _h(rng, (2 * H, D), D ** -0.5), _h(rng, (D, H), H ** -0.5)
    z = mx.array(rng.normal(size=(1, N, N, D)).astype(np.float32))
    l1, l2 = _LN(D, rng), _LN(D, rng)
    wz = _h(rng, (16, D), D ** -0.5)
    zs, xs, bs = N3.transition_lnx(xn, w12, w3, z, next_ln=l1, ln_s=l2, w_s=wz)
    zo, xo, bo = N3.transition3(xn, w12, w3, z, next_ln=l1, ln_s=l2, w_s=wz, cfg=cfg)
    rz = _cmp(zo - z, zs - z, 3e-3, 2e-2)
    rx = _cmp(xo, xs, 3e-3, 2e-2)
    rb = _cmp(bo, bs, 5e-3, 3e-2)
    zp = N3.transition3(xn, w12, w3, z, cfg=cfg)[0]          # no epilogue outputs
    _cmp(zp - z, zo - z, 1e-6, 1e-5)
    print(f"N={N} {cfg}: rel vs shipped z-update {rz:.1e} next-LN {rx:.1e} bias {rb:.1e}")


@pytest.mark.parametrize("N", [40, 117])
def test_pairformer_module_trans3(monkeypatch, N):
    from boltz_mlx import kernels as K
    from boltz_mlx.layers.pairformer import PairformerLayer, PairformerModule

    mx.random.seed(N + 3)
    chain = SimpleNamespace(layers=[PairformerLayer(384, 128, 16) for _ in range(2)])
    rng = np.random.default_rng(N + 3)
    s = mx.array(rng.normal(size=(1, N, 384)).astype(np.float32))
    z = mx.array(rng.normal(size=(1, N, N, 128)).astype(np.float32))
    mask = mx.ones((1, N))
    pm = mx.array((rng.random((1, N, N)) > 0.05).astype(np.float32))
    _f16(monkeypatch, True)
    monkeypatch.setitem(K.NAX_FLAGS, "TRANS3", "0")
    so, zo = PairformerModule.__call__(chain, s, z, mask, pm)
    monkeypatch.setitem(K.NAX_FLAGS, "TRANS3", "1")
    s3, z3 = PairformerModule.__call__(chain, s, z, mask, pm)
    rz = _cmp(z3 - z, zo - z, 3e-3, 3e-2)
    rs = _cmp(s3 - s, so - s, 3e-3, 3e-2)
    print(f"N={N}: chain update rel TRANS3 vs shipped z {rz:.1e} s {rs:.1e}")
