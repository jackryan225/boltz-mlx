"""kernels_nax3 (round-3 NAX GEMMs with fused epilogues) on the GPU vs their MLX-op references (same fp16 rounding
points, fp32 sums in another order -> a few fp16 ulps) and vs the shipped f16 paths they replace:

  BOLTZ_MLX_DEVICE=gpu python -m pytest -q tests/test_nax3_kernels.py

Shapes include ragged tails (row counts not multiples of any tile; N * N not a multiple of TM) and every config listed
in the module. The index logic was checked lane by lane on the CPU (tests/test_nax3_cpu.py); these tests check the
Metal semantics: the NAX fragment layout of the relaxed 16x32x16 descriptor, loads / stores, shuffles.
"""

import os

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
    rf = np.linalg.norm(o - r) / np.linalg.norm(r)
    mr = np.abs(o - r).max() / np.abs(r).max()
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


@pytest.mark.parametrize("cfg_i", range(4))
@pytest.mark.parametrize("N", [6, 37, 117])
@pytest.mark.parametrize("ty,to", [(False, True), (True, False), (False, False)])
def test_linear_addln(cfg_i, N, ty, to):
    N3 = _N3()
    cfg = N3.PAIR_CFGS[cfg_i]
    if N * N < cfg["tm"]:
        pytest.skip("rows < tile")
    rng = np.random.default_rng(N * 10 + cfg_i)
    D = 128
    x = _h(rng, (1, N, N, D))
    w = _h(rng, (D, D), D ** -0.5)
    z = mx.array(rng.normal(size=(1, N, N, D)).astype(np.float32))
    ln = _LN(D, rng)
    zo, lo = N3.linear_addln(x, w, z, ln, trans_y=ty, trans_out=to, **cfg)
    zr, lr = N3.linear_addln_ref(x, w, z, ln, trans_y=ty, trans_out=to)
    _cmp(zo, zr, 1e-3, 2e-3)
    _cmp(lo, lr, 2e-3, 1e-2)


def test_linear_addln_bias_and_no_ln_bias():
    N3 = _N3()
    rng = np.random.default_rng(5)
    D, N = 128, 23
    x, w = _h(rng, (1, N, N, D)), _h(rng, (D, D), D ** -0.5)
    z = mx.array(rng.normal(size=(1, N, N, D)).astype(np.float32))
    b = mx.array(rng.normal(size=D).astype(np.float32) * 0.1)
    ln = _LN(D, rng, bias=False)
    zo, lo = N3.linear_addln(x, w, z, ln, bias=b)
    zr, lr = N3.linear_addln_ref(x, w, z, ln, bias=b)
    _cmp(zo, zr, 1e-3, 2e-3)
    _cmp(lo, lr)


def test_linear_addln_vs_shipped_add_ln():
    """Against what pair_block_f16 runs today: lin16 (MLX fp16 GEMM) + kernels.add_ln. z_out is bitwise equal when the
    NAX accumulation order matches MLX's NAX GEMM (reported), and within fp16 ulps of y otherwise."""
    from boltz_mlx import kernels as K

    N3 = _N3()
    rng = np.random.default_rng(6)
    D, N = 128, 64
    x, w = _h(rng, (1, N, N, D)), _h(rng, (D, D), D ** -0.5)
    z = mx.array(rng.normal(size=(1, N, N, D)).astype(np.float32))
    ln = _LN(D, rng)
    zo, lo = N3.linear_addln(x, w, z, ln, trans_out=True)
    y = x @ w.T
    zs, ls = K.add_ln(z, y, ln, trans_out=True)
    print("z bitwise equal to lin16 + add_ln:", bool(mx.array_equal(zo, zs).item()),
          "ln bitwise:", bool(mx.array_equal(lo, ls).item()))
    _cmp(zo, zs, 1e-3, 2e-3)
    _cmp(lo, ls, 2e-3, 1e-2)


@pytest.mark.parametrize("cfg_i", range(3))
@pytest.mark.parametrize("N", [6, 37, 117])
def test_trimul_tail_addln(cfg_i, N):
    N3 = _N3()
    cfg = N3.TRIMUL_TAIL_CFGS[cfg_i]
    rng = np.random.default_rng(N + cfg_i)
    D = 128
    o = _h(rng, (D, N * N), 0.05)
    xn, wpo, wgo = _h(rng, (N * N, D)), _h(rng, (D, D), D ** -0.5), _h(rng, (D, D), D ** -0.5)
    z = mx.array(rng.normal(size=(1, N, N, D)).astype(np.float32))
    lo_, ln = _LN(D, rng), _LN(D, rng)
    es = 2.0 ** -16
    zo, lno = N3.trimul_tail_addln(o, lo_, es, wpo, xn, wgo, z, ln, **cfg)
    zr, lr = N3.trimul_tail_addln_ref(o, lo_, es, wpo, xn, wgo, z, ln)
    _cmp(zo, zr, 1e-3, 2e-3)
    _cmp(lno, lr, 2e-3, 1e-2)


def _gemm_case(rng, epi, M, K, NOUT, ntok, rep):
    ngrp = 3 if epi == 2 else 1
    x, w = _h(rng, (M, K)), _h(rng, (ngrp * NOUT, K), K ** -0.5)
    kw = {}
    if epi == 1:
        gl = 4 * NOUT
        nc = -(-M // (ntok * rep)) * ntok
        kw = dict(gate=_h(rng, (nc, gl)), gate_bias=mx.array(rng.normal(size=gl).astype(np.float32)), gate_off=NOUT,
                  resid=mx.array(rng.normal(size=(M, NOUT)).astype(np.float32)), ntok=ntok, rep=rep)
    return x, w, ngrp, kw


@pytest.mark.parametrize("cfg_i", range(12))
@pytest.mark.parametrize("epi", [0, 1])
@pytest.mark.parametrize("M,K,NOUT,ntok,rep", [(37, 768, 768, 37, 1), (1065, 768, 768, 213, 5),
                                                 (400, 1536, 768, 400, 1), (213, 768, 3072, 213, 1)])
def test_gemm16(cfg_i, epi, M, K, NOUT, ntok, rep):
    N3 = _N3()
    cfg = N3.GEMM_CFGS[cfg_i]
    if K % (16 * cfg["ku"]) or NOUT % (cfg["wn"] * cfg["sn"]):
        pytest.skip("cfg does not tile")
    rng = np.random.default_rng(cfg_i + 7 * epi + M)
    x, w, ngrp, kw = _gemm_case(rng, epi, M, K, NOUT, ntok, rep)
    y = N3.gemm16(x, w, epi=epi, ngrp=ngrp, **kw, **cfg)
    _cmp(y, N3.gemm16_ref(x, w, epi=epi, ngrp=ngrp, **kw), 2e-3, 1e-2)


@pytest.mark.parametrize("cfg_i", range(5))
@pytest.mark.parametrize("M", [37, 400, 1065])
def test_gemm16_swiglu(cfg_i, M):
    N3 = _N3()
    cfg = N3.SWIGLU_CFGS[cfg_i]
    rng = np.random.default_rng(cfg_i + M)
    x, w, ngrp, kw = _gemm_case(rng, 2, M, 768, 1536, M, 1)
    y = N3.gemm16(x, w, epi=2, ngrp=3, **cfg)
    _cmp(y, N3.gemm16_ref(x, w, epi=2, ngrp=3), 2e-3, 1e-2)


def test_gemm16_store_f32_bias():
    N3 = _N3()
    rng = np.random.default_rng(9)
    x, w = _h(rng, (100, 768)), _h(rng, (768, 768), 768 ** -0.5)
    b = mx.array(rng.normal(size=768).astype(np.float32))
    y = N3.gemm16(x, w, out_dtype=mx.float32, bias=b)
    _cmp(y, N3.gemm16_ref(x, w, out_dtype=mx.float32, bias=b), 1e-5, 1e-4)


def test_adaln16():
    N3 = _N3()
    rng = np.random.default_rng(10)
    L, D, ntok, rep = 24, 768, 41, 5
    M = rep * ntok
    a2 = mx.array(rng.normal(size=(M, D)).astype(np.float32) * 3 + 1)
    rc = N3.RawCond16(_h(rng, (ntok, L * 4 * D)), mx.array(rng.normal(size=L * 4 * D).astype(np.float32)),
                      _h(rng, (ntok, L * 2 * D)), mx.array(rng.normal(size=L * 2 * D).astype(np.float32)),
                      ntok=ntok, L=L, D=D, rep=rep)
    for layer, slot in ((0, 0), (23, 2), (7, 0)):
        _cmp(N3.adaln16(a2, rc, layer, slot), N3.adaln16_ref(a2, rc, layer, slot), 2e-3, 1e-2)


@pytest.mark.parametrize("B,brep,N", [(5, 5, 213), (6, 3, 50), (5, 5, 37)])
def test_attention_shared_bias_bitwise(B, brep, N):
    """attention_nax(brep): bias [B / brep] read for brep samples == the shipped kernel on the repeated bias, bitwise."""
    from boltz_mlx import kernels_nax2 as KN2

    rng = np.random.default_rng(B + N)
    h = _h(rng, (B * N, 4 * 768))
    bias = mx.array(rng.normal(size=(B // brep, 16, N, N)).astype(np.float32))
    bias = bias + mx.array(np.where(rng.random((B // brep, 1, 1, N)) < 0.1, -1e6, 0).astype(np.float32))
    qb = mx.array(rng.normal(size=768).astype(np.float32) * 0.1)
    o1 = KN2.attention_nax(h, bias, B=B, N=N, heads=16, c=48, qbias=qb, brep=brep)
    o2 = KN2.attention_nax(h, mx.repeat(bias, brep, axis=0), B=B, N=N, heads=16, c=48, qbias=qb)
    assert bool(mx.array_equal(o1, o2).item())


# ------------------------------------------------------------------------------------------------- integration
def _pair_layer(seed=0):
    from boltz_mlx.layers.pairformer import PairformerLayer

    mx.random.seed(seed)
    layer = PairformerLayer(384, 128, 16)
    return layer


@pytest.mark.parametrize("N", [40, 117])
def test_pair_block_epi_vs_shipped(monkeypatch, N):
    """pair_block_f16_epi == pair_block_f16 (f16, residual guards) within fp16 ulps, on a random-init layer."""
    from boltz_mlx import kernels as K

    monkeypatch.setattr(K, "PRECISION", "f16")
    layer = _pair_layer(N)
    rng = np.random.default_rng(N)
    z = mx.array(rng.normal(size=(1, N, N, 128)).astype(np.float32))
    mask = mx.array((rng.random((1, N, N)) > 0.05).astype(np.float32))
    monkeypatch.setitem(K.NAX_FLAGS, "PAIREPI", "0")
    ref = K.pair_block_f16(layer, z, mask)
    monkeypatch.setitem(K.NAX_FLAGS, "PAIREPI", "1")
    assert K.pair_epi_ok(z)
    got = K.pair_block_f16(layer, z, mask)
    rf = _cmp(got - z, ref - z, 3e-3, 2e-2)
    print(f"N={N}: pair block update rel_fro epi vs shipped {rf:.2e}")


def test_token_transformer_compiled_equals_eager(monkeypatch):
    """The nax3 token transformer under mx.compile == eager (the diffusion step is compiled), random-init model."""
    from boltz_mlx import kernels as K
    from boltz_mlx.modules.transformersv2 import DiffusionTransformer

    monkeypatch.setattr(K, "PRECISION", "f16")
    monkeypatch.setitem(K.NAX_FLAGS, "GEMM", "1")
    prev = K.DIFF_DTYPE
    K.set_diff_dtype("fp16")
    try:
        mx.random.seed(0)
        depth, D, N, mult = 3, 768, 70, 5
        tr = DiffusionTransformer(depth=depth, heads=16, dim=D, dim_single_cond=D)
        assert tr.nax3(None)
        rng = np.random.default_rng(0)
        s1 = mx.array(rng.normal(size=(1, N, D)).astype(np.float32))
        a = mx.array(rng.normal(size=(mult, N, D)).astype(np.float32))
        bias = mx.array(rng.normal(size=(1, N, N, 16 * depth)).astype(np.float32))
        mask = mx.array(np.tile(np.r_[np.ones(N - 3), np.zeros(3)].astype(np.float32), (mult, 1)))
        prep = tr.prepare(None, bias, mask, None, mult, with_cond=False)
        assert prep.get("attn_brep") == mult

        def f(a_, s_):
            return tr(a_, mx.repeat(s_, mult, 0), prepared={**prep, "cond": tr.precompute_cond_shared(s_, mult)})

        e = f(a, s1)
        c = mx.compile(f)(a, s1)
        assert bool(mx.array_equal(e, c).item())
        # vs the shipped f16 path (flag off): same model, update within fp16 noise
        monkeypatch.setitem(K.NAX_FLAGS, "GEMM", "0")
        prep0 = tr.prepare(None, bias, mask, None, mult, with_cond=False)
        r = tr(a, mx.repeat(s1, mult, 0), prepared={**prep0, "cond": tr.precompute_cond_shared(s1, mult)})
        rf = _cmp(e - a, r - a, 3e-3, 3e-2)
        print(f"token transformer update rel_fro nax3 vs shipped f16 {rf:.2e}")
    finally:
        K.DIFF_DTYPE = prev


@pytest.mark.skipif(os.environ.get("BOLTZ_MLX_PRECISION") == "f16", reason="flags are read at import")
def test_flags_off_outside_f16():
    """Without --precision f16 (strict / fast / turbo) the round-3 groups are off unless forced by env."""
    from boltz_mlx import kernels as K

    for g in ("GEMM", "PAIREPI"):
        if K.NAX_FLAGS.get(g) is None:
            assert not K.nax(g)
