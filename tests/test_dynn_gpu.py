"""R8-coldN DYNN (run-time token counts) on the GPU: every DYNN kernel bitwise == its static-size twin at several N, and
a 2-layer PairformerModule (f16, shipped pair flags) bitwise with DYNN on vs off.

  BOLTZ_MLX_DEVICE=gpu python -m pytest -q -s tests/test_dynn_gpu.py

(The instantiation logic, i.e. one Metal library for every N, is tested on the CPU: tests/test_dynn_cpu.py.)
"""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(mx.default_device() != mx.gpu, reason="custom kernels need the GPU device")


class _LN:
    def __init__(self, D, rng, bias=True):
        self.w = mx.array(rng.normal(1.0, 0.1, D).astype(np.float32))
        self.b = mx.array(rng.normal(0.0, 0.1, D).astype(np.float32)) if bias else None
        self.eps = 1e-5

    def get(self, k, d=None):
        return {"weight": self.w, "bias": self.b}.get(k, d)


def _h(rng, shape, s=1.0):
    return mx.array((rng.normal(size=shape) * s).astype(np.float16))


def _f(rng, shape, s=1.0):
    return mx.array((rng.normal(size=shape) * s).astype(np.float32))


def _diff(a, b) -> str:
    """How two same-shape outputs differ: element count, max |delta|, max distance in ULPs of the output dtype."""
    an, bn = np.array(a), np.array(b)
    ne = an != bn
    d = np.abs(an.astype(np.float64) - bn.astype(np.float64))
    ulp = np.spacing(np.abs(an)).astype(np.float64)
    return (f"{int(ne.sum())} of {an.size} elements differ ({a.dtype}), max |delta| {d.max():.3g}, "
            f"max {float((d / np.maximum(ulp, 1e-45)).max()):.3g} ulp, max |ref| {np.abs(an).max():.3g}")


def _both(monkeypatch, build):
    """build() with DYNN off (twice: the inputs and the static kernel must be deterministic, else the comparison is
    meaningless) and on; every output array must be bitwise identical. Failures report _diff."""
    from boltz_mlx import kernels as K

    outs = []
    for v in ("0", "0", "1"):
        monkeypatch.setitem(K.NAX_FLAGS, "DYNN", v)
        o = build()
        o = [x for x in (o if isinstance(o, (tuple, list)) else (o,)) if x is not None]
        mx.eval(*o)
        outs.append(o)
    off, off2, on = outs
    assert len(off) == len(off2) == len(on)
    for i, (a, a2, b) in enumerate(zip(off, off2, on)):
        assert a.shape == b.shape and a.dtype == b.dtype
        assert bool(mx.all(mx.isfinite(a.astype(mx.float32))).item())
        assert bool(mx.array_equal(a, a2).item()), f"output {i}: static run not reproducible (test bug?): {_diff(a, a2)}"
        if not bool(mx.array_equal(a, b).item()):
            raise AssertionError(f"output {i}: DYNN differs from the static-size kernel: {_diff(a, b)}")
    return on


def _pair_inputs(N, seed):
    rng = np.random.default_rng(seed)
    D, H = 128, 4
    return SimpleNamespace(
        rng=rng, D=D, H=H, z=_f(rng, (1, N, N, D)), x16=_h(rng, (1, N, N, D)), w16=_h(rng, (D, D), D ** -0.5),
        w16b=_h(rng, (D, D), D ** -0.5), o=_h(rng, (D, N * N), 4.0), wtb=_h(rng, (H, D), D ** -0.5),
        ln=_LN(D, rng), ln2=_LN(D, rng), ln_nob=_LN(D, rng, bias=False))


NS = [37, 117, 203]


@pytest.mark.parametrize("N", NS)
def test_tri_attention_nax_dynn(monkeypatch, N):
    from boltz_mlx import kernels as K
    from boltz_mlx import kernels_nax as KN

    rng = np.random.default_rng(N)
    H, C = 4, 32
    h = _h(rng, (1, N, N, 4 * H * C))
    tb = _h(rng, (1, H, N, N))
    m = mx.array((rng.random((1, N, N)) > 0.05).astype(np.float32))
    cfg = K.TRIATT_NAX_CFG
    _both(monkeypatch, lambda: KN.tri_attention_nax(h, H, C, m, 1e9, bias=tb, **cfg))
    _both(monkeypatch, lambda: KN.tri_attention_nax(h, H, C, m, 1e9, bias=tb, trans=True, **cfg))
    _both(monkeypatch, lambda: KN.tri_attention_nax(h, H, C, None, 1e9, bias=tb, **cfg))
    r = min(24, N)  # a row chunk (I != J), as the chunked large-N path
    _both(monkeypatch, lambda: KN.tri_attention_nax(h[:, :r], H, C, m[:, :r], 1e9, bias=tb, **cfg))


@pytest.mark.parametrize("N", NS)
def test_pair_epilogues_dynn(monkeypatch, N):
    from boltz_mlx import kernels_nax3 as N3

    p = _pair_inputs(N, N + 1)
    lin = N3.PAIR_SITE_CFG["lin"]
    for ty, to in ((False, False), (True, False), (False, True)):
        _both(monkeypatch, lambda: N3.linear_addln(p.x16, p.w16, p.z, p.ln, trans_y=ty, trans_out=to, **lin))
    _both(monkeypatch, lambda: N3.linear_addln(p.x16, p.w16, p.z, p.ln, trans_out=True, w_tb=p.wtb, **lin))
    bias = _f(p.rng, (p.D,))  # drawn once (R8 q18: drawn inside the lambda, each arm got a different bias)
    _both(monkeypatch, lambda: N3.linear_addln(p.x16, p.w16, p.z, p.ln_nob, bias=bias, **lin))
    tail = N3.PAIR_SITE_CFG["tail"]
    _both(monkeypatch, lambda: N3.trimul_tail_addln(p.o, p.ln2, 2.0 ** -16, p.w16, p.x16, p.w16b, p.z, p.ln, **tail))
    _both(monkeypatch, lambda: N3.trimul_tail_addln(p.o, p.ln2, 2.0 ** -16, p.w16, p.x16, p.w16b, p.z, p.ln,
                                                    w_tb=p.wtb))
    t3 = dict(N3.PAIR_SITE_CFG.get("tail3") or {})
    _both(monkeypatch, lambda: N3.trimul_tail3_addln(p.o, p.ln2, 2.0 ** -16, p.w16, p.x16, p.w16b, p.z, p.ln, **t3))
    _both(monkeypatch, lambda: N3.trimul_tail3_addln(p.o, p.ln2, 2.0 ** -16, p.w16, p.x16, p.w16b, p.z, p.ln,
                                                     w_tb=p.wtb, **t3))
    _both(monkeypatch, lambda: N3.gemm2_gate_addln(p.x16, p.w16, p.x16, p.w16b, p.z, p.ln, w_tb=p.wtb))


@pytest.mark.parametrize("N", NS)
def test_transition_reg_dynn(monkeypatch, N):
    from boltz_mlx import kernels_nax3 as N3

    p = _pair_inputs(N, N + 2)
    Hh = 512
    w12, w3 = _h(p.rng, (2 * Hh, p.D), p.D ** -0.5), _h(p.rng, (p.D, Hh), Hh ** -0.5)
    wz = _h(p.rng, (16, p.D), p.D ** -0.5)
    cfg = dict(N3.PAIR_SITE_CFG.get("trans3") or {})
    if cfg.pop("impl", "reg") != "reg":
        cfg = {}
    _both(monkeypatch, lambda: N3.transition_reg(p.x16, w12, w3, p.z, next_ln=p.ln, ln_s=p.ln2, w_s=wz, **cfg))
    _both(monkeypatch, lambda: N3.transition_reg(p.x16, w12, w3, p.z, **cfg))


@pytest.mark.parametrize("N", NS)
def test_attention_nax_dynn(monkeypatch, N):
    from boltz_mlx import kernels_nax2 as KN2

    rng = np.random.default_rng(N + 3)
    D = 768
    for B, brep in ((1, 1), (3, 1), (3, 3)):
        h = _h(rng, (B * N, 4 * D), 0.5)
        bias = _f(rng, (B // brep, 16, N, N))
        qb = _f(rng, (D,), 0.1)
        _both(monkeypatch, lambda: KN2.attention_nax(h, bias, B=B, N=N, heads=16, c=48, qbias=qb, brep=brep))
    h = _h(rng, (2 * N, 4 * D), 0.5)
    bias = _f(rng, (2, 16, N, N))
    _both(monkeypatch, lambda: KN2.attention_nax(h, None, B=2, N=N, heads=16, c=48,
                                                 prepared=KN2.attention_nax_bias(bias)))


@pytest.mark.parametrize("N", NS)
def test_add_ln_dynn(monkeypatch, N):
    from boltz_mlx import kernels as K

    p = _pair_inputs(N, N + 4)
    _both(monkeypatch, lambda: K.add_ln(p.z, p.x16, p.ln, trans_y=True))
    _both(monkeypatch, lambda: K.add_ln(p.z, p.z, p.ln_nob, trans_out=True))
    _both(monkeypatch, lambda: K.add_ln(p.z, K.Scaled(p.x16, 2.0 ** -8), p.ln))


@pytest.mark.parametrize("N", [40, 117])
def test_pairformer_module_dynn(monkeypatch, N):
    """2-layer PairformerModule (random init), f16 with the shipped pair flags: DYNN on == off bitwise (z and s)."""
    from boltz_mlx import kernels as K
    from boltz_mlx.layers.pairformer import PairformerLayer, PairformerModule

    monkeypatch.setattr(K, "PRECISION", "f16")
    mx.random.seed(N + 5)
    chain = SimpleNamespace(layers=[PairformerLayer(384, 128, 16) for _ in range(2)])
    rng = np.random.default_rng(N + 5)
    s = _f(rng, (1, N, 384))
    z = _f(rng, (1, N, N, 128))
    mask = mx.ones((1, N))
    pm = mx.array((rng.random((1, N, N)) > 0.05).astype(np.float32))
    _both(monkeypatch, lambda: PairformerModule.__call__(chain, s, z, mask, pm))
