"""Fused diffusion token-transformer kernels (BOLTZ_MLX_DIFFTX_IMPL=fused) vs the MLX op path. GPU only.

Compares, on random inputs, the attention half, the conditioned-transition half, a full multi-layer
DiffusionTransformer (incl. shared-noise-level conditioning with multiplicity) and the fallbacks.
Gate: fp32 max |d| / max |ref| <= 1e-5.
"""

import mlx.core as mx
import numpy as np
import pytest

from boltz_mlx import kernels as K
from boltz_mlx import kernels_difftx as KD
from boltz_mlx.modules.transformersv2 import DiffusionTransformer, DiffusionTransformerLayer

pytestmark = pytest.mark.skipif(mx.default_device() != mx.gpu, reason="custom kernels need the GPU device")

D, HEADS, L = 768, 16, 2
TOL = 1e-5


def _rnd(rng, shape, scale=1.0):
    return mx.array((rng.normal(size=shape) * scale).astype(np.float32))


def _layer(seed=0):
    rng = np.random.default_rng(seed)
    lay = DiffusionTransformerLayer(HEADS, D, D)
    at, tr = lay.pair_bias_attn, lay.transition
    for lin in (at.proj_q, at.proj_k, at.proj_v, at.proj_g, at.proj_o, tr.swish_gate[0], tr.a_to_b, tr.b_to_a):
        lin.weight = _rnd(rng, lin.weight.shape, lin.weight.shape[1] ** -0.5)
    at.proj_q.bias = _rnd(rng, at.proj_q.bias.shape, 0.2)
    mx.eval(lay.parameters())
    return lay


def _raw(rng, Bc, N, nl=1):
    return KD.RawCond(_rnd(rng, (Bc, N, nl, 4, D)), _rnd(rng, (Bc, N, nl, 2, D)))


def _mask_bias(rng, B, N):
    mask = (rng.uniform(size=(B, N)) < 0.85).astype(np.float32)
    mask[:, 0] = 1.0
    bias = _rnd(rng, (B, HEADS, N, N))
    return bias + (1 - mx.array(mask)[:, None, None, :]) * -1e6


def _rel(a, b):
    return float(mx.abs(a - b).max() / mx.abs(b).max())


@pytest.fixture(autouse=True)
def _restore(monkeypatch):
    monkeypatch.setattr(KD, "DIFFTX_IMPL", "mlx")
    yield


def _both(fn, monkeypatch, expect_fused=True):
    monkeypatch.setattr(KD, "DIFFTX_IMPL", "mlx")
    ref = fn()
    mx.eval(ref)
    calls = []
    for name in ("attn_fused", "transition_fused"):
        orig = getattr(KD, name)
        monkeypatch.setattr(KD, name, lambda *a, _o=orig, _n=name, **k: (calls.append(_n), _o(*a, **k))[1])
    monkeypatch.setattr(KD, "DIFFTX_IMPL", "fused")
    out = fn()
    mx.eval(out)
    if expect_fused:
        assert calls, "fused kernels were not used"
    else:
        assert not calls, "fused kernels used where a fallback was expected"
    return out, ref


@pytest.mark.parametrize("N", [1, 7, 117, 400])
@pytest.mark.parametrize("B", [1, 3])
def test_attention_half(N, B, monkeypatch):
    rng = np.random.default_rng(N * 10 + B)
    lay = _layer(1)
    a = _rnd(rng, (B, N, D))
    rc = _raw(rng, B, N)
    ab = _mask_bias(rng, B, N)
    out, ref = _both(lambda: lay.fast_attn(a, rc.layer(0), None, ab), monkeypatch)
    assert out.shape == ref.shape
    r = _rel(out, ref)
    print(f"attn N={N} B={B}: {r:.2e}")
    assert r <= TOL


@pytest.mark.parametrize("N", [1, 7, 117, 400])
@pytest.mark.parametrize("B", [1, 3])
def test_transition_half(N, B, monkeypatch):
    rng = np.random.default_rng(N * 10 + B + 1)
    lay = _layer(2)
    a = _rnd(rng, (B, N, D))
    rc = _raw(rng, B, N)
    out, ref = _both(lambda: lay.fast_transition(a, rc.layer(0)), monkeypatch)
    r = _rel(out, ref)
    print(f"transition N={N} B={B}: {r:.2e}")
    assert r <= TOL


@pytest.mark.parametrize("N", [117, 400])
def test_shared_conditioning_rep(N, monkeypatch):
    """One conditioning row-set shared by `rep` samples (shared noise level) == the repeated conditioning."""
    rng = np.random.default_rng(5)
    lay = _layer(3)
    B = 3
    a = _rnd(rng, (B, N, D))
    rc = _raw(rng, 1, N)
    rc.rep = B
    ab = _mask_bias(rng, B, N)
    out, ref = _both(lambda: (lay.fast_attn(a, rc.layer(0), None, ab), lay.fast_transition(a, rc.layer(0))), monkeypatch)
    for o, r in zip(out, ref):
        assert _rel(o, r) <= TOL


@pytest.mark.parametrize("N,B", [(117, 1), (117, 3), (400, 1), (1, 1)])
def test_full_transformer(N, B, monkeypatch):
    """DiffusionTransformer end to end (precompute_cond raw path, per-layer bias, mask, multiplicity)."""
    rng = np.random.default_rng(N + B)
    tr = DiffusionTransformer(depth=L, heads=HEADS, dim=D, dim_single_cond=D)
    for lay in tr.layers:
        ref_l = _layer(int(rng.integers(1 << 30)))
        lay.update(ref_l.parameters())
    for p in (tr.parameters(),):
        mx.eval(p)
    a = _rnd(rng, (B, N, D))
    s = _rnd(rng, (B, N, D))
    bias = _rnd(rng, (1, N, N, L * HEADS))
    m = (rng.uniform(size=(B, N)) < 0.9).astype(np.float32)
    m[:, 0] = 1.0
    mask = mx.array(m)
    out, ref = _both(lambda: tr(a, s, bias=bias, mask=mask, multiplicity=B), monkeypatch)
    r = _rel(out, ref)
    print(f"transformer N={N} B={B}: {r:.2e}")
    assert r <= TOL


def test_prepared_shared_times(monkeypatch):
    """Prepared path with precompute_cond_shared (the diffusion-loop configuration)."""
    N, B = 117, 3
    rng = np.random.default_rng(11)
    tr = DiffusionTransformer(depth=L, heads=HEADS, dim=D, dim_single_cond=D)
    for lay in tr.layers:
        lay.update(_layer(int(rng.integers(1 << 30))).parameters())
    mx.eval(tr.parameters())
    a = _rnd(rng, (B, N, D))
    s1 = _rnd(rng, (1, N, D))
    bias = _rnd(rng, (1, N, N, L * HEADS))
    mask = mx.ones((B, N))

    def run():
        prepared = tr.prepare(None, bias, mask, None, B, with_cond=False)
        prepared["cond"] = tr.precompute_cond_shared(s1, B)
        return tr(a, mx.repeat(s1, B, axis=0), prepared=prepared, multiplicity=B)

    out, ref = _both(run, monkeypatch)
    assert _rel(out, ref) <= TOL


def test_fallbacks(monkeypatch):
    """bf16 diffusion matmuls and non-768 dims silently use the MLX path (results identical to impl=mlx)."""
    rng = np.random.default_rng(3)
    N = 33
    tr = DiffusionTransformer(depth=1, heads=16, dim=D, dim_single_cond=D)
    tr.layers[0].update(_layer(4).parameters())
    mx.eval(tr.parameters())
    a, s = _rnd(rng, (1, N, D)), _rnd(rng, (1, N, D))
    bias = _rnd(rng, (1, N, N, HEADS))
    mask = mx.ones((1, N))
    monkeypatch.setattr(K, "DIFF_DTYPE", mx.bfloat16)
    out, ref = _both(lambda: tr(a, s, bias=bias, mask=mask), monkeypatch, expect_fused=False)
    assert float(mx.abs(out - ref).max()) == 0.0
    monkeypatch.setattr(K, "DIFF_DTYPE", mx.float32)
    small = DiffusionTransformer(depth=1, heads=4, dim=128, dim_single_cond=128)
    mx.eval(small.parameters())
    a, s = _rnd(rng, (1, N, 128)), _rnd(rng, (1, N, 128))
    bias = _rnd(rng, (1, N, N, 4))
    out, ref = _both(lambda: small(a, s, bias=bias, mask=mask), monkeypatch, expect_fused=False)
    assert float(mx.abs(out - ref).max()) == 0.0


@pytest.mark.parametrize("N", [1, 117, 400])
@pytest.mark.parametrize("B", [1, 3])
def test_fp16_halves(N, B, monkeypatch):
    """BOLTZ_MLX_DIFF_DTYPE=fp16: fused halves must be as close to the fp32 result as the MLX fp16 path is
    (fp16 GEMM operands/results dominate the error; the fused kernels add no measurable error)."""
    rng = np.random.default_rng(N * 7 + B)
    lay = _layer(5)
    a = _rnd(rng, (B, N, D))
    rc = _raw(rng, B, N)
    ab = _mask_bias(rng, B, N)
    fns = {"attn": lambda: lay.fast_attn(a, rc.layer(0), None, ab),
           "trans": lambda: lay.fast_transition(a, rc.layer(0))}
    for name, fn in fns.items():
        monkeypatch.setattr(KD, "DIFFTX_IMPL", "mlx")
        monkeypatch.setattr(K, "DIFF_DTYPE", mx.float32)
        lay._fw = None
        ref32 = fn()
        monkeypatch.setattr(K, "DIFF_DTYPE", mx.float16)
        lay._fw = None
        mlx16 = fn()
        monkeypatch.setattr(KD, "DIFFTX_IMPL", "fused")
        fused16 = fn()
        mx.eval(ref32, mlx16, fused16)
        e_mlx, e_fused, e_cross = _rel(mlx16, ref32), _rel(fused16, ref32), _rel(fused16, mlx16)
        print(f"fp16 {name} N={N} B={B}: mlx16-vs-fp32 {e_mlx:.2e}  fused16-vs-fp32 {e_fused:.2e}  fused16-vs-mlx16 {e_cross:.2e}")
        assert e_fused <= 1.5 * e_mlx + 1e-6
        assert e_cross <= 3e-3
    lay._fw = None
