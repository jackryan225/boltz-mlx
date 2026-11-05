"""Candidate NAX kernels (boltz_mlx/kernels_nax2.py) vs their MLX-op references (same rounding points). GPU only:

  BOLTZ_MLX_DEVICE=gpu python -m pytest -q tests/test_nax2_kernels.py

Shapes include ragged tails (rows / positions / N not multiples of the tiles or the K chunk) and every config listed
in the module. Tolerances: same fp16 rounding points as the reference, different fp32 summation order -> a few
fp16 ulps on outputs (rel Frobenius <= 2e-3, max |d| / max |ref| <= 1e-2).
"""

import mlx.core as mx
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(mx.default_device() != mx.gpu, reason="custom kernels need the GPU device")


def _KN2():
    from boltz_mlx import kernels_nax2 as KN2

    return KN2


def _cmp(out, ref, fro=2e-3, mx_rel=1e-2):
    o = np.array(out.astype(mx.float32), np.float64)
    r = np.array(ref.astype(mx.float32), np.float64)
    assert np.isfinite(o).all()
    rf = np.linalg.norm(o - r) / np.linalg.norm(r)
    mr = np.abs(o - r).max() / np.abs(r).max()
    assert rf <= fro and mr <= mx_rel, (rf, mr)


def _h(rng, shape, s=1.0):
    return mx.array((rng.normal(size=shape) * s).astype(np.float16))


@pytest.mark.parametrize("cfg_i", range(7))
@pytest.mark.parametrize("rows", [37, 300, 1029])
def test_transition_nax(cfg_i, rows):
    KN2 = _KN2()
    cfg = KN2.TRANS_CFGS[cfg_i]
    if rows < cfg["tm"]:
        pytest.skip("rows < tile")
    rng = np.random.default_rng(rows + cfg_i)
    D, H = 128, 512
    x = _h(rng, (1, rows, D))
    w12 = _h(rng, (2 * H, D), D ** -0.5)
    w3 = _h(rng, (D, H), H ** -0.5)
    _cmp(KN2.transition_nax(x, w12, w3, **cfg), KN2.transition_ref(x, w12, w3))


@pytest.mark.parametrize("D,H,hs", [(128, 256, 1.0), (64, 256, 0.25)])
def test_transition_nax_shapes_scales_resid(D, H, hs):
    KN2 = _KN2()
    rng = np.random.default_rng(D + H)
    x = _h(rng, (1, 5, 49, D))
    w12 = _h(rng, (2 * H, D), D ** -0.5)
    w3 = _h(rng, (D, H), H ** -0.5)
    y = KN2.transition_nax(x, w12, w3, hidden_scale=hs)
    _cmp(y, KN2.transition_ref(x, w12, w3, hidden_scale=hs))
    z = mx.array(rng.normal(size=x.shape).astype(np.float32))
    zr = KN2.transition_nax(x, w12, w3, hidden_scale=hs, out_scale=1.0 / hs, resid=z)
    assert bool(mx.array_equal(zr, z + y.astype(mx.float32) * (1.0 / hs)).item())


@pytest.mark.parametrize("cfg_i", range(5))
@pytest.mark.parametrize("NM", [40, 333, 2500])
def test_trimul_proj_nax(cfg_i, NM):
    KN2 = _KN2()
    cfg = KN2.TRIMUL_PROJ_CFGS[cfg_i]
    if NM < cfg["tp"]:
        pytest.skip("positions < tile")
    rng = np.random.default_rng(NM + cfg_i)
    D = 128
    x = _h(rng, (NM, D))
    w = _h(rng, (4 * D, D), D ** -0.5)
    mask = mx.array((rng.random(NM) > 0.2).astype(np.float32))
    wgo = _h(rng, (D, D), D ** -0.5)
    ab = KN2.trimul_proj_nax(x, w, mask, 1 / 16, **cfg)
    _cmp(ab, KN2.trimul_proj_ref(x, w, mask, 1 / 16))
    ab2, g = KN2.trimul_proj_nax(x, w, mask, 1 / 16, wgo, **cfg)
    ab_r, g_r = KN2.trimul_proj_ref(x, w, mask, 1 / 16, wgo)
    _cmp(ab2, ab_r)
    _cmp(g, g_r)
    m0 = np.array(mask) == 0
    assert (np.array(ab.astype(mx.float32))[:, m0] == 0).all()


@pytest.mark.parametrize("cfg_i", range(5))
@pytest.mark.parametrize("N", [64, 100, 117])
@pytest.mark.parametrize("outgoing", [True, False])
def test_trimul_contract_nax(cfg_i, N, outgoing):
    KN2 = _KN2()
    cfg = KN2.TRIMUL_CFGS[cfg_i]
    if N < max(cfg["tm"], cfg["tn"]):
        pytest.skip("N < tile")
    rng = np.random.default_rng(N + cfg_i + 7 * outgoing)
    a = _h(rng, (8, N, N))
    b = _h(rng, (8, N, N))
    _cmp(KN2.trimul_contract_nax(a, b, outgoing, **cfg), KN2.trimul_contract_ref(a, b, outgoing))


@pytest.mark.parametrize("B,N", [(1, 37), (1, 117), (2, 200)])
@pytest.mark.parametrize("use_qb", [True, False])
def test_attention_nax_hd48(B, N, use_qb):
    """Diffusion-transformer attention: 16 heads x 48, additive bias with masked keys (-1e9), q bias."""
    KN2 = _KN2()
    rng = np.random.default_rng(B * 1000 + N + use_qb)
    heads, c = 16, 48
    h = _h(rng, (B * N, 4 * heads * c))
    bias = rng.normal(size=(B, heads, N, N)).astype(np.float32)
    bias[..., 3] = -1e9  # a masked key
    bias = mx.array(bias)
    qb = mx.array(rng.normal(size=(heads * c,)).astype(np.float32) * 0.5) if use_qb else None
    for cfg in (dict(), dict(wq=2, nkf=4, order=0)):
        o = KN2.attention_nax(h, bias, B=B, N=N, heads=heads, c=c, qbias=qb, **cfg)
        _cmp(o, KN2.attention_ref(h, bias, B=B, N=N, heads=heads, c=c, qbias=qb), fro=3e-3, mx_rel=2e-2)


# ------------------------------------------------------------------------------------------------ f16 integration
@pytest.fixture
def f16_mode():
    from boltz_mlx import kernels as K

    old = (K.PRECISION, K.DIFF_DTYPE, dict(K.NAX_FLAGS))
    K.PRECISION = "f16"
    K.set_diff_dtype("fp16")
    yield K
    K.PRECISION, K.DIFF_DTYPE = old[0], old[1]
    K.NAX_FLAGS.clear()
    K.NAX_FLAGS.update(old[2])


def _rel(a, b):
    from boltz_mlx import kernels as K

    a, b = np.array(K.unscale(a).astype(mx.float32), np.float64), np.array(K.unscale(b).astype(mx.float32), np.float64)
    assert np.isfinite(a).all()
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


@pytest.mark.parametrize("D,H,shape", [(128, 512, (1, 40, 40)), (64, 256, (1, 7, 50))])
def test_f16_transition_nax_on_off(f16_mode, D, H, shape):
    from boltz_mlx.layers.transition import Transition

    K = f16_mode
    t = Transition(D, H)
    x = mx.random.normal((*shape, D))
    xn = t.norm.f16(x)
    K.NAX_FLAGS["TRANSITION"] = "0"
    y0 = t(x, xn=xn)
    K.NAX_FLAGS["TRANSITION"] = None
    y1 = t(x, xn=xn)
    assert type(y0) is type(y1) and _rel(y1, y0) <= 2e-3
    zr = t.f16_resid(x, xn)
    if D == 128:
        assert bool(mx.array_equal(zr, K.radd(x, y1)).item())
    else:
        assert zr is None  # scaled (Scaled output) transitions keep radd


@pytest.mark.parametrize("outgoing", [True, False])
def test_f16_trimul_nax_on_off(f16_mode, outgoing):
    from boltz_mlx.layers.triangular_mult import TriangleMultiplicationIncoming, TriangleMultiplicationOutgoing

    K = f16_mode
    m = (TriangleMultiplicationOutgoing if outgoing else TriangleMultiplicationIncoming)(128)
    x = mx.random.normal((1, 45, 45, 128))
    mask = (mx.random.uniform(shape=(1, 45, 45)) > 0.2).astype(mx.float32)
    K.NAX_FLAGS["TRIMUL"] = "0"
    y0 = m(x, mask=mask)
    K.NAX_FLAGS["TRIMUL"] = None
    y1 = m(x, mask=mask)
    assert _rel(y1, y0) <= 3e-3


def test_f16_diffattn_nax_on_off_and_compiled(f16_mode):
    from boltz_mlx.modules import transformersv2 as TV

    K = f16_mode
    layer = TV.DiffusionTransformerLayer(16, 768, 768, False)
    B, N = 2, 37
    a = mx.random.normal((B, N, 768))
    cond = tuple(mx.random.uniform(shape=(B, N, 768)) for _ in range(3))
    ab = mx.random.normal((B, 16, N, N))
    ab = mx.where(mx.arange(N) == 3, -1e9, ab)
    K.NAX_FLAGS["DIFFATTN"] = "0"
    y0 = layer.fast_attn(a, cond, None, ab)
    K.NAX_FLAGS["DIFFATTN"] = None
    y1 = layer.fast_attn(a, cond, None, ab)
    assert _rel(y1 - a, y0 - a) <= 3e-3
    f = mx.compile(lambda a_, ab_: layer.fast_attn(a_, cond, None, ab_))
    assert bool(mx.array_equal(f(a, ab), y1).item())
