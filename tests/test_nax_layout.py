"""NAX follow-up (task 1): fragment-layout assumptions of the register NAX kernels, N=1 compile fix, and
bit-identity of the fixed kernels with the validated ones on the recorded L1 inputs. GPU only:

  BOLTZ_MLX_DEVICE=gpu python -m pytest -q tests/test_nax_layout.py

* layout: MLX steel fragments are only valid for the matmul2d descriptors / element types MLX itself uses
  (16x32x16, relaxed_precision=true). The register kernels (kernels_nax.tri_attention_nax) use exactly these; this
  pins that on the device (an MPP/OS update that changes the layout fails here, not silently in the model).
* relax=False: the same product through the layout-agnostic MPP API is correct -> the "nax_norelax" failure is the
  layout, not the arithmetic.
* N=1 (B=1): the tri bias [1, H, 1, 1] has H < 8 elements, so metal_kernel binds it as `const constant float*` and
  the kernels' `const device float* brow = bias + ...` did not compile (strict fp32 MMA / scalar kernels and the f16
  fallback). Fixed with `auto` pointers; checked against numpy here.
* bit-identity: the fixed sources (auto / ld4 pointers) give bit-identical outputs to the pre-fix sources on the
  recorded prot117 / prot400 tri-att inputs (strict fp32 MMA kernel, f16 MMA kernel, NAX kernel).

Portability (M1-M5): the layout tests are device diagnostics of MetalPerformancePrimitives itself (the same checks gate
the nax / tensorops kernel paths: kernels._tensorops_check), so they run on every kernel path under kernels.probing()
(skipped below macOS 26: no Metal 4); the MPP tri-attention kernel is exercised only when kernels.mpp_enabled().
"""

import os
import contextlib
import pickle

import mlx.core as mx
import numpy as np
import pytest

from boltz_mlx import kernels as K

pytestmark = pytest.mark.skipif(mx.default_device() != mx.gpu, reason="custom kernels need the GPU device")
_MACOS = int((__import__("platform").mac_ver()[0] or "0").split(".")[0])
metal4 = pytest.mark.skipif(_MACOS < 26, reason="MetalPerformancePrimitives tensor ops need macOS 26 (Metal 4)")

H, C = 4, 32
HC = H * C


def _KN():
    from boltz_mlx import kernels_nax as KN

    return KN


# ------------------------------------------------------------------------------------------------ layout
# (TB, AT, PT-style combos) used by tri_attention_nax: S = Q K^T (half x half^T), O += P V (half|float x half)
USED = [(True, mx.float16), (False, mx.float16), (False, mx.float32)]


@metal4
@pytest.mark.parametrize("tb,at", USED)
def test_relaxed_layout_is_mlx_layout(tb, at):
    """Left and destination cooperative tensors of the relaxed 16x32x16 descriptor = MLX BaseNAXFrag layout."""
    KN = _KN()
    with K.probing():
        caps, coords = KN.nax_layout(True, False, tb, at, mx.float16, mx.float32)
    caps, coords = np.array(caps), np.array(coords)
    exp = KN.mlx_frag_coords()  # [32, 8, (row, col)]
    assert (caps[0] == 8).all() and (caps[2] == 16).all(), caps
    left = coords[0, :, :8]
    assert (left[..., ::-1] == exp).all() or (left == exp).all()  # idx order of a left operand: either convention
    dest = coords[2, :, :16][..., ::-1]
    assert (dest[:, :8] == exp).all()
    assert (dest[:, 8:] == exp + np.array([0, 16])).all()


@metal4
@pytest.mark.parametrize("tb", [True, False])
@pytest.mark.parametrize("at", [mx.float16, mx.float32])
def test_mlx_layout_mma_relaxed(tb, at):
    KN = _KN()
    rng = np.random.default_rng(0)
    a = rng.standard_normal((16, 16)).astype(np.float16).astype(np.float32)
    b = rng.standard_normal((32, 16) if tb else (16, 32)).astype(np.float16).astype(np.float32)
    with K.probing():
        c = np.array(KN.nax_mma_once(mx.array(a), mx.array(b), tb, at, relax=True))
    ref = a.astype(np.float64) @ (b.T if tb else b).astype(np.float64)
    assert np.abs(c - ref).max() / np.abs(ref).max() < 1e-3


@metal4
@pytest.mark.parametrize("tb", [True, False])
@pytest.mark.parametrize("relax", [True, False])
def test_generic_api_any_descriptor(tb, relax):
    """Layout-agnostic MPP path: correct with and without relaxed precision."""
    KN = _KN()
    rng = np.random.default_rng(1)
    a = rng.standard_normal((16, 16)).astype(np.float16)
    b = rng.standard_normal((32, 16) if tb else (16, 32)).astype(np.float16)
    with K.probing():
        c = np.array(KN.nax_mma_generic(mx.array(a), mx.array(b), tb, relax))
    ref = a.astype(np.float64) @ (b.T if tb else b).astype(np.float64)
    assert np.abs(c - ref).max() / np.abs(ref).max() < 1e-3


# ------------------------------------------------------------------------------------------------ N = 1, 2
def _np_triatt(h, mask, bias_qk=None):
    """numpy fp64 reference: h [B, I, J, F]; mask [B, I, J] or None; bias_qk [B, H, Q, K] or None (from h)."""
    h = h.astype(np.float64)
    B, I, J, _ = h.shape
    q = h[..., :HC].reshape(B, I, J, H, C)
    k = h[..., HC:2 * HC].reshape(B, I, J, H, C)
    v = h[..., 2 * HC:3 * HC].reshape(B, I, J, H, C)
    g = h[..., 3 * HC:4 * HC]
    if bias_qk is None:
        bias_qk = h[..., 4 * HC:].transpose(0, 3, 1, 2)  # [b, h, q, k] = h[b, q, k, h]
    s = np.einsum("biqhc,bikhc->bihqk", q, k) * C**-0.5
    if mask is not None:
        s = s + (1e9 * (mask - 1.0))[:, :, None, None, :]
    s = s + bias_qk[:, None]
    s = s - s.max(-1, keepdims=True)
    p = np.exp(s)
    p /= p.sum(-1, keepdims=True)
    o = np.einsum("bihqk,bikhc->biqhc", p, v).reshape(B, I, J, HC)
    return o / (1.0 + np.exp(-g))


@pytest.mark.parametrize("N", [1, 2, 3])
@pytest.mark.parametrize("use_mask", [False, True])
def test_tiny_n_b1_all_kernels(N, use_mask):
    rng = np.random.default_rng(10 + N)
    h32 = rng.normal(size=(1, N, N, 4 * HC + H)).astype(np.float16).astype(np.float32)
    mask = (rng.random((1, N, N)) > 0.3).astype(np.float32) if use_mask else None
    if mask is not None:
        mask[..., 0] = 1.0
    ref = _np_triatt(h32, mask)
    hm = mx.array(h32)
    mm = mx.array(mask) if mask is not None else None
    outs = {
        "scalar": K.tri_attention_fused(hm, H, C, mm, 1e9, impl="scalar"),
        "mma": K.tri_attention_fused(hm, H, C, mm, 1e9, impl="mma"),
        "mma_f16": K.tri_attention_fused(hm.astype(mx.float16), H, C, mm, 1e9, impl="mma", out_dtype=mx.float16),
        "mma_half": K.tri_attention_fused(hm.astype(mx.float16), H, C, mm, 1e9, impl="mma_half",
                                          out_dtype=mx.float16),
    }
    if K.mpp_enabled():  # the MPP kernel: nax / tensorops kernel paths only
        outs["nax"] = _KN().tri_attention_nax(hm.astype(mx.float16), H, C, mm, 1e9, order=1, wq=5)
    for name, o in outs.items():
        o = np.array(o.astype(mx.float32))
        tol = 1e-5 if name in ("scalar", "mma") else 5e-3
        assert np.isfinite(o).all(), name
        assert np.abs(o - ref).max() / np.abs(ref).max() <= tol, name


# ------------------------------------------------------------------------------------------------ bit-identity
DATA = __import__("pathlib").Path(os.environ.get("BOLTZ_MLX_DATA", __import__("pathlib").Path(__file__).resolve().parents[1] / "data")).expanduser() / "cache" / "l1"
CAPS = [("prot117", "TriangleAttention", 0), ("prot117", "TriangleAttentionEndingNode", 0),
        ("prot400", "TriangleAttention", 0), ("prot400", "TriangleAttentionEndingNode", 3)]


def _legacy_src(src: str, pairs) -> str:
    for new, old in pairs:
        assert new in src, new
        src = src.replace(new, old)
    return src


_MMA_PAIRS = [("auto brow = bias + ((size_t)(b * NH + hh) * J_DIM) * J_DIM;   // [K][J] for this (b,h); auto: bias may "
               "be bound in constant space (< 8 elements, N=1)",
               "const device float* brow = bias + ((size_t)(b * NH + hh) * J_DIM) * J_DIM;   // [K][J] for this (b,h)")]
_NAX_PAIRS = [("auto hrow = h + ", "const device HT* hrow = h + "),
              ("auto brow = bias + (size_t)(b * NH + hh) * J_DIM * KP;   // auto: any input may be bound in constant "
               "space (< 8 elements)", "const device BT* brow = bias + (size_t)(b * NH + hh) * J_DIM * KP;"),
              ("auto mrow = maskb + ", "const device float* mrow = maskb + "),
              ("auto pa = hrow", "const device HT* pa = hrow"), ("auto pb = hrow", "const device HT* pb = hrow"),
              ("auto gp = hrow", "const device HT* gp = hrow"),
              ("ld4(pa + 16)", "*(const device vec<HT, 4>*)(pa + 16)"),
              ("ld4(pb + 16)", "*(const device vec<HT, 4>*)(pb + 16)"),
              ("ld4(pa)", "*(const device vec<HT, 4>*)pa"), ("ld4(pb)", "*(const device vec<HT, 4>*)pb"),
              ("ld4(mrow + kc)", "*(const device float4*)(mrow + kc)"),
              ("ld4(brow + (size_t)qra * KP + kc)", "*(const device vec<BT, 4>*)(brow + (size_t)qra * KP + kc)"),
              ("ld4(brow + (size_t)qrb * KP + kc)", "*(const device vec<BT, 4>*)(brow + (size_t)qrb * KP + kc)")]


@contextlib.contextmanager
def _legacy(module, attr, pairs):
    """Swap a kernel source for its pre-fix text; kernels get a distinct name (MLX caches libraries by name)."""
    old_src = getattr(module, attr)
    saved = dict(module._kernels)
    module._kernels.clear()
    orig = mx.fast.metal_kernel

    def mk(*a, **kw):
        kw["name"] = kw["name"] + "_legacy"
        return orig(*a, **kw)

    setattr(module, attr, _legacy_src(old_src, pairs))
    mx.fast.metal_kernel = mk
    try:
        yield
    finally:
        setattr(module, attr, old_src)
        mx.fast.metal_kernel = orig
        module._kernels.clear()
        module._kernels.update(saved)


def _inputs(case, cls, call):
    from boltz_mlx.layers.triangular_attention.attention import TriangleAttention
    from boltz_mlx.weights import load_module

    f = DATA / case / f"trunk.pairformer__{cls}__{call}.pkl"
    if not f.exists():
        pytest.skip(f"missing L1 capture {f}")
    rec = pickle.load(open(f, "rb"))
    m = load_module(TriangleAttention(128, C, H, starting=rec["cls"] == "TriangleAttention", inf=1e9), rec["path"])
    x = mx.array(rec["args"][1][0][1])
    mask = mx.array(rec["kwargs"][1]["mask"][1])
    if not m.starting:
        x, mask = x.swapaxes(-2, -3), mask.swapaxes(-1, -2)
    w = mx.concatenate([m.mha.linear_q.weight, m.mha.linear_k.weight, m.mha.linear_v.weight, m.mha.linear_g.weight,
                        m.linear.weight], 0)
    h32 = m.layer_norm(x) @ w.T
    h16 = m.layer_norm.f16(x) @ w.astype(mx.float16).T
    mx.eval(h32, h16, mask)
    return h32, h16, mask


@pytest.mark.parametrize("case,cls,call", CAPS)
def test_fixed_kernels_bit_identical_on_recorded_inputs(case, cls, call):
    KN = _KN()
    h32, h16, mask = _inputs(case, cls, call)
    J = h16.shape[2]
    tb16 = h16[..., 4 * HC:]
    runs = {
        "scalar32": lambda: K.tri_attention_fused(h32, H, C, mask, 1e9, impl="scalar"),
        "mma32": lambda: K.tri_attention_fused(h32, H, C, mask, 1e9, impl="mma"),
        "mma16": lambda: K.tri_attention_fused(h16, H, C, mask, 1e9, impl="mma", out_dtype=mx.float16),
        "nax": lambda: KN.tri_attention_nax(h16[..., :4 * HC], H, C, mask, 1e9, bias=tb16.transpose(0, 3, 1, 2),
                                            **K.TRIATT_NAX_CFG),
    }
    if not K.mpp_enabled():  # portable kernel path: no MPP kernel
        del runs["nax"]
    new = {k: f() for k, f in runs.items()}
    mx.eval(new)
    with _legacy(K, "_TRI_ATT_SRC", _MMA_PAIRS):
        old = {"scalar32": runs["scalar32"]()}
        mx.eval(old)
    with _legacy(K, "_TRI_ATT_MMA_SRC", _MMA_PAIRS):
        old.update({k: runs[k]() for k in ("mma32", "mma16")})
        mx.eval(old)
    if "nax" in runs:
        with _legacy(KN, "_TRI_NAX_SRC", _NAX_PAIRS):
            old["nax"] = runs["nax"]()
            mx.eval(old["nax"])
    for k in runs:
        assert bool(mx.array_equal(new[k], old[k]).item()), f"{k} differs at {case} {cls} {call} (J={J})"
