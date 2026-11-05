"""Experimental NAX (MetalPerformancePrimitives matmul2d) triangle-attention kernel (boltz_mlx/kernels_nax.py) vs the
fp32 simdgroup-MMA kernel on the same fp16 inputs. GPU only (BOLTZ_MLX_DEVICE=gpu pytest tests/test_triatt_nax.py).
Tolerances: fp16 operands for Q K^T / P V (+ fp16 output) -> bounded like the shipped mma_half variant."""

import mlx.core as mx
import numpy as np
import pytest

from boltz_mlx import kernels as K

pytestmark = pytest.mark.skipif(mx.default_device() != mx.gpu, reason="custom kernels need the GPU device")

H, C = 4, 32
TOL = 5e-3  # max|d| / max|ref|; mma_half measured ~6e-4


def _nax():
    from boltz_mlx import kernels_nax as KN

    return KN


def _rand_mask(rng, shape):
    m = (rng.random(shape) > 0.3).astype(np.float32)
    if shape[-1] > 5:
        m[..., :, 5] = 0.0  # one masked key column
    m[..., :, 0] = 1.0  # at least one valid key per row (fully masked rows are padding, compared elsewhere)
    return mx.array(m)


def _rel(a, ref):
    a, ref = np.array(a.astype(mx.float32)), np.array(ref)
    assert np.isfinite(a).all()
    return float(np.abs(a - ref).max() / np.abs(ref).max())


def _h(rng, B, I, J, F, scale=1.0):
    return mx.array((rng.normal(size=(B, I, J, F)) * scale).astype(np.float16))


def test_nax_mma_layout():
    KN = _nax()
    rng = np.random.default_rng(0)
    a = rng.standard_normal((16, 16)).astype(np.float16).astype(np.float32)
    for tb in (True, False):
        b = rng.standard_normal((32, 16) if tb else (16, 32)).astype(np.float16).astype(np.float32)
        c = np.array(KN.nax_mma_once(mx.array(a), mx.array(b), tb))
        ref = a @ (b.T if tb else b)
        assert np.abs(c - ref).max() / np.abs(ref).max() < 1e-3


@pytest.mark.parametrize("N", [1, 2, 9, 17, 37, 117, 200])
@pytest.mark.parametrize("use_mask", [False, True])
def test_nax_vs_mma(N, use_mask):
    KN = _nax()
    rng = np.random.default_rng(N)
    h = _h(rng, 1, N, N, 4 * H * C + H)
    mask = _rand_mask(rng, (1, N, N)) if use_mask else None
    ref = K.tri_attention_fused(h.astype(mx.float32), H, C, mask, 1e9, impl="mma")
    for cfg in ({}, {"wq": 5, "nkf": 4}, {"hpt": 4, "wq": 1}, {"order": 1}, {"p_dtype": mx.float16}):
        out = KN.tri_attention_nax(h, H, C, mask, 1e9, **cfg)
        mx.eval(ref, out)
        assert _rel(out, ref) <= TOL, cfg


def test_nax_separate_bias_rows_batch():
    """Row-chunked call (I != J, tri bias passed separately, F = 4HC), B > 1."""
    KN = _nax()
    rng = np.random.default_rng(7)
    B, I, J = 2, 13, 61
    h = _h(rng, B, I, J, 4 * H * C)
    bias = mx.array(rng.normal(size=(B, H, J, J)).astype(np.float16))
    mask = _rand_mask(rng, (B, I, J))
    for m in (None, mask):
        # orientation: the MMA kernel takes the tri bias as [B, H, K, Q], the NAX kernel as [B, H, Q, K]
        # (attention.py passes tb.transpose(0, 3, 2, 1) resp. tb.transpose(0, 3, 1, 2)); same bias(b, h, q, k)
        ref = K.tri_attention_fused(h.astype(mx.float32), H, C, m, 1e9, bias=bias.swapaxes(-1, -2).astype(mx.float32),
                                    impl="mma")
        out = KN.tri_attention_nax(h, H, C, m, 1e9, bias=bias)
        mx.eval(ref, out)
        assert _rel(out, ref) <= TOL


def test_nax_trans_matches_swapaxes():
    """trans=True on h [B, J, I, F] == the default kernel on h.swapaxes(1, 2), output in h's orientation."""
    KN = _nax()
    rng = np.random.default_rng(9)
    N = 70
    h = _h(rng, 1, N, N, 4 * H * C + H)
    mask = _rand_mask(rng, (1, N, N))  # attention orientation
    ht = h.swapaxes(1, 2)
    ref = KN.tri_attention_nax(ht, H, C, mask, 1e9)  # contiguous copy of the transposed h
    out = KN.tri_attention_nax(h, H, C, mask, 1e9, trans=True)
    mx.eval(ref, out)
    assert _rel(out.swapaxes(1, 2), ref.astype(mx.float32)) <= 1e-6
