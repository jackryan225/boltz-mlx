"""Direct parity of the simdgroup-MMA tri-attention kernel against the scalar kernel. GPU only."""

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_map

from boltz_mlx import kernels as K
from boltz_mlx.layers.triangular_attention.attention import TriangleAttention

pytestmark = pytest.mark.skipif(mx.default_device() != mx.gpu, reason="custom kernels need the GPU device")

H, C = 4, 32


def _rand_mask(rng, shape):
    m = (rng.random(shape) > 0.3).astype(np.float32)
    m[..., 3, :] = 0.0  # fully-masked query rows (all keys masked)
    m[..., :, 5] = 0.0  # one masked key column
    m[..., 7, :] = 0.0
    return mx.array(m)


def _rel(a, ref):
    a, ref = np.array(a), np.array(ref)
    assert not np.isnan(a).any()
    return float(np.abs(a - ref).max() / np.abs(ref).max())


@pytest.mark.parametrize("N", [37, 117, 200])
@pytest.mark.parametrize("use_mask", [False, True])
def test_mma_vs_scalar(N, use_mask):
    rng = np.random.default_rng(N)
    h = mx.array(rng.normal(size=(1, N, N, 4 * H * C + H)).astype(np.float32))
    mask = _rand_mask(rng, (1, N, N)) if use_mask else None
    ref = K.tri_attention_fused(h, H, C, mask, 1e9, impl="scalar")
    out = K.tri_attention_fused(h, H, C, mask, 1e9, impl="mma")
    mx.eval(ref, out)
    assert _rel(out, ref) <= 1e-5


@pytest.mark.parametrize("N", [1, 2, 9, 17, 33])
def test_mma_tiny_and_ragged(N):
    rng = np.random.default_rng(100 + N)
    h = mx.array(rng.normal(size=(2, N, N, 4 * H * C + H)).astype(np.float32))
    mask = mx.array((rng.random((2, N, N)) > 0.3).astype(np.float32))
    for m in (None, mask):
        ref = K.tri_attention_fused(h, H, C, m, 1e9, impl="scalar")
        out = K.tri_attention_fused(h, H, C, m, 1e9, impl="mma")
        mx.eval(ref, out)
        assert _rel(out, ref) <= 1e-5


def test_mma_separate_bias_row_chunk():
    """Row-chunked call: I != J, tri bias passed separately (F = 4*H*C, no bias columns), B > 1."""
    rng = np.random.default_rng(7)
    B, I, J = 2, 13, 61
    h = mx.array(rng.normal(size=(B, I, J, 4 * H * C)).astype(np.float32))
    bias = mx.array(rng.normal(size=(B, H, J, J)).astype(np.float32))
    mask = _rand_mask(rng, (B, I, J))
    for m in (None, mask):
        ref = K.tri_attention_fused(h, H, C, m, 1e9, bias=bias, impl="scalar")
        out = K.tri_attention_fused(h, H, C, m, 1e9, bias=bias, impl="mma")
        mx.eval(ref, out)
        assert _rel(out, ref) <= 1e-5


def test_mma_unaligned_row_stride_and_heads():
    """F not a multiple of 4 (scalar tile-load path) and a different head count / head dim."""
    rng = np.random.default_rng(11)
    n_heads, c, N = 3, 16, 45
    h = mx.array(rng.normal(size=(1, N, N, 4 * n_heads * c + n_heads)).astype(np.float32))
    mask = _rand_mask(rng, (1, N, N))
    ref = K.tri_attention_fused(h, n_heads, c, mask, 1e9, impl="scalar")
    out = K.tri_attention_fused(h, n_heads, c, mask, 1e9, impl="mma")
    mx.eval(ref, out)
    assert _rel(out, ref) <= 1e-5


def test_mma_half_error_is_small():
    """fp16 Q/K/V/P operands, fp32 accumulate: reduced precision variant, loose bound (measured ~6e-4)."""
    rng = np.random.default_rng(3)
    N = 117
    h = mx.array(rng.normal(size=(1, N, N, 4 * H * C + H)).astype(np.float32))
    mask = _rand_mask(rng, (1, N, N))
    ref = K.tri_attention_fused(h, H, C, mask, 1e9, impl="scalar")
    out = K.tri_attention_fused(h, H, C, mask, 1e9, impl="mma_half")
    mx.eval(ref, out)
    err = _rel(out, ref)
    print("mma_half max|d|/max|ref|", err)
    assert err <= 5e-3


@pytest.mark.parametrize("starting", [True, False])
def test_layer_mma_vs_scalar_incl_chunked(monkeypatch, starting):
    rng = np.random.default_rng(5)
    N = 50
    layer = TriangleAttention(32, C, H, starting=starting, inf=1e9)
    layer.update(tree_map(lambda p: mx.array(rng.normal(size=p.shape).astype(np.float32) * 0.2), layer.parameters()))
    x = mx.array(rng.normal(size=(1, N, N, 32)).astype(np.float32))
    mask = _rand_mask(rng, (1, N, N))
    for chunked in (False, True):
        if chunked:
            monkeypatch.setattr(K, "CHUNK_N", 16)
            monkeypatch.setattr(K, "CHUNK_ROWS", 16)
        monkeypatch.setattr(K, "TRIATT_IMPL", "scalar")
        ref = layer(x, mask=mask)
        monkeypatch.setattr(K, "TRIATT_IMPL", "mma")
        layer._w = None
        out = layer(x, mask=mask)
        mx.eval(ref, out)
        assert _rel(out, ref) <= 1e-5
