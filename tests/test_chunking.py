"""Chunked pair ops (used above BOLTZ_MLX_CHUNK_N tokens) must equal the unchunked path. GPU only."""

import mlx.core as mx
import numpy as np
import pytest

from boltz_mlx import kernels as K
from boltz_mlx.layers.pairformer import PairformerLayer
from boltz_mlx.weights import load_module


@pytest.mark.skipif(mx.default_device() != mx.gpu, reason="custom kernels need the GPU device")
def test_chunked_layer_equals_unchunked(monkeypatch):
    N = 70
    rng = np.random.default_rng(0)
    s = mx.array(rng.normal(size=(1, N, 384)).astype(np.float32))
    z = mx.array(rng.normal(size=(1, N, N, 128)).astype(np.float32))
    mask = mx.ones((1, N)); mask[:, -5:] = 0
    pm = mask[:, :, None] * mask[:, None, :]
    L = load_module(PairformerLayer(384, 128, 16), "pairformer_module.layers.0")
    s0, z0 = L(s, z, mask, pm); mx.eval(s0, z0)
    monkeypatch.setattr(K, "CHUNK_N", 32)
    monkeypatch.setattr(K, "CHUNK_ROWS", 16)
    monkeypatch.setattr(K, "CHUNK_CH", 24)
    s1, z1 = L(s, z, mask, pm); mx.eval(s1, z1)
    np.testing.assert_allclose(np.array(z1), np.array(z0), rtol=0, atol=2e-4)
    np.testing.assert_allclose(np.array(s1), np.array(s0), rtol=0, atol=2e-3)
    print("max |dz|", float(mx.abs(z1 - z0).max()), "max |ds|", float(mx.abs(s1 - s0).max()))
