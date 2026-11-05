"""Fused Transition kernel (LN + GEMM + SwiGLU + GEMM in one Metal kernel) vs the MLX op path. GPU only."""

import mlx.core as mx
import numpy as np
import pytest
from conftest import load_dump, mxa

from boltz_mlx import kernels as K
from boltz_mlx.compare import assert_close
from boltz_mlx.layers.transition import Transition
from boltz_mlx.weights import load_module

pytestmark = pytest.mark.skipif(mx.default_device() != mx.gpu, reason="custom kernels need the GPU device")


def _random_transition(seed=0):
    rng = np.random.default_rng(seed)
    t = Transition(128, 512)
    t.norm.weight = mx.array(rng.uniform(0.5, 1.5, 128).astype(np.float32))
    t.norm.bias = mx.array(rng.normal(0, 0.3, 128).astype(np.float32))
    t.fc1.weight = mx.array((rng.normal(size=(512, 128)) / np.sqrt(128)).astype(np.float32))
    t.fc2.weight = mx.array((rng.normal(size=(512, 128)) / np.sqrt(128)).astype(np.float32))
    t.fc3.weight = mx.array((rng.normal(size=(128, 512)) / np.sqrt(512)).astype(np.float32))
    mx.eval(t.parameters())
    return t


def _run(t, x, impl, monkeypatch):
    monkeypatch.setattr(K, "TRANSITION_IMPL", impl)
    y = t(x)
    mx.eval(y)
    return y


@pytest.mark.parametrize("rows", [1, 37, 13689, 50000])
def test_fused_matches_mlx(rows, monkeypatch):
    t = _random_transition()
    rng = np.random.default_rng(rows)
    x = mx.array((rng.normal(size=(1, 1, rows, 128)) * 2.0 + 0.5).astype(np.float32))
    ref = _run(t, x, "mlx", monkeypatch)
    out = _run(t, x, "fused", monkeypatch)
    assert out.shape == ref.shape
    rel = float(mx.abs(out - ref).max() / mx.abs(ref).max())
    print(f"rows={rows}: max|d|/max|ref| = {rel:.2e}")
    assert rel <= 1e-5


def test_fused_unaligned_shapes(monkeypatch):
    """Non-4D input, rows not a multiple of the tile, and the module's other sizes fall back cleanly."""
    t = _random_transition(1)
    x = mx.array(np.random.default_rng(3).normal(size=(2, 7, 11, 128)).astype(np.float32))
    ref = _run(t, x, "mlx", monkeypatch)
    out = _run(t, x, "fused", monkeypatch)
    assert float(mx.abs(out - ref).max() / mx.abs(ref).max()) <= 1e-5
    # unsupported sizes (MSA D=64/H=256, s D=384/H=1536) take the MLX path even with fused selected
    for d, h in ((64, 256), (384, 1536)):
        m = Transition(d, h)
        xs = mx.random.normal((1, 9, d))
        a = _run(m, xs, "mlx", monkeypatch)
        b = _run(m, xs, "fused", monkeypatch)
        assert float(mx.abs(a - b).max()) == 0.0


def test_fused_chunked_path(monkeypatch):
    """Above CHUNK_N the MLX path row-chunks; the fused path (no big intermediates) must agree with it."""
    t = _random_transition(2)
    x = mx.array(np.random.default_rng(5).normal(size=(1, 45, 45, 128)).astype(np.float32))
    ref = _run(t, x, "mlx", monkeypatch)
    monkeypatch.setattr(K, "CHUNK_N", 16)
    monkeypatch.setattr(K, "CHUNK_ROWS", 8)
    ref_chunked = _run(t, x, "mlx", monkeypatch)
    out = _run(t, x, "fused", monkeypatch)
    assert float(mx.abs(ref_chunked - ref).max() / mx.abs(ref).max()) <= 1e-5
    assert float(mx.abs(out - ref).max() / mx.abs(ref).max()) <= 1e-5


def test_fused_transition_z_dump(monkeypatch):
    name = "pairformer_module.layers.0.transition_z"
    d = load_dump(name)
    mod = load_module(Transition(128, 512), name)
    monkeypatch.setattr(K, "TRANSITION_IMPL", "fused")
    out = mod(mxa(d["in_0"]))
    mx.eval(out)
    assert_close(out, d["out"], rel=1e-4, cos=0.99999, name=name)
