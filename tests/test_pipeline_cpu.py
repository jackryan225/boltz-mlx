"""R6-prof pipelined submission (numerics-neutral): CPU checks that the pipelined paths give bitwise-identical results.

- sampler: Euler and Heun loops with async step submission (sampler.PIPELINE) == blocking per-step evals;
- layer stacks: PairformerModule / PairformerNoSeqModule / MSAModule with per-layer mx.async_eval (kernels.PIPE_LAYERS)
  == without, and the submissions really happen (counted).
CPU device only; mx.fast.metal_kernel is refused (custom kernels run on the GPU stream whatever the default device).
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from boltz_mlx import kernels as K
from boltz_mlx import sampler as S


@pytest.fixture(autouse=True)
def _cpu_no_metal(monkeypatch):
    if mx.default_device() != mx.cpu:
        pytest.skip("CPU check")

    def refuse(*a, **kw):
        raise AssertionError(f"CPU test tried to build a Metal kernel: {kw.get('name', a[:1])}")

    monkeypatch.setattr(mx.fast, "metal_kernel", refuse)


def _denoiser(n_atoms, seed=0):
    rng = np.random.default_rng(seed)
    target = mx.array(rng.normal(size=(1, n_atoms, 3)).astype(np.float32) * 5)
    w = mx.array(rng.normal(size=(3, 3)).astype(np.float32) * 0.1)

    def denoise(x, t_hat):
        return mx.tanh(0.05 * x + target) * 8.0 + x @ w / (1.0 + t_hat)

    return denoise


@pytest.mark.parametrize("solver", ["euler", "heun"])
def test_pipelined_sampler_bitwise(monkeypatch, solver):
    n, mult = 96, 3
    mask = mx.ones((1, n))
    p = S.DiffusionParams(solver=solver, step_scale=1.0 if solver == "heun" else 1.5)
    steps = 7 if solver == "heun" else 12
    outs = {}
    for on in (False, True):
        monkeypatch.setattr(S, "PIPELINE", on)
        calls = {"async": 0}
        real = mx.async_eval

        def counting(*a):
            calls["async"] += 1
            return real(*a)

        monkeypatch.setattr(mx, "async_eval", counting)
        x = S.sample(_denoiser(n), mask, steps, mult, S.SeededNoise([11, 12, 13]), S.weighted_rigid_align, p)
        mx.eval(x)
        outs[on] = np.array(x)
        assert calls["async"] == (steps if on else 0)
        monkeypatch.setattr(mx, "async_eval", real)
    assert np.isfinite(outs[True]).all()
    np.testing.assert_array_equal(outs[True], outs[False])


def _count_async(monkeypatch):
    calls = {"n": 0}
    real = mx.async_eval

    def counting(*a):
        calls["n"] += 1
        return real(*a)

    monkeypatch.setattr(mx, "async_eval", counting)
    return calls


def test_pipelined_pairformer_stacks_bitwise(monkeypatch):
    from boltz_mlx.layers.pairformer import PairformerModule, PairformerNoSeqModule

    mx.random.seed(0)
    ts, tz, n = 32, 16, 20
    pf = PairformerModule(ts, tz, num_blocks=3, num_heads=2, pairwise_head_width=8, pairwise_num_heads=2)
    ns = PairformerNoSeqModule(tz, num_blocks=2, pairwise_head_width=8, pairwise_num_heads=2)
    mx.eval(pf.parameters(), ns.parameters())
    s0 = mx.random.normal((1, n, ts))
    z0 = mx.random.normal((1, n, n, tz))
    mask = mx.concatenate([mx.ones((1, n - 3)), mx.zeros((1, 3))], axis=1)
    pm = mask[:, :, None] * mask[:, None, :]
    res = {}
    for on in (False, True):
        monkeypatch.setattr(K, "PIPE_LAYERS", on)
        calls = _count_async(monkeypatch)
        s, z = pf(s0, z0, mask, pm)
        z2 = ns(z, pm)
        mx.eval(s, z, z2)
        res[on] = [np.array(a) for a in (s, z, z2)]
        assert calls["n"] == (3 + 2 if on else 0)
    for a, b in zip(res[True], res[False]):
        np.testing.assert_array_equal(a, b)


def test_pipelined_msa_module_bitwise(monkeypatch):
    from boltz_mlx.modules.trunkv2 import NUM_TOKENS, MSAModule

    mx.random.seed(1)
    ts, tz, n, sq = 32, 16, 12, 5
    m = MSAModule(msa_s=16, token_z=tz, token_s=ts, msa_blocks=2, msa_dropout=0.0, z_dropout=0.0)
    mx.eval(m.parameters())
    rng = np.random.default_rng(0)
    feats = {
        "msa": mx.array(rng.integers(0, NUM_TOKENS, size=(1, sq, n))),
        "has_deletion": mx.array(rng.integers(0, 2, size=(1, sq, n)).astype(np.float32)),
        "deletion_value": mx.array(rng.random(size=(1, sq, n)).astype(np.float32)),
        "msa_paired": mx.array(rng.integers(0, 2, size=(1, sq, n)).astype(np.float32)),
        "msa_mask": mx.array(np.ones((1, sq, n), np.float32)),
        "token_pad_mask": mx.array(np.ones((1, n), np.float32)),
    }
    z0 = mx.random.normal((1, n, n, tz))
    emb = mx.random.normal((1, n, ts))
    res = {}
    for on in (False, True):
        monkeypatch.setattr(K, "PIPE_LAYERS", on)
        calls = _count_async(monkeypatch)
        z = m(z0, emb, feats)
        mx.eval(z)
        res[on] = np.array(z)
        assert calls["n"] == (2 if on else 0)
    np.testing.assert_array_equal(res[True], res[False])


def test_submit_is_a_noop_while_suspended(monkeypatch):
    calls = _count_async(monkeypatch)
    monkeypatch.setattr(K, "PIPE_LAYERS", True)
    monkeypatch.setattr(K, "GUARD_SUSPEND", 1)
    K.submit(mx.ones((2,)))
    assert calls["n"] == 0
    monkeypatch.setattr(K, "GUARD_SUSPEND", 0)
    K.submit(mx.ones((2,)))
    assert calls["n"] == 1
