"""Sampler loop parity vs upstream AtomDiffusion.sample, using a stub score network.

Exercises schedule, noise order, random rotation, weighted_rigid_align and the update rule
independently of the score model port.
"""

import mlx.core as mx
import numpy as np
import torch

from boltz_mlx.compare import assert_close, rmsd
from boltz_mlx.sampler import (
    DiffusionParams,
    ReplayNoise,
    sample,
    sample_schedule,
    weighted_rigid_align,
)


class StubScore(torch.nn.Module):
    """Deterministic nonlinear stand-in for the score model: pulls atoms toward a fixed shape."""

    def __init__(self, target):
        super().__init__()
        self.target = target
        self.dummy = torch.nn.Parameter(torch.zeros(1))  # AtomDiffusion.device reads a parameter

    def forward(self, r_noisy, times, **kw):
        return torch.tanh(0.3 * r_noisy + self.target) - 0.1 * r_noisy * times[:, None, None]


def stub_np(r, times, target):
    return np.tanh(0.3 * r + target) - 0.1 * r * times[:, None, None]


def run_torch(n_atoms=40, steps=30, mult=2, seed=0):
    from boltz.model.modules.diffusionv2 import AtomDiffusion

    g = torch.Generator().manual_seed(123)
    target = torch.randn(1, n_atoms, 3, generator=g)
    mask = torch.ones(1, n_atoms)
    mask[0, -3:] = 0
    ad = AtomDiffusion.__new__(AtomDiffusion)
    torch.nn.Module.__init__(ad)
    p = DiffusionParams()
    for k, v in p.__dict__.items():
        setattr(ad, k, v)
    ad.num_sampling_steps = steps
    ad.step_scale_random = None
    ad.score_model = StubScore(target)
    ad.register_buffer("zero", torch.tensor(0.0))
    ad.training = False
    draws = []
    orig = torch.randn

    def rec(*a, **k):
        t = orig(*a, **k)
        draws.append(t.numpy().copy())
        return t

    torch.manual_seed(seed)
    torch.randn = rec
    try:
        out = ad.sample(
            atom_mask=mask,
            num_sampling_steps=steps,
            multiplicity=mult,
            steering_args={"fk_steering": False, "physical_guidance_update": False, "contact_guidance_update": False},
        )["sample_atom_coords"]
    finally:
        torch.randn = orig
    return out.numpy(), draws, target.numpy(), mask.numpy(), ad.sample_schedule(steps).numpy()


def test_schedule_matches_torch():
    _, _, _, _, sched = run_torch(steps=30)
    # float32 pow may differ by 1 ulp between numpy and torch
    np.testing.assert_allclose(sample_schedule(DiffusionParams(), 30), sched, rtol=1e-6)


def test_weighted_rigid_align():
    from boltz.model.loss.diffusionv2 import weighted_rigid_align as wra_t

    rng = np.random.default_rng(0)
    a = rng.normal(size=(2, 50, 3)).astype(np.float32) * 10
    b = rng.normal(size=(2, 50, 3)).astype(np.float32) * 10
    m = np.ones((2, 50), np.float32)
    m[1, 40:] = 0
    ref = wra_t(*(torch.from_numpy(x) for x in (a, b, m, m))).numpy()
    ours = weighted_rigid_align(*(mx.array(x) for x in (a, b, m, m)))
    assert_close(ours, ref, rel=1e-5, cos=0.999999, name="weighted_rigid_align")


def test_sampler_loop(tmp_path):
    ref, draws, target, mask, _ = run_torch()
    np.savez(tmp_path / "noise.npz", **{f"noise_{i:05d}": d for i, d in enumerate(draws)})
    noise = ReplayNoise(tmp_path / "noise.npz")
    p = DiffusionParams()
    tgt = mx.array(target)

    def denoise(x, t_hat):
        sd = p.sigma_data
        c_skip = sd**2 / (t_hat**2 + sd**2)
        c_out = t_hat * sd / np.sqrt(sd**2 + t_hat**2)
        c_in = 1 / np.sqrt(t_hat**2 + sd**2)
        times = np.log(t_hat / sd) * 0.25
        r = mx.tanh(0.3 * (c_in * x) + tgt) - 0.1 * (c_in * x) * times
        return c_skip * x + c_out * r

    ours = sample(denoise, mx.array(mask), 30, 2, noise, weighted_rigid_align, p)
    assert noise.i == len(draws)
    print("sampler rmsd", rmsd(ours, ref))
    assert_close(ours, ref, rel=1e-4, cos=0.99999, name="sampler_loop")
