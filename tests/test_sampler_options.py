"""Round-2 sampler options: defaults bit-identical, Heun order, seeding, presets, CLI / run_l3 flags.

CPU only, toy denoisers (no model weights); the whole file runs in a few seconds.
"""

import argparse
import sys
from math import sqrt
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from boltz_mlx.model import PredictArgs
from boltz_mlx.sampler import (
    SAMPLER_PRESETS,
    DiffusionParams,
    NoiseSource,
    SamplerConfig,
    affinity_config,
    nfe,
    random_rotation,
    record_noise,
    resolve_sampler,
    sample,
    weighted_rigid_align,
)

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------------------------ frozen copy of the round-1 sampler
# Verbatim from boltz_mlx/sampler.py before the sampler options (git 8d4dd76). The default path must stay bit-identical.


def _frozen_schedule(p, num_steps):
    inv_rho = np.float32(1.0 / p.rho)
    steps = np.arange(num_steps, dtype=np.float32)
    smax = np.float32(p.sigma_max) ** inv_rho
    smin = np.float32(p.sigma_min) ** inv_rho
    sig = (smax + steps / np.float32(num_steps - 1) * (smin - smax)) ** np.float32(p.rho)
    sig = (sig * np.float32(p.sigma_data)).astype(np.float32)
    return np.concatenate([sig, np.zeros(1, np.float32)])


@mx.compile
def _frozen_pre_step(x, q, tr, eps, eps_scale):
    R = random_rotation(q)
    x = x - x.mean(axis=-2, keepdims=True)
    x = mx.einsum("bmd,bds->bms", x, R) + tr
    return x + eps_scale * eps


@mx.compile
def _frozen_post_step(x_noisy, x_denoised, t_hat, coef):
    return x_noisy + coef * ((x_noisy - x_denoised) / t_hat)


def frozen_sample(denoise, atom_mask, num_steps, multiplicity, noise, align, p=DiffusionParams(), eval_every=1):
    atom_mask = mx.repeat(atom_mask, multiplicity, axis=0)
    shape = (*atom_mask.shape, 3)
    sigmas = _frozen_schedule(p, num_steps)
    gammas = np.where(sigmas > p.gamma_min, np.float32(p.gamma_0), np.float32(0.0)).astype(np.float32)

    x = float(sigmas[0]) * noise.randn(shape)
    for step in range(num_steps):
        sigma_tm, sigma_t, gamma = float(sigmas[step]), float(sigmas[step + 1]), float(gammas[step + 1])
        q = noise.randn((multiplicity, 4))
        tr = noise.randn((multiplicity, 1, 3))
        t_hat = sigma_tm * (1 + gamma)
        noise_var = p.noise_scale**2 * (t_hat**2 - sigma_tm**2)
        eps = noise.randn(shape)
        x_noisy = _frozen_pre_step(x, q, tr, eps, mx.array(sqrt(noise_var), mx.float32))

        x_denoised = denoise(x_noisy, t_hat)

        if p.alignment_reverse_diff and align is not None:
            x_noisy = align(x_noisy, x_denoised, atom_mask, atom_mask)

        x = _frozen_post_step(x_noisy, x_denoised, mx.array(t_hat, mx.float32),
                              mx.array(p.step_scale * (sigma_t - t_hat), mx.float32))
        if eval_every and (step + 1) % eval_every == 0:
            mx.eval(x)
    return x


# ------------------------------------------------------------------------------------------------------------ toys

N_ATOMS = 24


def _target(seed=1, n=N_ATOMS):
    return mx.array(np.random.default_rng(seed).normal(size=(1, n, 3)).astype(np.float32) * 5)


def stub_denoiser(target):
    """Nonlinear, non-equivariant synthetic denoiser with Boltz's EDM preconditioning (sigma_data 16)."""
    sd = 16.0

    def denoise(x, t_hat):
        c_skip = sd**2 / (t_hat**2 + sd**2)
        c_out = t_hat * sd / sqrt(sd**2 + t_hat**2)
        c_in = 1 / sqrt(t_hat**2 + sd**2)
        times = np.log(t_hat / sd) * 0.25
        r = mx.tanh(0.3 * (c_in * x) + target) - 0.1 * (c_in * x) * times
        return c_skip * x + c_out * r

    return denoise


def _mask(n=N_ATOMS, pad=3):
    m = np.ones((1, n), np.float32)
    m[0, n - pad:] = 0
    return mx.array(m)


class RecordingNoise(NoiseSource):
    def __init__(self, inner):
        self.inner, self.shapes = inner, []

    def randn(self, shape):
        self.shapes.append(tuple(shape))
        return self.inner.randn(shape)


class DeterministicNoise(NoiseSource):
    """Random init, then identity rotations, zero translations, zero eps: the sampler reduces to its ODE solver."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.first = True

    def randn(self, shape):
        if self.first:
            self.first = False
            return mx.array(self.rng.normal(size=shape).astype(np.float32))
        if shape[-1] == 4:
            q = np.zeros(shape, np.float32)
            q[..., 0] = 1.0
            return mx.array(q)
        return mx.zeros(shape, mx.float32)


# ---------------------------------------------------------------------------------------- 1. defaults bit-identical


@pytest.mark.parametrize("p", [DiffusionParams(), DiffusionParams(rho=5.0, step_scale=1.25), DiffusionParams(rho=9.0, step_scale=1.75)],
                         ids=["default", "rho5_ss1.25", "rho9_ss1.75"])
def test_euler_bit_identical_to_round1_sampler(p):
    den, mask = stub_denoiser(_target()), _mask()
    for steps, mult in ((12, 2), (5, 1)):
        ref = frozen_sample(den, mask, steps, mult, record_noise(0, "rec", "structure", mult), weighted_rigid_align, p)
        new = sample(den, mask, steps, mult, record_noise(0, "rec", "structure", mult), weighted_rigid_align, p)
        assert np.array_equal(np.array(ref), np.array(new))


def test_default_args_unchanged():
    d = DiffusionParams()
    assert (d.gamma_0, d.gamma_min, d.noise_scale, d.rho, d.step_scale, d.sigma_min, d.sigma_max, d.sigma_data) == \
        (0.8, 1.0, 1.003, 7, 1.5, 0.0001, 160.0, 16.0)
    assert d.solver == "euler" and d.heun_euler_first == 0 and d.alignment_reverse_diff
    pa = PredictArgs()
    assert (pa.recycling_steps, pa.sampling_steps, pa.diffusion_samples, pa.diffusion) == (3, 200, 1, DiffusionParams())
    assert SamplerConfig().params() == DiffusionParams() and SamplerConfig().tag() == ""
    assert PredictArgs.from_sampler(None, 3, 1) == PredictArgs()
    assert PredictArgs.from_sampler(SAMPLER_PRESETS["s200"].structure, 3, 1) == PredictArgs()
    assert PredictArgs.from_sampler(SAMPLER_PRESETS["aff200"].affinity, 5, 5) == PredictArgs(5, 200, 5)
    # Euler presets at the upstream schedule are plain step-count changes
    assert PredictArgs.from_sampler(SAMPLER_PRESETS["s25"].structure, 3, 1) == PredictArgs(3, 25, 1)


def test_unknown_solver_rejected():
    with pytest.raises(ValueError):
        sample(stub_denoiser(_target()), _mask(), 3, 1, record_noise(0, "r", "structure", 1), None, DiffusionParams(solver="rk4"))


# -------------------------------------------------------------------------------------------- 2. Heun order on a toy

S_DATA = 16.0


def gaussian_denoiser(rotation=None):
    """Exact denoiser for x0 ~ N(0, S_DATA^2 I): D(x, sigma) = S^2/(S^2+sigma^2) x; optionally rotated by a fixed R0
    (a non-equivariant network output that the rigid alignment must absorb)."""

    def denoise(x, sigma):
        d = (S_DATA**2 / (S_DATA**2 + sigma**2)) * x
        return d if rotation is None else mx.einsum("bnd,ed->bne", d, rotation)

    return denoise


def _ode_run(steps, solver, step_scale=1.0, align=None, rotation=None, heun_euler_first=0):
    p = DiffusionParams(gamma_0=0.0, step_scale=step_scale, solver=solver, heun_euler_first=heun_euler_first)
    mask = mx.ones((1, 16))
    x = sample(gaussian_denoiser(rotation), mask, steps, 1, DeterministicNoise(0), align, p)
    init = np.random.default_rng(0).normal(size=(1, 16, 3)).astype(np.float64) * float(_frozen_schedule(p, steps)[0])
    init -= init.mean(axis=-2, keepdims=True)  # the first step centres x
    sig0 = float(_frozen_schedule(p, steps)[0])
    # x(sigma) solves dx/dsigma = s (x - D)/sigma = s sigma/(S^2+sigma^2) x  ->  x(0) = x(sig0) (S^2/(S^2+sig0^2))^(s/2)
    exact = init * (S_DATA**2 / (S_DATA**2 + sig0**2)) ** (step_scale / 2)
    return np.array(x).astype(np.float64), exact


def _rel_err(steps, solver, **kw):
    x, exact = _ode_run(steps, solver, **kw)
    return float(np.abs(x - exact).max() / np.abs(exact).max())


def test_heun_second_order_euler_first_order():
    ns = (8, 16, 32, 64)
    e_eu = [_rel_err(n, "euler") for n in ns]
    e_he = [_rel_err(n, "heun") for n in ns]
    order_eu = [np.log2(a / b) for a, b in zip(e_eu, e_eu[1:])]
    order_he = [np.log2(a / b) for a, b in zip(e_he, e_he[1:])]
    print("euler errors", e_eu, "orders", order_eu)
    print("heun  errors", e_he, "orders", order_he)
    # measured (float32, rho 7, sigma 2560 -> 0.0016 -> 0): Euler 0.19 / 0.097 / 0.049 (order 0.97); Heun 0.091 / 0.020 / 0.0046 (2.1-2.2)
    assert all(0.7 < o < 1.4 for o in order_eu[1:])
    assert all(1.7 < o < 2.6 for o in order_he[1:])
    assert e_he[-1] < 0.15 * e_eu[-1]
    # at equal NFE (Heun N steps = 2N - 1 NFE) Heun is still more accurate on this smooth problem
    assert _rel_err(32, "heun") < _rel_err(63, "euler") / 2


def test_heun_with_step_scale_is_consistent_and_stable():
    """step_scale 1.5: Heun (s on the averaged PF-ODE slope) is first-order consistent with Boltz's s-scaled ODE, like Euler,
    and does not blow up at 6 steps (Heun on the s-scaled ODE itself would; see sampler._sample_heun)."""
    ns = (8, 16, 32, 64, 128)
    e_he = [_rel_err(n, "heun", step_scale=1.5) for n in ns]
    e_eu = [_rel_err(n, "euler", step_scale=1.5) for n in ns]
    assert all(b < a for a, b in zip(e_he, e_he[1:]))
    orders = [np.log2(a / b) for a, b in zip(e_he[2:], e_he[3:])]
    assert all(0.6 < o < 1.5 for o in orders)
    assert all(h <= e for h, e in zip(e_he[1:], e_eu[1:]))
    x6, exact = _ode_run(6, "heun", step_scale=1.5)
    assert np.isfinite(x6).all() and np.abs(x6).max() < 3 * np.abs(exact).max()


def test_heun_alignment_is_frame_consistent():
    """A denoiser whose output is rotated by R0: with the rigid alignment, each step's result is R0 times the unrotated one
    (x_hat aligned onto D1; D2 aligned onto x_e). After N steps the sample is R0^N @ (sample of the equivariant denoiser)."""
    th = 0.3
    R0 = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]], np.float32)
    for solver, steps in (("heun", 6), ("euler", 6)):
        x_rot, _ = _ode_run(steps, solver, step_scale=1.5, align=weighted_rigid_align, rotation=mx.array(R0))
        x_ref, _ = _ode_run(steps, solver, step_scale=1.5, align=None)
        RN = np.linalg.matrix_power(R0.astype(np.float64), steps)
        np.testing.assert_allclose(x_rot, x_ref @ RN.T, rtol=0, atol=2e-4 * np.abs(x_ref).max())


def test_heun_euler_first_and_nfe_count():
    calls = []
    den = stub_denoiser(_target())

    def counting(x, t):
        calls.append(t)
        return den(x, t)

    for steps, ef, smin, want in ((6, 0, 0.0, 11), (6, 1, 0.0, 10), (7, 1, 0.0, 12), (2, 0, 0.0, 3),
                                  (7, 0, 16.0, 10), (8, 1, 16.0, 11), (9, 1, 16.0, 12)):
        calls.clear()
        p = DiffusionParams(solver="heun", step_scale=1.0, heun_euler_first=ef, heun_sigma_min=smin)
        sample(counting, _mask(), steps, 1, record_noise(0, "r", "structure", 1), weighted_rigid_align, p)
        assert len(calls) == want == nfe(steps, p)
        assert calls[-1] > 0  # never evaluated at sigma = 0
    assert nfe(25, DiffusionParams()) == 25
    # exact evaluation sequence of the sigma_min family: 8 steps, first Euler, correct steps ending >= 16 A -> steps 1-3
    calls.clear()
    p = DiffusionParams(solver="heun", step_scale=1.0, heun_euler_first=1, heun_sigma_min=16.0)
    sample(counting, _mask(), 8, 1, record_noise(0, "r", "structure", 1), None, p)
    sig = _frozen_schedule(p, 8)
    want = []
    for i in range(8):
        want.append(float(sig[i]) * (1 + (0.8 if sig[i + 1] > 1.0 else 0.0)))  # t_hat
        if i in (1, 2, 3):
            want.append(float(sig[i + 1]))                                      # corrector at sigma_t (346, 98, 21 A)
    np.testing.assert_allclose(calls, want, rtol=1e-6)
    assert [round(float(sig[i + 1])) for i in (1, 2, 3)] == [346, 98, 21]


# ------------------------------------------------------------------------------------------------- 3. noise / seeding


def test_heun_consumes_exactly_eulers_noise():
    den, mask = stub_denoiser(_target()), _mask()
    shapes = {}
    for solver in ("euler", "heun"):
        rn = RecordingNoise(record_noise(0, "rec", "structure", 2))
        sample(den, mask, 6, 2, rn, weighted_rigid_align, DiffusionParams(solver=solver))
        shapes[solver] = rn.shapes
    assert shapes["euler"] == shapes["heun"] and len(shapes["heun"]) == 1 + 3 * 6


def test_heun_per_sample_noise_independent_of_sample_count():
    den, mask = stub_denoiser(_target()), _mask()
    p = DiffusionParams(solver="heun")
    one = np.array(sample(den, mask, 6, 1, record_noise(3, "rec", "structure", 1), None, p))
    three = np.array(sample(den, mask, 6, 3, record_noise(3, "rec", "structure", 3), None, p))
    np.testing.assert_array_equal(one[0], three[0])
    assert not np.array_equal(three[0], three[1])


@pytest.mark.parametrize("solver", ["euler", "heun"])
def test_records_independent_of_input_order_and_resume(solver):
    p = DiffusionParams(solver=solver)
    targets = {"rec_a": _target(1), "rec_b": _target(2), "rec_c": _target(3)}

    def run(order):
        out = {}
        for rid in order:  # one noise stream per record, as in cli.predict / run_l3
            out[rid] = np.array(sample(stub_denoiser(targets[rid]), _mask(), 6, 1, record_noise(0, rid, "structure", 1),
                                       weighted_rigid_align, p))
        return out

    fwd, rev, resume = run(["rec_a", "rec_b", "rec_c"]), run(["rec_c", "rec_b", "rec_a"]), run(["rec_c"])
    for rid in targets:
        np.testing.assert_array_equal(fwd[rid], rev[rid])
    np.testing.assert_array_equal(fwd["rec_c"], resume["rec_c"])
    again = run(["rec_a"])
    np.testing.assert_array_equal(fwd["rec_a"], again["rec_a"])


# ------------------------------------------------------------------------------------------------------- 4. presets


def test_prereg_f_candidates():
    P = SAMPLER_PRESETS
    for name, steps in (("s25", 25), ("s20", 20), ("s10", 10), ("s50", 50), ("s200", 200)):
        c = P[name].structure
        assert (c.steps, c.rho, c.step_scale, c.solver, c.nfe, c.tag()) == (steps, 7, 1.5, "euler", steps, "")
        assert P[name].affinity is None
    grid = {n: p.structure for n, p in P.items() if p.candidate == "F4"}
    assert len(grid) == 9
    assert {(c.rho, c.step_scale) for c in grid.values()} == {(r, s) for r in (5.0, 7.0, 9.0) for s in (1.25, 1.5, 1.75)}
    assert all(c.steps == 25 and c.solver == "euler" for c in grid.values())
    assert P["s25_rho5_ss1.25"].structure.tag() == "rho5_ss1.25" and P["s25_rho7_ss1.5"].structure.tag() == ""
    heun = {n: p.structure for n, p in P.items() if p.candidate == "F5"}
    assert {n: (c.steps, c.heun_euler_first, c.heun_sigma_min, c.nfe) for n, c in heun.items()} == {
        "heun10": (6, 1, 0.0, 10), "heun11": (6, 0, 0.0, 11), "heun12": (7, 1, 0.0, 12),
        "heun10_n7": (7, 0, 16.0, 10), "heun11_n8": (8, 1, 16.0, 11), "heun12_n9": (9, 1, 16.0, 12)}
    # EDM step size 1.0 (Boltz's 1.5 diverges at <= 8 steps), Boltz churn and schedule
    assert all(c.solver == "heun" and c.step_scale == 1.0 and c.rho == 7 and 10 <= c.nfe <= 12 for c in heun.values())
    assert heun["heun11"].tag() == "heun_ss1" and heun["heun10"].tag() == "heunef1_ss1"
    assert heun["heun11_n8"].tag() == "heunef1_smin16_ss1"
    assert len({c.tag() + str(c.steps) for c in heun.values()}) == len(heun)
    for n in (100, 50, 25):
        pr = P[f"aff{n}"]
        assert pr.structure is None and pr.affinity == SamplerConfig(n) and pr.candidate == "F6"
    assert {p.candidate for p in P.values()} - {"reference", "F (screen)"} == {"F1", "F2", "F3", "F4", "F5", "F6"}


def test_resolve_and_compose():
    pr = resolve_sampler("s50+aff25")
    assert pr.name == "s50+aff25" and pr.structure.steps == 50 and pr.affinity.steps == 25
    assert resolve_sampler("heun11") is SAMPLER_PRESETS["heun11"]
    for bad in ("nope", "s25+s50", "aff25+aff50", "", "+"):
        with pytest.raises(ValueError):
            resolve_sampler(bad)
    assert affinity_config("aff50") == SamplerConfig(50) and affinity_config("s25") == SamplerConfig(25)
    assert affinity_config("heun11").solver == "heun"
    d = SAMPLER_PRESETS["heun11"].structure.describe()
    assert d == {"steps": 6, "rho": 7.0, "step_scale": 1.0, "solver": "heun", "heun_euler_first": 0, "heun_sigma_min": 0.0,
                 "nfe": 11, "gamma_0": 0.8, "gamma_min": 1.0, "noise_scale": 1.003}
    p = SAMPLER_PRESETS["s25_rho9_ss1.75"].structure.params()
    assert (p.rho, p.step_scale, p.gamma_0, p.noise_scale, p.solver) == (9.0, 1.75, 0.8, 1.003, "euler")


# ---------------------------------------------------------------------------------------------------- 5. CLI flags


def _cli(*flags):
    from boltz_mlx.cli import parse_args, resolve_samplers

    args = parse_args(["predict", "x.yaml", *flags])
    return args, resolve_samplers(args)


def test_cli_defaults_and_turbo():
    args, (st, af) = _cli()
    assert st == SamplerConfig(200) and af == SamplerConfig(200)
    assert PredictArgs.from_sampler(st, args.recycling_steps, args.diffusion_samples) == PredictArgs()
    _, (st, af) = _cli("--turbo")
    assert st == SamplerConfig(50) and af == SamplerConfig(200)  # turbo lowers the structure pass only
    _, (st, af) = _cli("--turbo", "--sampling_steps_affinity", "50")
    assert (st.steps, af.steps) == (50, 50)
    _, (st, af) = _cli("--sampling_steps", "100", "--step_scale", "1.25")
    assert st == SamplerConfig(100, step_scale=1.25) and af == SamplerConfig(200, step_scale=1.25)  # upstream: both passes


def test_cli_sampler_presets():
    _, (st, af) = _cli("--sampler", "heun11")
    assert st == SAMPLER_PRESETS["heun11"].structure and af == SamplerConfig(200)
    _, (st, af) = _cli("--sampler", "s50+aff25")
    assert (st.steps, af.steps) == (50, 25)
    _, (st, af) = _cli("--sampler_affinity", "aff50", "--turbo")
    assert (st.steps, af.steps) == (50, 50)
    _, (st, af) = _cli("--sampler_affinity", "s25_rho5_ss1.25")
    assert st == SamplerConfig(200) and af == SamplerConfig(25, rho=5.0, step_scale=1.25)
    _, (st, af) = _cli("--sampler", "s25", "--sampling_steps", "25")  # agreeing explicit value is fine
    assert st.steps == 25
    for bad in (("--sampler", "s25", "--sampling_steps", "50"), ("--sampler", "s50+aff25", "--sampler_affinity", "aff50"),
                ("--sampler", "aff25", "--sampling_steps_affinity", "50"), ("--sampler", "nope")):
        with pytest.raises(SystemExit):
            _cli(*bad)


def test_cli_precision_flag_parses():
    args, _ = _cli("--precision", "f16")
    assert args.precision == "f16"
    args, _ = _cli()
    assert args.precision is None


# --------------------------------------------------------------------------------------------------- 6. run_l3 wiring

sys.path.insert(0, str(ROOT / "tools" / "validation"))
sys.path.insert(0, str(ROOT / "tools"))


def test_run_l3_sampler_resolution_and_labels():
    import run_l3 as r

    st, af = r.resolve_run_samplers(None, None, None)
    assert (st, af) == (SamplerConfig(200), SamplerConfig(200))
    st, af = r.resolve_run_samplers("s25", None, 200)
    assert (st.steps, af.steps) == (25, 200)
    st, af = r.resolve_run_samplers("aff50", 50, None)
    assert (st.steps, af.steps) == (50, 50)
    st, af = r.resolve_run_samplers("heun11", None, None)
    assert st.solver == "heun" and st.steps == 6
    with pytest.raises(ValueError):
        r.resolve_run_samplers("s25", 200, None)
    with pytest.raises(ValueError):
        r.resolve_run_samplers("aff25", None, 200)
    assert r.mode_label("f16", "s25") == "f16+s25" and r.mode_label("strict", None) == "strict"
    assert r.run_name("strict", 200) == "strict_s200"  # unchanged for the upstream sampler: existing work dirs stay valid
    assert r.run_name("f16", 25, "", "rho5_ss1.25") == "f16_s25_rho5_ss1.25"
    assert r.run_name("strict", 6, "x", "heun") == "strict_s6_heun_x"
    assert r.aff_dir(Path("/w"), 50) == Path("/w/aff_a50") and r.aff_dir(Path("/w"), 6, "heun") == Path("/w/aff_a6_heun")
    assert "heun11" in r.sampler_table() and "F5" in r.sampler_table()


def test_run_l3_cli_parses_sampler(tmp_path, capsys):
    import subprocess

    root = Path(__file__).resolve().parents[1]
    if subprocess.run(["git", "-C", str(root), "rev-parse", "--verify", "-q", "HEAD:boltz_mlx"],
                      capture_output=True).returncode != 0:
        pytest.skip("run_l3 records the git tree of boltz_mlx: needs a git checkout with a commit")
    import run_l3 as r

    assert r.main(["--list-samplers"]) == 0
    assert "s25_rho9_ss1.75" in capsys.readouterr().out
    rc = r.main(["--set", "parity", "--cases", "prot117", "--sampler", "heun11", "--mode", "f16", "--dry-run",
                 "--out", str(tmp_path / "rows.jsonl")])
    out = capsys.readouterr().out
    assert rc == 0 and "rows: mode=f16+heun11" in out and "steps=6 (NFE 11)" in out and "todo: prot117" in out
    with pytest.raises(SystemExit):
        r.main(["--set", "parity", "--sampler", "s25", "--steps", "200", "--dry-run", "--out", str(tmp_path / "x.jsonl")])


def _sinfo(tmp_path, case_id, **kw):
    import json

    cdir = tmp_path / "seed0" / case_id
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / f"{case_id}_model_0.cif").write_text("data_x\n")
    (cdir / f"pre_affinity_{case_id}.npz").write_bytes(b"x")
    d = dict(case=case_id, seed=0, mode="strict", steps=200, recycling=3, device="gpu", env={}, pkg_tree="abc1234", git="deadbee",
             n_tokens=250, ptm=0.8, iptm=0.7, ligand_iptm=0.65, complex_plddt=0.85, nan=False, inf=False, guard_counts=None,
             seconds=12.5, model_cif=str(cdir / f"{case_id}_model_0.cif"), pre_affinity=str(cdir / f"pre_affinity_{case_id}.npz"),
             date="2026-10-28 10:00:00")
    d.update(kw)
    (cdir / "l3_struct.json").write_text(json.dumps(d))
    return d


def test_run_l3_rows_record_sampler_and_keys_differ(tmp_path):
    import gates
    import run_l3 as r

    pc = r.Case(id="9ssd", kind="protein", target="9ssd", exp_cif=None, n_res=494)
    lc = r.Case(id="tyk2_ejm_31", kind="ligand", target="tyk2", smiles="C", exp_value=-9.1, ligand_chain="L", has_affinity=True)
    old = _sinfo(tmp_path, "9ssd")                                       # sidecar written before sampler options
    heun = _sinfo(tmp_path, "9ssd_h", steps=6, sampler="heun11", sampler_tag="heun",
                  sampler_cfg=SAMPLER_PRESETS["heun11"].structure.describe())
    row_old = r.build_protein_row(pc, 0, "strict", 200, old)
    row_h = r.build_protein_row(pc, 0, "strict", 6, heun, sampler="heun11")
    assert row_old["mode"] == "strict" and row_old["sampler"] is None and row_old["sampler_structure"]["nfe"] == 200
    assert row_old["sampler_structure"]["solver"] == "euler"
    assert row_h["mode"] == "strict+heun11" and row_h["precision_mode"] == "strict" and row_h["sampler"] == "heun11"
    assert row_h["sampler_structure"]["nfe"] == 11
    assert r.row_key(row_h) == r.row_key_for(pc, 0, r.mode_label("strict", "heun11"), 6, 200)
    # the same steps under different presets never share a resume key
    a = r.build_protein_row(pc, 0, "strict", 25, _sinfo(tmp_path, "a", steps=25), sampler="s25")
    b = r.build_protein_row(pc, 0, "strict", 25, _sinfo(tmp_path, "b", steps=25), sampler="s25_rho5_ss1.25")
    assert r.row_key(a) != r.row_key(b)
    # gates pick the sampler arm by its label even in a mixed file
    assert gates.select([row_old, row_h], "strict+heun11", 6) == [row_h] and gates.select([row_old, row_h], "strict", 200) == [row_old]
    # affinity-only preset: structure as strict@200, affinity sampler recorded
    ainfo = dict(case=lc.id, seed=0, mode="strict", steps=200, affinity_steps=50, recycling=5, pkg_tree="abc1234",
                 affinity_pred_value=0.5, affinity_probability_binary=0.8, mw=300.0, value_mw_corrected=0.1, sampler="aff50",
                 sampler_tag="", sampler_cfg=SamplerConfig(50).describe(), seconds=3.0)
    lrow = r.build_ligand_row(lc, 0, "strict", 200, 50, _sinfo(tmp_path, lc.id), ainfo, sampler="aff50")
    assert lrow["mode"] == "strict+aff50" and lrow["affinity_steps"] == 50 and lrow["sampler_affinity"]["steps"] == 50
    assert lrow["sampler_structure"]["steps"] == 200
    assert r.row_key(lrow) == r.row_key_for(lc, 0, "strict+aff50", 200, 50)


def test_run_l3_struct_reuse_respects_sampler_tag(tmp_path):
    import run_l3 as r

    ident = {"pkg_tree": "abc1234", "git": "deadbee"}
    case = r.Case(id="x", kind="protein", target="x")
    old = _sinfo(tmp_path, "x", steps=25)
    assert r.struct_reusable(old, case, "strict", 25, 3, ident)               # pre-option sidecar = upstream sampler
    assert not r.struct_reusable(old, case, "strict", 25, 3, ident, "rho5_ss1.25")
    tagged = _sinfo(tmp_path, "y", steps=25, sampler_tag="rho5_ss1.25")
    assert r.struct_reusable(tagged, case, "strict", 25, 3, ident, "rho5_ss1.25")
    assert not r.struct_reusable(tagged, case, "strict", 25, 3, ident)


def test_run_l3_collect_rows_with_sampler(tmp_path):
    import json

    import run_l3 as r

    run_root = tmp_path / "strict_s200"
    sdir = r.seed_dir(run_root, 0)
    lc = r.Case(id="tyk2_ejm_31", kind="ligand", target="tyk2", smiles="C", exp_value=-9.1, ligand_chain="L", has_affinity=True)
    cdir = sdir / lc.id
    cdir.mkdir(parents=True)
    r.struct_sidecar(sdir, lc.id).write_text(json.dumps({"case": lc.id, "seed": 0, "mode": "strict", "steps": 200, "recycling": 3,
                                                         "pkg_tree": "abc", "model_cif": str(cdir / "m.cif")}))
    ap = r.aff_sidecar(sdir, lc.id, 25)
    ap.parent.mkdir(parents=True)
    ap.write_text(json.dumps({"affinity_pred_value": 0.1, "affinity_probability_binary": 0.5, "value_mw_corrected": 0.0,
                              "sampler_cfg": SamplerConfig(25).describe()}))
    got = list(r.collect_rows([lc], [0], "strict", 200, 25, run_root, sampler="aff25"))
    row = got[0][0]
    assert row["mode"] == "strict+aff25" and row["sampler_affinity"]["steps"] == 25 and row["affinity_steps"] == 25
    assert list(r.collect_rows([lc], [0], "strict", 200, 25, run_root, sampler="x", aff_tag="heun"))[0][3] == "no affinity sidecar"
