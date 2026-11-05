"""Boltz-2 reverse-diffusion sampler (AtomDiffusion.sample) with replayable noise.

Mirrors boltz/model/modules/diffusionv2.py::AtomDiffusion.sample for inference with
fk_steering / physical guidance off. Contact/template guidance potentials are skipped:
they only act when the input carries contact constraints or templates (asserted by the
caller via `has_guidance_targets`).

Noise draws happen in exactly the PyTorch order so a recorded stream can be replayed:
  init:      randn(M, N, 3)                (times sigma_0)
  each step: randn(M, 4)   quaternions      (random rotation)
             randn(M, 1, 3) translation
             randn(M, N, 3) eps

Solvers (DiffusionParams.solver; round-2 sampler candidates, PREREG F):
  euler (default, = upstream Boltz-2): one network evaluation (NFE) per step. The default path is the
        upstream loop unchanged (bit-identical to every earlier build).
  heun  2nd-order (EDM Algorithm 2, Karras et al. 2022) with Boltz's churn kept as is. See `_sample_heun`.
`rho` and `step_scale` (DiffusionParams) set the Karras schedule exponent and the step multiplier for both solvers.
Named presets for the pre-registered candidates: SAMPLER_PRESETS / resolve_sampler.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from math import sqrt
from pathlib import Path
from typing import Callable

import mlx.core as mx
import numpy as np

# Pipelined steps (R6-prof; numerics-neutral): step i is submitted with mx.async_eval and the host waits for step i-1
# only after that, so the GPU runs step i while Python builds step i+1 (noise draws, augmentation, the compiled
# network's graph replay, alignment). With a blocking mx.eval per step the GPU idles for that host time every step.
# At most two steps are in flight; same graphs, kernels and order -> bitwise-identical coordinates.
PIPELINE = os.environ.get("BOLTZ_MLX_PIPE_SAMPLER", "1") == "1"


class _StepPipe:
    """Evaluates sampler steps: blocking (PIPELINE off) or one step ahead (async_eval, then wait for the previous)."""

    def __init__(self, on: bool | None = None):
        self.on = PIPELINE if on is None else on
        self.prev = None

    def submit(self, x: mx.array) -> None:
        if not self.on:
            mx.eval(x)
            return
        mx.async_eval(x)
        if self.prev is not None:
            mx.eval(self.prev)  # already finished or queued just ahead of x: bounds the queue to two steps
        self.prev = x


@dataclass
class DiffusionParams:  # boltz.main.Boltz2DiffusionParams
    gamma_0: float = 0.8
    gamma_min: float = 1.0
    noise_scale: float = 1.003
    rho: float = 7
    step_scale: float = 1.5
    sigma_min: float = 0.0001
    sigma_max: float = 160.0
    sigma_data: float = 16.0
    alignment_reverse_diff: bool = True
    # boltz-mlx sampler options (not upstream fields; defaults = upstream behaviour)
    solver: str = "euler"        # euler | heun
    heun_euler_first: int = 0    # heun: the first k steps are plain Euler steps (NFE knob; see _sample_heun)
    heun_sigma_min: float = 0.0  # heun: correct only steps ending at sigma >= this (A); 0 = EDM (all but the last)


class NoiseSource:
    def randn(self, shape) -> mx.array:  # pragma: no cover - interface
        raise NotImplementedError


class MLXNoise(NoiseSource):
    def __init__(self, seed: int = 0):
        self.key = mx.random.key(seed)

    def randn(self, shape):
        self.key, sub = mx.random.split(self.key)
        return mx.random.normal(shape, key=sub)


class ReplayNoise(NoiseSource):
    """Replays torch.randn draws recorded by tools/make_reference.py, in order."""

    def __init__(self, path: Path | str):
        z = np.load(path)
        self.draws = [z[k] for k in sorted(z.files)]
        self.i = 0

    def randn(self, shape):
        d = self.draws[self.i]
        if tuple(d.shape) != tuple(shape):
            raise ValueError(f"noise draw {self.i}: recorded {d.shape} != requested {tuple(shape)}")
        self.i += 1
        return mx.array(d.astype(np.float32))


def sample_schedule(p: DiffusionParams, num_steps: int) -> np.ndarray:
    """Karras schedule computed in float32 like torch; final sigma 0 appended."""
    inv_rho = np.float32(1.0 / p.rho)
    steps = np.arange(num_steps, dtype=np.float32)
    smax = np.float32(p.sigma_max) ** inv_rho
    smin = np.float32(p.sigma_min) ** inv_rho
    sig = (smax + steps / np.float32(num_steps - 1) * (smin - smax)) ** np.float32(p.rho)
    sig = (sig * np.float32(p.sigma_data)).astype(np.float32)
    return np.concatenate([sig, np.zeros(1, np.float32)])


def random_rotation(o: mx.array) -> mx.array:
    """(n,4) normals -> (n,3,3) rotations; same math as boltz random_quaternions + quaternion_to_matrix."""
    s = (o * o).sum(1)
    norm = mx.sqrt(s)
    norm = mx.where((norm < 0) != (o[:, 0] < 0), -norm, norm)  # _copysign(sqrt(s), o[:,0])
    q = o / norm[:, None]
    r, i, j, k = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    two_s = 2.0 / (q * q).sum(-1)
    m = mx.stack(
        [
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ],
        -1,
    )
    return m.reshape(q.shape[0], 3, 3)


def sample(
    denoise: Callable[[mx.array, float], mx.array],
    atom_mask: mx.array,
    num_steps: int,
    multiplicity: int,
    noise: NoiseSource,
    align: Callable | None,
    p: DiffusionParams = DiffusionParams(),
    eval_every: int = 1,
) -> mx.array:
    """Run the reverse diffusion.

    denoise(x_noisy (M,N,3), t_hat float) -> x_denoised (M,N,3): the preconditioned network.
    align(noisy, denoised, weights, mask) -> noisy aligned onto denoised (weighted_rigid_align).
    atom_mask: (B, N) float; repeated to multiplicity.
    """
    if p.solver == "heun":
        return _sample_heun(denoise, atom_mask, num_steps, multiplicity, noise, align, p, eval_every)
    if p.solver != "euler":
        raise ValueError(f"unknown solver {p.solver!r} (euler | heun)")
    atom_mask = mx.repeat(atom_mask, multiplicity, axis=0)
    shape = (*atom_mask.shape, 3)
    sigmas = sample_schedule(p, num_steps)
    gammas = np.where(sigmas > p.gamma_min, np.float32(p.gamma_0), np.float32(0.0)).astype(np.float32)

    x = float(sigmas[0]) * noise.randn(shape)
    pipe = _StepPipe()
    for step in range(num_steps):
        sigma_tm, sigma_t, gamma = float(sigmas[step]), float(sigmas[step + 1]), float(gammas[step + 1])
        q = noise.randn((multiplicity, 4))
        tr = noise.randn((multiplicity, 1, 3))
        t_hat = sigma_tm * (1 + gamma)
        noise_var = p.noise_scale**2 * (t_hat**2 - sigma_tm**2)
        eps = noise.randn(shape)
        x_noisy = _pre_step(x, q, tr, eps, mx.array(sqrt(noise_var), mx.float32))

        x_denoised = denoise(x_noisy, t_hat)

        if p.alignment_reverse_diff and align is not None:
            x_noisy = align(x_noisy, x_denoised, atom_mask, atom_mask)

        x = _post_step(x_noisy, x_denoised, mx.array(t_hat, mx.float32), mx.array(p.step_scale * (sigma_t - t_hat), mx.float32))
        if eval_every and (step + 1) % eval_every == 0:
            pipe.submit(x)
    if eval_every:
        mx.eval(x)
    return x


@mx.compile
def _pre_step(x, q, tr, eps, eps_scale):
    R = random_rotation(q)
    x = x - x.mean(axis=-2, keepdims=True)
    x = mx.einsum("bmd,bds->bms", x, R) + tr
    return x + eps_scale * eps


@mx.compile
def _post_step(x_noisy, x_denoised, t_hat, coef):
    return x_noisy + coef * ((x_noisy - x_denoised) / t_hat)


def heun_corrected(step: int, sigma_t: float, p: DiffusionParams) -> bool:
    """Does Heun step `step` (ending at sigma_t) get a corrector evaluation? Never on the last step (sigma_t = 0: the
    corrector slope (x - D)/sigma is undefined; EDM Algorithm 2 takes an Euler step there too), nor on the first
    `heun_euler_first` steps, nor on steps ending below `heun_sigma_min` (A)."""
    return sigma_t > 0 and step >= p.heun_euler_first and sigma_t >= p.heun_sigma_min


def nfe(num_steps: int, p: DiffusionParams = DiffusionParams()) -> int:
    """Network evaluations of one sampler run (per sample): Euler N; Heun N + number of corrected steps."""
    if p.solver == "euler":
        return int(num_steps)
    if p.solver == "heun":
        sig = sample_schedule(p, num_steps)
        return int(num_steps) + sum(heun_corrected(i, float(sig[i + 1]), p) for i in range(num_steps))
    raise ValueError(f"unknown solver {p.solver!r}")


def _sample_heun(denoise, atom_mask, num_steps, multiplicity, noise, align, p, eval_every):
    """2nd-order Heun sampler with Boltz's stochastic churn, augmentation and rigid alignment (EDM Algorithm 2).

    Per step i (sigma_tm = sigmas[i] -> sigma_t = sigmas[i+1], h = sigma_t - t_hat, s = step_scale), identical to the
    Boltz Euler step up to the slope:
      1. random augmentation: centre x, rotate by R(q), translate by tr;          noise draws q, tr (as Euler)
      2. churn: t_hat = sigma_tm (1 + gamma), gamma = gamma_0 if sigma_t > gamma_min else 0 (Boltz indexes gamma by
         the NEXT sigma), x_hat = x + noise_scale * sqrt(t_hat^2 - sigma_tm^2) * eps;  noise draw eps (as Euler)
      3. D1 = denoise(x_hat, t_hat); x_hat <- rigid-align(x_hat onto D1)  (as Euler);  d1 = (x_hat - D1) / t_hat
      4. if heun_corrected(i, sigma_t): (sigma_t > 0, i >= heun_euler_first, sigma_t >= heun_sigma_min)
           predictor (probability-flow ODE Euler):  x_e = x_hat + h d1
           D2 = denoise(x_e, sigma_t); D2 <- rigid-align(D2 onto x_e); d2 = (x_e - D2) / sigma_t
           x_{i+1} = x_hat + s h (d1 + d2) / 2
         else (the last step to sigma 0, or a step excluded by the two knobs): the Boltz Euler step x_hat + s h d1.
    Semantics:
      - Noise: the corrector draws NO noise. Draws per step are exactly Euler's (q, tr, eps), so a Heun run with N
        steps consumes the same per-sample stream as an Euler run with N steps (SeededNoise: one key per sample, so
        per-record determinism across input order / resume / sample count is unchanged).
      - Churn (gamma_0, gamma_min, noise_scale) is Boltz's, unchanged: Heun integrates from t_hat to sigma_t, as in EDM.
        The corrector evaluation is not separately augmented or noised (it is the 2nd stage of the same ODE step).
        Note Boltz's churn is per step (gamma_0 = 0.8 whatever N is; EDM scales S_churn / N), so Boltz's sampler has no
        N -> infinity limit: every step count samples a slightly different distribution (Euler or Heun alike).
      - step_scale s: Boltz's update is x_hat + s h (slope). Heun keeps that update and replaces the slope d1 by the
        2nd-order (trapezoidal) slope (d1 + d2)/2 of the probability-flow ODE dx/dsigma = (x - D)/sigma. With s = 1 this
        is exactly EDM Heun (2nd order; tests/test_sampler_options.py checks order 2 vs 1 on a Gaussian toy). With s != 1
        it is first-order consistent with Boltz's s-scaled ODE, and at sigma >> sigma_data (D ~ 0, d2 = d1) it reduces
        to the Boltz Euler step exactly. Rejected alternative (measured on the Gaussian toy, gamma/noise as Boltz,
        s = 1.5): Heun on the s-scaled ODE dx/dsigma = s (x - D)/sigma (s on both slopes, predictor x_hat + s h d1) is
        unstable at the large sigma ratios of a 6-7 step Karras schedule (sample std x1748 / x708 of the data std at
        6 / 7 steps, vs 0.97 / 1.9 for this form and 0.68 for Euler at 200 steps).
      - Rigid alignment: as in Boltz, x_hat is aligned onto D1 before d1. For d2, D2 is aligned onto x_e (not x_e onto
        D2): this removes the same rigid-body disagreement (the residual x - D is the same up to a rotation) while
        keeping both slopes and the base point x_hat in one frame, so their average is frame-consistent.
      - NFE = N + corrected steps (nfe()). EDM Heun (all but the last step): 6 steps -> 11. heun_euler_first = 1 skips the
        first correction, which costs little: at sigma_0 ~ 2560-4600 A, D << x and d2 = d1 to O(sigma_data^2/sigma^2).
        heun_sigma_min (A) restricts the corrector to steps ending at sigma >= heun_sigma_min, so more (smaller) steps fit
        the NFE budget: 8 steps, first Euler, heun_sigma_min = sigma_data (16 A) -> corrected steps 1-3 (ending at 346,
        98, 21 A) -> 11 NFE.
      - step_scale: with s = 1.5 (Boltz's value, tuned for 200 Euler steps) every step leaves residual noise
        |1 - s (1 - r)| t_hat instead of r t_hat (r = sigma_t / t_hat), e.g. 0.41 t_hat vs 0.06 t_hat at a 6-step
        schedule's middle step, so the network sees inputs far noisier than the sigma it is given. CPU diagnostic
        (prot117, 0 recycles, 2 seeds; engineering check, not a gate): 6 Euler steps at s = 1.5 give Rg 99 A (Euler@200:
        13.3 A), EDM Heun 6 steps at s = 1.5 Rg 222 A. The Heun presets therefore use EDM's step s = 1.0.
    """
    atom_mask = mx.repeat(atom_mask, multiplicity, axis=0)
    shape = (*atom_mask.shape, 3)
    sigmas = sample_schedule(p, num_steps)
    gammas = np.where(sigmas > p.gamma_min, np.float32(p.gamma_0), np.float32(0.0)).astype(np.float32)
    use_align = p.alignment_reverse_diff and align is not None

    x = float(sigmas[0]) * noise.randn(shape)
    pipe = _StepPipe()
    for step in range(num_steps):
        sigma_tm, sigma_t, gamma = float(sigmas[step]), float(sigmas[step + 1]), float(gammas[step + 1])
        q = noise.randn((multiplicity, 4))
        tr = noise.randn((multiplicity, 1, 3))
        t_hat = sigma_tm * (1 + gamma)
        noise_var = p.noise_scale**2 * (t_hat**2 - sigma_tm**2)
        eps = noise.randn(shape)
        x_noisy = _pre_step(x, q, tr, eps, mx.array(sqrt(noise_var), mx.float32))

        d1 = denoise(x_noisy, t_hat)
        if use_align:
            x_noisy = align(x_noisy, d1, atom_mask, atom_mask)
        coef = mx.array(p.step_scale * (sigma_t - t_hat), mx.float32)
        t_hat_a = mx.array(t_hat, mx.float32)
        if heun_corrected(step, sigma_t, p):
            x_e =_post_step(x_noisy, d1, t_hat_a, mx.array(sigma_t - t_hat, mx.float32))  # PF-ODE Euler predictor
            d2 = denoise(x_e, sigma_t)
            if use_align:
                d2 = align(d2, x_e, atom_mask, atom_mask)
            x = _heun_step(x_noisy, d1, x_e, d2, t_hat_a, mx.array(sigma_t, mx.float32), coef)
        else:
            x = _post_step(x_noisy, d1, t_hat_a, coef)  # the Boltz Euler step
        if eval_every and (step + 1) % eval_every == 0:
            pipe.submit(x)
    if eval_every:
        mx.eval(x)
    return x


@mx.compile
def _heun_step(x_noisy, d1, x_pred, d2, t_hat, sigma_t, coef):
    return x_noisy + (0.5 * coef) * ((x_noisy - d1) / t_hat + (x_pred - d2) / sigma_t)


def _det3(m: mx.array) -> mx.array:
    return (
        m[..., 0, 0] * (m[..., 1, 1] * m[..., 2, 2] - m[..., 1, 2] * m[..., 2, 1])
        - m[..., 0, 1] * (m[..., 1, 0] * m[..., 2, 2] - m[..., 1, 2] * m[..., 2, 0])
        + m[..., 0, 2] * (m[..., 1, 0] * m[..., 2, 1] - m[..., 1, 1] * m[..., 2, 0])
    )


def weighted_rigid_align(true_coords, pred_coords, weights, mask):
    """boltz/model/loss/diffusionv2.py::weighted_rigid_align (inference). Aligns true onto pred.

    The 3x3 SVD runs on the CPU stream (mx.linalg.svd is CPU-only), inside the lazy graph.
    """
    w = (mask * weights)[..., None]
    wsum = w.sum(axis=-2, keepdims=True)
    true_c = (true_coords * w).sum(axis=-2, keepdims=True) / wsum
    pred_c = (pred_coords * w).sum(axis=-2, keepdims=True) / wsum
    tc = true_coords - true_c
    pc = pred_coords - pred_c
    cov = mx.einsum("bni,bnj->bij", w * pc, tc)
    U, _, Vh = mx.linalg.svd(cov, stream=mx.cpu)
    rot = U @ Vh
    d = _det3(rot)
    F = mx.concatenate([mx.ones((*d.shape, 2)), d[..., None]], axis=-1)
    rot = (U * F[..., None, :]) @ Vh
    return mx.einsum("bni,bji->bnj", tc, rot) + pred_c


class SeededNoise(NoiseSource):
    """Per-sample noise streams: sample k draws from its own key (seeds[k]), so a sample's noise does not
    depend on how many samples run together, the input order, or resume."""

    def __init__(self, seeds: list[int]):
        self.keys = [mx.random.key(int(s)) for s in seeds]

    def randn(self, shape):
        if shape[0] != len(self.keys):
            raise ValueError(f"noise for {shape[0]} samples requested, {len(self.keys)} seeds")
        draws = []
        for k in range(len(self.keys)):
            self.keys[k], sub = mx.random.split(self.keys[k])
            draws.append(mx.random.normal(tuple(shape[1:]), key=sub))
        return mx.stack(draws, axis=0)


def record_noise(global_seed: int, record_id: str, pass_: str, n_samples: int) -> SeededNoise:
    from boltz_mlx.seeding import record_seed

    return SeededNoise([record_seed(global_seed, record_id, pass_, k) for k in range(n_samples)])


# ---------------------------------------------------------------------------------------------------- sampler presets


@dataclass(frozen=True)
class SamplerConfig:
    """One diffusion pass's sampler: steps + Karras rho + step_scale + solver. Defaults = upstream Boltz-2 (200 steps)."""

    steps: int = 200
    rho: float = DiffusionParams.rho
    step_scale: float = DiffusionParams.step_scale
    solver: str = "euler"
    heun_euler_first: int = 0
    heun_sigma_min: float = 0.0

    def params(self, base: DiffusionParams | None = None) -> DiffusionParams:
        """DiffusionParams for this config (churn gamma_0 / gamma_min / noise_scale and sigmas from `base`, default upstream)."""
        return replace(base or DiffusionParams(), rho=self.rho, step_scale=self.step_scale, solver=self.solver,
                       heun_euler_first=self.heun_euler_first, heun_sigma_min=self.heun_sigma_min)

    @property
    def nfe(self) -> int:
        return nfe(self.steps, self.params())

    def tag(self) -> str:
        """Identity of everything except the step count ('' = upstream schedule and Euler): run dirs / resume keys."""
        parts = []
        if self.solver != "euler":
            parts.append(self.solver + (f"ef{self.heun_euler_first}" if self.heun_euler_first else "")
                         + (f"_smin{float(self.heun_sigma_min):g}" if self.heun_sigma_min else ""))
        if float(self.rho) != float(DiffusionParams.rho):
            parts.append(f"rho{float(self.rho):g}")
        if float(self.step_scale) != float(DiffusionParams.step_scale):
            parts.append(f"ss{float(self.step_scale):g}")
        return "_".join(parts)

    def describe(self) -> dict:
        d = DiffusionParams()
        return {"steps": int(self.steps), "rho": float(self.rho), "step_scale": float(self.step_scale), "solver": self.solver,
                "heun_euler_first": int(self.heun_euler_first), "heun_sigma_min": float(self.heun_sigma_min), "nfe": self.nfe,
                "gamma_0": d.gamma_0, "gamma_min": d.gamma_min, "noise_scale": d.noise_scale}


@dataclass(frozen=True)
class SamplerPreset:
    """A named sampler candidate: a structure-pass config and/or an affinity-pass config (None = that pass unchanged)."""

    name: str
    structure: SamplerConfig | None = None
    affinity: SamplerConfig | None = None
    candidate: str = ""  # PREREG F item
    note: str = ""


def _build_presets() -> dict[str, SamplerPreset]:
    P: dict[str, SamplerPreset] = {}

    def add(name, structure=None, affinity=None, candidate="", note=""):
        P[name] = SamplerPreset(name, structure, affinity, candidate, note)

    add("s200", SamplerConfig(200), candidate="reference", note="upstream default (identity); the strict reference arm")
    add("s50", SamplerConfig(50), candidate="F (screen)", note="screen preset's structure pass, confirmed as a sampler change")
    add("s25", SamplerConfig(25), candidate="F1", note="current Karras schedule, 25 steps")
    add("s20", SamplerConfig(20), candidate="F2", note="current schedule, 20 steps")
    add("s10", SamplerConfig(10), candidate="F3", note="current schedule, 10 steps")
    for rho in (5, 7, 9):  # F4 grid, fixed before tuning (PREREG F4)
        for ss in (1.25, 1.5, 1.75):
            add(f"s25_rho{rho}_ss{ss:g}", SamplerConfig(25, rho=float(rho), step_scale=ss), candidate="F4",
                note="25 steps, rho / step_scale grid point (retuned on T)")
    # F5: Heun at 10-12 NFE, EDM step (step_scale 1.0; Boltz's 1.5 diverges at <= 8 steps, see _sample_heun), Boltz churn,
    # rho 7. Two families, tuned on T: EDM (corrector on every step but the last) and sigma_min = sigma_data (corrector only
    # on steps ending at sigma >= 16 A, so 7-9 smaller steps fit the budget).
    sd = DiffusionParams.sigma_data
    heun = dict(solver="heun", step_scale=1.0)
    add("heun11", SamplerConfig(6, **heun), candidate="F5",
        note="EDM Heun (Alg. 2): 6 steps, corrector on steps 0-4, Euler last step to sigma 0: 11 NFE")
    add("heun10", SamplerConfig(6, heun_euler_first=1, **heun), candidate="F5",
        note="EDM Heun, 6 steps, first and last steps Euler: 10 NFE")
    add("heun12", SamplerConfig(7, heun_euler_first=1, **heun), candidate="F5",
        note="EDM Heun, 7 steps, first and last steps Euler: 12 NFE")
    add("heun10_n7", SamplerConfig(7, heun_sigma_min=sd, **heun), candidate="F5",
        note="Heun, 7 steps, corrector on steps ending >= 16 A (0-2): 10 NFE")
    add("heun11_n8", SamplerConfig(8, heun_euler_first=1, heun_sigma_min=sd, **heun), candidate="F5",
        note="Heun, 8 steps, corrector on steps 1-3 (ending >= 16 A, first step Euler): 11 NFE")
    add("heun12_n9", SamplerConfig(9, heun_euler_first=1, heun_sigma_min=sd, **heun), candidate="F5",
        note="Heun, 9 steps, corrector on steps 1-3 (ending >= 16 A, first step Euler): 12 NFE")
    for n in (200, 100, 50, 25):
        add(f"aff{n}", affinity=SamplerConfig(n), candidate="reference" if n == 200 else "F6",
            note=f"affinity pass {n} steps x 5 samples, structure pass unchanged")
    return P


SAMPLER_PRESETS: dict[str, SamplerPreset] = _build_presets()


def resolve_sampler(name: str) -> SamplerPreset:
    """Preset by name; '+' composes a structure preset with an affinity preset (e.g. 's50+aff25')."""
    parts = [x.strip() for x in str(name).split("+") if x.strip()]
    if not parts:
        raise ValueError("empty sampler preset name")
    structure = affinity = None
    cands, notes = [], []
    for part in parts:
        if part not in SAMPLER_PRESETS:
            raise ValueError(f"unknown sampler preset {part!r}; known: {sorted(SAMPLER_PRESETS)} (compose with '+')")
        pr = SAMPLER_PRESETS[part]
        if pr.structure is not None:
            if structure is not None:
                raise ValueError(f"sampler {name!r}: two structure-pass presets")
            structure = pr.structure
        if pr.affinity is not None:
            if affinity is not None:
                raise ValueError(f"sampler {name!r}: two affinity-pass presets")
            affinity = pr.affinity
        cands.append(pr.candidate)
        notes.append(pr.note)
    if len(parts) == 1:
        return SAMPLER_PRESETS[parts[0]]
    return SamplerPreset("+".join(parts), structure, affinity, "+".join(cands), "; ".join(notes))


def affinity_config(name: str) -> SamplerConfig:
    """Config for the affinity pass from a preset name: its affinity part, else its structure part ('s25' == 'aff25')."""
    pr = resolve_sampler(name)
    cfg = pr.affinity or pr.structure
    if cfg is None:  # pragma: no cover - every preset has one part
        raise ValueError(f"sampler {name!r} has no config")
    return cfg
