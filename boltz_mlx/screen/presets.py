"""Screen presets: numerics environment (applied by re-exec before MLX initialises, as tools/validation/run_l3.py MODES)
plus pass settings. Recycles and the 5 affinity samples are never reduced.

Time estimates are ESTIMATES derived from bench rows (warm model-predict times, median of runs 2..N), not measurements
of a screen run. strict / turbo / f16: M5 Pro 24 GB rows of 2026-10-27 (development bench rows). screen: NAX-build
rows on the M5 Pro (R7-combined, the preset's est_* fields) and on the M5 Max (r2-speed-R8p, EST_CHIP_INPUTS; the "bring-up" rows). Those rows were measured at 50 structure / 200 affinity steps (est_steps /
est_affinity_steps); other step counts are scaled (struct_estimate / affinity_estimate). They do not transfer to other
chips: the dry-run plan says "not measured on this chip" anywhere else (estimate_measured_here), and on GPUs without NAX
the f16 presets map to their portable variants (for_device).

The default preset depends on the kernel path (default_preset; project decision, 2026-10-29): "screen" on M5-class
GPUs (NAX path), "strict" on Macs without NAX (M1-M4), where strict reproduces the M5 results to rung-1 tolerance
(cross-chip check PASS) and the f16 path does not (FAIL). The f16 presets stay available there by name (--preset screen
runs as screen-tensorops / screen-portable, not validated).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

# Default on the NAX path (M5-class GPUs): the screen preset (f16 + 20 + 25 steps), which passed sampler confirmation
# attempt 1 under PREREG-B and the rung 5 screen gates on the M5 Max. Default without NAX (M1-M4): strict.
# Tiny-ligand rule (boltz_mlx/tiny_ligand.py): Fast runs 200 structure steps for a ligand with < 10 heavy atoms when the steps are
# not set (a 4-atom ligand, BCT / 8gpp, collapsed in every 20-step run, not at 50 or 200); f16 + 200 + 25 passed PREREG-B 8/8.
DEFAULT_PRESET = "screen"
DEFAULT_PRESET_NON_NAX = "strict"


@dataclass(frozen=True)
class Preset:
    name: str
    env: dict = field(default_factory=dict)  # numerics switches (BOLTZ_MLX_* / MLX_ENABLE_TF32)
    steps: int = 200                        # structure-pass sampling steps
    recycling: int = 3                      # structure-pass recycles
    affinity_steps: int = 200
    affinity_recycling: int = 5
    affinity_samples: int = 5
    validated: bool = False                 # full gate set passed
    note: str = ""
    # estimate inputs (seconds, M5 Pro): structure predict at 117 / 400 tokens with est_steps (default `steps`), affinity
    # pass at ~213 crop tokens with est_affinity_steps (default `affinity_steps`)
    est_struct_117: float = 0.0
    est_struct_400: float = 0.0
    est_affinity: float = 0.0
    est_steps: int | None = None            # structure steps the est_* inputs were measured at (None: `steps`)
    est_affinity_steps: int | None = None   # affinity steps the est_affinity input was measured at (None: `affinity_steps`)


PRESETS: dict[str, Preset] = {
    "strict": Preset(
        "strict", {}, 200, 3, 200, 5, 5, validated=True,
        note="fp32 everywhere; the reference build (fixed-noise parity with upstream)",
        est_struct_117=4.65, est_struct_400=30.28, est_affinity=28.06),
    "turbo": Preset(
        "turbo", {"BOLTZ_MLX_DIFF_DTYPE": "fp16", "BOLTZ_MLX_TF32": "1"}, 50, 3, 200, 5, 5, validated=False,
        note="round-1 turbo (TF32 + fp16 diffusion, 50 structure steps); validated on single chains only, NOT on ligands/affinity",
        est_struct_117=1.82, est_struct_400=18.47, est_affinity=18.28),
    "f16": Preset(
        "f16", {"BOLTZ_MLX_PRECISION": "f16"}, 200, 3, 200, 5, 5, validated=False,
        note="fp16 GEMMs (rung 1, BOLTZ_MLX_PRECISION=f16), 200 steps; L3 validation pending",
        # estimate: f16 @50 (1.50 / 14.17 s) + 150 extra steps at the strict per-step diffusion cost (~0.016 / 0.065 s)
        est_struct_117=1.50 + 150 * 0.016, est_struct_400=14.17 + 150 * 0.065, est_affinity=20.0),
    # The stage C composition f16 + s20 + aff25: f16 numerics, 20-step Euler
    # structure pass, 25-step affinity pass; 3 recycles and the 5 affinity samples unchanged. Sampler confirmation attempt
    # 1 under PREREG-B (docs/report.md section 7): PASS, 8 of 8 gated gates against strict on the confirmation set
    # (56 chains, 107 ligands; Apple M5 Max, f16 fingerprint 8497b1df3c77).
    # L3L.chirality and A3.kendall are untestable on this data: reported only and not blocking (project decision);
    # L3L.chirality reads 10 candidate vs 3 strict stereocentre flips (excess 7), A3.kendall tau +0.884 (PASS). Rung 5
    # screen gates G1-G6 on the M5 Max: benchmark jobs not included in this repository. MEASURED T400 rate on the M5
    # Max: full cold 199-ligand screens 6.32-7.21 s/ligand, pooled 6.835 over 4 passes (rung 5 G1, AC power; G3 peak
    # footprint 10.46 GB); 5.38-5.39 s/ligand on the 5-ligand --dry-run --measure 5 (job boltz-val-t400-stageBC-20261029a).
    # Until 2026-10-29 this preset was f16 + 50 + 200 steps (provisional, never gated).
    # estimate inputs: M5 Pro bench rows of the R7-combined build (prot117 / prot400 f16 @50, lig_aff --affinity f16
    # @200: 0.843 / 6.128 / 7.634 s; benchmark speed summary), scaled to 20 / 25 steps by struct_estimate /
    # affinity_estimate. The old inputs (1.50 / 14.17 / 20.0, pre-NAX rows of 2026-10-27) gave ~36 s/ligand for T400 where
    # the 50 + 200 screen measured 15.5 s on the M5 Pro and 8.0 s on the M5 Max (--dry-run --measure 5).
    "screen": Preset(
        "screen", {"BOLTZ_MLX_PRECISION": "f16"}, 20, 3, 25, 5, 5, validated=True,
        note="f16 + 20-step structure pass + 25-step affinity pass; passed PREREG-B sampler confirmation attempt 1 (8/8 "
             "gated gates vs strict) and the rung 5 screen gates on the Apple M5 Max (NAX path); L3L.chirality and "
             "A3.kendall untestable on this data, reported only",
        est_struct_117=0.843, est_struct_400=6.128, est_affinity=7.634, est_steps=50, est_affinity_steps=200),
}

# The chip the est_* inputs were measured on (bench rows with the NAX kernels on). Anything else - another chip without
# an EST_CHIP_INPUTS row, or the portable f16 path on this one - gets estimates labelled "not measured on this chip".
ESTIMATE_CHIP = "Apple M5 Pro"
# (preset, chip) -> this chip's estimate points (NAX path only), each: structure seconds at 117 / 400 tokens with `steps`
# structure steps, affinity-pass seconds with `affinity_steps`, and the per-ligand `overhead` added on top. A run's
# structure estimate starts from the point nearest its structure steps, the affinity estimate from the point nearest its
# affinity steps (the step-scaling rules are rough far from the measured counts: at T400 on the M5 Max 20 vs 50 structure
# steps measured 3.25 vs 3.47 s, the rule says 0.76x).
# screen @20 / 25 on the M5 Max: recalibrated from MEASURED s20 / aff25 rows, not scaled from the 50 / 200-step bench rows
# (scaled, those gave 4.9 s/ligand for T400 where full cold screens take 6.7-7.1 s). Rows: G1 passes 1 and 2 (job boltz-val-g1-t400-20261029a, class 1,
# alone on the GPU, AC power, boltz_mlx tree 35b4ba8): two full cold T400 screens, 2 x 199 ligands (406-421 tokens),
# f16 + 20 + 25 steps, chunk 50. Means over the 398 rows: seconds_structure normalised to 400 tokens with the bench power
# law (exponent log(3.095 / 0.557) / log(400 / 117) = 1.395) 4.25 s; seconds_affinity 2.40 s; per-ligand wall minus
# structure + affinity 0.098 / 0.101 s (startup, both checkpoint loads per chunk, writing: prefetch hides featurization)
# -> overhead 0.08 s (+ the plan's 2 x 0.6 s loads per chunk = 0.024 s/ligand at chunk 50). 117-token point: the 400
# point scaled by the bench power law (ESTIMATE: no measured s20 row at 117 tokens alone). Fixed before G1 passes 3-4:
# target, the plan's T400 s/ligand within +-10 % of the G1 pooled cold rate (tests/test_screen.py). That test is NOT
# independent: passes 1-2 are both the calibration rows and half of the pooled rate. Result (MEASURED rates): plan 6.90
# s/ligand vs pooled 4 passes 6.835 (+0.9 %, in-sample); vs passes 3-4 alone 6.765 (+2.0 %, out-of-sample: 1434.2 +
# 1258.2 s for 2 x 199 ligands).
# screen @50 / 200 on the M5 Max (the earlier screen settings): bench rows, bench rows "bring-up" (prot117 0.557,
# prot400 3.095 @50, lig_aff --affinity @200 3.784 s; build r2-speed-R8p) + the generic 1.0 s overhead; this reproduced
# the 5-ligand --measure (8.0 s/ligand), no full cold screen was measured at these settings.
EST_CHIP_INPUTS: dict = {
    ("screen", "Apple M5 Max"): [
        {"struct_117": 4.25 * (117 / 400) ** 1.395, "struct_400": 4.25, "affinity": 2.40, "steps": 20,
         "affinity_steps": 25, "overhead": 0.08,
         "source": "screen rows, G1 passes 1-2 (boltz-val-g1-t400-20261029a: full cold T400 screens)"},
        {"struct_117": 0.557, "struct_400": 3.095, "affinity": 3.784, "steps": 50, "affinity_steps": 200,
         "overhead": 1.0, "source": "bench rows of the model passes (bring-up, r2-speed-R8p)"},
    ],
}


PATH_NOTES = {"portable": "FALLBACK f16 path (internal, unvalidated: no MPP kernels; see python -m boltz_mlx.capability)",
              "tensorops": "M1-M4 f16 path (the MPP kernels as Metal 4 tensor ops on the shader cores, no NAX)"}


def default_preset(path: str | None = None) -> str:
    """The default preset for a kernel path (kernels.kernel_path() when None): DEFAULT_PRESET on the NAX path (M5-class
    GPUs), DEFAULT_PRESET_NON_NAX (strict) on tensorops / portable (M1-M4, or BOLTZ_MLX_KERNEL_PATH / BOLTZ_MLX_NAX=0)."""
    if path is None:
        from boltz_mlx import kernels as K

        path = K.kernel_path()
    return DEFAULT_PRESET if path == "nax" else DEFAULT_PRESET_NON_NAX


def for_device(p: Preset, path: str) -> Preset:
    """Device-aware preset: off the NAX kernel path (kernels.kernel_path(): M1-M4, or BOLTZ_MLX_KERNEL_PATH) a preset
    running --precision f16 runs another code path, so it gets a distinct name - 'f16' -> 'f16-portable' /
    'f16-tensorops', 'screen' -> 'screen-portable' / ... - which goes into every results row, the config key (no mixing
    of paths in one out dir) and the plan. Same environment otherwise; never a gated preset."""
    if path == "nax" or p.env.get("BOLTZ_MLX_PRECISION") != "f16" or p.name.endswith(("-portable", "-tensorops")):
        return p
    return replace(p, name=f"{p.name}-{path}", validated=False, note=p.note + "; " + PATH_NOTES[path])


def estimate_measured_here(p: Preset, chip: str) -> bool:
    """Were p's time estimates measured on this chip with this code path?"""
    return (chip == ESTIMATE_CHIP or (p.name, chip) in EST_CHIP_INPUTS) and not p.name.endswith(("-portable", "-tensorops"))


def est_point(p: Preset, chip: str | None = None, steps: int | None = None, affinity_steps: int | None = None,
              by: str = "both") -> dict:
    """The estimate point for p: this chip's (EST_CHIP_INPUTS, NAX path only) nearest the given steps (default the
    preset's; log distance in structure steps (by="steps"), affinity steps (by="affinity") or both, structure first),
    else the preset's own (ESTIMATE_CHIP bench rows). Keys: struct_117 / struct_400 / affinity seconds, the steps /
    affinity_steps they hold for, per-ligand overhead."""
    import math

    if chip is not None and (p.name, chip) in EST_CHIP_INPUTS:
        s, a = steps or p.steps, affinity_steps or p.affinity_steps
        pts = EST_CHIP_INPUTS[(p.name, chip)]
        pts = [pts] if isinstance(pts, dict) else pts

        def dist(q):
            ds, da = abs(math.log(s / q["steps"])), abs(math.log(a / q["affinity_steps"]))
            return {"steps": (ds, da), "affinity": (da, ds)}.get(by, (ds + da, ds))

        return dict(min(pts, key=dist))
    return {"struct_117": p.est_struct_117, "struct_400": p.est_struct_400, "affinity": p.est_affinity,
            "steps": p.est_steps or p.steps, "affinity_steps": p.est_affinity_steps or p.affinity_steps,
            "overhead": EST_OVERHEAD_S}


def est_inputs(p: Preset, chip: str | None = None) -> tuple:
    """(structure s at 117 tokens, at 400 tokens, affinity pass s) for this chip at the preset's steps (est_point)."""
    e = est_point(p, chip)
    return e["struct_117"], e["struct_400"], e["affinity"]


def est_overhead(p: Preset, chip: str | None = None, steps: int | None = None, affinity_steps: int | None = None) -> float:
    """Per-ligand seconds the plan adds on top of the model passes (est_point nearest these steps)."""
    return est_point(p, chip, steps, affinity_steps)["overhead"]


# per-ligand overhead outside model predict (featurization, RDKit conformer, writing) and per-checkpoint load: estimates
# from the T400 --measure runs (screen preset): per-ligand structure + affinity seconds minus the bench model times above
# ~0.9-1.0 s on both chips; checkpoint loads conf 0.57-0.59 s, aff 0.33-0.34 s (M5 Pro R7, M5 Max)
EST_OVERHEAD_S = 1.0
EST_LOAD_S = 0.6


def struct_estimate(p: Preset, n_tokens: int, steps: int | None = None, chip: str | None = None,
                    affinity_steps: int | None = None) -> float:
    """Power law through the 117- and 400-token points; `steps` (default: the preset's) other than the step count the
    inputs were measured at (est_steps, default the preset's) scale the diffusion share (~40%). `affinity_steps` (the run's,
    default the preset's) only breaks ties between estimate points with the same structure steps."""
    import math

    pt = est_point(p, chip, steps, affinity_steps, by="steps")
    a, b = pt["struct_117"], pt["struct_400"]
    e = math.log(b / a) / math.log(400 / 117)
    t = b * (max(n_tokens, 16) / 400.0) ** e
    ref, steps = pt["steps"], p.steps if steps is None else steps
    if steps != ref:
        t = 0.6 * t + 0.4 * t * steps / ref
    return t


def affinity_estimate(p: Preset, steps: int | None = None, chip: str | None = None,
                      structure_steps: int | None = None) -> float:
    pt = est_point(p, chip, structure_steps, steps, by="affinity")
    t = pt["affinity"]  # crop is <= 256 tokens: roughly independent of the target size
    ref, steps = pt["affinity_steps"], p.affinity_steps if steps is None else steps
    if steps != ref:
        t = 0.3 * t + 0.7 * t * steps / ref  # lig_aff: diffusion ~70% of the affinity pass (measured)
    return t
