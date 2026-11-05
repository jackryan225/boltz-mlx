"""Time estimates per Apple chip, for the GUI's plan panel, preset cards and run ETAs.

Every number here is a bench measurement on one chip (warm model-predict times, median of runs 2..N), never a guess for
another chip: a chip without an entry gets "no estimate yet for this chip" instead of a number that would be wrong.

  struct_117 / struct_400   structure predict (trunk + diffusion + confidence) at 117 / 400 tokens, with `steps` steps
  affinity                  the affinity pass (crop <= 256 tokens, 5 samples) with `affinity_steps` steps
  overhead                  per-ligand work outside model predict (featurization, RDKit conformer, writing)
  load                      one checkpoint load (a screen loads conf + aff once per chunk; predict loads each once)

Steps other than the entry's are scaled with the screen's own rules (boltz_mlx.screen.presets.struct_estimate /
affinity_estimate). The chip name is what `sysctl machdep.cpu.brand_string` prints (e.g. "Apple M5 Pro");
BOLTZ_MLX_GUI_CHIP overrides it (tests, or to borrow another chip's table knowingly).
"""

from __future__ import annotations

import functools
import os
import subprocess
from dataclasses import replace

# f16 numerics (BOLTZ_MLX_PRECISION=f16, NAX kernels), build r2-speed-R8p, M5 Pro 24 GB (
# bench rows "R7-combined"): prot117 0.843 s and prot400 ~6.0 s at 50 structure steps, lig_aff affinity pass 7.63 s at
# 200 steps; T400 (lig_aff receptor + 249-row MSA, 199 ligands, screen preset) 15.5 s per ligand.
_M5PRO_F16 = {"struct_117": 0.843, "struct_400": 6.0, "steps": 50, "affinity": 7.63, "affinity_steps": 200,
              "source": "M5 Pro 24 GB, f16 build r2-speed-R8p (bench rows, 2026-10-28)"}

CHIP_TABLE: dict[str, dict] = {
    "Apple M5 Pro": {
        "overhead": 1.5,
        "load": 6.0,
        "presets": {
            # fp32 / TF32 rows were not re-measured after the round-2 speed work: older builds, likely pessimistic
            "strict": {"struct_117": 4.65, "struct_400": 30.28, "steps": 200, "affinity": 28.06, "affinity_steps": 200,
                       "source": "M5 Pro 24 GB, fp32 reference build (bench rows of 2026-10-27; not re-measured since)"},
            "turbo": {"struct_117": 1.82, "struct_400": 18.47, "steps": 50, "affinity": 18.28, "affinity_steps": 200,
                      "source": "M5 Pro 24 GB, round-1 turbo (bench rows of 2026-10-27; not re-measured since)"},
            "f16": dict(_M5PRO_F16),
            "screen": dict(_M5PRO_F16),
        },
        # whole-ligand screen measurements, shown next to the estimate as a sanity reference (the earlier screen
        # settings, f16 + 50 + 200 steps; the current screen preset, f16 + 20 + 25, is not measured on the M5 Pro yet)
        "reference": {"t400_s_per_ligand": 15.5, "preset": "screen", "steps": 50, "affinity_steps": 200,
                      "what": "T400: lig_aff receptor, 199 ligands, earlier screen settings f16 + 50 + 200 steps (3-ligand measure)"},
    },
    # TODO: fill from the M5 Max benchmark rows; until then every
    # preset shows "no estimate yet for this chip".
    "Apple M5 Max": {"overhead": 1.5, "load": 6.0, "presets": {}},
}

NO_ESTIMATE = "no estimate yet for this chip"

# Predict-only rows for chips whose CHIP_TABLE entry is empty (the Screen estimate keeps its own measured per-ligand times there).
# Same shape as a CHIP_TABLE preset row, plus "load" (seconds per checkpoint). The M5 Max's screen row is the screen's own G1 point
# (boltz_mlx.screen.presets.EST_CHIP_INPUTS); "strict" comes from the P38 trial (job boltz-val-strict-trial-20261029a, 3
# ligands in one process, ~360-residue kinase + ligand): structure 14.92 s and affinity pass 14.77 s per ligand (median per pass,
# compute only), at 200 steps / 200 affinity steps; 94.4 s wall - 89.1 s compute = ~5 s for the one model load and the featurization.
# The 117-token point is extrapolated with the exponent of the G1 rows (1.395), as the G1 entry does.
def _m5max_rows() -> dict:
    from boltz_mlx.screen.presets import EST_CHIP_INPUTS

    g1 = dict(EST_CHIP_INPUTS[("screen", "Apple M5 Max")][0])
    g1["load"] = 0.6
    strict = {"struct_117": round(14.92 * (117 / 390) ** 1.395, 3), "struct_400": 14.92, "steps": 200, "affinity": 14.77,
              "affinity_steps": 200, "overhead": 1.5, "load": 2.5,
              "source": "M5 Max, strict trial of the P38 kinase (3 ligands, one process; boltz-val-strict-trial-20261029a)"}
    return {"screen": g1, "strict": strict}


def predict_entry(preset_name: str, chip: str | None = None) -> dict | None:
    """The bench row a structure job's estimate uses: this chip's CHIP_TABLE row, else a predict-only row (M5 Max), else None."""
    chip = chip or chip_name()
    e = entry(preset_name, chip)
    if e is not None:
        return e
    if chip == "Apple M5 Max" and "-" not in preset_name:
        return _m5max_rows().get(preset_name)
    return None


# Peak memory of one job (MLX peak, bench rows): f16 prot117 4.95 GB and prot400 8.55 GB (@50 steps, round 2); fp32 prot400 9.13 GB
# and the 964-token ex_pocket at 14.8 GB process footprint (1000 tokens: 16 GB MLX peak before the row-chunking, 12.3 GB after).
# A line through each pair of points, continued beyond them (the pair tensors grow with the square, but the row-chunked kernels
# keep the curve near linear in this range); a rough guide, not a limit.
MEM_POINTS = {"f16": [(117, 4.95), (400, 8.55)], "fp32": [(400, 9.13), (964, 14.8)]}
MEM_BASIS = "Measured peak-memory rows (f16: 117 and 400 tokens; fp32: 400 and 964 tokens), extended linearly"


@functools.lru_cache(maxsize=1)
def ram_gb() -> float | None:
    try:
        r = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5)
        return round(int(r.stdout.strip()) / 2**30) if r.stdout.strip() else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def predict_memory_gb(n_tokens: int, precision: str = "fp32") -> dict:
    """ESTIMATE of the peak memory of one structure job: {"peak_gb", "ram_gb", "basis"} (the line through the two MEM_POINTS of the precision)."""
    pts = MEM_POINTS["f16" if precision == "f16" else "fp32"]
    n = max(int(n_tokens), 1)
    (x0, y0), (x1, y1) = pts
    peak = max(y0 + (y1 - y0) * (n - x0) / (x1 - x0), 2.6)   # never below the weights (2.6 GB)
    return {"peak_gb": round(peak, 1), "ram_gb": ram_gb(), "basis": MEM_BASIS}


@functools.lru_cache(maxsize=1)
def _sysctl_chip() -> str:
    try:
        r = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=5)
        return r.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def chip_name() -> str:
    """This Mac's chip ("Apple M5 Max"); BOLTZ_MLX_GUI_CHIP overrides it."""
    return os.environ.get("BOLTZ_MLX_GUI_CHIP") or _sysctl_chip()


def entry(preset_name: str, chip: str | None = None) -> dict | None:
    """The bench entry for a preset on a chip, or None (no numbers for this chip / preset)."""
    chip = chip or chip_name()
    base = preset_name.split("-")[0]  # f16-portable etc. run another code path: no numbers for those
    if base != preset_name:
        return None
    return (CHIP_TABLE.get(chip) or {}).get("presets", {}).get(base)


def chip_constants(chip: str | None = None) -> dict:
    from boltz_mlx.screen.presets import EST_LOAD_S, EST_OVERHEAD_S

    c = CHIP_TABLE.get(chip or chip_name()) or {}
    return {"overhead": c.get("overhead", EST_OVERHEAD_S), "load": c.get("load", EST_LOAD_S)}


def reference_preset(p, chip: str | None = None):
    """A copy of preset `p` whose est_* inputs (and the step counts they were measured at) are this chip's bench numbers,
    for the screen's estimate functions only (never used to run anything); None when the chip has no entry. Call
    struct_estimate / affinity_estimate with explicit steps so a run's own step counts are scaled from the entry's."""
    e = entry(p.name, chip)
    if not e:
        return None
    return replace(p, est_struct_117=e["struct_117"], est_struct_400=e["struct_400"], est_affinity=e["affinity"],
                   steps=e["steps"], affinity_steps=e["affinity_steps"], est_steps=e["steps"],
                   est_affinity_steps=e["affinity_steps"])


def predict_seconds(p, n_tokens: int, *, steps: int, affinity: bool, affinity_steps: int, samples: int = 1,
                    chip: str | None = None) -> dict | None:
    """ESTIMATE for one `boltz-mlx predict` job: {"struct_s", "affinity_s", "load_s", "total_s"}, or None (no entry)."""
    from boltz_mlx.screen.presets import affinity_estimate, struct_estimate

    ref = reference_preset(p, chip)
    k = chip_constants(chip)
    if ref is None:   # a predict-only row (M5 Max), same est_* shape as the table rows
        e = predict_entry(p.name, chip)
        if e is None:
            return None
        ref = replace(p, est_struct_117=e["struct_117"], est_struct_400=e["struct_400"], est_affinity=e["affinity"], steps=e["steps"],
                      affinity_steps=e["affinity_steps"], est_steps=e["steps"], est_affinity_steps=e["affinity_steps"])
        k = {"overhead": e.get("overhead", k["overhead"]), "load": e.get("load", k["load"])}
    s = struct_estimate(ref, n_tokens, steps)
    if samples > 1:  # extra diffusion samples scale the diffusion share (~40%, as struct_estimate's step scaling)
        s = 0.6 * s + 0.4 * s * samples
    a = affinity_estimate(ref, affinity_steps) if affinity else 0.0
    load = k["load"] * (2 if affinity else 1)
    return {"struct_s": round(s, 2), "affinity_s": round(a, 2), "load_s": round(load, 1),
            "total_s": round(s + a + load + k["overhead"], 1)}


def chip_info(chip: str | None = None) -> dict:
    """What the page shows about estimates on this Mac."""
    chip = chip or chip_name()
    c = CHIP_TABLE.get(chip) or {}
    presets = sorted(c.get("presets", {}))
    return {"name": chip, "has_estimates": bool(presets), "presets": presets, "reference": c.get("reference"),
            "known": chip in CHIP_TABLE,
            "note": (f"Time estimates are bench numbers measured on this chip ({chip})." if presets else
                     f"{NO_ESTIMATE} ({chip}): times are shown once the first job measures them.")}
