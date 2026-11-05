"""The GUI's two presets and how they map onto the CLI's (boltz_mlx/screen/presets.py).

The UI names are "Fast" (key "standard") and "Full" (key "reference"); the keys stay because stored runs, drafts and scripts use them.
Fast is the default of Predict and Screen.

  standard (Fast) -> CLI "screen" on M5-class GPUs (NAX kernel path): f16, 20 structure steps + 25 affinity steps (5 samples,
                recycles unchanged); 200 structure steps when any ligand has < 10 heavy atoms and the steps are not set
                (boltz_mlx/tiny_ligand.py; decided after the 4-atom BCT collapse at 20 steps; f16 + 200 + 25 also passed 8/8).
                The 20-step version passed PREREG-B sampler confirmation attempt 1 (8 of 8 gated gates against strict)
                and the rung 5 screen gates on the M5 Max; L3L.chirality and A3.kendall are untestable on that data and
                reported only (project decision).
                On Macs without NAX (M1-M4) Fast runs CLI "strict" (project decision): there the f16 path does
                not reproduce the M5 results (cross-chip check FAIL) and strict does. The f16 path stays available from
                the command line (--preset screen, labelled screen-tensorops; not validated).
  reference (Full) -> CLI "strict": upstream Boltz-2's own defaults at fp32 (200 structure steps, 3 recycles, affinity 200 x 5).
                A screen setup choice, and the preset of the results table's Re-check.

Measured per-ligand screen times (whole ligands, cold, incl. one load of each checkpoint) are kept apart from the bench
rows in estimates.py: they are what a run on that chip took, quoted with their source, never scaled to other chips.
"""

from __future__ import annotations

from boltz_mlx.gui.common import ApiError

# "line": the one sentence under Fast | Full; "note": the plain details behind the info button (and in About).
GUI_PRESETS: list[dict] = [
    {"key": "standard", "label": "Fast", "cli": "screen", "status": "validated", "enabled": True, "screen": True,
     "tooltip": "Matches 200-step output in our accuracy checks.",
     "summary": "f16, 20 structure steps, 3 recycles, affinity 25 steps × 5 samples",
     "line": "Fast: half precision, fewer steps. Same results as Full on our checks, 2–3× quicker.",
     "note": "Fast runs in half precision (f16) with 20 structure steps and 25 affinity steps instead of 200. On an M5 Max it "
             "agreed with Full on every check we set before testing (56 proteins, 107 ligands). Two small differences remain: "
             "ligand stereo centres flip a little more often (10 vs 3 in that set), and affinity rankings agree closely with "
             "Full but not exactly. Very small ligands (under 10 heavy atoms) automatically get 200 structure steps, because "
             "20 is too few for them.",
     # Macs without NAX (M1-M4): Fast runs strict
     "non_nax": {"cli": "strict", "status": "validated",
                 "tooltip": "Validated on M5 (NAX) Macs only; on this Mac Fast runs the Full settings.",
                 "summary": "on this Mac: fp32, 200 structure steps, 3 recycles, affinity 200 steps × 5 samples",
                 "line": "Fast: on this Mac, the same as Full.",
                 "note": "The quicker half-precision settings are checked on M5-class Macs only; on this Mac they do not give the "
                         "same results, so Fast runs the Full settings here. They stay available from the command line "
                         "(boltz-mlx screen --preset screen)."}},
    {"key": "reference", "label": "Full", "cli": "strict", "status": "validated", "enabled": True, "screen": True,
     "predict_label": "Full",
     "tooltip": "Upstream Boltz-2 defaults at fp32: 200 structure steps, 3 recycles, affinity 200 steps × 5 samples.",
     "summary": "fp32, 200 structure steps, 3 recycles, affinity 200 steps × 5 samples",
     "line": "Full: the original Boltz-2 settings.",
     "note": "The original Boltz-2 settings in full precision (fp32): 200 structure steps, 3 recycles, 200 affinity steps. "
             "The reference, and what Re-check uses."},
]
BY_KEY = {p["key"]: p for p in GUI_PRESETS}
SCREEN_DEFAULT = "standard"   # "Fast": the default of Predict and Screen on NAX Macs (M5-class)
PREDICT_DEFAULT = "standard"


def default_key(path: str | None = None) -> str:
    """The preset the forms start on: Fast where f16 is the default (NAX, M5-class), Full on M1-M4 (the f16 path failed the
    cross-chip check there; same rule as screen.presets.default_preset)."""
    path = path or kernel_path()
    return SCREEN_DEFAULT if path == "nax" else "reference"

# Measured on the M5 Max, CLI preset "screen" = f16 + 20 + 25 steps, T400 receptor (lig_aff, 384 aa, 249-row MSA):
#   what a real screen takes (the card's time): rung 5 G1, job boltz-val-g1-t400-20261029a (2026-10-29, class 1,
#   alone on the GPU, AC power in every pmset snapshot): 4 consecutive full cold screens of the 199 T400 ligands, each a
#   fresh process (featurization, both passes, checkpoint loads, writing included): 6.683 / 7.123 / 7.207 / 6.323
#   s/ligand, pooled 6.835 s/ligand (796 ligands in 5440.9 s; run g1_t400_20261029a; docs/report.md section 12).
#   secondary, "warm": `--dry-run --measure 5` (5 ligands, cold process, one load of each checkpoint): 5.39 / 5.38
#   s/ligand (job boltz-val-t400-stageBC-20261029a, comp_s20+aff25 arms A / B). Long screens run slower per ligand
#   than the 5-ligand measure (structure 4.2-4.5 vs 3.25 s, affinity 2.3-2.5 vs 1.8 s per ligand).
# The earlier screen settings (f16 + 50 + 200) measured 8.2-8.7 s/ligand there (5-ligand measure) and 15.5 s on the M5
# Pro; those are not this preset's times, and the new preset has no measurement on the M5 Pro yet.
# "what" says what was measured, without "measured on this chip": every place that quotes it adds that once (api.py
# estimate caveat and plan line, the setup page), so the text never reads "measured on this chip (measured on this chip: ...)".
MEASURED: dict[tuple[str, str], dict] = {
    ("Apple M5 Max", "screen"): {
        "lo": 6.32, "hi": 7.21, "runs": [6.68, 7.12, 7.21, 6.32], "pooled": 6.84,
        "display": "~6.8 s per ligand (measured: 4 full cold screens of 199 ligands, ~400-residue target, AC power; "
                   "5.4 s on a 5-ligand warm-up measure)",
        "what": "4 full cold screens of 199 ligands, T400 receptor (384 aa), AC power, Fast",
        "warm": {"lo": 5.38, "hi": 5.39, "runs": [5.39, 5.38],
                 "what": "5 ligands, --dry-run --measure 5", "source": "benchmark log boltz-val-t400-stageBC-20261029a.log"},
        "source": "run g1_t400_20261029a; docs/report.md section 12 (rung 5 G1, job boltz-val-g1-t400-20261029a)"},
}
# Reference (strict) on the M5 Max (job boltz-val-strict-trial-20261029a, AC power): P38 (FEP+ seed 1, ~360-residue kinase
# + ligand), 3 ligands in one process: 94.4 s wall incl. one model load = 31.5 s/ligand; compute 29.7 s/ligand (structure
# 14.92 s + affinity 14.77 s, median per pass). Sources: benchmark log boltz-val-strict-trial-20261029a.log (phase N=1 line),
# the strict concurrency trial.
MEASURED[("Apple M5 Max", "strict")] = {
    "lo": 29.7, "hi": 31.5, "runs": [31.5],
    "display": "~30 s per ligand (measured, ~360-residue target)",
    "what": "3 ligands in one process incl. one model load, P38 (~360-residue kinase), Full",
    "source": "benchmark log boltz-val-strict-trial-20261029a.log, strict concurrency trial"}
MEASURING: dict[tuple[str, str], str] = {}


def measured(chip: str, cli_preset: str) -> dict | None:
    m = MEASURED.get((chip, cli_preset))
    return dict(m) if m else None


def kernel_path() -> str:
    from boltz_mlx import kernels as K

    return K.kernel_path()


def resolved(p: dict, path: str | None = None) -> dict:
    """GUI preset p as it runs on this kernel path: off NAX (M1-M4) its non_nax fields replace the defaults."""
    path = path or kernel_path()
    d = {k: v for k, v in p.items() if k != "non_nax"}
    if path != "nax" and p.get("non_nax"):
        d.update(p["non_nax"])
    return d


def to_cli(name: str | None, default_key: str, path: str | None = None) -> str:
    """A GUI preset key (or, for scripts and older pages, a CLI preset name) -> the CLI preset name on this kernel path
    (Fast -> screen on NAX, strict on M1-M4). A disabled card -> 400."""
    name = (str(name) if name is not None else "").strip() or default_key
    if name in BY_KEY:
        p = resolved(BY_KEY[name], path)
        if not p["enabled"]:
            raise ApiError(400, f"the {p['label']} preset is not available yet ({p['tooltip']})")
        return p["cli"]
    return name  # a CLI preset name; checked against the CLI's presets by the caller


def gui_key(cli_preset: str | None) -> str | None:
    for p in GUI_PRESETS:
        if p["cli"] and p["cli"] == cli_preset:
            return p["key"]
    return None


def run_label(cfg: dict | None, cli_preset: str | None = None) -> str | None:
    """What a screen ran, as the results pages name it (cfg: its screen_config.json). The GUI name ("Fast",
    "Full") only when the run's step counts are that preset's today: CLI "screen" was f16 + 50 + 200 steps until
    2026-10-29 and Fast is f16 + 20 + 25 since, so an older screen run, or one with changed steps, is named by what it
    ran ("screen, 50 + 200 steps (Fast is 20 + 25)"), not as Fast. Not started yet: the preset chosen."""
    from boltz_mlx.screen.presets import PRESETS

    cfg = cfg or {}
    name = cfg.get("preset") or cli_preset
    if not name:
        return None
    key = gui_key(name)
    label = BY_KEY[key]["label"] if key else name
    p = PRESETS.get(name)
    ran, rec = (cfg.get("steps"), cfg.get("affinity_steps")), cfg.get("recycling")
    if p is None or None in ran or (ran == (p.steps, p.affinity_steps) and rec in (None, p.recycling)):
        return label
    short = label.split(" (")[0]
    if rec in (None, p.recycling):
        return f"{name}, {ran[0]} + {ran[1]} steps ({short} is {p.steps} + {p.affinity_steps})"

    def recs(n: int) -> str:
        return f"{n} recycle{'' if n == 1 else 's'}"

    return f"{name}, {ran[0]} + {ran[1]} steps, {recs(rec)} ({short} is {p.steps} + {p.affinity_steps}, {recs(p.recycling)})"


def payload(chip: str, path: str | None = None) -> list[dict]:
    """The presets as the page shows them on this kernel path, with this chip's measured per-ligand time (or why there is
    none)."""
    out = []
    for p in GUI_PRESETS:
        p = resolved(p, path)
        d = dict(p)
        d["default_screen"] = d["default_predict"] = p["key"] == default_key(path)
        m = measured(chip, p["cli"]) if p["cli"] else None
        d["measured"] = m
        d["time_note"] = (m["display"] if m and m.get("display") else
                          f"{m['lo']:g}–{m['hi']:g} s / ligand (measured on this chip: {m['what']})" if m and m["lo"] != m["hi"] else
                          f"{m['lo']:g} s / ligand (measured on this chip: {m['what']})" if m else
                          MEASURING.get((chip, p["cli"])) if p["cli"] else None) or (
                          "not measured on this chip yet" if p["enabled"] else None)
        out.append(d)
    return out
