"""`boltz-mlx check`: is this Mac set up to run boltz-mlx? Never touches the GPU and never loads the model.

Checks: macOS 26+, Apple silicon (arm64), the Python packages, the converted weights (boltz2_{conf,aff}.safetensors +
.json; the mixed fp16 copies the fast preset prefers are a warning only) and the upstream molecule dictionary (<cache>/mols). Exit 0 when
everything needed is there, 3 otherwise. Missing weights / mols are fixed by `boltz-mlx setup` (boltz_mlx/assets.py); the
result's "setup" entry lists its steps and download size (no network is used here).
"""

from __future__ import annotations

import importlib.util
import os
import platform
import subprocess
import sys
from pathlib import Path

WEIGHT_FILES = ("boltz2_conf.safetensors", "boltz2_conf.json", "boltz2_aff.safetensors", "boltz2_aff.json")
MIXED_FILES = ("boltz2_conf_mixed.safetensors", "boltz2_aff_mixed.safetensors")
REINSTALL = "run ./install.sh in the boltz-mlx checkout again"
PACKAGES = (("mlx", REINSTALL), ("boltz", REINSTALL + " (featurizer + writers come from upstream boltz)"),
            ("rdkit", REINSTALL), ("yaml", REINSTALL), ("torch", REINSTALL + " (torch is installed with boltz)"))
LOW_RAM_GB = 16


def _sysctl(name: str) -> str:
    try:
        return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return ""


def _check(name: str, ok: bool, detail: str, fix: str | None = None) -> dict:
    return {"name": name, "ok": bool(ok), "detail": detail, "fix": None if ok else fix}


def run_checks(cache: str | os.PathLike | None = None, weights: str | os.PathLike | None = None) -> dict:
    from boltz_mlx import MIN_MACOS, macos_version
    from boltz_mlx.paths import default_cache, weights_dir

    checks, warnings = [], []
    sysname, machine = platform.system(), platform.machine()
    mac = macos_version()
    try:
        major = int(mac.split(".")[0])
    except ValueError:
        major = 0
    checks.append(_check("macos", sysname == "Darwin" and major >= MIN_MACOS,
                         f"macOS {mac}" if mac else f"{sysname or 'unknown OS'} (not macOS)",
                         f"update to macOS {MIN_MACOS} or later (Metal 4); every Apple-silicon Mac can."))
    chip = _sysctl("machdep.cpu.brand_string") if sysname == "Darwin" else ""
    checks.append(_check("apple_silicon", sysname == "Darwin" and machine == "arm64",
                         f"{chip or machine}" + ("" if machine == "arm64" else " (this Python is not arm64)"),
                         "run on an Apple-silicon Mac with an arm64 Python (not Rosetta/x86_64); boltz-mlx needs the Apple GPU."))
    try:
        ram = int(_sysctl("hw.memsize")) / 2**30
    except ValueError:
        ram = 0.0
    if 0 < ram < LOW_RAM_GB:
        warnings.append(f"{ram:.0f} GB unified memory: complexes above ~600 tokens may run out of memory (see AGENTS.md).")
    for mod, how in PACKAGES:
        found = importlib.util.find_spec(mod) is not None
        checks.append(_check(f"package:{mod}", found, "installed" if found else "missing", how))

    if weights:
        os.environ["BOLTZ_MLX_WEIGHTS"] = str(Path(weights).expanduser())
    from boltz_mlx import assets

    wdir = weights_dir(cache)
    try:
        sp = assets.plan(cache, wdir)
    except Exception as ex:  # noqa: BLE001  (a broken file must not stop the report)
        sp = {"steps": [], "ready": False, "download_bytes": 0, "disk_bytes": 0, "error": str(ex)}
    setup_fix = "run `boltz-mlx setup` (" + (f"downloads {sp['download_bytes'] / 1e9:.1f} GB, " if sp["download_bytes"] else "") + \
        "then converts; resumable), or set BOLTZ_MLX_WEIGHTS / BOLTZ_MLX_CACHE to directories that already hold them."
    missing = [f for f in WEIGHT_FILES if not (wdir / f).is_file()]
    checks.append(_check("weights", not missing, str(wdir) if not missing else f"{wdir}: missing {', '.join(missing)}",
                         setup_fix))
    if not missing:  # the fast preset falls back to fp32 weights + runtime fp16 casts without them (slower load, more memory)
        stale = [f for f, n in zip(MIXED_FILES, assets.MODELS)
                 if not (wdir / f).is_file() or not assets.mixed_is_current(wdir / f, n)]
        if stale:
            warnings.append(f"{wdir}: {', '.join(stale)} missing or written for another build (the fast preset then casts "
                            "the fp32 weights at load time); `boltz-mlx setup` writes them.")
    cdir = default_cache(cache)
    mols = cdir / "mols"
    has_mols = assets.has_mols(cdir)
    checks.append(_check("cache_mols", has_mols, f"{mols}" if has_mols else f"{mols}: not found or empty",
                         setup_fix.replace("BOLTZ_MLX_WEIGHTS / BOLTZ_MLX_CACHE to directories that already hold them",
                                           "BOLTZ_MLX_CACHE (or pass --cache) to a Boltz cache that has mols/")))
    ok = all(c["ok"] for c in checks)
    return {"status": "ok" if ok else "fail", "command": "check", "ok": ok,
            "system": {"macos": mac or None, "machine": machine, "chip": chip or None,
                       "memory_gb": round(ram, 1) if ram else None, "python": platform.python_version()},
            "paths": {"cache": str(cdir), "weights": str(wdir), "python": sys.executable},
            "checks": checks, "warnings": warnings,
            "setup": {"steps": sp["steps"], "download_bytes": sp["download_bytes"], "disk_bytes": sp["disk_bytes"]},
            "next": None if ok else next(c["fix"] for c in checks if not c["ok"])}


def format_text(r: dict) -> str:
    lines = [f"{'ok  ' if c['ok'] else 'FAIL'} {c['name']:<14} {c['detail']}" + ("" if c["ok"] else f"\n       fix: {c['fix']}")
             for c in r["checks"]]
    lines += [f"warn {w}" for w in r["warnings"]]
    steps = r.get("setup", {}).get("steps") or []
    if steps:
        lines.append("`boltz-mlx setup` would:")
        for s in steps:
            size = s.get("bytes") or s.get("bytes_written") or 0
            lines.append(f"       {s['step']:<8} {s.get('file') or s.get('model') or '':<17} {size / 1e9:5.2f} GB  {s['detail']}")
    lines.append("boltz-mlx is ready." if r["ok"] else "boltz-mlx is NOT ready: fix the FAIL lines above.")
    return "\n".join(lines)


def preflight(cache: str | os.PathLike | None = None, weights: str | os.PathLike | None = None) -> None:
    """Raise a CliError (exit 3) naming the first missing piece of a real run's setup; returns when everything is there."""
    from boltz_mlx.cliout import setup_problem

    r = run_checks(cache, weights)
    bad = [c for c in r["checks"] if not c["ok"]]
    if bad:
        raise setup_problem(f"{bad[0]['name']}: {bad[0]['detail']}.", f"{bad[0]['fix']} (`boltz-mlx check` lists everything).")
