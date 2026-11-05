"""Output helpers of the `boltz-mlx` command line: exit codes, one-line errors, `--json` result objects, preset names.

Kept free of MLX / torch imports so `boltz-mlx check` and the error paths stay cheap.

Exit codes (AGENTS.md):  0 ok | 1 runtime failure | 2 bad input or arguments | 3 setup problem (macOS, weights, cache).
With --json, stdout carries exactly one JSON object (also for errors); logs and progress go to stderr.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import sys
from pathlib import Path

EXIT_OK, EXIT_RUNTIME, EXIT_BAD_INPUT, EXIT_SETUP = 0, 1, 2, 3
KINDS = {EXIT_RUNTIME: "runtime", EXIT_BAD_INPUT: "bad_input", EXIT_SETUP: "setup"}
AFFINITY_UNIT = "log10(IC50 in uM)"


class CliError(SystemExit):
    """A failure with an exit code, a one-line message and the fix. Subclasses SystemExit, so code that only expects
    sys.exit() (tests, callers of screen.main.run) still sees a SystemExit."""

    def __init__(self, code: int, message: str, fix: str | None = None):
        super().__init__(code)
        self.code, self.message, self.fix = code, message, fix

    @property
    def kind(self) -> str:
        return KINDS.get(self.code, "runtime")

    def line(self) -> str:
        msg = self.message if self.message.endswith((".", "!", "?")) else self.message + "."
        return f"boltz-mlx: error: {msg}" + (f" Fix: {self.fix}" if self.fix else "")

    def __str__(self) -> str:
        return self.line()


def bad_input(message: str, fix: str | None = None) -> CliError:
    return CliError(EXIT_BAD_INPUT, message, fix)


def setup_problem(message: str, fix: str | None = None) -> CliError:
    return CliError(EXIT_SETUP, message, fix)


def classify_exception(ex: BaseException) -> CliError:
    """Map an unexpected exception to a CliError (exit 1 unless it is recognisably an input or a setup problem)."""
    if isinstance(ex, CliError):
        return ex
    text = str(ex)
    if isinstance(ex, FileNotFoundError):
        if "manifest.json" in text:
            return bad_input("no input record could be processed (the reason is printed above on stderr).",
                             "fix the YAML/FASTA; every protein needs `msa: empty`, an .a3m/.csv path, or --use_msa_server.")
        if "safetensors" in text or "boltz2_" in text:
            return setup_problem(f"model weights not found ({text}).", "run `boltz-mlx setup` (downloads + converts "
                                 "them), or point BOLTZ_MLX_WEIGHTS at the weights directory.")
        if "mols" in text or "ccd" in text:
            return setup_problem(f"molecule dictionary not found ({text}).",
                                 "run `boltz-mlx setup` (downloads mols/ into the cache), or point --cache / $BOLTZ_MLX_CACHE at a Boltz cache.")
        return bad_input(f"file not found: {ex.filename or text}.", "check the path.")
    if isinstance(ex, RuntimeError) and "requires macOS" in text:
        return setup_problem(text, "update macOS to 26 or later, then run `boltz-mlx check`.")
    if type(ex).__module__.startswith("yaml"):
        return bad_input(f"invalid YAML: {text}", "fix the YAML and run again.")
    if isinstance(ex, MemoryError) or "out of memory" in text.lower() or "OutOfMemory" in text or "Insufficient Memory" in text:
        return CliError(EXIT_RUNTIME, "ran out of memory (unified memory).",
                        "use a smaller input (fewer tokens), close other apps, or --preset fast; see AGENTS.md limits.")
    return CliError(EXIT_RUNTIME, f"{type(ex).__name__}: {text}".strip(": "),
                    "set BOLTZ_MLX_TRACEBACK=1 and rerun for the traceback.")


# ----------------------------------------------------------------------------------------------- stdout handling
_SAVED_FD: list[int] = []


@contextlib.contextmanager
def json_stdout(enabled: bool):
    """With `enabled`, everything written to stdout (Python prints, upstream libraries, child processes) goes to stderr for the
    duration; the context yields a file object that still reaches the real stdout, for the single JSON object."""
    if not enabled:
        yield sys.stdout
        return
    sys.stdout.flush()
    sys.stderr.flush()
    saved = os.dup(1)
    _SAVED_FD.append(saved)
    real = os.fdopen(os.dup(saved), "w", buffering=1)
    os.dup2(2, 1)
    old = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield real
    finally:
        sys.stdout = old
        real.flush()
        os.dup2(saved, 1)
        os.close(saved)
        _SAVED_FD.pop()


def restore_stdout() -> None:
    """Undo the fd-level redirect before an exec (the new process must see the real stdout on fd 1)."""
    if _SAVED_FD:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(_SAVED_FD[-1], 1)


def _default(o):
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, "item"):
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def clean(o):
    """NaN / inf -> None, recursively (strict JSON)."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    return o


def emit(obj: dict, stream=None) -> None:
    """Write the one JSON object (plus a newline) to `stream` (default stdout)."""
    s = stream or sys.stdout
    s.write(json.dumps(clean(obj), indent=2, default=_default) + "\n")
    s.flush()


def error_object(command: str | None, err: CliError) -> dict:
    return {"status": "error", "command": command, "exit_code": err.code,
            "error": {"kind": err.kind, "message": err.message, "fix": err.fix}}


# ------------------------------------------------------------------------------------------------------ presets
PUBLIC_PRESETS = ("fast", "full")
# user-facing name -> internal preset of boltz_mlx/screen/presets.py; the old names (screen, strict, turbo, f16) keep working, hidden
PRESET_ALIASES = {"fast": "screen", "full": "strict"}


def canonical_preset(name: str, known) -> str:
    """fast -> screen, full -> strict; any other name must be one of `known` (the old names and custom --presets-file ones)."""
    n = PRESET_ALIASES.get(name, name)
    if n not in known:
        raise bad_input(f"unknown preset {name!r}.", "use --preset fast (default) or --preset full.")
    return n


def public_name(internal: str) -> str:
    inv = {v: k for k, v in PRESET_ALIASES.items()}
    return inv.get(internal, internal)


def preset_settings(internal: str, presets: dict | None = None) -> dict:
    """precision / steps / recycles / samples of an internal preset (boltz_mlx/screen/presets.py)."""
    from boltz_mlx.screen.presets import PRESETS

    p = (presets or PRESETS)[internal]
    return {"precision": p.env.get("BOLTZ_MLX_PRECISION", "fp32"), "tf32": p.env.get("BOLTZ_MLX_TF32") == "1",
            "diff_fp16": p.env.get("BOLTZ_MLX_DIFF_DTYPE") == "fp16", "structure_steps": p.steps,
            "affinity_steps": p.affinity_steps, "recycles": p.recycling, "affinity_recycles": p.affinity_recycling,
            "affinity_samples": p.affinity_samples}


# ------------------------------------------------------------------------------------------------- result readers
def _load(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def confidence_score(complex_plddt, iptm, ptm):
    """Boltz's ranking score: (4 * complex_plddt + (iptm if iptm != 0 else ptm)) / 5."""
    if complex_plddt is None or (iptm is None and ptm is None):
        return None
    return (4 * complex_plddt + (iptm if iptm else ptm)) / 5


def affinity_numbers(d: dict | None) -> dict | None:
    if not d or d.get("affinity_pred_value") is None:
        return None
    v = float(d["affinity_pred_value"])
    return {"affinity_pred_value": v, "unit": AFFINITY_UNIT, "ic50_uM": 10.0 ** v,
            "affinity_probability_binary": d.get("affinity_probability_binary")}


def collect_predict(results_dir: Path, ids: list[str] | None, timings: dict) -> list[dict]:
    """Per-target outputs of a predict run, read from <results_dir>/predictions/<id>/ (Boltz layout)."""
    pred = Path(results_dir) / "predictions"
    if ids is None:
        ids = sorted(p.name for p in pred.iterdir() if p.is_dir()) if pred.is_dir() else []
    targets = []
    for rid in ids:
        d = pred / rid
        models = []
        for cif in sorted(d.glob(f"{rid}_model_*.cif")) + sorted(d.glob(f"{rid}_model_*.pdb")):
            k = int(cif.stem.rsplit("_", 1)[1])
            cj = d / f"confidence_{rid}_model_{k}.json"
            c = _load(cj) or {}
            models.append({"model": k, "structure": str(cif), "confidence_json": str(cj) if cj.exists() else None,
                           "confidence_score": c.get("confidence_score"), "ptm": c.get("ptm"), "iptm": c.get("iptm"),
                           "complex_plddt": c.get("complex_plddt")})
        models.sort(key=lambda m: m["model"])
        t = {"id": rid, "status": "ok" if models else "error"}
        if models:
            best = models[0]  # model_0 is the best-ranked sample
            t.update(structure=best["structure"], confidence_json=best["confidence_json"],
                     confidence_score=best["confidence_score"], ptm=best["ptm"], iptm=best["iptm"],
                     complex_plddt=best["complex_plddt"])
            if len(models) > 1:
                t["models"] = models
        else:
            t["error"] = "no structure was written"
        aj = d / f"affinity_{rid}.json"
        t["affinity_json"] = str(aj) if aj.exists() else None
        t["affinity"] = affinity_numbers(_load(aj)) if aj.exists() else None
        secs = [timings[k] for k in (f"predict_{rid}_s", f"affinity_{rid}_s") if k in timings]
        t["wall_s"] = round(sum(secs), 2) if secs else None
        targets.append(t)
    return targets


def ligand_row(r: dict) -> dict:
    """A screen results row in the JSON shape."""
    aff = {"affinity_pred_value": r.get("affinity_pred_value"), "unit": AFFINITY_UNIT,
           "affinity_probability_binary": r.get("affinity_probability_binary")}
    if aff["affinity_pred_value"] is not None:
        aff["ic50_uM"] = 10.0 ** float(aff["affinity_pred_value"])
    return {"rank": r.get("rank"), "ligand_id": r.get("ligand_id"), "smiles": r.get("smiles"), "affinity": aff,
            "confidence_score": confidence_score(r.get("complex_plddt"), r.get("iptm"), r.get("ptm")),
            "ptm": r.get("ptm"), "iptm": r.get("iptm"), "ligand_iptm": r.get("ligand_iptm"),
            "complex_plddt": r.get("complex_plddt"), "ligand_plddt": r.get("ligand_plddt"), "seconds": r.get("seconds"),
            "structure_steps": r.get("steps")}
