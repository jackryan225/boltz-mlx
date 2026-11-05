"""`boltz-mlx screen`: rank a ligand list against one protein target with Boltz-2 affinity, offline, in one command.

  boltz-mlx screen --target receptor.yaml --ligands ligands.csv --out results/
  boltz-mlx screen --target target.fasta --msa target.a3m --ligands library.sdf --out lib1/ --rank-by value
  boltz-mlx screen --target receptor.yaml --ligands ligands.smi --out results/ --dry-run

Inputs
  --target   Boltz YAML (protein chains, optional MSA paths / co-factors / pocket constraint), FASTA (protein chains), or a raw
             amino-acid sequence. Chains without an MSA run single-sequence (msa: empty) with a warning; the MSA server is
             never contacted (ligands and target stay on this machine).
  --ligands  .csv/.tsv (header with a SMILES column + optional id column), .smi ("SMILES id" per line) or .sdf (_Name as id).
Outputs (in --out)
  results.csv     ranked table (one row per ligand; invalid/error rows at the bottom with the reason)
  results.jsonl   full per-ligand rows, appended as each ligand finishes (resume reads it)
  structures/     top --save-structures poses;   work/  inputs, preprocessing, per-ligand predictions (Boltz layout)
  screen_config.json / screen_status.json
Resume is the default: rerun the same command after a kill; finished ligands are skipped and every result is identical
(per-record seeds; every SMILES ligand's RDKit reference conformer embedded with the fixed seeding.CONFORMER_SEED),
whatever the input order, chunking or the point of the kill. A ligand's result equals its solo-screen result.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from boltz_mlx import tiny_ligand
from boltz_mlx.cliout import CliError, bad_input, canonical_preset, json_stdout, public_name, restore_stdout
from boltz_mlx.screen.presets import DEFAULT_PRESET, DEFAULT_PRESET_NON_NAX, PRESETS, Preset, default_preset, for_device

CHILD_ENV = "BOLTZ_MLX_SCREEN_CHILD"
# BOLTZ_MLX_* variables kept from the caller's environment (location / memory knobs, not numerics)
# (BOLTZ_MLX_KERNEL_PATH / BOLTZ_MLX_NAX select the kernel path; the preset is then labelled <name>-<path>,
# presets.for_device; BOLTZ_MLX_CFG_CHIP selects the per-device launch-config entry, kernels.cfg_chip: speed only,
# outputs bitwise equal (tests/test_cfg_bitwise.py), so it is not part of the config key either)
ENV_PASSTHROUGH = {"BOLTZ_MLX_CACHE_GB", "BOLTZ_MLX_MEM_LIMIT_GB", "BOLTZ_MLX_WEIGHTS", "BOLTZ_MLX_CACHE", "BOLTZ_MLX_NAX",
                   "BOLTZ_MLX_KERNEL_PATH", "BOLTZ_MLX_CFG_CHIP", CHILD_ENV}


SCREEN_EPILOG = """\
examples:
  boltz-mlx screen --target receptor.yaml --ligands ligands.csv --out results/ --json
  boltz-mlx screen --target receptor.yaml --ligands ligands.smi --out results/ --dry-run   # plan + time estimate, no model

--ligands: .csv/.tsv (SMILES column + optional id column), .smi ("SMILES id" per line) or .sdf.
--target: Boltz YAML (protein chains, MSA paths), FASTA, or a raw sequence; chains without an MSA run single-sequence.
The MSA server is never contacted. Rerun the same command to resume after a stop. Output: out/results.csv (ranked),
results.jsonl, structures/ (top poses), screen_status.json. Exit codes: 0 ok, 1 runtime failure, 2 bad input, 3 setup problem."""


def add_parser(sub) -> argparse.ArgumentParser:
    H = argparse.SUPPRESS  # expert flags keep working but stay out of --help
    p = sub.add_parser("screen", help="rank ligands against one target (structure + affinity, offline, resumable)",
                       description="Rank a ligand list against one protein target by predicted Boltz-2 binding affinity.",
                       epilog=SCREEN_EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", required=True, help="receptor .yaml / .fasta, or an amino-acid sequence")
    p.add_argument("--ligands", required=True, help=".csv / .tsv / .smi / .sdf")
    p.add_argument("--out", required=True, help="output directory (resumable)")
    p.add_argument("--preset", default=None, metavar="{fast,full}",
                   help="fast: f16, 20 structure + 25 affinity steps (200 structure steps when a ligand has < 10 heavy atoms and "
                        "the steps are not set; default; fp32 'full' on Macs without the M5 neural "
                        "accelerators). full: fp32, 200 + 200 steps (upstream Boltz-2 defaults)")
    p.add_argument("--json", action="store_true", help="print one JSON result object on stdout; logs go to stderr")
    p.add_argument("--top", type=int, default=20, metavar="N", help="ligands listed in the --json result (default 20)")
    p.add_argument("--presets-file", default=None, help=H)
    p.add_argument("--msa", default=None, help="'empty' or an .a3m/.csv MSA for the protein chain(s) (default: as in the target YAML, else empty)")
    p.add_argument("--pocket", action="append", default=None, metavar="CHAIN:RES[,CHAIN:RES...]",
                   help="pocket residues (1-based residue index) for a pocket constraint on the screened ligand")
    p.add_argument("--ligand-chain", default=None, help=H)
    p.add_argument("--rank-by", choices=["prob", "value"], default="prob",
                   help="prob: affinity_probability_binary (hit finding, default); value: affinity_pred_value (lead optimisation)")
    p.add_argument("--save-structures", default="10", help="copy the top-K poses to out/structures ('all', or 0 for none; default 10)")
    p.add_argument("--chunk", type=int, default=50, help=H)
    p.add_argument("--dry-run", action="store_true", help="validate inputs, print the plan and a time ESTIMATE; no model")
    p.add_argument("--measure", type=int, default=0, metavar="K", help=H)
    p.add_argument("--seed", type=int, default=0, help="global seed (per-ligand seeds derive from it and the ligand id)")
    p.add_argument("--resume", dest="resume", action="store_true", default=True, help=H)
    p.add_argument("--no-resume", dest="resume", action="store_false", help="recompute every ligand (rows are appended; latest wins)")
    p.add_argument("--retry-errors", action="store_true", help="rerun ligands whose previous row is status=error")
    p.add_argument("--steps", type=int, default=None, help="override the preset's structure-pass sampling steps")
    p.add_argument("--recycling", type=int, default=None, help="override the preset's structure-pass recycles")
    p.add_argument("--affinity-steps", type=int, default=None, help="override the preset's affinity-pass sampling steps")
    p.add_argument("--cache", default=None, help="dir holding the upstream mols/ dictionary (default $BOLTZ_MLX_CACHE, $BOLTZ_CACHE, ~/.boltz)")
    p.add_argument("--weights", default=None, help="dir with boltz2_{conf,aff}.safetensors (default $BOLTZ_MLX_WEIGHTS, see boltz_mlx/paths.py)")
    p.add_argument("--id-col", default=None, help="id column (csv) or SDF property")
    p.add_argument("--smiles-col", default=None, help="SMILES column (csv)")
    p.add_argument("--max-msa-seqs", type=int, default=8192, help=H)
    p.add_argument("--output-format", choices=["mmcif", "pdb"], default="mmcif", help="pose format (default mmcif)")
    p.add_argument("--prefetch", dest="prefetch", action="store_true", default=True, help=H)
    p.add_argument("--no-prefetch", dest="prefetch", action="store_false", help=H)
    p.add_argument("--keep-work", action="store_true", help="keep per-ligand pae/pde/plddt npz and pre_affinity files")
    p.add_argument("--max-minutes", type=float, default=None, help="stop cleanly at a ligand boundary after this many minutes")
    p.add_argument("--device", choices=["gpu", "cpu"], default="gpu", help=H)
    p.add_argument("--mem-limit-gb", type=float, default=None, help="MLX memory limit (default $BOLTZ_MLX_MEM_LIMIT_GB, else none)")
    return p


def load_presets(path: str | None) -> dict[str, Preset]:
    presets = dict(PRESETS)
    if path:
        for name, d in json.loads(Path(path).read_text()).items():
            base = presets.get(d.get("base", "strict"), PRESETS["strict"])
            kw = {k: v for k, v in d.items() if k in Preset.__dataclass_fields__ and k != "name"}
            kw.setdefault("note", f"custom preset from {path}")
            kw.setdefault("validated", False)
            presets[name] = Preset(**{**base.__dict__, **kw, "name": name})
    return presets


def preset_environ(base: dict, preset: Preset) -> dict:
    """The caller's env without MLX numerics switches, then the preset's (as run_l3.mode_environ)."""
    env = {k: v for k, v in base.items()
           if not ((k.startswith("BOLTZ_MLX_") and k not in ENV_PASSTHROUGH) or k == "MLX_ENABLE_TF32")}
    env.update(preset.env)
    return env


def _kernel_path() -> str:
    from boltz_mlx import kernels as K

    return K.kernel_path()


def _reexec(argv: list[str], env: dict) -> None:
    import boltz_mlx

    env = dict(env)
    env[CHILD_ENV] = "1"
    pkg_parent = str(Path(boltz_mlx.__file__).resolve().parents[1])  # the child imports this same boltz_mlx tree
    env["PYTHONPATH"] = os.pathsep.join([pkg_parent] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    sys.stdout.flush()  # lines logged before the re-exec (e.g. the default preset) must not die in the buffer
    sys.stderr.flush()
    restore_stdout()  # with --json the parent redirected fd 1 to stderr; the child needs the real stdout
    os.execve(sys.executable, [sys.executable, "-m", "boltz_mlx.cli", "screen", *argv], env)


def build_config(a, presets) -> "ScreenConfig":  # noqa: F821
    from boltz_mlx.screen.runner import ScreenConfig

    if a.preset is None:
        a.preset = default_preset()  # by kernel path: screen on NAX (M5-class), strict on M1-M4
    a.preset = canonical_preset(a.preset, presets)  # fast -> screen, full -> strict; old names pass through
    p = presets[a.preset]
    ss = str(a.save_structures).strip().lower()
    save = -1 if ss == "all" else int(ss)
    if a.chunk < 1:
        raise bad_input("--chunk must be >= 1.", "pass --chunk 1 or more.")
    rule = a.preset == tiny_ligand.RULE_PRESET and not a.steps   # Fast and the user did not set --steps: tiny ligands get 200
    return ScreenConfig(
        out=Path(a.out).expanduser(), preset=p, steps=a.steps or p.steps, recycling=a.recycling if a.recycling is not None else p.recycling,
        affinity_steps=a.affinity_steps or p.affinity_steps, affinity_recycling=p.affinity_recycling,
        affinity_samples=p.affinity_samples, seed=a.seed, rank_by=a.rank_by, chunk=a.chunk, save_structures=save, resume=a.resume,
        retry_errors=a.retry_errors, prefetch=a.prefetch, keep_work=a.keep_work, max_minutes=a.max_minutes,
        max_msa_seqs=a.max_msa_seqs, output_format=a.output_format, device=a.device,
        env={k: v for k, v in sorted(os.environ.items()) if k.startswith("BOLTZ_MLX_") and k not in ENV_PASSTHROUGH
             or k == "MLX_ENABLE_TF32"},
        tiny_steps=tiny_ligand.TINY_STEPS if rule else None, tiny_heavy=tiny_ligand.TINY_HEAVY_ATOMS,
    )


def steps_info(cfg, ligands) -> dict:
    """What the tiny-ligand rule does in this screen (for the log line, the --json settings and the GUI estimate)."""
    valid = [l for l in ligands if l.status != "invalid"]
    tiny = [l for l in valid if cfg.steps_for(l) != cfg.steps]
    return {"rule": bool(cfg.tiny_steps), "structure_steps": cfg.steps, "tiny_steps": cfg.tiny_steps if cfg.tiny_steps else None,
            "heavy_atoms_lt": cfg.tiny_heavy, "n_tiny": len(tiny), "n_valid": len(valid),
            "tiny_ids": [l.ligand_id for l in tiny[:20]]}


def steps_note(info: dict) -> str | None:
    """One line for the log and the GUI: None when no ligand is affected."""
    if not info.get("n_tiny"):
        return None
    ids = ", ".join(info["tiny_ids"][:3]) + (f" and {info['n_tiny'] - 3} more" if info["n_tiny"] > 3 else "")
    return (f"Fast preset: {info['n_tiny']} of {info['n_valid']} ligands have < {info['heavy_atoms_lt']} heavy atoms ({ids}) and run "
            f"{info['tiny_steps']} structure steps; the other {info['n_valid'] - info['n_tiny']} run {info['structure_steps']}")


def load_inputs(a):
    from boltz_mlx.screen.inputs import load_target, read_ligands, validate_ligands

    target = load_target(a.target, msa=a.msa, pocket=a.pocket, ligand_chain=a.ligand_chain)
    ligands = validate_ligands(read_ligands(a.ligands, a.id_col, a.smiles_col))
    return target, ligands


def init_mlx(cfg) -> None:
    import boltz_mlx  # noqa: F401  (TF32 handling before MLX initialises)
    import mlx.core as mx

    mx.set_default_device(mx.gpu if cfg.device == "gpu" else mx.cpu)
    from boltz_mlx import cachelimit

    cachelimit.install_startup()  # 4 GB or $BOLTZ_MLX_CACHE_GB; UpstreamEngine.model_inputs adapts it once N is known
    lim = os.environ.get("BOLTZ_MLX_MEM_LIMIT_GB")
    if lim and cfg.device == "gpu":
        mx.set_memory_limit(int(float(lim) * 1e9))


def run(a, argv: list[str] | None = None, engine=None, log=print) -> int:
    """Entry for `boltz-mlx screen`. `engine` (tests) replaces the upstream featurizer/model glue and skips the re-exec."""
    from boltz_mlx.paths import default_cache
    from boltz_mlx.screen import results as R
    from boltz_mlx.screen.inputs import InputError
    from boltz_mlx.screen.runner import ScreenRunner, plan_lines

    presets = load_presets(a.presets_file)
    if a.preset is None:
        # the default depends on the kernel path (presets.default_preset: screen on NAX, strict on M1-M4); resolved once
        # here and passed on explicitly, so the re-exec'd child and a --measure run use the same preset
        a.preset = default_preset()
        argv = [*(argv if argv is not None else sys.argv[2:]), "--preset", a.preset]
        log(f"screen: preset {a.preset} (the default for kernel path {_kernel_path()})")
    else:
        asked = a.preset
        a.preset = canonical_preset(asked, presets)  # fast -> screen, full -> strict; the old names keep working
        if asked != a.preset:
            log(f"screen: preset {asked} runs as {a.preset}")
    if a.mem_limit_gb:
        os.environ["BOLTZ_MLX_MEM_LIMIT_GB"] = str(a.mem_limit_gb)
    if a.weights:
        os.environ["BOLTZ_MLX_WEIGHTS"] = str(Path(a.weights).expanduser().resolve())
    # numerics env must be in place before MLX / boltz_mlx.kernels initialise: re-exec the CLI with the preset applied
    if engine is None and not a.dry_run and os.environ.get(CHILD_ENV) != "1":
        from boltz_mlx.check import preflight

        preflight(a.cache, a.weights)  # exit 3 with the fix when weights / mols are missing (before any model work)
        _reexec(list(argv if argv is not None else sys.argv[2:]), preset_environ(os.environ, presets[a.preset]))
    try:
        target, ligands = load_inputs(a)
    except InputError as ex:
        raise bad_input(str(ex), "check --target / --ligands (see `boltz-mlx screen --help`).") from None
    from boltz_mlx import kernels as K

    hw = K.device_tag()  # chip / architecture / kernel path (tiny GPU checks on M5-class GPUs; none on M1-M4 by default)
    presets[a.preset] = for_device(presets[a.preset], hw["path"])  # f16 -> f16-portable / f16-tensorops off NAX
    cfg = build_config(a, presets)
    cfg.hw = hw
    a.steps_info = steps_info(cfg, ligands)   # read by json_result
    done = R.done_ids(R.read_rows(cfg.out / "results.jsonl"), cfg.retry_errors) if cfg.resume else set()
    if a.dry_run:
        if engine is None:
            cfg.env = dict(presets[a.preset].env)
        for line in plan_lines(cfg, target, ligands, done):
            log(line)
        if a.measure > 0:
            return measure(a, argv, cfg, target, ligands, done, log)
        return 0
    if steps_note(a.steps_info):   # a dry run prints it in the plan
        log("screen: " + steps_note(a.steps_info))
    if os.environ.get("BOLTZ_MLX_CFG_CHIP"):   # A/B runs of launch configs (speed only): say which entry ran
        log(f"screen: per-device kernel config entry {K.cfg_chip()!r} (BOLTZ_MLX_CFG_CHIP; speed only, outputs bitwise equal)")
    if engine is None:
        from boltz_mlx.screen.engine import UpstreamEngine

        init_mlx(cfg)
        cache = default_cache(a.cache)
        engine = UpstreamEngine(cfg.out / "work", cache, target, cfg.max_msa_seqs, cfg.output_format, cfg.seed,
                                keep_extras=cfg.keep_work)
    for w in target.warnings:
        log(f"[target] warning: {w}")
    runner = ScreenRunner(cfg, target, ligands, engine, log=log)
    try:
        status = runner.run()
    except KeyboardInterrupt:
        return 130
    # per-ligand errors are rows in results.csv (status=error), not a failed run
    return 0


def measure(a, argv, cfg, target, ligands, done, log) -> int:
    """--dry-run --measure K: run K pending ligands for real in OUT/_measure (a separate out dir, same config) and project."""
    import subprocess

    pending = [l for l in ligands if l.status != "invalid" and l.ligand_id not in done]
    if not pending:
        log("measure: nothing pending")
        return 0
    k = min(a.measure, len(pending))
    mdir = cfg.out / "_measure"
    mdir.mkdir(parents=True, exist_ok=True)
    sub = mdir / "ligands.csv"
    import csv

    with open(sub, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "smiles"])
        for l in pending[:k]:
            w.writerow([l.ligand_id, l.smiles])
    args = [x for x in (argv or []) if x not in ("--dry-run",)]
    # drop --measure K, --out X, --ligands X from the forwarded args
    fwd, skip = [], 0
    for x in args:
        if skip:
            skip -= 1
            continue
        if x in ("--measure", "--out", "--ligands"):
            skip = 1
            continue
        if x.startswith(("--measure=", "--out=", "--ligands=")):
            continue
        fwd.append(x)
    cmd = [sys.executable, "-m", "boltz_mlx.cli", "screen", *fwd, "--ligands", str(sub), "--out", str(mdir / "run"),
           "--save-structures", "0", "--chunk", str(k)]
    log("measure: " + " ".join(cmd))
    rc = subprocess.call(cmd)
    st = json.loads((mdir / "run" / "screen_status.json").read_text()) if (mdir / "run" / "screen_status.json").exists() else {}
    if rc != 0 or not st.get("ok_this_run"):
        log(f"measure: run failed (rc {rc}); see output above")
        return rc or 1
    wall = float(st["wall_s"])
    per = wall / st["ok_this_run"]
    import math

    n_chunks = math.ceil(len(pending) / cfg.chunk)
    loads = sum(st.get("load_s", {}).values())
    proj = (wall - loads) / st["ok_this_run"] * len(pending) + loads * n_chunks
    log(f"MEASURED on this machine: {st['ok_this_run']} ligands in {wall:.0f} s cold ({per:.1f} s/ligand incl. one load of each "
        f"checkpoint, {loads:.1f} s). Projected for {len(pending)} ligands: ~{proj / 3600:.2f} h "
        f"(projection from {k} ligands; per-ligand time varies with ligand size)")
    return 0


# ------------------------------------------------------------------------------------------------ `screen --json`
def json_result(a, rc: int, plan: list[str], presets: dict | None = None) -> dict:
    """The one JSON object of `boltz-mlx screen --json` (built from the files the run wrote)."""
    from boltz_mlx.cliout import ligand_row, preset_settings
    from boltz_mlx.screen import results as R

    presets = presets or load_presets(a.presets_file)
    asked = a.preset
    internal = canonical_preset(asked, presets) if asked else default_preset()
    st = preset_settings(internal, presets)
    out = Path(a.out).expanduser().resolve()
    settings = {"preset": public_name(internal), "mode": internal, "precision": st["precision"],
                "structure_steps": a.steps or st["structure_steps"], "affinity_steps": a.affinity_steps or st["affinity_steps"],
                "recycles": a.recycling if a.recycling is not None else st["recycles"], "affinity_recycles": st["affinity_recycles"],
                "affinity_samples": st["affinity_samples"], "seed": a.seed, "rank_by": a.rank_by}
    info = getattr(a, "steps_info", None) or {}
    if info.get("rule"):   # Fast, --steps not given: tiny ligands run more structure steps (per ligand; rows carry their own steps)
        settings["structure_steps_tiny_ligands"] = info["tiny_steps"]
        settings["structure_steps_reason"] = None
        if info["n_tiny"]:
            ids = ", ".join(info["tiny_ids"][:3]) + (f" and {info['n_tiny'] - 3} more" if info["n_tiny"] > 3 else "")
            settings["structure_steps_reason"] = (f"{info['n_tiny']} of {info['n_valid']} ligands have < {info['heavy_atoms_lt']} "
                                                  f"heavy atoms ({ids}) and run {info['tiny_steps']} structure steps")
        settings["ligands_at_tiny_steps"] = info["n_tiny"]
    obj = {"status": "ok", "command": "screen", "exit_code": rc,
           "inputs": {"target": a.target, "ligands": str(Path(a.ligands).expanduser().resolve()), "msa": a.msa,
                      "pocket": a.pocket, "out": str(out)},
           "settings": settings}
    if rc:
        obj.update(status="interrupted" if rc == 130 else "error", exit_code=rc)
    if a.dry_run:
        obj.update(status="dry_run", plan=plan)
        return obj
    rows = R.read_rows(out / "results.jsonl")
    ranked = [r for r in R.rank_rows(rows, a.rank_by) if r.get("rank") is not None]
    latest = R.latest_rows(rows)
    status = {}
    try:
        status = json.loads((out / "screen_status.json").read_text())
    except (OSError, ValueError):
        pass
    count = lambda k: sum(1 for r in latest.values() if r.get("status") == k)  # noqa: E731
    obj["settings"]["mode"] = status.get("mode", internal)  # as labelled in the results (f16-tensorops off the M5 path)
    obj["counts"] = {"ligands": status.get("ligands", len(latest)), "ok": count("ok"), "invalid": count("invalid"),
                     "error": count("error"), "remaining": status.get("remaining")}
    obj.update(top_n=a.top, top=[ligand_row(r) for r in ranked[:max(a.top, 0)]],
               results_csv=str(out / "results.csv"), results_jsonl=str(out / "results.jsonl"),
               structures_dir=str(out / "structures") if (out / "structures").is_dir() else None,
               wall_s=status.get("wall_s"), complete=status.get("complete"))
    if status.get("complete") is False and not rc:
        obj["status"] = "incomplete"  # stopped by --max-minutes / a signal: rerun the same command to resume
    if obj["counts"]["ok"] == 0 and rc == 0 and not a.dry_run and obj["counts"]["ligands"]:
        bad = obj["counts"]["invalid"] == obj["counts"]["ligands"]
        obj["status"] = "error"
        obj["error"] = {"kind": "bad_input" if bad else "runtime",
                        "message": "no ligand was valid." if bad else "every ligand failed (see the error column of results.csv).",
                        "fix": "check the SMILES column of --ligands." if bad else "see results.csv; rerun with --retry-errors."}
        obj["exit_code"] = 2 if bad else 1
    return obj


def command(a, argv: list[str] | None = None) -> int:
    """`boltz-mlx screen`: run(), with --json support (one JSON object on stdout, everything else on stderr)."""
    from boltz_mlx.cliout import emit

    plan: list[str] = []

    def log(line=""):
        plan.append(str(line))
        print(line)  # stdout is stderr while --json is on

    with json_stdout(a.json):
        rc = run(a, argv, log=log)
    if a.json:
        obj = json_result(a, rc, plan)
        emit(obj)
        return obj["exit_code"] if rc == 0 else rc
    return rc
