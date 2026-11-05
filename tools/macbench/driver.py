"""macbench driver: runs the timed arms of PREREG_MACBENCH.md, one fresh process per (arm, input, seed), stdlib only.

  python tools/macbench/driver.py run   --tag TAG --group GROUP [--idle-s 120] [--dry-run]     # arms x inputs x seeds of a group
  python tools/macbench/driver.py t400up --tag TAG --part first5|chunk --index K [--dry-run]   # upstream on the T400 ligands
  python tools/macbench/driver.py list                                                          # groups and arms

Never touches the GPU by itself beyond starting the model processes it is asked to start; --dry-run prints the commands only.
Results: logs/macbench/<tag>/<arm>/<input>/seed<k>/{run.json, stderr_time.txt, out/...} (PAE / PDE / pLDDT npz and processed features are
deleted after the run). run.json: W_s (whole-process wall, monotonic, spawn to exit), M_* (model-call seconds), peak footprint, machine
state (power source, W, GPU %, thermal, swap) before and after, the command and versions.

Rules (PREREG_MACBENCH.md sections 3, 4, 6): seeds 0,1,2; inputs ordered small to large; after a failed run the remaining seeds of that
(arm, input) are skipped and recorded; the job stops after the current run when a run's peak footprint exceeds GUARD_PEAK_GB or swap has
grown by more than GUARD_SWAP_GB since the job started (exit 0, STOPPED_BY_GUARD in the log).
"""
from __future__ import annotations

import argparse
import importlib.util
import signal
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # the editable install points at the main checkout: t400up must use THIS tree's boltz_mlx
    sys.path.insert(0, str(ROOT))
# Paths (environment overrides): BOLTZ_MLX_CACHE = Boltz cache with the checkpoints and mols/ (default ~/.boltz); MACBENCH_MLX_PY =
# python with boltz_mlx installed (default: this interpreter); MACBENCH_ENV = the comparison venvs + msa/ (default <checkout>/macbench-env,
# see docs/macbench/ENV.md); BOLTZ_MLX_EXT_DATA = external datasets (default <checkout>/data/external).
CACHE = Path(os.environ.get("BOLTZ_MLX_CACHE", Path.home() / ".boltz")).expanduser()
MLX_PY = Path(os.environ.get("MACBENCH_MLX_PY", sys.executable))
ENV_DIR = Path(os.environ.get("MACBENCH_ENV", ROOT / "macbench-env")).expanduser()
EXT_DATA = Path(os.environ.get("BOLTZ_MLX_EXT_DATA", ROOT / "data" / "external")).expanduser()
VENV_PY = {"rbvi": ENV_DIR / "venv-rbvi" / "bin" / "python", "stock": ENV_DIR / "venv-stock" / "bin" / "python",
           "community": ENV_DIR / "venv-community" / "bin" / "python"}
CASES = ROOT / "tools" / "macbench" / "cases"
LOGS = ROOT / "logs" / "macbench"
SEEDS = (0, 1, 2)
GUARD_PEAK_GB = 80.0
GUARD_SWAP_GB = 2.0

# arm -> how to run it
ARMS = {
    "up200": dict(kind="up", venv="rbvi", steps=200, aff_steps=200),
    "stock200": dict(kind="up", venv="stock", steps=200, aff_steps=200),
    # boltz-community (Novel-Therapeutics) at its defaults: --accelerator mps (fp32 forced), 200 steps, 3 recycles, 1 sample, its default
    # --subsample_msa (1024 rows). comm200_full = the same with --no_subsample_msa (all MSA rows, like upstream / RBVI / MLX).
    "comm200": dict(kind="up", venv="community", steps=200, aff_steps=200, accel="mps"),
    "comm200_full": dict(kind="up", venv="community", steps=200, aff_steps=200, accel="mps", extra=["--no_subsample_msa"]),
    "strict200": dict(kind="mlx", precision="fp32", steps=200, aff_steps=200),
    "f16_200": dict(kind="mlx", precision="f16", steps=200, aff_steps=200),
    "screen": dict(kind="mlx", precision="f16", steps=20, aff_steps=25),
    "f16_50": dict(kind="mlx", precision="f16", steps=50, aff_steps=50),
    # the Fast preset for part of 2026-11-03: f16, 50 structure steps, 25 affinity steps (presets.py `screen` in the gui-sci-loop tree; this
    # tree b3a0ef2 still has the 20-step `screen`, so the steps are passed explicitly, as for every other arm). NOT f16_50, whose affinity pass is 50 steps.
    "f16_50_aff25": dict(kind="mlx", precision="f16", steps=50, aff_steps=25),
    # Fast for a ligand with < 10 heavy atoms (rule added 2026-11-03): f16, 200 structure steps, 25 affinity steps. NOT f16_200 (its
    # affinity pass is 200 steps). Passed explicitly like every arm; the benchmark tree b3a0ef2 predates the rule and has no such preset.
    "f16_200_aff25": dict(kind="mlx", precision="f16", steps=200, aff_steps=25),
    # precision isolated from steps: the screen preset's 20 / 25 steps and sampler (the plain --sampling_steps path, as `screen`) in fp32
    "strict20": dict(kind="mlx", precision="fp32", steps=20, aff_steps=25),
}
MLX_ARMS = ["strict200", "f16_200", "screen", "f16_50"]

SMALL = ["prot117_empty", "8rf4_empty", "8rf4_msa", "8rf4aff_empty", "8rf4aff_msa", "lig_aff_empty", "lig_aff_msa"]
MID = ["prot400_empty", "ntca_aff_empty", "ntca_aff_msa"]
G9MOJ = ["9moj_empty", "9moj_msa"]
G9B3H = ["9b3h_empty", "9b3h_msa"]
NIM = ["nim_8eil_msa", "nim_7qsj_msa", "nim_8cpk_msa", "nim_8gpp_msa"]
GROUPS = {
    "mlx-1": dict(arms=MLX_ARMS, inputs=SMALL),
    "mlx-2": dict(arms=MLX_ARMS, inputs=MID),
    "mlx-3": dict(arms=MLX_ARMS, inputs=G9MOJ + G9B3H),
    "up-1": dict(arms=["up200"], inputs=SMALL),
    "up-2": dict(arms=["up200"], inputs=MID),
    "up-3": dict(arms=["up200"], inputs=G9MOJ),
    "up-4": dict(arms=["up200"], inputs=G9B3H),
    "stock-1": dict(arms=["stock200"], inputs=["prot117_empty", "8rf4_empty", "8rf4_msa", "9moj_msa"]),
    "comm-1": dict(arms=["comm200"], inputs=["9moj_msa", "9b3h_msa"]),  # withdrawn 2026-11-02 (PREREG_MACBENCH amendment 3)
    "comm-1b": dict(arms=["comm200_full"], inputs=["9b3h_msa"]),  # withdrawn
    # NVIDIA-size jobs (2026-11-02): structure + affinity of 8eil 186 / 7qsj 375 / 8cpk 384 / 8gpp 628 (tokens 194 / 399 / 410 / 628 with
    # the affinity ligand), ONE timed run per case (seed 0), no sweeps. An untimed warm-up on 8eil precedes the timed runs; a run that exceeds
    # timeout_s is killed and recorded as a failure (no retry).
    # job_limit_s: no new run is started after this many seconds since the job started (idle and warm-up included); with timeout_s it bounds the job.
    "nim-ours": dict(arms=["f16_50"], inputs=NIM, seeds=(0,), warmup="nim_8eil_msa", timeout_s=300, job_limit_s=900),
    "nim-ours-screen": dict(arms=["screen"], inputs=NIM, seeds=(0,), warmup="nim_8eil_msa", timeout_s=300, job_limit_s=900),  # variant of nim-ours: Standard preset
    # repeats: warm-up once, then 3 rounds of the 4 cases (8eil, 7qsj, 8cpk, 8gpp, again x3), same seed 0, dirs seed0_r1..r3
    "nim-ours-rep": dict(arms=["screen"], inputs=NIM, seeds=(0,), rounds=3, warmup="nim_8eil_msa", timeout_s=300, job_limit_s=900),
    # the same repeats for the NEW Fast (f16, 50 structure + 25 affinity steps): identical inputs, order, seed, warm-up, rounds and limits as nim-ours-rep
    "nim-ours-rep50": dict(arms=["f16_50_aff25"], inputs=NIM, seeds=(0,), rounds=3, warmup="nim_8eil_msa", timeout_s=300, job_limit_s=900),
    # re-timing of the cases whose ligand has < 10 heavy atoms (8eil SIN 8, 8gpp BCT 4; 7qsj / 8cpk celecoxib 26 stay at 20 steps) at the Fast steps for such a ligand:
    # same inputs' order (8eil, 8gpp), seed 0, warm-up (8eil, here at 200 steps), 3 interleaved rounds, limits and reporting as nim-ours-rep
    "nim-ours-rep200": dict(arms=["f16_200_aff25"], inputs=["nim_8eil_msa", "nim_8gpp_msa"], seeds=(0,), rounds=3, warmup="nim_8eil_msa", timeout_s=300, job_limit_s=900),
    # Full arm: strict200 = fp32, 200 structure steps, 200 affinity steps (5 recycles x 5 samples as always); one run per case
    "nim-full": dict(arms=["strict200"], inputs=NIM, seeds=(0,), warmup="nim_8eil_msa", timeout_s=600, job_limit_s=1200),
    # BCT-collapse diagnostic on 8gpp: explicit (arm, input, seed) runs after the 8eil warm-up; screen seed 0 already exists (nim-ours-screen)
    "bct-diag": dict(arms=["f16_50", "f16_200", "screen", "strict20"], inputs=["nim_8gpp_msa"], seeds=(0,), warmup="nim_8eil_msa", timeout_s=300, job_limit_s=600,
                     runs=[("f16_50", "nim_8gpp_msa", 0), ("f16_200", "nim_8gpp_msa", 0), ("screen", "nim_8gpp_msa", 1), ("screen", "nim_8gpp_msa", 2),
                           ("strict20", "nim_8gpp_msa", 0)]),
    "nim-comm": dict(arms=["comm200_full"], inputs=NIM, seeds=(0,), warmup="nim_8eil_msa", timeout_s=900, job_limit_s=2400),
}


def _state_mod():
    spec = importlib.util.spec_from_file_location("bench_r2_state", ROOT / "tools" / "bench_r2" / "state.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def snap(label: str) -> dict:
    try:
        return _state_mod().snapshot(label)
    except Exception as e:  # noqa: BLE001
        return {"label": label, "error": repr(e)}


def swap_used_gb() -> float | None:
    try:
        return _state_mod().swap_gb()[0]
    except Exception:  # noqa: BLE001
        return None


def git(*a) -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


# ------------------------------------------------------------------------------------------------------------------- commands
def build_cmd(arm: str, yaml: Path, out: Path, seed: int, timings: Path) -> tuple[list, dict]:
    a = ARMS[arm]
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    common = ["--out_dir", str(out), "--cache", str(CACHE), "--recycling_steps", "3", "--sampling_steps", str(a["steps"]),
              "--diffusion_samples", "1", "--sampling_steps_affinity", str(a["aff_steps"]), "--diffusion_samples_affinity", "5",
              "--seed", str(seed)]
    if a["kind"] == "mlx":
        cmd = [str(MLX_PY), "-m", "boltz_mlx.cli", "predict", str(yaml), *common, "--precision", a["precision"]]
        env.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
    else:
        cmd = [str(VENV_PY[a["venv"]]), "-I", str(ROOT / "tools" / "macbench" / "up_predict.py"), "predict", str(yaml), *common,
               "--accelerator", a.get("accel", "gpu"), *a.get("extra", [])]
        env.update(PYTORCH_ENABLE_MPS_FALLBACK="1", PYTORCH_MPS_HIGH_WATERMARK_RATIO="0.6", PYTORCH_MPS_LOW_WATERMARK_RATIO="0.5",
                   UP_TIMINGS=str(timings))
    return cmd, env


def parse_time_report(text: str) -> dict:
    d = {}
    m = re.search(r"(\d+)\s+peak memory footprint", text)
    if m:
        d["peak_footprint_gb"] = round(int(m.group(1)) / 2**30, 3)
    m = re.search(r"(\d+)\s+maximum resident set size", text)
    if m:
        d["max_rss_gb"] = round(int(m.group(1)) / 2**30, 3)
    m = re.search(r"([\d.]+)\s+real", text)
    if m:
        d["time_real_s"] = float(m.group(1))
    return d


def collect_timings(arm: str, inp: str, run_dir: Path) -> dict:
    a = ARMS[arm]
    out = run_dir / "out"
    if a["kind"] == "mlx":
        p = out / f"boltz_results_{inp}" / "boltz_mlx_timings.json"
        if not p.exists():
            return {}
        t = json.loads(p.read_text())
        ms = sum(v for k, v in t.items() if k.startswith("predict_") and k.endswith("_s"))
        ma = sum(v for k, v in t.items() if k.startswith("affinity_") and k.endswith("_s"))
        return {"M_structure_s": round(ms, 3) if ms else None, "M_affinity_s": round(ma, 3) if ma else None,
                "M_s": round(ms + ma, 3), "load_s": [t.get("load_conf_s"), t.get("load_aff_s")], "process_total_inside_s": t.get("total_wall_s"),
                "timings_raw": t}
    p = run_dir / "up_timings.json"
    if not p.exists():
        return {}
    t = json.loads(p.read_text())
    calls = t.get("calls", [])
    return {"M_structure_s": t.get("structure_model_s"), "M_affinity_s": t.get("affinity_model_s"), "M_s": t.get("model_s"),
            "load_s": t.get("loads"), "process_inputs_s": t.get("process_inputs_s"), "import_s": t.get("import_s"),
            "call_exceptions": sum(1 for c in calls if c.get("exception")), "n_calls": len(calls),
            "versions": {"boltz": t.get("boltz_version"), "torch": t.get("torch"), "boltz_file": t.get("boltz_file")},
            "calls": calls}


def cleanup(run_dir: Path, inp: str) -> None:
    res = run_dir / "out" / f"boltz_results_{inp}"
    for d in ("processed", "msa"):
        shutil.rmtree(res / d, ignore_errors=True)
    for p in res.rglob("*.npz"):
        p.unlink()


def _power_ok(*s) -> bool:
    return all((x or {}).get("power_source") == "AC Power" for x in s)


def _spawn(full, env, outf, err, timeout):
    """(rc, timed_out). With a timeout the child runs in its own process group and the whole group is killed when it is exceeded."""
    if not timeout:
        return subprocess.run(full, cwd=ROOT, env=env, stdout=outf, stderr=err).returncode, False
    p = subprocess.Popen(full, cwd=ROOT, env=env, stdout=outf, stderr=err, start_new_session=True)
    try:
        return p.wait(timeout=timeout), False
    except subprocess.TimeoutExpired:
        for sig, wait in ((signal.SIGTERM, 30), (signal.SIGKILL, 30)):
            try:
                os.killpg(p.pid, sig)
                p.wait(timeout=wait)
                break
            except (ProcessLookupError, subprocess.TimeoutExpired):
                continue
        return p.returncode if p.returncode is not None else -9, True


def run_one(tag: str, arm: str, inp: str, seed: int, dry: bool, timeout: float | None = None, warm: bool = False, rep: int = 0) -> dict:
    """warm=True: the untimed warm-up run; written under <tag>/warmup/..., never in runs.jsonl, so no summary row reads it."""
    run_dir = (LOGS / tag / "warmup" if warm else LOGS / tag) / arm / inp / (f"seed{seed}_r{rep}" if rep else f"seed{seed}")
    yaml = CASES / f"{inp}.yaml"
    cmd, env = build_cmd(arm, yaml, run_dir / "out", seed, run_dir / "up_timings.json")
    full = ["/usr/bin/time", "-l", *cmd]
    if dry:
        print("DRY", " ".join(full))
        return {"ok": True, "dry": True}
    if run_dir.exists():
        raise SystemExit(f"{run_dir} exists: use a new --tag (a rerun would mix files)")
    run_dir.mkdir(parents=True)
    s0 = snap(f"{arm}/{inp}/seed{seed} start")
    t_epoch0 = time.time()
    t0 = time.monotonic()
    with open(run_dir / "stderr_time.txt", "w") as err, open(run_dir / "stdout.txt", "w") as outf:
        rc, timed_out = _spawn(full, env, outf, err, timeout)
    W = time.monotonic() - t0
    t_epoch1 = time.time()
    s1 = snap(f"{arm}/{inp}/seed{seed} end")
    rep = parse_time_report((run_dir / "stderr_time.txt").read_text(errors="replace"))
    tim = collect_timings(arm, inp, run_dir)
    pd_ = run_dir / "out" / f"boltz_results_{inp}" / "predictions" / inp
    has_cif = (pd_ / f"{inp}_model_0.cif").exists()
    ok = rc == 0 and has_cif and not tim.get("call_exceptions") and not timed_out
    cleanup(run_dir, inp)
    rec = {"tag": tag, "arm": arm, "input": inp, "seed": seed, "cmd": cmd, "rc": rc, "ok": ok, "has_cif": has_cif, "W_s": round(W, 3),
           "timed_out": timed_out, "timeout_s": timeout, "warmup": warm, "repeat": rep,
           "start_epoch": t_epoch0, "end_epoch": t_epoch1, **rep, **{k: v for k, v in tim.items() if k not in ("calls",)},
           "calls": tim.get("calls"), "state_start": s0, "state_end": s1, "all_on_ac": _power_ok(s0, s1),
           "git_head": git("rev-parse", "HEAD"), "boltz_mlx_tree": git("rev-parse", "HEAD:boltz_mlx")}
    (run_dir / "run.json").write_text(json.dumps(rec, indent=1) + "\n")
    if not warm:
        with open(LOGS / tag / "runs.jsonl", "a") as f:
            f.write(json.dumps({k: v for k, v in rec.items() if k not in ("calls", "timings_raw", "state_start", "state_end", "cmd")}) + "\n")
    print(f"{'WARMUP (untimed) ' if warm else ''}RUN {arm}/{inp}/seed{seed}: rc {rc} ok {ok}{' TIMED OUT after ' + str(timeout) + ' s' if timed_out else ''} W {W:.1f} s, M {tim.get('M_s')} s, peak {rep.get('peak_footprint_gb')} GB, "
          f"AC {rec['all_on_ac']}", flush=True)
    return rec


def cmd_run(a) -> int:
    g = GROUPS[a.group]
    t_job = time.monotonic()
    if not a.dry_run:
        (LOGS / a.tag).mkdir(parents=True, exist_ok=True)
        (LOGS / a.tag / "job.json").write_text(json.dumps({
            "tag": a.tag, "group": a.group, "arms": g["arms"], "inputs": g["inputs"], "seeds": g.get("seeds", SEEDS),
            "warmup": g.get("warmup"), "timeout_s": g.get("timeout_s"), "job_limit_s": g.get("job_limit_s"), "git_head": git("rev-parse", "HEAD"),
            "git_branch": git("branch", "--show-current"), "boltz_mlx_tree": git("rev-parse", "HEAD:boltz_mlx"),
            "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "state": snap("job start"),
            "freeze_sha": {k: _sha(ENV_DIR / f"freeze_{k}.txt") for k in ("rbvi", "stock", "community")}}, indent=1, default=str) + "\n")
        if a.idle_s:
            print(f"GPU idle {a.idle_s} s", flush=True)
            time.sleep(a.idle_s)
    if g.get("warmup"):  # one untimed warm-up with the group's first arm; not in runs.jsonl, not in any table
        run_one(a.tag, g["arms"][0], g["warmup"], 0, a.dry_run, g.get("timeout_s"), warm=True)
    swap0 = swap_used_gb()
    failed = []
    rounds = g.get("rounds", 1)  # rounds > 1: repeats interleaved round-robin over the inputs (round 1: every input, then round 2: ...)
    order = [(r, inp, arm, seed) for r in range(rounds) for inp in g["inputs"] for arm in g["arms"] for seed in g.get("seeds", SEEDS)]
    if g.get("runs"):  # explicit (arm, input, seed) list, in this order
        order = [(0, inp, arm, seed) for arm, inp, seed in g["runs"]]
    dead = set()
    for rnd, inp, arm, seed in order:
        if (arm, inp) in dead:  # a failed (arm, input): its remaining seeds / repeats are skipped
            continue
        if g.get("job_limit_s") and not a.dry_run and time.monotonic() - t_job > g["job_limit_s"]:
            print(f"STOPPED_BY_JOB_LIMIT before {arm}/{inp}/seed{seed}: {time.monotonic() - t_job:.0f} s > {g['job_limit_s']} s since job start", flush=True)
            return 0
        rec = run_one(a.tag, arm, inp, seed, a.dry_run, g.get("timeout_s"), rep=(rnd + 1 if rounds > 1 else 0))
        if rec.get("dry"):
            continue
        stop = None
        if (rec.get("peak_footprint_gb") or 0) >= GUARD_PEAK_GB:
            stop = f"peak footprint {rec['peak_footprint_gb']} GB >= {GUARD_PEAK_GB} GB"
        sw = swap_used_gb()
        if sw is not None and swap0 is not None and sw - swap0 > GUARD_SWAP_GB:
            stop = (stop or "") + f" swap {swap0:.2f} -> {sw:.2f} GB (> {GUARD_SWAP_GB} GB growth)"
        if stop:
            print(f"STOPPED_BY_GUARD after {arm}/{inp}/seed{seed}: {stop}", flush=True)
            return 0
        if not rec["ok"]:
            dead.add((arm, inp))
            failed.append(f"{arm}/{inp}/seed{seed}/r{rnd + 1}" + (" (timeout)" if rec.get("timed_out") else ""))
            print(f"FAILED {arm}/{inp}/seed{seed}: recorded as a failure, not retried; remaining seeds / repeats of this (arm, input) skipped", flush=True)
    print("FAILED RUNS:" if failed else "all runs ok", failed, flush=True)
    return 0


def _sha(p: Path) -> str | None:
    import hashlib
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else None


# ------------------------------------------------------------------------------------------------------------ T400 upstream
T400_CSV = EXT_DATA / "fep" / "t400_ligands.csv"
T400_TARGET = ROOT / "cases" / "lig_aff.yaml"
T400_MSA = ROOT / "cases" / "msa" / "seq1.a3m"
EST_LIMIT_S = 21600.0


def t400_inputs(tag: str) -> list[Path]:
    """Per-ligand YAMLs exactly as the screen writes them (Target.ligand_yaml(smiles_used)), in CSV order; written once per tag."""
    d = LOGS / tag / "t400up_inputs"
    if not d.exists():
        import yaml
        from boltz_mlx.screen.inputs import load_target, read_ligands, validate_ligand

        d.mkdir(parents=True)
        tgt = load_target(str(T400_TARGET), msa=str(T400_MSA))
        ligs = [validate_ligand(x) for x in read_ligands(T400_CSV)]
        n = 0
        for lig in ligs:
            if lig.status == "invalid":
                continue
            (d / f"{lig.record_id}.yaml").write_text(yaml.safe_dump(tgt.ligand_yaml(lig.smiles_used), sort_keys=False))
            n += 1
        (d / "_order.json").write_text(json.dumps([x.record_id for x in ligs if x.status != "invalid"]))
    return [d / f"{r}.yaml" for r in json.loads((d / "_order.json").read_text())]


def cmd_t400up(a) -> int:
    base_tag = a.tag
    root = LOGS / base_tag
    root.mkdir(parents=True, exist_ok=True)
    ys = t400_inputs(base_tag)
    dec = root / "t400up_decision.json"
    if a.part == "first5":
        sel, name = ys[:5], "first5"
    else:
        if not dec.exists():
            print("no decision file: first5 job has not run; chunk job skipped")
            return 0
        d = json.loads(dec.read_text())
        if d["verdict"] != "CONTINUE":
            print(f"SKIPPED (estimate {d['est_wall_199_s']:.0f} s >= {EST_LIMIT_S:.0f} s): the measurement stays the 5-ligand set")
            return 0
        rest = ys[5:]
        k, n = a.index, 4
        sz = -(-len(rest) // n)
        sel, name = rest[(k - 1) * sz: k * sz], f"chunk{k}"
    if not sel:
        print("nothing to do")
        return 0
    run_dir = root / "t400up" / name
    indir = run_dir / "inputs"
    cmd, env = build_cmd("up200", indir, run_dir / "out", 0, run_dir / "up_timings.json")
    cmd += ["--preprocessing-threads", "4"]
    full = ["/usr/bin/time", "-l", *cmd]
    if a.dry_run:
        print("DRY", len(sel), "ligands", " ".join(full))
        return 0
    if run_dir.exists():
        raise SystemExit(f"{run_dir} exists: new --tag")
    indir.mkdir(parents=True)
    for y in sel:
        shutil.copy(y, indir / y.name)
    s0 = snap(f"t400up {name} start")
    t0 = time.monotonic()
    with open(run_dir / "stderr_time.txt", "w") as err, open(run_dir / "stdout.txt", "w") as o:
        rc = subprocess.run(full, cwd=ROOT, env=env, stdout=o, stderr=err).returncode
    W = time.monotonic() - t0
    s1 = snap(f"t400up {name} end")
    rep = parse_time_report((run_dir / "stderr_time.txt").read_text(errors="replace"))
    tim = json.loads((run_dir / "up_timings.json").read_text()) if (run_dir / "up_timings.json").exists() else {}
    calls = tim.get("calls", [])
    rec = {"part": name, "n_ligands": len(sel), "rc": rc, "W_s": round(W, 2), **rep, "n_calls": len(calls),
           "call_exceptions": sum(1 for c in calls if c.get("exception")), "M_s": tim.get("model_s"), "loads": tim.get("loads"),
           "process_inputs_s": tim.get("process_inputs_s"), "state_start": s0, "state_end": s1, "all_on_ac": _power_ok(s0, s1)}
    if calls:
        span = max(c["t1"] for c in calls) - min(c["t0"] for c in calls)
        rec["span_s"] = round(span, 2)
        rec["calls"] = calls
    (run_dir / "run.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k not in ("calls", "state_start", "state_end")}), flush=True)
    if a.part == "first5" and calls and rc == 0:
        # est_wall_199 = (W5 - span5') + 199 x span5' / 5, with span5' = first-call-start to last-call-end of this run (model calls
        # of all 5 ligands: 5 structure + 5 affinity, plus the checkpoint loads between them: loads are inside the W5 - span part only
        # once per run, which is what a single 199-ligand run would also pay) -> fixed rule of PREREG_MACBENCH.md section 8
        span = rec["span_s"]
        est = (W - span) + 199.0 * span / 5.0
        verdict = "CONTINUE" if est < EST_LIMIT_S else "STOP"
        dec.write_text(json.dumps({"verdict": verdict, "est_wall_199_s": round(est, 1), "W5_s": round(W, 2), "span5_s": span,
                                   "limit_s": EST_LIMIT_S, "label": "ESTIMATE (from the 5-ligand MEASURED run)"}, indent=1) + "\n")
        print(f"ESTIMATE est_wall_199 = {est:.0f} s ({est / 3600:.2f} h) -> {verdict} (limit {EST_LIMIT_S:.0f} s)", flush=True)
    return 0 if rc == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--tag", required=True)
    r.add_argument("--group", required=True, choices=sorted(GROUPS))
    r.add_argument("--idle-s", type=int, default=120)
    r.add_argument("--dry-run", action="store_true")
    t = sub.add_parser("t400up")
    t.add_argument("--tag", required=True)
    t.add_argument("--part", choices=["first5", "chunk"], required=True)
    t.add_argument("--index", type=int, default=1)
    t.add_argument("--dry-run", action="store_true")
    sub.add_parser("list")
    a = ap.parse_args()
    if a.cmd == "list":
        print(json.dumps({"arms": ARMS, "groups": GROUPS}, indent=1))
        return 0
    return cmd_run(a) if a.cmd == "run" else cmd_t400up(a)


if __name__ == "__main__":
    sys.exit(main())
