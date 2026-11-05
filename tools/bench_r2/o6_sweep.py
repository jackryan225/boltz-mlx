"""O6 size sweep (bench-r2): f16 model predict time and MLX peak memory from ~100 tokens up to the memory limit.

  python tools/bench_r2/o6_sweep.py run  --sweep-dir DIR --sizes 117,400,800,1000,1600 --note TAG [--steps-log FILE]
  python tools/bench_r2/o6_sweep.py plan --sweep-dir DIR --sizes ...          # what run would do; no prep, no GPU
  python tools/bench_r2/o6_sweep.py table --sweep-dir DIR                     # points so far, with the ESTIMATEs

Sizes (tokens -> case): 117 prot117, 400 prot400, 800 prot800, 1000 prot1000 (cases/, all msa: empty) and the generated
homo-oligomers of the prot400 construct 1600 (x4), 2400 (x6), 3200 (x8) in tools/bench_r2/cases/ (msa: empty). Missing
data/<case>/feats.npz is featurized first with tools/prep_case.py (CPU, before the GPU idle); the token count of every
case is checked against its size (exit 3 on mismatch).

Per size: one tools/bench_mlx.py process under /usr/bin/time -l: --mode f16 --steps 50 (3 recycles, 1 sample)
--runs R --mem-limit-gb 100 --cache-limit-gb 4 (MLX buffer cache capped at 4 GB like the CLI, so the process footprint
is the shipped one; the memory limit is the dispatcher budget, so MLX does not throttle below it). R = 3 up to 1000
tokens, 2 at 1600 / 2400, 1 at 3200 (--runs N=R overrides). bench_mlx's row goes to logs/bench_mlx.jsonl (note
"TAG o6 N=<size>"); this sweep's row (time, MLX peak, footprint, swap, power at both ends) to DIR/points.jsonl.

Guards (the sweep stops; no larger size runs in this or a later job of the same DIR, via DIR/STOP):
  * swap: a watchdog samples vm.swapusage every 2 s and kills the size's process group when swap used is more than
    --swap-kill-gb (2.0) above the value at this job's start (the dispatcher kills a class 1 job only at +4 GB +1 GB);
    after a size, swap grown more than --swap-stop-gb (0.5) since the job start also stops the sweep.
  * projection: before a size, its footprint is projected from the two largest finished sizes (activation memory =
    MLX peak - load_active_gb, power law with exponent max(measured, 2.0) capped at 3.0, ESTIMATE); above
    --max-proj-gb (100 = dispatcher budget) the size is skipped and the sweep stops.
  * time: a size whose projected time (same rule, exponent clamped to [2.0, 3.2]) does not fit before the job's
    deadline (--deadline-min after BENCH_R2_T0, minus 3 min) runs with fewer runs, or is skipped (no STOP: larger sizes
    in this job are skipped too); a size still running 60 s before the deadline is killed and recorded.
After a size whose wall time is >= --heavy-s (60) s the GPU idles --idle-s (300) s before the next one; the job's
first idle (300 s) comes after the prep. Exit 0 = ran or stopped/skipped by a guard (results), 1 = a size failed for
another reason, 3 = token count mismatch.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import state  # noqa: E402  (tools/bench_r2/state.py)

DATA = Path(os.environ.get("BOLTZ_MLX_DATA", Path(__file__).resolve().parents[2] / "data")).expanduser()  # = bench_mlx.DATA
CASES = {
    117: ("prot117", "cases/prot117.yaml"),
    400: ("prot400", "cases/prot400.yaml"),
    800: ("prot800", "cases/prot800.yaml"),
    1000: ("prot1000", "cases/prot1000.yaml"),
    1600: ("prot1600", "tools/bench_r2/cases/prot1600.yaml"),
    2400: ("prot2400", "tools/bench_r2/cases/prot2400.yaml"),
    3200: ("prot3200", "tools/bench_r2/cases/prot3200.yaml"),
}
RUNS = {117: 3, 400: 3, 800: 3, 1000: 3, 1600: 2, 2400: 2, 3200: 1}
EXPECT = json.loads((Path(__file__).resolve().parent / "expectations.json").read_text())["o6"]["sizes"]
STOP_FILE = "STOP"
CHILD: dict = {"p": None}


def log(msg: str) -> None:
    print(f"=== [{time.strftime('%H:%M:%SZ', time.gmtime())}] o6: {msg}", flush=True)


def snap(steps_log: str | None, label: str, **extra) -> dict:
    rec = state.snapshot(label)
    rec.update(extra)
    if steps_log:
        with open(steps_log, "a") as f:
            f.write(json.dumps(rec) + "\n")
    return rec


def load_points(d: Path) -> list:
    p = d / "points.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def write_point(d: Path, row: dict) -> None:
    with open(d / "points.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    print("o6 point: " + json.dumps({k: row.get(k) for k in ("size", "status", "reason", "runs", "median_runs2N_s",
                                                             "mlx_peak_gb", "footprint_gb", "swap_grew_gb",
                                                             "power_start", "power_end")}), flush=True)


def project(points: list, n: int) -> dict | None:
    """Footprint and per-run time at n tokens from the two largest finished sizes (ESTIMATE), or None."""
    ok = sorted((p for p in points if p.get("status") == "ok" and p.get("mlx_peak_gb")), key=lambda p: p["size"])
    ok = [p for p in ok if p["size"] < n]
    if len(ok) < 2:
        return None
    a, b = ok[-2], ok[-1]
    r = n / b["size"]
    act_a = max(a["mlx_peak_gb"] - (a.get("load_active_gb") or 0), 1e-3)
    act_b = max(b["mlx_peak_gb"] - (b.get("load_active_gb") or 0), 1e-3)
    e_m = math.log(act_b / act_a) / math.log(b["size"] / a["size"])
    e_m = min(max(e_m, 2.0), 3.0)
    act_n = act_b * r ** e_m
    fp_b = max(b.get("footprint_gb") or 0.0, b["mlx_peak_gb"])  # footprint includes the MLX (Metal) allocations
    fp_n = fp_b + (act_n - act_b)
    ta, tb = a.get("median_runs2N_s"), b.get("median_runs2N_s")
    e_t = min(max(math.log(tb / ta) / math.log(b["size"] / a["size"]), 2.0), 3.2) if ta and tb else 3.0
    t_n = tb * r ** e_t if tb else None
    return {"from_sizes": [a["size"], b["size"]], "mem_exponent": round(e_m, 3), "time_exponent": round(e_t, 3),
            "mlx_peak_gb": round(act_n + (b.get("load_active_gb") or 0), 2), "footprint_gb": round(fp_n, 2),
            "run_s": round(t_n, 1) if t_n else None, "label": "ESTIMATE"}


def tokens_of(case: str) -> int | None:
    f = DATA / case / "feats.npz"
    if not f.exists():
        return None
    import numpy as np
    with np.load(f) as z:
        return int(z["token_pad_mask"].sum())


def prep(n: int) -> int:
    case, yaml = CASES[n]
    if (DATA / case / "feats.npz").exists():
        return 0
    log(f"prep {case}: python tools/prep_case.py {yaml} (CPU featurization; data/{case}/ is new)")
    t = time.time()
    rc = subprocess.run([sys.executable, "tools/prep_case.py", yaml], cwd=ROOT).returncode
    log(f"prep {case} rc={rc} ({time.time() - t:.0f} s)")
    return rc


def idle(seconds: float, why: str, steps_log: str | None) -> None:
    s0 = snap(steps_log, f"o6 idle start ({why})", phase="idle_start", idle_s=seconds)
    log(f"GPU idle {seconds:.0f} s ({why}); power {s0['power_source']}, swap {s0['swap_used_gb']} GB")
    time.sleep(seconds)
    s1 = snap(steps_log, f"o6 idle end ({why})", phase="idle_end", idle_s=seconds)
    log(f"idle end; power {s1['power_source']}, GPU util {s1['gpu_util_pct']} %, swap {s1['swap_used_gb']} GB")


def _kill_child(sig=signal.SIGTERM) -> None:
    p = CHILD["p"]
    if p is not None and p.poll() is None:
        try:
            os.killpg(p.pid, sig)
        except (ProcessLookupError, PermissionError):
            pass


def _on_term(signum, _frame):
    _kill_child(signal.SIGTERM)
    time.sleep(2)
    _kill_child(signal.SIGKILL)
    sys.exit(128 + signum)


def run_size(a, d: Path, n: int, runs: int, swap0: float, deadline: float) -> dict:
    case, _ = CASES[n]
    stem = f"N{n}_{re.sub(r'[^A-Za-z0-9._-]', '_', a.note)}"
    out_f, err_f = d / f"{stem}.out", d / f"{stem}.err"
    note = f"{a.note} o6 N={n}"
    cmd = ["/usr/bin/time", "-l", sys.executable, "tools/bench_mlx.py", case, "--mode", a.mode, "--steps", str(a.steps),
           "--runs", str(runs), "--mem-limit-gb", str(a.mem_limit_gb), "--cache-limit-gb", str(a.cache_limit_gb),
           "--note", note]
    s0 = state.snapshot(f"o6 N={n} start")
    log(f"N={n} ({case}) runs {runs}: {' '.join(cmd[2:])}")
    t0 = time.time()
    swap_max, killed = s0["swap_used_gb"] or 0.0, None
    pw_counts, t_pw = {s0["power_source"]: 1}, time.time()
    with open(out_f, "w") as fo, open(err_f, "w") as fe:
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=fo, stderr=fe, start_new_session=True)
        CHILD["p"] = p
        while p.poll() is None:
            time.sleep(2)
            used, _ = state.swap_gb()
            if used is not None:
                swap_max = max(swap_max, used)
                if killed is None and used - swap0 > a.swap_kill_gb:
                    killed = f"swap used {used:.2f} GB = +{used - swap0:.2f} GB over the job start (> {a.swap_kill_gb})"
            if time.time() - t_pw >= 30:
                t_pw = time.time()
                src = state.power_source()
                pw_counts[src] = pw_counts.get(src, 0) + 1
            if killed is None and time.time() > deadline - 60:
                killed = "time: 60 s before the job deadline"
            if killed:
                log(f"N={n}: KILL ({killed})")
                _kill_child(signal.SIGTERM)
                for _ in range(20):
                    if p.poll() is not None:
                        break
                    time.sleep(1)
                _kill_child(signal.SIGKILL)
                p.wait()
                break
        CHILD["p"] = None
    wall = time.time() - t0
    s1 = state.snapshot(f"o6 N={n} end")
    out, err = out_f.read_text(errors="replace"), err_f.read_text(errors="replace")
    bench = None
    for line in reversed(out.splitlines()):
        if line.startswith("{"):
            try:
                bench = json.loads(line)
                break
            except json.JSONDecodeError:
                pass
    m_fp = re.search(r"(\d+)\s+peak memory footprint", err)
    m_rss = re.search(r"(\d+)\s+maximum resident set size", err)
    status = "ok" if (p.returncode == 0 and bench and not killed) else (
        "killed_swap" if killed and killed.startswith("swap") else "killed_time" if killed else "failed")
    row = {
        "sweep": d.name, "note": note, "size": n, "case": case, "status": status, "reason": killed,
        "rc": p.returncode, "runs_planned": runs, "mode": a.mode, "steps": a.steps,
        "mem_limit_gb": a.mem_limit_gb, "cache_limit_gb": a.cache_limit_gb,
        "utc_start": s0["utc"], "utc_end": s1["utc"], "wall_s": round(wall, 1),
        "footprint_gb": round(int(m_fp.group(1)) / 1e9, 3) if m_fp else None,
        "max_rss_gb_time": round(int(m_rss.group(1)) / 1e9, 3) if m_rss else None,
        "swap_job_start_gb": swap0, "swap_start_gb": s0["swap_used_gb"], "swap_end_gb": s1["swap_used_gb"],
        "swap_max_gb": swap_max, "swap_grew_gb": round((s1["swap_used_gb"] or swap0) - swap0, 3),
        "swap_max_grew_gb": round(max(swap_max, s1["swap_used_gb"] or 0.0) - swap0, 3),
        "power_start": s0["power_source"], "power_end": s1["power_source"],
        "power_samples": {**pw_counts, s1["power_source"]: pw_counts.get(s1["power_source"], 0) + 1},
        "system_in_w_start": s0["system_in_w"], "system_in_w_end": s1["system_in_w"],
        "mem_level_pct_end": s1["mem_level_pct"], "expect": EXPECT.get(str(n)),
        "out": str(out_f.relative_to(ROOT)) if out_f.is_relative_to(ROOT) else str(out_f),
    }
    if bench:
        row.update({
            "tokens": bench.get("tokens"), "atoms": bench.get("atoms"), "runs": len(bench.get("runs", [])),
            "median_runs2N_s": bench.get("median_runs2N_s"), "min_s": bench.get("min_s"),
            "run_s": [x.get("predict_s") for x in bench.get("runs", [])],
            "phases": bench.get("median_phases_runs2N"),
            "mlx_peak_gb": max((x.get("peak_mem_gb", 0) for x in bench.get("runs", [])), default=None),
            "load_active_gb": bench.get("load_active_gb"), "max_rss_gb": bench.get("max_rss_gb"),
            "git": bench.get("git"), "pkg_tree": bench.get("pkg_tree"), "fingerprint": bench.get("fingerprint"),
            "bench_power_start": (bench.get("machine_start") or {}).get("power"),
            "bench_power_end": (bench.get("machine_end") or {}).get("power"),
        })
    else:
        row["stderr_tail"] = err.strip().splitlines()[-12:]
    row["timing_clean"] = bool(status == "ok" and row["swap_max_grew_gb"] <= 0.25
                               and set(row["power_samples"]) == {"AC Power"})
    return row


def parse_runs(items: list) -> dict:
    out = dict(RUNS)
    for it in items or []:
        k, v = it.split("=")
        out[int(k)] = int(v)
    return out


def table(d: Path) -> None:
    pts = load_points(d)
    print(f"O6 sweep {d.name}: f16 @50 steps, 3 recycles, 1 sample; median of runs 2..N (1 run: that run); "
          f"MLX peak = max over runs of mx.get_peak_memory; footprint = /usr/bin/time -l peak memory footprint")
    print("| tokens | case | status | runs | median s | min s | trunk s | diffusion s | MLX peak GB | footprint GB | "
          "swap grew GB | power (samples, every 30 s) | timing clean | ESTIMATE s/run | ESTIMATE MLX peak GB |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p in sorted(pts, key=lambda p: (p.get("size", 0), p.get("utc_start", ""))):
        ph = p.get("phases") or {}
        e = p.get("expect") or EXPECT.get(str(p.get("size"))) or {}
        print(f"| {p.get('size')} | {p.get('case')} | {p.get('status')}{(' (' + p['reason'] + ')') if p.get('reason') else ''} | "
              f"{p.get('runs', '-')} | {p.get('median_runs2N_s', '-')} | {p.get('min_s', '-')} | {ph.get('trunk_s', '-')} | "
              f"{ph.get('diffusion_s', '-')} | {p.get('mlx_peak_gb', '-')} | {p.get('footprint_gb', '-')} | "
              f"{p.get('swap_max_grew_gb', '-')} | {p.get('power_samples', '-')} | "
              f"{p.get('timing_clean', '-')} | {e.get('run_s', '-')} | {e.get('mlx_peak_gb', '-')} |")
    if (d / STOP_FILE).exists():
        print(f"STOP: {(d / STOP_FILE).read_text().strip()}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "plan", "table"])
    ap.add_argument("--sweep-dir", required=True)
    ap.add_argument("--sizes", default="117,400,800,1000,1600")
    ap.add_argument("--note", default="bench-r2-o6")
    ap.add_argument("--steps-log", default=None)
    ap.add_argument("--runs", action="append", help="N=R: runs at size N (default 3 up to 1000, 2 at 1600/2400, 1 at 3200)")
    ap.add_argument("--mode", default="f16")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--mem-limit-gb", type=float, default=100)
    ap.add_argument("--cache-limit-gb", type=float, default=4)
    ap.add_argument("--idle-s", type=float, default=300)
    ap.add_argument("--heavy-s", type=float, default=60)
    ap.add_argument("--swap-kill-gb", type=float, default=2.0)
    ap.add_argument("--swap-stop-gb", type=float, default=0.5)
    ap.add_argument("--max-proj-gb", type=float, default=100)
    ap.add_argument("--deadline-min", type=float, default=45)
    a = ap.parse_args()
    d = Path(a.sweep_dir)
    if not d.is_absolute():
        d = ROOT / d
    if a.cmd == "table":
        table(d)
        return 0
    sizes = [int(x) for x in a.sizes.split(",")]
    for n in sizes:
        if n not in CASES:
            sys.exit(f"o6: no case for size {n} (have {sorted(CASES)})")
    runs = parse_runs(a.runs)
    points = load_points(d)
    stop = (d / STOP_FILE).read_text().strip() if (d / STOP_FILE).exists() else None
    if a.cmd == "plan":
        print(f"o6 plan: sweep {d} (exists: {d.exists()}, points {len(points)}, STOP: {stop})")
        for n in sizes:
            case, yaml = CASES[n]
            print(f"  N={n}: {case} ({yaml}; data {'present' if (DATA / case / 'feats.npz').exists() else 'MISSING -> prep'}) "
                  f"runs {runs[n]}; ESTIMATE {EXPECT.get(str(n))}; projection from points: {project(points, n)}")
        return 0

    t_job0 = float(os.environ.get("BENCH_R2_T0", time.time()))
    deadline = t_job0 + a.deadline_min * 60 - 180
    d.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)
    if stop:
        log(f"SKIPPED: the sweep was stopped by an earlier size: {stop}")
        for n in sizes:
            write_point(d, {"sweep": d.name, "note": f"{a.note} o6 N={n}", "size": n, "case": CASES[n][0],
                            "status": "skipped_stop", "reason": stop, "utc_start": state.snapshot()["utc"]})
        table(d)
        return 0
    first = project(points, sizes[0])
    if first and first["footprint_gb"] > a.max_proj_gb:
        reason = (f"N={sizes[0]} projected footprint {first['footprint_gb']} GB > {a.max_proj_gb} GB "
                  f"(ESTIMATE from sizes {first['from_sizes']})")
        log(f"SKIPPED before prep: {reason}")
        (d / STOP_FILE).write_text(reason + "\n")
        for n in sizes:
            write_point(d, {"sweep": d.name, "note": f"{a.note} o6 N={n}", "size": n, "case": CASES[n][0],
                            "status": "skipped_projection", "reason": reason, "projection": project(points, n),
                            "utc_start": state.snapshot()["utc"]})
        table(d)
        return 0

    for n in sizes:  # CPU first, then the GPU idle
        if prep(n) != 0:
            log(f"prep failed for N={n}")
            return 1
        tok = tokens_of(CASES[n][0])
        if tok != n:
            log(f"TOKEN MISMATCH: data/{CASES[n][0]}/feats.npz has {tok} tokens, expected {n}")
            return 3
    swap0 = state.swap_gb()[0] or 0.0
    snap(a.steps_log, "o6 job start", phase="o6_start", swap_job_start_gb=swap0)
    idle(a.idle_s, "job start, after prep", a.steps_log)
    failed = False
    for i, n in enumerate(sizes):
        proj = project(load_points(d), n)
        if proj and proj["footprint_gb"] > a.max_proj_gb:
            reason = f"N={n} projected footprint {proj['footprint_gb']} GB > {a.max_proj_gb} GB (ESTIMATE from sizes {proj['from_sizes']})"
            log(f"STOP: {reason}")
            (d / STOP_FILE).write_text(reason + "\n")
            write_point(d, {"sweep": d.name, "note": f"{a.note} o6 N={n}", "size": n, "case": CASES[n][0],
                            "status": "skipped_projection", "reason": reason, "projection": proj,
                            "utc_start": state.snapshot()["utc"]})
            break
        r = runs[n]
        if proj and proj.get("run_s"):
            left = deadline - time.time()
            while r > 1 and 30 + r * proj["run_s"] > left:
                r -= 1
            if 30 + r * proj["run_s"] > left:
                reason = f"N={n} projected {proj['run_s']} s/run does not fit in the {left:.0f} s left (ESTIMATE)"
                log(f"SKIPPED (time; no STOP): {reason}")
                for m in sizes[i:]:
                    write_point(d, {"sweep": d.name, "note": f"{a.note} o6 N={m}", "size": m, "case": CASES[m][0],
                                    "status": "skipped_time", "reason": reason, "projection": project(load_points(d), m),
                                    "utc_start": state.snapshot()["utc"]})
                break
            if r != runs[n]:
                log(f"N={n}: runs {runs[n]} -> {r} (projected {proj['run_s']} s/run, {left:.0f} s left)")
        row = run_size(a, d, n, r, swap0, deadline)
        row["projection"] = proj
        write_point(d, row)
        if row.get("tokens") not in (None, n):
            log(f"TOKEN MISMATCH in the bench row: {row.get('tokens')} != {n}")
            return 3
        if row["status"] == "killed_swap" or row["swap_max_grew_gb"] > a.swap_stop_gb:
            reason = (f"N={n}: {row['reason']}" if row["status"] == "killed_swap"
                      else f"N={n}: swap grew {row['swap_max_grew_gb']} GB since the job start (> {a.swap_stop_gb})")
            log(f"STOP on swap growth: {reason}")
            (d / STOP_FILE).write_text(reason + "\n")
            break
        if row["status"] == "killed_time":
            break
        if row["status"] == "failed":
            failed = True
            log(f"N={n} failed rc={row['rc']}: {row.get('stderr_tail')}")
            break
        if i + 1 < len(sizes) and row["wall_s"] >= a.heavy_s:
            idle(a.idle_s, f"after N={n} ({row['wall_s']:.0f} s)", a.steps_log)
    snap(a.steps_log, "o6 job end", phase="o6_end")
    table(d)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
