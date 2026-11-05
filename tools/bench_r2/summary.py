"""Summaries of the bench-r2 jobs against tools/bench_r2/expectations.json (information only, exit 0 unless no data).

  python tools/bench_r2/summary.py speed    --tag TAG --out OUT     # logs/bench_mlx.jsonl rows with note TAG
  python tools/bench_r2/summary.py t400m    --out OUT               # OUT/screen_{preset,50_200,strict}/_measure/run
  python tools/bench_r2/summary.py t400cold --out OUT               # OUT/cold (full cold screen) + OUT/cold.time

Power per row: bench rows carry pmset at their start and end (machine_start / machine_end); screen rows (one per
ligand) take every power source of the OUT/power.jsonl samples (tools/bench_r2/state.py sample, every 30 s) over the
span the row's work ran in: the whole run for t400m, the row's chunk for t400cold (a chunk's structure stage runs before
any of its rows end, so the power at t_done alone can miss it). Steps carry OUT/steps.jsonl snapshots at both ends.
Writes OUT/summary_<job>.json.
"""

from __future__ import annotations

import argparse
import bisect
import json
import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = json.loads((Path(__file__).resolve().parent / "expectations.json").read_text())


def jl(path: Path) -> list:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def delta(x, ref):
    return f"{100 * (x / ref - 1):+.1f} %" if (x and ref) else "-"


def footprint(time_file: Path):
    if not time_file.exists():
        return None
    m = re.search(r"(\d+)\s+peak memory footprint", time_file.read_text(errors="replace"))
    return round(int(m.group(1)) / 1e9, 3) if m else None


class Power:
    def __init__(self, out: Path):
        self.s = sorted(jl(out / "power.jsonl"), key=lambda r: r["t"])
        self.t = [r["t"] for r in self.s]

    def at(self, t: float):
        i = bisect.bisect_right(self.t, t) - 1
        return self.s[i]["power_source"] if i >= 0 else None

    def window(self, t0: float, t1: float) -> list:
        return sorted({r["power_source"] for r in self.s if t0 <= r["t"] <= t1})


def steps(out: Path) -> dict:
    d: dict = {}
    for r in jl(out / "steps.jsonl"):
        if r.get("step"):
            d.setdefault(r["step"], {})[r.get("phase")] = r
    return d


def speed(tag: str, out: Path) -> int:
    rows = [r for r in jl(ROOT / "logs" / "bench_mlx.jsonl") if r.get("note") == tag]
    e = EXP["speed"]
    res = []
    print(f"bench-r2 SPEED ({tag}): f16, median of runs 2..N; expectations tools/bench_r2/expectations.json (97d7b4a)")
    print("| case | runs | median 2..N s | min s | trunk / diffusion / confidence s | MLX peak GB | fingerprint | "
          "power start / end | expected s | delta | label |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        key = f"{r['case']}_{'aff' + str(r['steps']) if r['affinity'] else 's' + str(r['steps'])}"
        x = e.get(key, {})
        ph = r.get("median_phases_runs2N", {})
        peak = max(q["peak_mem_gb"] for q in r["runs"])
        p0 = (r.get("machine_start") or {}).get("power", "")
        p1 = (r.get("machine_end") or {}).get("power", "")
        p0 = re.findall(r"'([^']+)'", p0)[0] if "'" in p0 else p0
        p1 = re.findall(r"'([^']+)'", p1)[0] if "'" in p1 else p1
        print(f"| {key} | {len(r['runs'])} | {r['median_runs2N_s']:.3f} | {r['min_s']:.3f} | "
              f"{ph.get('trunk_s')} / {ph.get('diffusion_s')} / {ph.get('confidence_s')} | {peak} | {r.get('fingerprint')} | "
              f"{p0} / {p1} | {x.get('s', '-')} | {delta(r['median_runs2N_s'], x.get('s'))} | {x.get('label', '-')} |")
        res.append({"key": key, "median_runs2N_s": r["median_runs2N_s"], "min_s": r["min_s"], "phases": ph,
                    "mlx_peak_gb": peak, "fingerprint": r.get("fingerprint"), "power_start": p0, "power_end": p1,
                    "expected": x, "date": r.get("date"), "git": r.get("git"), "pkg_tree": r.get("pkg_tree")})
    if rows:
        d = rows[-1].get("device") or {}
        print(f"device: {d.get('chip')} {d.get('architecture')} path {d.get('path')} macOS {d.get('macos')}; git "
              f"{rows[-1]['git']} pkg_tree {rows[-1]['pkg_tree']}; swap start {rows[0]['machine_start'].get('swap')}; "
              f"swap end {rows[-1]['machine_end'].get('swap')}")
    else:
        print(f"no bench rows with note {tag!r}")
    (out / "summary_speed.json").write_text(json.dumps({"tag": tag, "rows": res}, indent=1) + "\n")
    return 0 if rows else 1


def _screen_rows(run: Path) -> list:
    return [r for r in jl(run / "results.jsonl") if r.get("status") == "ok"]


def t400m(out: Path) -> int:
    pw, stp = Power(out), steps(out)
    e = EXP["t400m"]
    res, any_ok = [], False
    print("bench-r2 T400 --measure 5 (cold process, s/ligand incl. one load of each checkpoint)")
    print("| run | preset | structure / affinity steps | ligands ok | wall s | s/ligand | median structure s | "
          "median affinity s | peak footprint GB | power (rows) | expected s/ligand | delta | label |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, key in (("screen_preset", "preset_20_25"), ("screen_50_200", "old_50_200"), ("screen_strict", "strict_200_200")):
        run = out / name / "_measure" / "run"
        x = e.get(key, {})
        try:
            stt = json.loads((run / "screen_status.json").read_text())
        except FileNotFoundError:
            print(f"| {name} | - | - | no screen_status.json (not run or failed) | | | | | | | {x.get('s_per_ligand')} | - | {x.get('label')} |")
            continue
        ok = _screen_rows(run)
        any_ok = any_ok or bool(ok)
        spl = stt.get("s_per_ligand_this_run")
        # power over the whole run (one process), not at each row's t_done: the structure stage runs before any row ends
        s = stp.get(name, {})
        a, b = (s.get("start") or {}).get("t"), (s.get("end") or {}).get("t")
        pows = pw.window(a, b) if (a and b) else []
        if not pows and s:
            pows = sorted({s.get("start", {}).get("power_source"), s.get("end", {}).get("power_source")} - {None})
        steps_ = (sorted({r.get("steps") for r in ok}), sorted({r.get("affinity_steps") for r in ok}))
        presets = sorted({r.get("preset") for r in ok})
        fp = footprint(out / f"{name}.time")
        print(f"| {name} | {presets} | {steps_[0]} / {steps_[1]} | {stt.get('ok_this_run')} | {stt.get('wall_s')} | {spl} | "
              f"{st.median(r['seconds_structure'] for r in ok):.2f} | {st.median(r['seconds_affinity'] for r in ok):.2f} | "
              f"{fp} | {pows} | {x.get('s_per_ligand')} | {delta(spl, x.get('s_per_ligand'))} | {x.get('label')} |")
        res.append({"run": name, "key": key, "status": stt, "s_per_ligand": spl, "presets": presets, "steps": steps_,
                    "peak_footprint_gb": fp, "power_rows": pows, "expected": x,
                    "per_ligand": [{k: r.get(k) for k in ("ligand_id", "n_tokens", "seconds_structure", "seconds_affinity",
                                                          "affinity_pred_value", "affinity_probability_binary", "iptm")}
                                   for r in ok]})
    (out / "summary_t400m.json").write_text(json.dumps({"runs": res}, indent=1) + "\n")
    return 0 if any_ok else 1


def t400cold(out: Path) -> int:
    pw, stp = Power(out), steps(out)
    x = EXP["t400cold"]
    run = out / "cold"
    try:
        stt = json.loads((run / "screen_status.json").read_text())
    except FileNotFoundError:
        stt = {}
    ok = _screen_rows(run)
    t0 = (stp.get("cold", {}).get("start") or {}).get("t")
    csv_t = (run / "results.csv").stat().st_mtime if (run / "results.csv").exists() else None
    wall = (csv_t - t0) if (csv_t and t0) else None
    done = sorted(r["t_done"] for r in ok if "t_done" in r)

    def rate(a, b):
        n = sum(1 for t in done if a <= t < b)
        return round((b - a) / n, 3) if n else None

    # A chunked screen runs structure for a whole chunk, then affinity row by row, so a row's work starts when its chunk
    # starts, well before its t_done. Label each row by every power source sampled over its chunk's span (chunk k runs
    # from the end of chunk k-1, or t0, to its last t_done). Chunk boundaries: t_done gaps above max(30 s, 5x the median).
    gaps = [b - a for a, b in zip(done, done[1:])]
    cut = max(30.0, 5 * st.median(gaps)) if gaps else 30.0
    spans, a = [], t0 if t0 else (done[0] if done else 0)
    for i, t in enumerate(done):
        if i + 1 == len(done) or done[i + 1] - t > cut:
            spans.append((a, t))
            a = t
    by_src: dict = {}
    for r in ok:
        t = r.get("t_done")
        lo, hi = next(((x, y) for x, y in spans if x <= t <= y), (t, t)) if t else (0, 0)
        src = pw.window(lo, hi) or ([pw.at(hi)] if pw.at(hi) else [])
        by_src.setdefault(" + ".join(src) or "?", []).append(r["ligand_id"])
    spl = round(wall / len(ok), 3) if (wall and ok) else None
    res = {
        "ligands_ok": len(ok), "rows_total": len(jl(run / "results.jsonl")), "status": stt,
        "cold_wall_s": round(wall, 1) if wall else None, "s_per_ligand_cold": spl,
        "s_per_ligand_status": stt.get("s_per_ligand_this_run"),
        "s_per_ligand_first_20min": rate(t0, t0 + 1200) if t0 else None,
        "s_per_ligand_last_20min": rate(done[-1] - 1200, done[-1] + 1e-6) if done else None,
        "peak_footprint_gb": footprint(out / "cold.time"),
        "presets": sorted({r.get("preset") for r in ok}),
        "steps": [sorted({r.get("steps") for r in ok}), sorted({r.get("affinity_steps") for r in ok})],
        "seeds": sorted({r.get("seed") for r in ok}),
        "power_rows": {k: len(v) for k, v in by_src.items()},
        "rows_not_on_ac": sorted(i for k, v in by_src.items() if k != "AC Power" for i in v),
        "median_structure_s": round(st.median(r["seconds_structure"] for r in ok), 3) if ok else None,
        "median_affinity_s": round(st.median(r["seconds_affinity"] for r in ok), 3) if ok else None,
        "expected": x,
    }
    print("bench-r2 T400 full cold pass (wall = process start -> results.csv written, as rung 5 G1)")
    print(json.dumps({k: v for k, v in res.items() if k not in ("status", "rows_not_on_ac")}, indent=1))
    print(f"vs expected {x['s_per_ligand']} s/ligand ({x['label']}): {delta(spl, x['s_per_ligand'])}; peak footprint "
          f"{res['peak_footprint_gb']} GB vs {x['peak_footprint_gb']} GB")
    (out / "summary_t400cold.json").write_text(json.dumps(res, indent=1) + "\n")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("job", choices=["speed", "t400m", "t400cold"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    out = Path(a.out)
    if not out.is_absolute():
        out = ROOT / out
    if a.job == "speed":
        return speed(a.tag, out)
    return t400m(out) if a.job == "t400m" else t400cold(out)


if __name__ == "__main__":
    sys.exit(main())
