"""In-model A/B of one f16 kernel flag: fresh processes of tools/prof/f16_smoke.py (load model, 3 predictions of the
case @50 steps, 3 recycles) with the flag on / off in ABBA order (on, off, off, on), per case; the warm runs (1, 2) of
every process are the samples. GPU, one process at a time.

  python tools/prof/pair_ab_inmodel.py                       # BOLTZ_MLX_NAX_TAIL3, prot400 + msa384 (~5 GPU min)
  FLAG=BOLTZ_MLX_NAX_TAIL3 python tools/prof/pair_ab_inmodel.py prot400

Rows -> logs/pair_profile.jsonl (exp pair_ab_inmodel / pair_ab_inmodel_summary). Verdict: the flag ships only if the
median of its warm runs is below the off median on every case (a slower or equal case = no win).
TOL=0.01 (R8-coldN, flags whose point is not steady-state speed): also prints NO REGRESSION when every case's on-median
is within +TOL (relative) of the off-median (|delta| <= 1% or faster), else REGRESSION.
"""

from __future__ import annotations

import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "logs" / "pair_profile.jsonl"
FLAG = os.environ.get("FLAG", "BOLTZ_MLX_NAX_TAIL3")
TOL = float(os.environ.get("TOL", "0"))
GIT = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()


def emit(row: dict):
    row = {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "git": GIT, **row}
    print(json.dumps(row), flush=True)
    with open(OUT, "a") as fo:
        fo.write(json.dumps(row) + "\n")


def one(case: str, value: str) -> list[float]:
    env = {**os.environ, "BOLTZ_MLX_PRECISION": "f16", FLAG: value, "PYTHONPATH": str(ROOT)}
    r = subprocess.run([sys.executable, "tools/prof/f16_smoke.py", case], cwd=ROOT, env=env, capture_output=True,
                       text=True)
    runs = {int(m.group(1)): float(m.group(2)) for m in re.finditer(r"^run(\d) ([\d.]+)s", r.stdout, re.M)}
    warm = [runs[k] for k in sorted(runs) if k >= 1]
    row = {"exp": "pair_ab_inmodel", "case": case, "flag": FLAG, "value": value, "runs": runs, "warm": warm,
           "rc": r.returncode}
    if r.returncode != 0 or not warm:
        row["tail"] = (r.stdout + r.stderr)[-1500:]
    emit(row)
    return warm


def main():
    cases = sys.argv[1:] or ["prot400", "msa384"]
    verdicts, noreg = {}, {}
    for case in cases:
        res = {"1": [], "0": []}
        for v in ("1", "0", "0", "1"):  # ABBA
            res[v] += one(case, v)
        if res["1"] and res["0"]:
            on, off = statistics.median(res["1"]), statistics.median(res["0"])
            verdicts[case] = on < off
            noreg[case] = on <= off * (1.0 + TOL)
            emit({"exp": "pair_ab_inmodel_summary", "case": case, "flag": FLAG, "on_warm": res["1"], "off_warm": res["0"],
                  "median_on_s": round(on, 3), "median_off_s": round(off, 3), "delta_s": round(on - off, 3),
                  "delta_rel": round(on / off - 1.0, 4), "win": on < off, "tol": TOL, "no_regression": noreg[case]})
        else:
            verdicts[case] = False
            noreg[case] = False
    win = bool(verdicts) and all(verdicts.values())
    ok = bool(noreg) and all(noreg.values())
    emit({"exp": "pair_ab_inmodel_summary", "flag": FLAG, "cases": verdicts, "WIN": win, "tol": TOL,
          "no_regression_cases": noreg, "NO_REGRESSION": ok})
    print("WIN" if win else "NO WIN")
    if TOL > 0:
        print("NO REGRESSION" if ok else "REGRESSION")


if __name__ == "__main__":
    main()
