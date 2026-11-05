"""Reads logs/cachelimit_ab.jsonl (written by tools/prof/cachelimit_ab.py) and applies the PRE-REGISTERED decision rule.

  python tools/prof/cachelimit_jobs/summary.py [--rows logs/cachelimit_ab.jsonl] [--git HEAD-short] [--part b|c|all] [--require-b]

Rule (fixed before any data; no threshold tuning afterwards):
  SHIP  iff  (1) bit-identity: job b's A and B predictions (1600 tokens, 50 steps, f16) have identical digests for every
                 output array (coordinates, confidence outputs, pdistogram, s, z), AND the B arm really ran a larger cache
                 (limit and cache held after the run above arm A's; otherwise the test is vacuous -> INCONCLUSIVE),
             (2) speed: at BOTH 1600 and 2400 tokens, the pooled median predict_s of arm B (all B runs of that size) is lower
                 than the pooled median of arm A,
             and no digest differs between any two runs of the same size in job c (a stricter reading of (1); it can only block).
  NO-SHIP if (1) fails (hard stop), or (2) fails at either size.
  INCONCLUSIVE if rows are missing, fewer than 2 runs per arm at a size, or (1) is vacuous.
Effect size: (median B / median A - 1) in %, per-arm min / max / MAD, per-ABBA-block B-vs-A, system-CPU share of the wall.
Rows are selected by the git short hash (default: HEAD of this checkout); the last row of each label wins.
"""
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SIZES = ("1600", "2400")


def load_rows(path: Path, git: str) -> dict:
    rows = {}
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(r.get("git", "")).split("-")[0] == git and r.get("label"):
                rows[r["label"]] = r  # last wins
    return rows


def _med(xs):
    return statistics.median(xs)


def _mad(xs):
    m = _med(xs)
    return statistics.median([abs(x - m) for x in xs])


def analyse_b(rows: dict) -> dict:
    """Bit-identity (job b): labels cl-b-A (arm A alone), cl-b-B (arm B alone), cl-b-BA (B then A in one process)."""
    ra, rb, rba = rows.get("cl-b-A"), rows.get("cl-b-B"), rows.get("cl-b-BA")
    if not ra or not rb:
        return {"verdict": "MISSING", "why": "need rows cl-b-A and cl-b-B"}
    a, b = ra["runs"][0], rb["runs"][0]
    if "digests" not in a or "digests" not in b:
        return {"verdict": "MISSING", "why": "rows carry no digests (--hash)"}
    pairs = [("A alone", a, "B alone", b)]
    if rba and len(rba["runs"]) == 2:
        pairs += [("B alone", b, "B (1st of BA)", rba["runs"][0]), ("A alone", a, "A (2nd of BA)", rba["runs"][1])]
    diffs = []
    for na, x, nb, y in pairs:
        if x["digests"].keys() != y["digests"].keys():
            diffs.append(f"{na} vs {nb}: key sets differ")
        diffs += [f"{na} vs {nb}: {k}" for k in sorted(x["digests"]) if x["digests"].get(k) != y["digests"].get(k)]
    nonvac = b["limit_gb"] > a["limit_gb"] and b["cache_after_gb"] > a["cache_after_gb"]
    out = {"A_limit_gb": a["limit_gb"], "B_limit_gb": b["limit_gb"], "A_cache_after_gb": a["cache_after_gb"],
           "B_cache_after_gb": b["cache_after_gb"], "arrays_compared": len(a["digests"]), "differing": diffs,
           "tokens": ra.get("tokens"), "steps": ra.get("steps")}
    if diffs:
        out["verdict"] = "DIFFERENT"
    elif not nonvac:
        out["verdict"] = "VACUOUS"
        out["why"] = "arm B did not run a larger cache than arm A (policy value == 4 GB or limit not effective)"
    else:
        out["verdict"] = "IDENTICAL"
    return out


def analyse_c(rows: dict) -> dict:
    res = {}
    for n in SIZES:
        r = rows.get(f"cl-c-{n}")
        if not r:
            res[n] = {"verdict": "MISSING"}
            continue
        A = [x for x in r["runs"] if x["arm"] == "A"]
        B = [x for x in r["runs"] if x["arm"] == "B"]
        if min(len(A), len(B)) < 2:
            res[n] = {"verdict": "TOO_FEW", "runs_A": len(A), "runs_B": len(B)}
            continue
        ta, tb = [x["predict_s"] for x in A], [x["predict_s"] for x in B]
        blocks = []
        for i in range(0, len(r["runs"]) - 3, 4):  # ABBA blocks: B-vs-A within each block
            blk = r["runs"][i:i + 4]
            xa = [x["predict_s"] for x in blk if x["arm"] == "A"]
            xb = [x["predict_s"] for x in blk if x["arm"] == "B"]
            if xa and xb:
                blocks.append(round(100 * (statistics.mean(xb) / statistics.mean(xa) - 1), 1))
        digs = {x.get("digest") for x in r["runs"] if x.get("digest")}
        ma, mb = _med(ta), _med(tb)
        res[n] = {"verdict": "FASTER" if mb < ma else "NOT_FASTER", "tokens": r.get("tokens"), "runs_A": len(A), "runs_B": len(B),
                  "A_s": ta, "B_s": tb, "A_median": round(ma, 2), "B_median": round(mb, 2),
                  "effect_pct": round(100 * (mb / ma - 1), 2), "A_min_max_mad": [min(ta), max(ta), round(_mad(ta), 2)],
                  "B_min_max_mad": [min(tb), max(tb), round(_mad(tb), 2)], "block_effect_pct": blocks,
                  "trunk_median_A": round(_med([x["trunk_s"] for x in A]), 2), "trunk_median_B": round(_med([x["trunk_s"] for x in B]), 2),
                  "diffusion_median_A": round(_med([x["diffusion_s"] for x in A]), 2),
                  "diffusion_median_B": round(_med([x["diffusion_s"] for x in B]), 2),
                  "sys_share_A": round(_med([x["sys_share"] for x in A]), 3), "sys_share_B": round(_med([x["sys_share"] for x in B]), 3),
                  "B_limit_gb": sorted({x["limit_gb"] for x in B}), "digests_distinct": len(digs),
                  "fp16_fallbacks": sorted({json.dumps(x.get("guard_counts")) for x in r["runs"]})}
    return res


def decide(b: dict, c: dict) -> str:
    if b["verdict"] == "DIFFERENT" or any(v.get("digests_distinct", 1) > 1 for v in c.values()):
        return "NO-SHIP (bit-identity failed)"
    if b["verdict"] in ("MISSING", "VACUOUS") or any(v["verdict"] in ("MISSING", "TOO_FEW") for v in c.values()):
        return "INCONCLUSIVE"
    if all(v["verdict"] == "FASTER" for v in c.values()):
        return "SHIP"
    return "NO-SHIP (B not faster at every size)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=str(ROOT / "logs" / "cachelimit_ab.jsonl"))
    ap.add_argument("--git", default=None)
    ap.add_argument("--part", default="all", choices=["b", "c", "all"])
    ap.add_argument("--require-b", action="store_true", help="exit 4 unless job b says IDENTICAL (gate for running job c)")
    a = ap.parse_args()
    git = a.git or subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], text=True).strip()
    rows = load_rows(Path(a.rows), git)
    print(f"cachelimit summary: git {git}, labels {sorted(rows)}")
    b = analyse_b(rows) if a.part in ("b", "all") or a.require_b else None
    c = analyse_c(rows) if a.part in ("c", "all") else None
    if b is not None:
        print("BIT-IDENTITY (job b): " + json.dumps(b))
    if c is not None:
        for n, v in c.items():
            print(f"SPEED {n} tokens: " + json.dumps(v))
    if a.part == "all":
        print("DECISION: " + decide(b, c))
    if a.require_b and (b is None or b["verdict"] != "IDENTICAL"):
        print("require-b: job b is not IDENTICAL; not spending GPU time on job c")
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
