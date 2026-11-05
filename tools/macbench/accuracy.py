"""Accuracy rules S, E, C, A of PREREG_MACBENCH.md section 5 over the runs of tools/macbench/driver.py (CPU, numpy + gemmi only).

  python tools/macbench/accuracy.py --logs logs/macbench [--tags TAG ...] [--ref up200] [--out accuracy.json] [--md accuracy.md]
  python tools/macbench/accuracy.py --selftest        # synthetic checks on the committed reference files, no run data needed

Run layout: <logs>/<tag>/<arm>/<input>/seed<k>/{run.json, out/boltz_results_<input>/predictions/<input>/...}. Several tags may supply the
same (arm, input, seed): the latest tag (sorted by name) wins and the choice is reported. Thresholds are the pre-registered ones and are
constants below; they are never fit. Metrics are tools/validation/score.py's (score_arrays: CA-lDDT, TM-score, CA RMSD).
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "tools" / "validation"))
import score  # noqa: E402

REF_DIR = HERE / "ref"

# ---- pre-registered constants (PREREG_MACBENCH.md section 5) -------------------------------------------------------------------
S_RATIO = 1.25        # rule S: dissimilarity to upstream <= 1.25 x upstream's own seed-to-seed dissimilarity ...
S_FLOOR = 0.01        # ... or that plus 0.01, whichever is larger
E_FLOOR = 0.02        # rule E: candidate experimental score >= upstream's - max(0.02, range of upstream seed values)
C_WIDEN = 0.02        # rule C: candidate seed-mean inside [min, max] of the reference seeds, widened by 0.02
A_RATIO = 1.25        # rule A: RMS(group mean diffs) <= 1.25 x sigma_pool x sqrt(2/3)
A_FLOOR = {"affinity_pred_value": 0.02, "affinity_probability_binary": 0.01}
SEEDS = (0, 1, 2)

# chain groups (identical-sequence chains share a group), reference structure and chains for rule E, ligand / affinity flags
CASES = {
    "prot117": dict(groups=[["A"]], ref=None, ligand=False, affinity=False),
    "prot400": dict(groups=[["A"]], ref=None, ligand=False, affinity=False),
    "lig_aff": dict(groups=[["A"]], ref=None, ligand=True, affinity=True),
    "8rf4": dict(groups=[["A"]], ref="8RF4", ligand=True, affinity=False),
    "8rf4aff": dict(groups=[["A"]], ref="8RF4", ligand=True, affinity=True),
    "ntca_aff": dict(groups=[["A", "B"]], ref="9GUI", ligand=True, affinity=True),
    "9moj": dict(groups=[["A", "B"], ["C", "D"]], ref="9MOJ", ligand=False, affinity=False),
    "9b3h": dict(groups=[["A"], ["B"]], ref="9B3H", ligand=False, affinity=False),
}


def chain_lengths(case: str) -> dict:
    """{chain id: protein sequence length} from the case YAML (tools/macbench/cases/<case>_empty.yaml)."""
    import yaml

    doc = yaml.safe_load((HERE / "cases" / f"{case}_empty.yaml").read_text())
    out = {}
    for e in doc["sequences"]:
        if "protein" in e:
            ids = e["protein"]["id"]
            for c in (ids if isinstance(ids, list) else [ids]):
                out[c] = len(e["protein"]["sequence"])
    return out


def case_of(input_name: str) -> str:
    for suf in ("_empty", "_msa"):
        if input_name.endswith(suf):
            return input_name[: -len(suf)]
    return input_name


# ---- coordinates ---------------------------------------------------------------------------------------------------------------
_CACHE: dict = {}


def chain_ca(path: Path, chain: str, length: int | None = None) -> np.ndarray:
    key = (str(path), chain)
    if key not in _CACHE:
        _CACHE[key] = score.load_ca(path, chain, length=length)
    return _CACHE[key]


def complex_ca(path: Path, groups: list, lengths: dict | None = None) -> dict:
    """{chain: (n, 3) CA by SEQRES index} for every chain of every group."""
    return {c: chain_ca(path, c, (lengths or {}).get(c)) for g in groups for c in g}


def _cat(ca: dict, order: list) -> np.ndarray:
    return np.vstack([ca[c] for c in order])


def sim(x: dict, y: dict, groups: list) -> dict:
    """Complex similarity of prediction x to reference y (both {chain: CA}): the within-group chain order of x with the best TM-score.
    Returns tm_score, lddt_ca, rmsd_ca of that order."""
    best = None
    perms = [list(itertools.permutations(g)) for g in groups]
    ref = _cat(y, [c for g in groups for c in g])
    for combo in itertools.product(*perms):
        order = [c for g in combo for c in g]
        # chains keep their positions in the concatenation: position i of the reference is paired with chain `order[i]` of x
        pred = _cat(x, order)
        if len(pred) != len(ref):
            raise ValueError(f"length mismatch {len(pred)} vs {len(ref)}")
        r = score.score_arrays(pred, ref)
        if best is None or r["tm_score"] > best["tm_score"]:
            best = r
    return {"tm_score": best["tm_score"], "lddt_ca": best["lddt_ca"], "rmsd_ca": best["rmsd_ca"]}


# ---- run discovery -------------------------------------------------------------------------------------------------------------
def find_runs(logs: Path, tags: list | None) -> dict:
    """{(arm, input, seed): dict(dir, tag, run)} - latest tag wins."""
    out = {}
    for tagdir in sorted(p for p in Path(logs).iterdir() if p.is_dir()):
        if tags and tagdir.name not in tags:
            continue
        for rj in sorted(tagdir.glob("*/*/seed*/run.json")):
            d = rj.parent
            try:
                run = json.loads(rj.read_text())
            except Exception:  # noqa: BLE001
                continue
            seed = int(d.name[4:])
            out[(d.parent.parent.name, d.parent.name, seed)] = dict(dir=d, tag=tagdir.name, run=run)
    return out


def pred_dir(rd: dict, input_name: str) -> Path:
    return rd["dir"] / "out" / f"boltz_results_{input_name}" / "predictions" / input_name


def load_outputs(rd: dict, input_name: str) -> dict | None:
    """dict(cif, conf, aff) or None when the run did not produce a usable structure."""
    pd_ = pred_dir(rd, input_name)
    cif, cj = pd_ / f"{input_name}_model_0.cif", pd_ / f"confidence_{input_name}_model_0.json"
    if not (cif.exists() and cj.exists()):
        return None
    conf = json.loads(cj.read_text())
    aj = pd_ / f"affinity_{input_name}.json"
    aff = json.loads(aj.read_text()) if aj.exists() else None
    return dict(cif=cif, conf=conf, aff=aff)


def finite(d) -> bool:
    if isinstance(d, dict):
        return all(finite(v) for v in d.values())
    if isinstance(d, (int, float)):
        return math.isfinite(d)
    return True


# ---- rules ---------------------------------------------------------------------------------------------------------------------
def mean(v):
    return float(np.mean(v))


def rule_s(cand: list, ref: list, groups: list) -> dict:
    """cand / ref: lists of {chain: CA} for the 3 seeds. Dissimilarity = 1 - sim, TM and lDDT separately."""
    res = {}
    base = [sim(ref[i], ref[j], groups) for i, j in itertools.combinations(range(len(ref)), 2)]
    cross = [sim(c, r, groups) for c in cand for r in ref]
    for m, key in (("tm", "tm_score"), ("lddt", "lddt_ca")):
        R = mean([b[key] for b in base])
        C = mean([x[key] for x in cross])
        lim = max(S_RATIO * (1 - R), (1 - R) + S_FLOOR)
        res[m] = dict(resampling_sim=R, cross_sim=C, dissim_cross=1 - C, limit=lim, ok=bool((1 - C) <= lim))
    res["ok"] = res["tm"]["ok"] and res["lddt"]["ok"]
    return res


def rule_e(cand: list, ref: list, exp: dict, groups: list) -> dict:
    res = {}
    sc = lambda arm: [sim(x, exp, groups) for x in arm]  # noqa: E731
    sc_c, sc_r = sc(cand), sc(ref)
    for m, key in (("tm", "tm_score"), ("lddt", "lddt_ca")):
        ec, er = mean([x[key] for x in sc_c]), mean([x[key] for x in sc_r])
        rng = max(x[key] for x in sc_r) - min(x[key] for x in sc_r)
        lim = er - max(E_FLOOR, rng)
        res[m] = dict(candidate=ec, reference=er, ref_range=rng, limit=lim, ok=bool(ec >= lim))
    res["ok"] = res["tm"]["ok"] and res["lddt"]["ok"]
    return res


def rule_c(cand_conf: list, ref_conf: list, keys: list) -> dict:
    res = {}
    for k in keys:
        c = mean([x[k] for x in cand_conf])
        rv = [x[k] for x in ref_conf]
        lo, hi = min(rv) - C_WIDEN, max(rv) + C_WIDEN
        res[k] = dict(candidate=c, ref_min=min(rv), ref_max=max(rv), lo=lo, hi=hi, ok=bool(lo <= c <= hi))
    res["ok"] = all(v["ok"] for v in res.values())
    return res


def rule_a(groups: dict) -> dict:
    """groups: {name: (cand_affinity_jsons, ref_affinity_jsons)}. RMS over groups of seed-mean differences vs 1.25 sigma sqrt(2/3)."""
    out = {"per_key": {}}
    ok = True
    for k, floor in A_FLOOR.items():
        var, diffs, rows = [], [], {}
        for name, (cj, rj) in groups.items():
            rv = [x[k] for x in rj]
            var.append(float(np.var(rv, ddof=1)) if len(rv) > 1 else float("nan"))
            d = mean([x[k] for x in cj]) - mean(rv)
            diffs.append(d)
            rows[name] = dict(candidate=mean([x[k] for x in cj]), reference=mean(rv), diff=d, ref_sd=math.sqrt(var[-1]))
        sig = max(math.sqrt(float(np.nanmean(var))), floor)
        rms = float(np.sqrt(np.mean(np.square(diffs))))
        lim = A_RATIO * sig * math.sqrt(2 / 3)
        out["per_key"][k] = dict(rms=rms, sigma_pool=sig, limit=lim, ok=bool(rms <= lim), groups=rows)
        ok = ok and rms <= lim
    out["ok"] = ok
    return out


# ---- driver --------------------------------------------------------------------------------------------------------------------
def conf_keys(case: dict) -> list:
    ks = ["complex_plddt", "ptm", "confidence_score"]
    if len(case["groups"]) > 1 or len(case["groups"][0]) > 1 or case["ligand"]:
        ks.append("iptm")
    if case["ligand"]:
        ks.append("ligand_iptm")
    return ks


def evaluate(logs: Path, tags: list | None, ref_arm: str, cand_arms: list) -> dict:
    runs = find_runs(logs, tags)
    inputs = sorted({k[1] for k in runs})
    result = {"reference_arm": ref_arm, "candidates": {}, "integrity": {}, "tags_used": sorted({v["tag"] for v in runs.values()})}
    exp_cache: dict = {}
    # integrity
    for (arm, inp, seed), rd in sorted(runs.items()):
        o = load_outputs(rd, inp)
        c = CASES.get(case_of(inp))
        bad = o is None or not finite(o["conf"]) or (c and c["affinity"] and (o["aff"] is None or not finite(o["aff"])))
        result["integrity"][f"{arm}/{inp}/seed{seed}"] = {"ok": not bad, "rc": rd["run"].get("rc")}
    for arm in cand_arms:
        res = {}
        for inp in inputs:
            case = CASES.get(case_of(inp))
            if case is None:
                continue
            have = {s: runs.get((arm, inp, s)) for s in SEEDS}
            rhave = {s: runs.get((ref_arm, inp, s)) for s in SEEDS}
            if any(v is None for v in have.values()) or any(v is None for v in rhave.values()):
                if any(v is not None for v in have.values()):
                    res[inp] = {"status": "INCOMPLETE", "seeds_candidate": [s for s, v in have.items() if v], "seeds_reference": [s for s, v in rhave.items() if v]}
                continue
            outs = {s: load_outputs(have[s], inp) for s in SEEDS}
            routs = {s: load_outputs(rhave[s], inp) for s in SEEDS}
            if any(v is None for v in list(outs.values()) + list(routs.values())):
                res[inp] = {"status": "FAIL (missing structure output)"}
                continue
            groups = case["groups"]
            lens = chain_lengths(case_of(inp))
            ca = lambda o: complex_ca(o["cif"], groups, lens)  # noqa: E731
            cand, ref = [ca(outs[s]) for s in SEEDS], [ca(routs[s]) for s in SEEDS]
            r = {"S": rule_s(cand, ref, groups)}
            if case["ref"]:
                ek = (case["ref"], inp)
                if ek not in exp_cache:
                    exp_cache[ek] = complex_ca(REF_DIR / f"{case['ref']}.cif", groups, lens)
                r["E"] = rule_e(cand, ref, exp_cache[ek], groups)
            r["C"] = rule_c([outs[s]["conf"] for s in SEEDS], [routs[s]["conf"] for s in SEEDS], conf_keys(case))
            r["status"] = "PASS" if all(v["ok"] for v in r.values() if isinstance(v, dict)) else "FAIL"
            res[inp] = r
        # rule A over the affinity inputs present
        ag = {}
        for inp in inputs:
            case = CASES.get(case_of(inp))
            if not case or not case["affinity"]:
                continue
            ca_ = [runs.get((arm, inp, s)) for s in SEEDS]
            ra_ = [runs.get((ref_arm, inp, s)) for s in SEEDS]
            if any(v is None for v in ca_ + ra_):
                continue
            co = [load_outputs(v, inp) for v in ca_]
            ro = [load_outputs(v, inp) for v in ra_]
            if any(o is None or o["aff"] is None for o in co + ro):
                continue
            ag[inp] = ([o["aff"] for o in co], [o["aff"] for o in ro])
        if ag:
            res["__affinity__"] = rule_a(ag)
            res["__affinity__"]["status"] = "PASS" if res["__affinity__"]["ok"] else "FAIL"
        statuses = [v.get("status") for v in res.values()]
        res["__arm_verdict__"] = ("MATCH (every input, every rule family)" if statuses and all(s == "PASS" for s in statuses)
                                  else "NO MATCH or incomplete: " + ", ".join(f"{k}={v.get('status')}" for k, v in res.items() if v.get("status") != "PASS"))
        result["candidates"][arm] = res
    return result


def to_md(result: dict) -> str:
    L = [f"# Accuracy vs `{result['reference_arm']}` (PREREG_MACBENCH.md section 5)\n", f"tags used: {', '.join(result['tags_used'])}\n"]
    for arm, res in result["candidates"].items():
        L += [f"\n## {arm}: {res.get('__arm_verdict__')}\n", "| input | status | S tm (dissim / limit) | S lddt (dissim / limit) | E tm (cand / limit) | E lddt (cand / limit) | C failing keys |",
              "|---|---|---|---|---|---|---|"]
        for inp, r in res.items():
            if inp.startswith("__"):
                continue
            if "S" not in r:
                L.append(f"| {inp} | {r.get('status')} | | | | | |")
                continue
            f = lambda d: f"{d['dissim_cross']:.4f} / {d['limit']:.4f}" if 'dissim_cross' in d else f"{d['candidate']:.3f} / {d['limit']:.3f}"  # noqa: E731
            e = r.get("E")
            badc = ",".join(k for k, v in r["C"].items() if isinstance(v, dict) and not v["ok"]) or "-"
            L.append(f"| {inp} | {r['status']} | {f(r['S']['tm'])} | {f(r['S']['lddt'])} | {f(e['tm']) if e else '-'} | {f(e['lddt']) if e else '-'} | {badc} |")
        a = res.get("__affinity__")
        if a:
            L.append(f"\nRule A ({a['status']}): " + "; ".join(f"{k}: RMS {v['rms']:.4f} vs limit {v['limit']:.4f} (sigma_pool {v['sigma_pool']:.4f})" for k, v in a["per_key"].items()))
    return "\n".join(L) + "\n"


# ---- self-test (committed reference files only) ---------------------------------------------------------------------------------
def selftest() -> int:
    ok = True
    groups = [["A", "B"], ["C", "D"]]
    exp = complex_ca(REF_DIR / "9MOJ.cif", groups)
    s = sim(exp, exp, groups)
    print("9MOJ vs itself:", {k: round(v, 4) for k, v in s.items()})
    ok &= s["tm_score"] > 0.999 and s["lddt_ca"] > 0.999
    swapped = dict(exp)
    swapped["A"], swapped["B"], swapped["C"], swapped["D"] = exp["B"], exp["A"], exp["D"], exp["C"]
    s2 = sim(swapped, exp, groups)
    print("9MOJ chain-swapped vs itself (best within-entity order):", {k: round(v, 4) for k, v in s2.items()})
    ok &= s2["tm_score"] > 0.999
    rng = np.random.default_rng(0)
    noisy = {c: v + rng.normal(0, 1.0, v.shape) for c, v in exp.items()}
    s3 = sim(noisy, exp, groups)
    print("9MOJ with 1 A coordinate noise:", {k: round(v, 4) for k, v in s3.items()})
    ok &= 0.5 < s3["tm_score"] < 0.999 and s3["lddt_ca"] < s["lddt_ca"]
    # rules on toy data: identical candidate and reference pass S; a displaced candidate fails
    g1 = [["A"]]
    e1 = complex_ca(REF_DIR / "8RF4.cif", g1)
    mk = lambda sd: [{c: v + np.random.default_rng(sd + i).normal(0, 0.3, v.shape) for c, v in e1.items()} for i in range(3)]  # noqa: E731
    ref, cand_same = mk(10), mk(20)
    far = [{c: v + np.random.default_rng(30 + i).normal(0, 4.0, v.shape) for c, v in e1.items()} for i in range(3)]
    r_ok, r_bad = rule_s(cand_same, ref, g1), rule_s(far, ref, g1)
    print("rule S toy: same-noise candidate", r_ok["ok"], "| 4 A noise candidate", r_bad["ok"])
    ok &= r_ok["ok"] and not r_bad["ok"]
    cj = [{"complex_plddt": 0.90, "ptm": 0.9, "confidence_score": 0.9}] * 3
    rj = [{"complex_plddt": 0.91, "ptm": 0.9, "confidence_score": 0.9}] * 3
    ok &= rule_c(cj, rj, ["complex_plddt", "ptm", "confidence_score"])["ok"]
    ok &= not rule_c([{"complex_plddt": 0.5, "ptm": 0.9, "confidence_score": 0.9}] * 3, rj, ["complex_plddt"])["ok"]
    ra = {"g": ([{"affinity_pred_value": 1.0 + d, "affinity_probability_binary": 0.5} for d in (0.0, 0.1, -0.1)],
                [{"affinity_pred_value": 1.0 + d, "affinity_probability_binary": 0.5} for d in (0.0, 0.1, -0.1)])}
    ok &= rule_a(ra)["ok"]
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs/macbench")
    ap.add_argument("--tags", nargs="*")
    ap.add_argument("--ref", default="up200")
    ap.add_argument("--cands", nargs="*", default=["strict200", "f16_200", "screen", "f16_50"])
    ap.add_argument("--out")
    ap.add_argument("--md")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    res = evaluate(Path(a.logs), a.tags, a.ref, a.cands)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1, default=float) + "\n")
    md = to_md(res)
    if a.md:
        Path(a.md).write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
