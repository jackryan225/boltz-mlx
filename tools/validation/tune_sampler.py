"""Rung 4 sampler tuning on T (PREREG deviation 17): T surrogates for every candidate label in the tuning rows.

  python tools/validation/tune_sampler.py [--json logs/tune_T.json] [--prot F ...] [--tyk2 F ...]

--prot / --tyk2: candidate row files (default logs/l3/tune_T_prot.jsonl / logs/l3/tune_T_tyk2.jsonl; e.g. the M5 Max
stage A files logs/l3/tune_T_{prot,tyk2}_m5max.jsonl, kept apart from the M5 Pro rows).

T = the first 10 round-1 chains + the 16 TYK2 ligands. Reference = strict@200 rows on disk (logs/l3/v36_strict.jsonl,
logs/l3/fep_strict.jsonl). Candidates = every `mode` label in logs/l3/tune_T_{prot,tyk2}.jsonl, seeds 0-2.

Surrogates (point estimates against the PREREG-A thresholds, deviation 17):
  L3-P  mean over chains of [mean_cand(seeds 0-2) - mean_strict(seeds 0-4)] CA-lDDT >= -0.005, TM >= -0.015
  L3-L  median / P90 ratio of RMSD(cand s, strict s) over RMSD(strict i, strict j), seeds 0-2: <= 1.25 / <= 1.5;
        chirality flips <= strict's
  A2'   RMS_l [mean_cand(0-2) - mean_strict(0-2)] value <= 1.25 * sigma_seed * sqrt(2/3), sigma_seed pooled over the
        TYK2 strict seeds 0-5
  A3    delta Pearson vs experiment (TYK2 only, 16 ligands: noisy) >= -0.05
Within-family score (F4, F5): max(-dlDDT/0.005, -dTM/0.015, median ratio/1.25, P90 ratio/1.5); tie -> fewer NFE.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "validation"))
import gates  # noqa: E402
import score  # noqa: E402

T_CHAINS = ["8j9r", "8fj8", "8oxk", "8ti5", "8t98", "8ta1", "8jqu", "8tsz", "8c0m", "8ywo"]
CAND_SEEDS = (0, 1, 2)
PROT_REF_SEEDS = (0, 1, 2, 3, 4)
SIGMA_SEEDS = (0, 1, 2, 3, 4, 5)
THR = {"dlddt": -0.005, "dtm": -0.015, "med": 1.25, "p90": 1.5, "a2p": 1.25, "a3": -0.05}


def _rows(path):
    p = ROOT / path
    return [json.loads(line) for line in open(p)] if p.exists() else []


def _by_case_seed(rows, key, seeds):
    d = {}
    for r in rows:
        if r.get(key) is not None and int(r["seed"]) in seeds:
            d.setdefault(r["case"], {})[int(r["seed"])] = float(r[key])
    return d


def protein(cand, strict):
    out = {}
    for key, name in (("ca_lddt", "dlddt"), ("tm", "dtm")):
        c, s = _by_case_seed(cand, key, CAND_SEEDS), _by_case_seed(strict, key, PROT_REF_SEEDS)
        deltas = {k: np.mean(list(c[k].values())) - np.mean(list(s[k].values())) for k in c if k in s}
        out[name] = float(np.mean(list(deltas.values()))) if deltas else None
        out[name + "_n"] = len(deltas)
        out[name + "_worst"] = (min(deltas.items(), key=lambda kv: kv[1]) if deltas else None)
    return out


class _PoseSource(gates.PoseSource):
    """gates.PoseSource, but a candidate pose whose ligand bonds cannot be matched to the SMILES (geometry too distorted)
    counts as a broken pose: RMSD = inf (fails the L3-L surrogates) and counted, instead of raising."""

    broken: set

    def rmsd(self, case, a, b):
        try:
            return super().rmsd(case, a, b)
        except ValueError:
            self.broken = getattr(self, "broken", set()) | {(case, a[0], a[1])}
            self.cache[(case, a, b)] = float("inf")
            return float("inf")

    def flips(self, case, mode, seed, steps=None):
        try:
            return super().flips(case, mode, seed, steps)
        except ValueError:
            self.broken = getattr(self, "broken", set()) | {(case, mode, seed)}
            return float("nan")


def pose(cand_rows, strict_rows, label, cfg):
    src = _PoseSource(strict_rows + cand_rows, cfg)
    cases = sorted({r["case"] for r in cand_rows})
    pairs = [(s, s) for s in CAND_SEEDS]
    d = gates.build_pose(src, cases, ["tyk2"] * len(cases), label, None, "strict", None, pairs)
    if not len(d["cases"]):  # every candidate pose broken (or no strict pairs): an explicit L3-L failure
        return {"med": float("inf"), "p90": float("inf"), "chir_cand": float("nan"), "chir_strict": float("nan"), "n_lig": 0,
                "broken_poses": sorted(f"{c} s{sd}" for c, m, sd in getattr(src, "broken", set()))}
    st = gates.pose_stats(d["cs"], d["ss"], d["target"], 1000, gates.make_rng(0, "pose"))
    return {"med": st["l3l_med"], "p90": st["l3l_p90"], "med_cand_A": st["l3l_med_cand"], "med_strict_pair_A": st["l3l_med_strict_pair"],
            "chir_cand": float(d["chir_c"].sum()), "chir_strict": float(d["chir_s"].sum()), "n_lig": len(d["cases"]),
            "broken_poses": sorted(f"{c} s{sd}" for c, m, sd in getattr(src, "broken", set()))}


def affinity(cand_rows, strict_rows):
    vc = _by_case_seed(cand_rows, "affinity_pred_value", CAND_SEEDS)
    if not vc:
        return {}
    vr = _by_case_seed(strict_rows, "affinity_pred_value", CAND_SEEDS)
    vs = _by_case_seed(strict_rows, "affinity_pred_value", SIGMA_SEEDS)
    pc = _by_case_seed(cand_rows, "affinity_probability_binary", CAND_SEEDS)
    pr = _by_case_seed(strict_rows, "affinity_probability_binary", CAND_SEEDS)
    exp = {r["case"]: r.get("exp_value") for r in strict_rows}
    lig = sorted(k for k in vc if k in vr and len(vc[k]) == len(CAND_SEEDS))
    ca = np.array([np.mean(list(vc[k].values())) for k in lig])
    ra = np.array([np.mean(list(vr[k].values())) for k in lig])
    d = ca - ra
    sigma = float(np.sqrt(np.mean([np.var(list(vs[k].values()), ddof=1) for k in lig if len(vs.get(k, {})) >= 2])))
    e = np.array([np.nan if exp.get(k) is None else float(exp[k]) for k in lig])
    ok = np.isfinite(e)
    g = np.array(["tyk2"] * len(lig))
    dp = np.array([np.mean(list(pc[k].values())) - np.mean(list(pr[k].values())) for k in lig if k in pc and k in pr])
    return {"n_lig": len(lig), "a1_dvalue": float(d.mean()), "a1_dprob": float(dp.mean()) if len(dp) else None,
            "a2p": float(np.sqrt(np.mean(d * d)) / (sigma * np.sqrt(2 / 3))), "sigma_seed": sigma,
            "a3": float(score.weighted_pearson(ca[ok], e[ok], g[ok]) - score.weighted_pearson(ra[ok], e[ok], g[ok])),
            "pearson_cand": float(score.weighted_pearson(ca[ok], e[ok], g[ok])),
            "pearson_strict": float(score.weighted_pearson(ra[ok], e[ok], g[ok]))}


def _median(rows, key):
    v = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    return float(np.median(v)) if v else None


def evaluate(prot_files=("logs/l3/tune_T_prot.jsonl",), tyk2_files=("logs/l3/tune_T_tyk2.jsonl",)) -> dict:
    cfg = gates.load_config()
    sp = [r for r in _rows("logs/l3/v36_strict.jsonl") if r["case"] in T_CHAINS and r["mode"] == "strict"]
    sl = [r for r in _rows("logs/l3/fep_strict.jsonl") if r.get("target") == "tyk2" and r["mode"] == "strict"]
    cp = [r for f in prot_files for r in _rows(f)]
    cl = [r for f in tyk2_files for r in _rows(f)]
    labels = list(dict.fromkeys([r["mode"] for r in cp + cl]))
    res = {}
    for lab in labels:
        p = [r for r in cp if r["mode"] == lab]
        l_all = [r for r in cl if r["mode"] == lab]
        nfe = next(((r.get("sampler_structure") or {}).get("nfe") for r in p + l_all if r.get("sampler_structure")), None)
        steps = next((r.get("steps") for r in p + l_all), None)
        rec = {"label": lab, "steps": steps, "nfe": nfe if nfe is not None else steps,
               "n_prot_rows": len(p), "n_lig_rows": len(l_all), "sec_struct_tyk2": _median(l_all, "seconds_structure"),
               "sec_prot": _median(p, "seconds")}
        if p:
            rec.update(protein(p, sp))
        if l_all:
            rec["pose"] = pose(l_all, sl, lab, cfg)
        aff_by_steps = {}
        for r in l_all:
            if r.get("affinity_pred_value") is not None:
                aff_by_steps.setdefault(int(r["affinity_steps"]), []).append(r)
        rec["affinity"] = {a: {**affinity(rows, sl), "sec_affinity": _median(rows, "seconds_affinity")}
                           for a, rows in sorted(aff_by_steps.items())}
        rec.update(verdict(rec))
        res[lab] = rec
    return res


def verdict(rec) -> dict:
    po = rec.get("pose") or {}
    terms = {}
    if rec.get("dlddt") is not None:
        terms["lddt"] = -rec["dlddt"] / -THR["dlddt"]
    if rec.get("dtm") is not None:
        terms["tm"] = -rec["dtm"] / -THR["dtm"]
    if po:
        terms["med"], terms["p90"] = po["med"] / THR["med"], po["p90"] / THR["p90"]
    struct_ok = (len(terms) == 4 and all(v <= 1.0 for v in terms.values()) and po["chir_cand"] <= po["chir_strict"])
    aff_ok = {a: (v["a2p"] <= THR["a2p"] and v["a3"] >= THR["a3"]) for a, v in rec.get("affinity", {}).items()}
    return {"score": max(terms.values()) if len(terms) == 4 else None, "score_terms": terms,
            "struct_T_pass": struct_ok if len(terms) == 4 else None, "aff_T_pass": aff_ok}


def table(res) -> str:
    h = ("| label | NFE | ΔlDDT | ΔTM | L3-L med / P90 | chir c/s | score | struct T | aff steps: A2′ ratio, ΔPearson, Δvalue (T) | "
         "s struct (TYK2) | s aff |")
    lines = [h, "|" + "---|" * 11]

    def f(x, fmt="+.4f"):
        return "—" if x is None else format(x, fmt)

    for lab, r in sorted(res.items(), key=lambda kv: (-(kv[1]["nfe"] or 0), kv[0])):
        po = r.get("pose") or {}
        aff = "; ".join(f"a{a}: {v['a2p']:.2f}, {v['a3']:+.3f}, {v['a1_dvalue']:+.3f} {'✓' if r['aff_T_pass'][a] else '✗'}"
                        for a, v in r.get("affinity", {}).items())
        sa = "; ".join(f(v.get("sec_affinity"), ".1f") for v in r.get("affinity", {}).values())
        st = "—" if r["struct_T_pass"] is None else ("PASS" if r["struct_T_pass"] else "fail")
        lines.append(f"| {lab} | {r['nfe']} | {f(r.get('dlddt'))} | {f(r.get('dtm'))} | {f(po.get('med'), '.2f')} / {f(po.get('p90'), '.2f')} | "
                     f"{po.get('chir_cand', '—')}/{po.get('chir_strict', '—')} | {f(r['score'], '.2f')} | {st} | {aff or '—'} | "
                     f"{f(r['sec_struct_tyk2'], '.1f')} | {sa or '—'} |")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", type=Path, default=ROOT / "logs" / "tune_T.json")
    ap.add_argument("--prot", nargs="+", default=["logs/l3/tune_T_prot.jsonl"], help="candidate rows, T chains")
    ap.add_argument("--tyk2", nargs="+", default=["logs/l3/tune_T_tyk2.jsonl"], help="candidate rows, TYK2 ligands")
    a = ap.parse_args(argv)
    res = evaluate(a.prot, a.tyk2)
    a.json.write_text(json.dumps(res, indent=1, default=lambda o: float(o) if isinstance(o, np.floating) else str(o)))
    print(table(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
