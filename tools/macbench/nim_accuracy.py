"""Consistency scoring of the 4 NVIDIA-size cases (8eil, 7qsj, 8cpk, 8gpp): ours Fast / Full, boltz-community, the experimental structures.

  nice -n 15 python -I tools/macbench/nim_accuracy.py [--logs logs/macbench] [--md docs/macbench/nim_accuracy.md] [--json OUT.json]
  python tools/macbench/nim_accuracy.py --selftest        # one prediction scored against itself (needs a finished run in --logs)

CPU only (numpy, gemmi, rdkit through tools/validation/score.py); nothing here starts a model. n = 1 seed per build per case, so every number is a
CONSISTENCY CHECK, not a pass/fail gate: no threshold is applied, before or after seeing data (PREREG_MACBENCH.md amendment 3).

Builds (run directory = <logs>/<tag>/<arm>/<input>/seed0[_rK]/out/boltz_results_<input>/predictions/<input>/):
  Fast       ours, arm screen (f16, 20 + 25 steps), job nim-ours-screen-20261102c, seed 0
  Fast r1-3  ours, arm screen, job nim-ours-rep-20261103a, seed 0 repeats (same seed: identical or near-identical expected)
  Full       ours, arm strict200 (fp32, 200 + 200 steps), job nim-full-20261103a
  Community  boltz-community, arm comm200_full (fp32, 200 + 200 steps, --no_subsample_msa), job nim-comm-20261102b
Metrics (tools/validation/score.py definitions): TM-score and CA-lDDT (global, radius 15 A) over the protein CA atoms of all chains, the chain
order of the prediction that maximises TM (identical chains of 8gpp); CA RMSD after Kabsch on the commonly modelled residues; ligand heavy-atom RMSD
after superposing the pocket CA atoms (10 A pocket of the reference ligand, score.ligand_pose_rmsd's recipe), symmetry-aware, lowest over the
reference ligand copies. Normalisation is by the residues modelled in the reference (experimental: the deposited coordinates). Reference chain: the
copy of the asymmetric unit that carries the scored ligand (8eil: chain D, SIN), else chain A; 8gpp: the A, B, C trimer. Ligand RMSD against the
experiment exists only where the entry contains the ligand: 8eil (SIN) and 8gpp (BCT); 7qsj and 8cpk have no celecoxib. Between two predictions the
ligand RMSD is computed for all four cases.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools" / "validation"))
import score  # noqa: E402

# RCSB mmCIFs of the NIM test ids, fetched by tools/data/nim_cases.py (default <checkout>/data/external/nim; BOLTZ_MLX_EXT_DATA)
NIM_DIR = Path(os.environ.get("BOLTZ_MLX_EXT_DATA", ROOT / "data" / "external")).expanduser() / "nim"
CELE = "Cc1ccc(cc1)-c1cc(nn1-c1ccc(cc1)S(N)(=O)=O)C(F)(F)F"
CASES = {
    "nim_8eil": dict(pid="8eil", tokens=194, chains=["A"], lig_chain="B", lig="SIN", smiles="O=C(O)CCC(=O)O", exp_ccd="SIN", exp_chains=["D"]),
    "nim_7qsj": dict(pid="7qsj", tokens=399, chains=["A"], lig_chain="B", lig="celecoxib", smiles=CELE, exp_ccd=None, exp_chains=["A"]),
    "nim_8cpk": dict(pid="8cpk", tokens=410, chains=["A"], lig_chain="B", lig="celecoxib", smiles=CELE, exp_ccd=None, exp_chains=["A"]),
    "nim_8gpp": dict(pid="8gpp", tokens=628, chains=["A", "B", "C"], lig_chain="D", lig="BCT", smiles="O=C([O-])O", exp_ccd="BCT", exp_chains=["A", "B", "C"]),
}
# label, job tag, arm, seed-dir suffix
BUILDS = [
    ("Fast", "nim-ours-screen-20261102c", "screen", ""),
    ("Fast r1", "nim-ours-rep-20261103a", "screen", "_r1"),
    ("Fast r2", "nim-ours-rep-20261103a", "screen", "_r2"),
    ("Fast r3", "nim-ours-rep-20261103a", "screen", "_r3"),
    ("Full", "nim-full-20261103a", "strict200", ""),
    ("Community", "nim-comm-20261102b", "comm200_full", ""),
]
CONF_KEYS = ["confidence_score", "iptm", "ligand_iptm", "ptm", "complex_plddt"]
AFF_KEYS = ["affinity_pred_value", "affinity_probability_binary"]


# ------------------------------------------------------------------------------------------------------------------ loading
def chain_lengths(case: str) -> dict:
    import yaml

    doc = yaml.safe_load((HERE / "cases" / f"{case}_msa.yaml").read_text())
    out = {}
    for e in doc["sequences"]:
        if "protein" in e:
            ids = e["protein"]["id"]
            for c in ids if isinstance(ids, list) else [ids]:
                out[c] = len(e["protein"]["sequence"])
    return out


def run_outputs(logs: Path, tag: str, arm: str, case: str, suffix: str, seed: int = 0):
    """dict(cif, conf, aff, run) of a finished ok run, or None."""
    inp = f"{case}_msa"
    d = logs / tag / arm / inp / f"seed{seed}{suffix}"
    rj = d / "run.json"
    if not rj.exists():
        return None
    run = json.loads(rj.read_text())
    pd_ = d / "out" / f"boltz_results_{inp}" / "predictions" / inp
    cif, cj = pd_ / f"{inp}_model_0.cif", pd_ / f"confidence_{inp}_model_0.json"
    if not (run.get("ok") and cif.exists() and cj.exists()):
        return dict(failed=True, run=run)
    aj = pd_ / f"affinity_{inp}.json"
    return dict(cif=cif, conf=json.loads(cj.read_text()), aff=json.loads(aj.read_text()) if aj.exists() else None, run=run)


_CA: dict = {}


def chain_ca(cif: Path, chain: str, length: int):
    k = (str(cif), chain)
    if k not in _CA:
        _CA[k] = score.load_ca(cif, chain, length=length)
    return _CA[k]


def ligand_copies(cif: Path, resname: str | None = None, chain: str | None = None) -> list:
    """Heavy atoms of each ligand copy: by residue name (experimental entries) or by non-polymer chain (predictions)."""
    import gemmi

    st = gemmi.read_structure(str(cif))
    st.setup_entities()
    out = []
    for ch in st[0]:
        for r in ch:
            hit = (resname is not None and r.name == resname and r.het_flag == "H") or (chain is not None and ch.name == chain and r.het_flag == "H")
            if not hit:
                continue
            atoms = [a for a in r if a.element.name not in ("H", "D")]
            out.append(dict(chain=ch.name, resname=r.name, symbols=[a.element.name for a in atoms],
                            xyz=np.array([[a.pos.x, a.pos.y, a.pos.z] for a in atoms])))
    return out


def all_atom_xyz(cif: Path) -> np.ndarray:
    import gemmi

    st = gemmi.read_structure(str(cif))
    return np.array([[a.pos.x, a.pos.y, a.pos.z] for ch in st[0] for r in ch for a in r])


# ------------------------------------------------------------------------------------------------------------------ metrics
def best_match(x: dict, y: dict) -> dict:
    """x: prediction {chain: CA by SEQRES index}; y: reference {chain: CA}. Chains of x are permuted (identical sequences) to maximise TM.
    Returns the score.score_arrays dict plus 'chain_map' {y chain: x chain}."""
    yorder = list(y)
    ref = np.vstack([y[c] for c in yorder])
    best = None
    for perm in itertools.permutations(list(x), len(yorder)):
        pred = np.vstack([x[c] for c in perm])
        if len(pred) != len(ref):
            raise ValueError(f"length mismatch {len(pred)} vs {len(ref)}")
        r = score.score_arrays(pred, ref)
        if best is None or r["tm_score"] > best["tm_score"]:
            best = dict(r, chain_map=dict(zip(yorder, perm)))
    return best


def ligand_rmsd(pred_cif: Path, ref_cif: Path, case: dict, chain_map: dict, ref_resname: str | None, ref_chain: str | None) -> dict | None:
    """Pocket-superposed, symmetry-aware ligand heavy-atom RMSD of the prediction's ligand against the reference ligand copies (lowest wins;
    all copies are listed). ref_resname: experimental entry (residue name); ref_chain: a prediction used as reference (non-polymer chain)."""
    smi = case["smiles"]
    pc = ligand_copies(pred_cif, chain=case["lig_chain"])
    rc = ligand_copies(ref_cif, resname=ref_resname) if ref_resname else ligand_copies(ref_cif, chain=ref_chain)
    if not pc or not rc:
        return None
    pred = score.load_complex(pred_cif, None)
    ref = score.load_complex(ref_cif, None)
    method = "smiles graph (score.symmetric_rmsd_in_place)"
    try:
        pm, _ = score.ligand_mol_from_coords(smi, pc[0]["symbols"], pc[0]["xyz"])
        pred_x, pred_sym = pm.GetConformer().GetPositions(), None
    except ValueError:  # distorted geometry: the bond graph cannot be perceived from the coordinates
        pred_x, pred_sym, method = np.asarray(pc[0]["xyz"], float), list(pc[0]["symbols"]), "element-wise optimal assignment (no bond graph)"
    res = []
    for c in rc:
        if pred_sym is None:
            try:
                rm, _ = score.ligand_mol_from_coords(smi, c["symbols"], c["xyz"])
                ref_x, ref_sym = rm.GetConformer().GetPositions(), None
            except ValueError:
                pred_sym, method = [a.GetSymbol() for a in score._template(smi).GetAtoms()], "element-wise optimal assignment (no bond graph)"
                ref_x, ref_sym = np.asarray(c["xyz"], float), list(c["symbols"])
                pred_x = np.asarray(pc[0]["xyz"], float)
                pred_sym = list(pc[0]["symbols"])
        else:
            ref_x, ref_sym = np.asarray(c["xyz"], float), list(c["symbols"])
        A, B = [], []
        for k in sorted(score.pocket_keys(ref, ref_x, score.POCKET_RADIUS)):
            pc_name = chain_map.get(k[0])
            if pc_name is None or k not in ref.ca or (pc_name, k[1]) not in pred.ca:
                continue
            A.append(pred.ca[(pc_name, k[1])])
            B.append(ref.ca[k])
        if len(A) < 3:
            continue
        R, t = score.kabsch(np.array(A), np.array(B))
        moved = score.apply(R, t, pred_x)
        if pred_sym is None:
            rms = float(score.symmetric_rmsd_in_place(score._template(smi), moved, ref_x, True))
        else:
            rms = assignment_rmsd(pred_sym, moved, ref_sym, ref_x)
        res.append(dict(ref_chain=c["chain"], ligand_rmsd=rms, n_pocket_ca=len(A), n_ref_atoms=len(ref_x), method=method))
    if not res:
        return None
    best = min(res, key=lambda d: d["ligand_rmsd"])
    return dict(best, copies=res)


def assignment_rmsd(sym_a, xyz_a, sym_b, xyz_b) -> float:
    """Heavy-atom RMSD with atoms matched one-to-one within each element (Hungarian assignment on distances): the fallback when no bond graph can
    be perceived (a distorted predicted ligand); it ignores connectivity, so for anything but a tiny, symmetric ligand it can under-estimate."""
    from scipy.optimize import linear_sum_assignment

    sa, sb = np.array(sym_a), np.array(sym_b)
    tot, n = 0.0, 0
    for el in sorted(set(sa) | set(sb)):
        ia, ib = np.where(sa == el)[0], np.where(sb == el)[0]
        if len(ia) != len(ib):
            raise ValueError(f"element count mismatch for {el}")
        d = np.linalg.norm(xyz_a[ia][:, None, :] - xyz_b[ib][None, :, :], axis=2)
        r, c = linear_sum_assignment(d)
        tot += float((d[r, c] ** 2).sum())
        n += len(r)
    return float(np.sqrt(tot / n))


def compare(logs, case_name: str, pred: dict, ref: dict | None = None, exp: bool = False) -> dict:
    """Structure + ligand metrics of prediction `pred` against the reference prediction `ref`, or (exp=True) the experimental entry."""
    case = CASES[case_name]
    L = chain_lengths(case_name)
    x = {c: chain_ca(pred["cif"], c, L[c]) for c in case["chains"]}
    if exp:
        ecif = NIM_DIR / f"{case['pid']}.cif"
        y = {c: chain_ca(ecif, c, L[case["chains"][0]]) for c in case["exp_chains"]}
        m = best_match(x, y)
        lig = ligand_rmsd(pred["cif"], ecif, case, m["chain_map"], case["exp_ccd"], None) if case["exp_ccd"] else None
    else:
        y = {c: chain_ca(ref["cif"], c, L[c]) for c in case["chains"]}
        m = best_match(x, y)
        lig = ligand_rmsd(pred["cif"], ref["cif"], case, m["chain_map"], None, case["lig_chain"])
        a, b = all_atom_xyz(pred["cif"]), all_atom_xyz(ref["cif"])
        # coordinate difference in place, no superposition: only meaningful (and only reported) for outputs that agree to the last digit's order
        m["max_atom_shift"] = float(np.abs(a - b).max()) if (a.shape == b.shape and m["tm_score"] > 0.99999) else None
    return dict(tm=m["tm_score"], lddt=m["lddt_ca"], rmsd_ca=m["rmsd_ca"], n_scored=m["n_scored"], n_ref=m["n_ref_modeled"],
                ligand_rmsd=(lig or {}).get("ligand_rmsd"), ligand_n_pocket_ca=(lig or {}).get("n_pocket_ca"),
                ligand_ref_copy=(lig or {}).get("ref_chain"), ligand_method=(lig or {}).get("method"), max_atom_shift=m.get("max_atom_shift"))


# ------------------------------------------------------------------------------------------------------------------ report
def f(v, d=3):
    return "-" if v is None else (v if isinstance(v, str) else f"{v:.{d}f}")


def evaluate(logs: Path) -> dict:
    out = dict(cases={}, builds=[b[0] for b in BUILDS])
    for cn, case in CASES.items():
        runs = {lab: run_outputs(logs, tag, arm, cn, suf) for lab, tag, arm, suf in BUILDS}
        ok = {k: v for k, v in runs.items() if v and not v.get("failed")}
        rec = dict(info={k: case[k] for k in ("pid", "tokens", "lig")}, confidence={}, vs_exp={}, pairs={}, ligand_geometry={})
        for lab, r in runs.items():
            if r is None:
                rec["confidence"][lab] = "not run"
            elif r.get("failed"):
                rec["confidence"][lab] = "FAILED"
            else:
                rec["confidence"][lab] = {k: r["conf"].get(k) for k in CONF_KEYS} | {k: (r["aff"] or {}).get(k) for k in AFF_KEYS}
        for lab, r in ok.items():
            rec["vs_exp"][lab] = compare(logs, cn, r, exp=True)
            lc = ligand_copies(r["cif"], chain=case["lig_chain"])
            if lc:
                d = np.linalg.norm(lc[0]["xyz"][:, None] - lc[0]["xyz"][None], axis=2)
                iu = np.triu_indices(len(d), 1)
                rec["ligand_geometry"][lab] = dict(n_atoms=len(d), min_pair_A=float(d[iu].min()), n_pairs_below_1A=int((d[iu] < 1.0).sum()), max_pair_A=float(d[iu].max()))
        for a, b in (("Fast", "Fast r1"), ("Fast", "Fast r2"), ("Fast", "Fast r3"), ("Fast", "Full"), ("Fast", "Community"), ("Full", "Community")):
            if a in ok and b in ok:
                rec["pairs"][f"{a} vs {b}"] = compare(logs, cn, ok[a], ref=ok[b])
        out["cases"][cn] = rec
    return out


def to_md(res: dict, logs: Path) -> str:
    L = ["# NIM-size cases: Fast vs Full vs boltz-community vs experiment (consistency check, n = 1 seed)\n",
         "**Read this first.** One seed per build per case (seed 0), 1 diffusion sample: these numbers are a *consistency check*, not a pass/fail gate. "
         "No threshold is applied, before or after seeing the data. A difference between two builds of one seed cannot be separated from seed-to-seed noise "
         "here (the repeats of Fast share the seed and only measure run-to-run determinism). Builds: Fast = ours `screen` (f16, 20 structure + 25 affinity steps); "
         "Full = ours `strict200` (fp32, 200 + 200); Community = boltz-community `comm200_full` (fp32, 200 + 200, MPS, `--no_subsample_msa`); all with the same ColabFold "
         "MSA csv, the same ligand, 3 recycles, 1 sample. Fast and Full differ in precision AND steps. Experimental structures: RCSB mmCIFs in `data/external/nim/` (tools/data/nim_cases.py) "
         "(sha256 in `tools/macbench/ref/nim_sources.json`); reference chain = the copy carrying the scored ligand (8eil: chain D), else chain A; 8gpp = the A, B, C trimer. "
         "TM / lDDT / CA RMSD are over protein CA atoms and normalised by the residues modelled in the reference. Ligand RMSD: heavy atoms, symmetry-aware, after superposing the "
         "10 A pocket CA atoms, lowest over the reference ligand copies; against the experiment only where the entry contains the ligand (8eil SIN, 8gpp BCT; 7qsj and 8cpk "
         "have no celecoxib). The experimental ligand of 8eil sits at chain D and 8gpp has one BCT per chain, so those ligand RMSDs also depend on which site the model chose.\n"]
    cases = res["cases"]
    L += ["## 1. Confidence and affinity side by side (model 0)\n",
          "| case | build | confidence_score | iptm | ligand_iptm | ptm | complex_plddt | affinity_pred_value | affinity_probability_binary |", "|---|---|---|---|---|---|---|---|---|"]
    for cn, rec in cases.items():
        for lab, v in rec["confidence"].items():
            if isinstance(v, str):
                L.append(f"| {cn[4:]} | {lab} | {v} | | | | | | |")
            else:
                L.append(f"| {cn[4:]} | {lab} | " + " | ".join(f(v[k]) for k in CONF_KEYS + AFF_KEYS) + " |")
    L += ["\n## 2. Prediction against prediction (same case, same MSA and ligand)\n",
          "| case | pair | TM | CA-lDDT | CA RMSD A | ligand RMSD A (pocket-superposed) | max atom shift A |", "|---|---|---|---|---|---|---|"]
    for cn, rec in cases.items():
        for pair, v in rec["pairs"].items():
            L.append(f"| {cn[4:]} | {pair} | {f(v['tm'])} | {f(v['lddt'])} | {f(v['rmsd_ca'], 2)} | {f(v['ligand_rmsd'], 2)} | {f(v['max_atom_shift'], 2)} |")
    L += ["\nRows `Fast vs Fast rK` compare repeats of the same seed (determinism); `Fast vs Full` is f16 20/25 steps against fp32 200/200; `Fast vs Community` and `Full vs Community` "
          "are the two implementations. 'max atom shift' is the largest coordinate difference over all atoms in the CIFs in place (no superposition), so it is only meaningful when two outputs "
          "are near-identical.\n",
          "\n## 3. Each build against the experimental structure\n",
          "| case | build | TM | CA-lDDT | CA RMSD A | modelled / reference residues | ligand RMSD A | ligand pocket CA used (ref copy) |", "|---|---|---|---|---|---|---|---|"]
    for cn, rec in cases.items():
        for lab, v in rec["vs_exp"].items():
            lig = "n/a (ligand not in entry)" if CASES[cn]["exp_ccd"] is None else f(v["ligand_rmsd"], 2)
            L.append(f"| {cn[4:]} | {lab} | {f(v['tm'])} | {f(v['lddt'])} | {f(v['rmsd_ca'], 2)} | {v['n_scored']} / {v['n_ref']} | {lig} | {v['ligand_n_pocket_ca'] or '-'} ({v['ligand_ref_copy'] or '-'}) |")
    L += ["\n## 4. Predicted ligand internal geometry (sanity, not a metric)\n",
          "Smallest and largest distance between any two ligand heavy atoms and the number of pairs closer than 1.0 A (no covalent bond is that short: such pairs mean a collapsed ligand).\n",
          "| case | build | heavy atoms | min pair distance A | pairs < 1.0 A | max pair distance A |", "|---|---|---|---|---|---|"]
    for cn, rec in cases.items():
        for lab, g in rec["ligand_geometry"].items():
            L.append(f"| {cn[4:]} | {lab} | {g['n_atoms']} | {f(g['min_pair_A'], 2)} | {g['n_pairs_below_1A']} | {f(g['max_pair_A'], 2)} |")
    fb = [f"{cn[4:]} {k}" for cn, rec in cases.items() for k, v in list(rec["pairs"].items()) + [(f"{a} vs experiment", x) for a, x in rec["vs_exp"].items()]
          if v.get("ligand_method") and "assignment" in v["ligand_method"]]
    if fb:
        L.append("\nLigand RMSD by element-wise optimal assignment (the predicted ligand's geometry was too distorted to perceive a bond graph; ignores connectivity, adequate only for the "
                 "3-oxygen BCT): " + "; ".join(fb) + ". All other ligand RMSDs use the SMILES graph.")
    L.append("\nMissing rows mean the run did not exist or failed (see the build's `run.json`). Sources of every number: `logs/macbench/<job tag>/<arm>/nim_<id>_msa/seed0[_rK]/` "
             "(not committed); code: `tools/macbench/nim_accuracy.py`.\n")
    return "\n".join(L)


# ------------------------------------------------------------------------------------------------------------------ BCT diagnostic
# (label, tag, arm, seed): the bct-diag arms and the reference rows. Rule fixed in PREREG_MACBENCH.md before the data: collapse = any ligand
# heavy-atom pair < 1.0 A; pose error = ligand RMSD against the experimental ligand (pocket-superposed).
BCT_ROWS = [
    ("screen seed 0 (reference)", "nim-ours-screen-20261102c", "screen", 0),
    ("strict200 seed 0 (reference)", "nim-full-20261103a", "strict200", 0),
    ("boltz-community 200 seed 0 (reference)", "nim-comm-20261102b", "comm200_full", 0),
    ("f16_50 seed 0: f16, 50 / 50", "bct-diag-20261103b", "f16_50", 0),
    ("f16_200 seed 0: f16, 200 / 200", "bct-diag-20261103b", "f16_200", 0),
    ("screen seed 1: f16, 20 / 25", "bct-diag-20261103b", "screen", 1),
    ("screen seed 2: f16, 20 / 25", "bct-diag-20261103b", "screen", 2),
    ("strict20 seed 0: fp32, 20 / 25", "bct-diag-20261103b", "strict20", 0),
]


def bct_diag(logs: Path) -> tuple[str, list]:
    cn = "nim_8gpp"
    L = ["\n## 6. BCT collapse diagnostic on 8gpp (job `bct-diag-20261103b`; rule pre-registered in PREREG_MACBENCH.md before the data)\n",
         "Collapse = any pair of ligand heavy atoms closer than 1.0 A in the predicted BCT (4 atoms, 6 pairs). Pose error = ligand RMSD against the experimental BCT after pocket-CA "
         "superposition (lowest over the three experimental copies). n = 1 run per row, a diagnostic, no gate. Protein TM against the experimental trimer (A, B, C). "
         "`screen` is f16 with 20 structure / 25 affinity steps; `strict20` is fp32 at the same 20 / 25 steps (precision isolated from steps).\n",
         "| run | collapse | min pair A | pairs < 1.0 A | ligand RMSD vs exp A | protein TM vs exp | affinity_pred_value | affinity_probability_binary | peak GB |",
         "|---|---|---|---|---|---|---|---|---|"]
    rows = []
    for lab, tag, arm, seed in BCT_ROWS:
        r = run_outputs(logs, tag, arm, cn, "", seed)
        if r is None or r.get("failed"):
            L.append(f"| {lab} | {'not run' if r is None else 'FAILED'} | | | | | | | |")
            continue
        lc = ligand_copies(r["cif"], chain=CASES[cn]["lig_chain"])[0]
        d = np.linalg.norm(lc["xyz"][:, None] - lc["xyz"][None], axis=2)
        iu = np.triu_indices(len(d), 1)
        mn, nb = float(d[iu].min()), int((d[iu] < 1.0).sum())
        v = compare(logs, cn, r, exp=True)
        a = r["aff"] or {}
        rows.append(dict(run=lab, collapse=nb > 0, min_pair=mn, n_below=nb, ligand_rmsd=v["ligand_rmsd"], tm=v["tm"], affinity=a.get("affinity_pred_value"),
                         prob=a.get("affinity_probability_binary"), peak_gb=r["run"].get("peak_footprint_gb")))
        L.append(f"| {lab} | {'YES' if nb > 0 else 'no'} | {f(mn, 2)} | {nb} / 6 | {f(v['ligand_rmsd'], 2)} | {f(v['tm'])} | {f(a.get('affinity_pred_value'))} | "
                 f"{f(a.get('affinity_probability_binary'))} | {f(r['run'].get('peak_footprint_gb'), 1)} |")
    return "\n".join(L) + "\n", rows


def selftest(logs: Path) -> int:
    for cn in CASES:
        for lab, tag, arm, suf in BUILDS:
            r = run_outputs(logs, tag, arm, cn, suf)
            if r and not r.get("failed"):
                v = compare(logs, cn, r, ref=r)
                print(cn, lab, v)
                assert v["tm"] > 0.999 and v["lddt"] > 0.999 and (v["ligand_rmsd"] is None or v["ligand_rmsd"] < 1e-3), v
                print("SELFTEST PASS")
                return 0
    print("no finished run to test on")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=str(ROOT / "logs" / "macbench"))
    ap.add_argument("--md")
    ap.add_argument("--json")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--bct-diag", action="store_true", help="score job bct-diag (8gpp) and print the section; with --append FILE add it to that markdown file")
    ap.add_argument("--append")
    a = ap.parse_args()
    logs = Path(a.logs)
    if a.bct_diag:
        md, rows = bct_diag(logs)
        print(md)
        if a.append:
            with open(a.append, "a") as fh:
                fh.write(md)
        return 0
    if a.selftest:
        return selftest(logs)
    res = evaluate(logs)
    res["experimental_sha256"] = {c["pid"]: hashlib.sha256((NIM_DIR / f"{c['pid']}.cif").read_bytes()).hexdigest() for c in CASES.values()}
    md = to_md(res, logs)
    print(md)
    if a.md:
        Path(a.md).write_text(md + "\n")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
