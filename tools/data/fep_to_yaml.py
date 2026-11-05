"""Build Boltz-2 input YAMLs, labels and ligand lists for the FEP+ JACS-set benchmark (round 2 DATA).

Inputs (all data only; no script from the cloned repo is run):
  data/external/fep_benchmark/                 git clone of schrodinger/public_binding_free_energy_benchmark (MIT)
  data/external/rcsb_entities/<PDB>.json       RCSB entity sequences (tools/data/fetch_rcsb_entities.py)
  data/external/fep/chain_decisions.json       chain handling (tools/data/fep_chain_check.py, run first)

Outputs (default --out data/external/fep):
  yaml/<target>/<target>_<name>.yaml   one Boltz YAML per complex (receptor chains + ligand chain L + affinity binder L)
  labels.csv                           id,target,name,exp_dg_kcal_mol,fep_plus_pred_dg,smiles,n_atoms_upstream,size_flag
  fep_plus_tau.csv                     FEP+ own per-target statistics from jacs_set_results.csv (context only)
  name_map.csv                         original ligand name <-> sanitised id <-> SDF index
  t400_ligands.csv                     all JACS-set ligands (8 targets), sorted by id: id,smiles
  summary.json                         counts, receptors, chain decisions, size-check tallies

YAML targets: cdk2, tyk2, jnk1, p38, bace (the five gated targets) and thrombin (smoke test). mcl1 and ptp1b only enter
labels.csv / t400_ligands.csv (they are in the 199-ligand throughput list, not in any gate).

Receptor MSA: `msa: empty` by default. With --msa-dir DIR, DIR/<target>.a3m is used for the first (target) protein chain
and DIR/<target>_<chain>.a3m for the other chains, each only if the file exists (otherwise `empty`).

Ligand size check follows upstream (boltz/data/parse/schema.py:1263): count atoms of RemoveHs(AddHs(MolFromSmiles(smi)),
sanitize=False); reject > 128, warn > 56. Atom names (symbol + canonical rank + 1) must be <= 4 characters.

Usage: python -I tools/data/fep_to_yaml.py [--out DIR] [--msa-dir DIR]
"""
import os
import argparse
import csv
import json
import re
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

HOME = Path(os.environ.get("BOLTZ_MLX_EXT_DATA", Path(__file__).resolve().parents[2] / "data" / "external")).expanduser()
BENCH = HOME / "fep_benchmark"
STRUCT = BENCH / "fep_benchmark_inputs" / "structure_inputs" / "jacs_set"
PRED = BENCH / "21_4_results" / "ligand_predictions" / "jacs_set"
SUMMARY = BENCH / "21_4_results" / "summary_statistics" / "jacs_set_results.csv"
ENT = HOME / "rcsb_entities"

# target id -> (SDF/PDB prefix, *_out.csv stem, reference PDB, key in chain_decisions.json or None)
TARGETS = {
    "bace": ("bace", "bace_out", "4DJW", "bace"),
    "cdk2": ("cdk2", "cdk2_out", "1H1Q", "cdk2"),
    "jnk1": ("jnk1_manual_flips", "jnk1_manual_flips_symbmcorr_out", "2GMX", "jnk1"),
    "mcl1": ("mcl1_extra_flips", "mcl1_extra_flips_bmcorr_out", "4HW3", None),
    "p38": ("p38", "p38_out", "3FLN", "p38"),
    "ptp1b": ("ptp1b", "ptp1b_out", "2QBS", None),
    "thrombin": ("thrombin_core", "thrombin_core_out", "2ZFF", "thrombin"),
    "tyk2": ("tyk2", "tyk2_out", "4GIH", "tyk2"),
}
YAML_TARGETS = ["cdk2", "tyk2", "jnk1", "p38", "bace", "thrombin"]
EXPECTED = {"mcl1": 42, "cdk2": 16, "p38": 34, "tyk2": 16, "bace": 36, "jnk1": 21, "thrombin": 11, "ptp1b": 23}
CUTOFF = 10.0
LIG_CHAIN = "L"


def sanitize(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip()).strip("_.-").lower()
    return s


def upstream_size(smiles: str) -> tuple[int, bool]:
    """(atom count as upstream counts it, atom-name-length ok)."""
    mol = AllChem.AddHs(Chem.MolFromSmiles(smiles))
    ranks = AllChem.CanonicalRankAtoms(mol)
    names_ok = all(len(a.GetSymbol().upper() + str(r + 1)) <= 4 for a, r in zip(mol.GetAtoms(), ranks))
    return AllChem.RemoveHs(mol, sanitize=False).GetNumAtoms(), names_ok


def receptor_chains(target: str, decisions: dict) -> list[tuple[str, int]]:
    """[(chain id, RCSB entity id)] to include, in order. Pre-registered rules; see module docstring / PREREG."""
    if target in ("mcl1", "ptp1b"):
        return []
    ch = decisions[target]["chains"]
    pick = {"cdk2": ["A", "B"], "tyk2": ["A"], "p38": ["C"], "bace": ["A"]}.get(target)
    if target == "jnk1":  # chain F only if any atom within 10 A of any ligand (decided by fep_chain_check.py)
        pick = ["A"] + (["F"] if decisions["decisions"]["jnk1_include_chain_F"] else [])
    elif target == "thrombin":  # smoke test: not pre-registered upstream of this script; same 10 A rule applied to every chain
        pick = [c for c, v in ch.items() if v["min_dist_to_any_ligand_A"] <= CUTOFF]
    return [(c, ch[c]["best_entity_match"]["entity"]) for c in pick]


def yaml_str(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def write_yaml(path: Path, chains: list[tuple[str, str, str]], smiles: str) -> None:
    lines = ["version: 1", "sequences:"]
    for cid, seq, msa in chains:
        lines += ["  - protein:", f"      id: {cid}", f"      sequence: {seq}", f"      msa: {msa}"]
    lines += ["  - ligand:", f"      id: {LIG_CHAIN}", f"      smiles: {yaml_str(smiles)}"]
    lines += ["properties:", "  - affinity:", f"      binder: {LIG_CHAIN}", ""]
    path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HOME / "fep")
    ap.add_argument("--msa-dir", type=Path, default=None)
    a = ap.parse_args()
    out = a.out
    decisions = json.loads((out / "chain_decisions.json").read_text())
    msa_found = {}

    labels, name_map, t400, tally = [], [], [], {}
    receptors = {}
    for tgt, (prefix, stem, pdb, _) in sorted(TARGETS.items()):
        ent = {e["entity_id"]: e for e in json.loads((ENT / f"{pdb}.json").read_text())["entities"]}
        rc = receptor_chains(tgt, decisions)
        chains = []
        for k, (cid, eid) in enumerate(rc):
            seq = ent[eid]["seq"]
            assert set(seq) <= set("ACDEFGHIKLMNPQRSTVWY"), (tgt, cid, set(seq) - set("ACDEFGHIKLMNPQRSTVWY"))
            msa = "empty"
            if a.msa_dir is not None:
                f = a.msa_dir / (f"{tgt}.a3m" if k == 0 else f"{tgt}_{cid}.a3m")
                if f.exists():
                    msa = str(f)
                msa_found[f.name] = f.exists()
            chains.append((cid, seq, msa))
        if rc:
            receptors[tgt] = {"pdb": pdb, "chains": [{"chain": c, "entity": e, "length": len(ent[e]["seq"]), "description": ent[e]["description"]} for c, e in rc]}

        sup = {}
        for idx, m in enumerate(Chem.SDMolSupplier(str(STRUCT / f"{prefix}_ligands.sdf"), removeHs=False)):
            assert m is not None, (tgt, idx)
            sup[m.GetProp("_Name")] = (idx, m)
        rows = list(csv.DictReader(open(PRED / f"{stem}.csv")))
        assert len(rows) == EXPECTED[tgt], (tgt, len(rows), EXPECTED[tgt])
        used_ids = {}
        n_warn = n_reject = n_namebad = n_unassigned = 0
        for r in rows:
            name = r["Ligand name"]
            idx, m = sup[name]  # entry with the exact label name (the ' flip' entries are alternative poses of the same ligand)
            Chem.AssignStereochemistryFrom3D(m)
            unassigned = [c for c in Chem.FindMolChiralCenters(m, includeUnassigned=True, useLegacyImplementation=False) if c[1] == "?"]
            n_unassigned += bool(unassigned)
            smi = Chem.MolToSmiles(Chem.RemoveHs(m), isomericSmiles=True)
            assert Chem.MolToSmiles(Chem.MolFromSmiles(smi)) == smi, (tgt, name)
            lid = f"{tgt}_{sanitize(name)}"
            assert lid not in used_ids, (lid, name, used_ids[lid])
            used_ids[lid] = name
            n_at, names_ok = upstream_size(smi)
            flag = "reject" if (n_at > 128 or not names_ok) else "warn" if n_at > 56 else "ok"
            n_reject += flag == "reject"
            n_warn += flag == "warn"
            n_namebad += not names_ok
            exp = round(float(r["Exp. dG (kcal/mol)"]), 3)
            sdf_exp = float(m.GetProp("r_exp_dg")) if m.HasProp("r_exp_dg") else None
            if sdf_exp is not None:
                assert abs(sdf_exp - exp) < 0.01, (lid, sdf_exp, exp)
            labels.append(
                {"id": lid, "target": tgt, "name": name, "exp_dg_kcal_mol": exp,
                 "fep_plus_pred_dg": round(float(r["Pred. dG (kcal/mol)"]), 3), "smiles": smi, "n_atoms_upstream": n_at,
                 "size_flag": flag, "unassigned_stereocentres": len(unassigned)}
            )
            name_map.append({"id": lid, "target": tgt, "original_name": name, "sdf_file": f"{prefix}_ligands.sdf", "sdf_index": idx})
            t400.append((lid, smi))
            if tgt in YAML_TARGETS and flag != "reject":
                d = out / "yaml" / tgt
                d.mkdir(parents=True, exist_ok=True)
                write_yaml(d / f"{lid}.yaml", chains, smi)
        tally[tgt] = {"n_ligands": len(rows), "n_sdf_entries": len(sup), "warn_gt56": n_warn, "reject": n_reject,
                      "atom_name_too_long": n_namebad, "with_unassigned_stereocentres": n_unassigned}

    def write_csv(p: Path, rows: list[dict]) -> None:
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "labels.csv", sorted(labels, key=lambda r: r["id"]))
    write_csv(out / "name_map.csv", sorted(name_map, key=lambda r: r["id"]))
    with open(out / "t400_ligands.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "smiles"])
        w.writerows(sorted(t400))
    # FEP+ own statistics (context only)
    stem2tgt = {v[1]: k for k, v in TARGETS.items()}
    srows = []
    for r in csv.DictReader(open(SUMMARY)):
        t = stem2tgt.get(r["entries"])
        if t:
            srows.append({"target": t, "n_compounds": r["number of compounds"], "pairwise_rmse": r["Pairwise RMSE"],
                          "pairwise_mue": r["Pairwise MUE"], "r_squared": r["R-squared"], "kendall_tau": r["Kendall tau"]})
    write_csv(out / "fep_plus_tau.csv", sorted(srows, key=lambda r: r["target"]))

    n_yaml = len(list((out / "yaml").rglob("*.yaml")))
    summary = {"n_ligands_total": len(t400), "n_yaml": n_yaml, "per_target": tally, "receptors": receptors,
               "chain_decisions": decisions["decisions"], "msa_dir": str(a.msa_dir) if a.msa_dir else None, "msa_files_found": msa_found}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    assert len(t400) == 199, len(t400)


if __name__ == "__main__":
    main()
