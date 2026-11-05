"""Pre-registered chain handling for the FEP+ JACS-set receptors (decided by script before any run).

For each target:
  * reads the prepared protein PDB from the cloned benchmark (data only; no repo scripts are run),
  * lists chains with CA counts and the sequence of the modelled residues,
  * aligns each prepared chain to every RCSB entity sequence of the reference PDB (data.rcsb.org, fetched by
    tools/data/fetch_rcsb_entities.py) with a local (Smith-Waterman-like) alignment and reports identity/coverage,
    which verifies the ID -> target mapping against the repo files,
  * computes, per chain, the minimum heavy-atom distance to ANY atom of ANY ligand in <target>_ligands.sdf
    (all structural inputs, including 'flip' variants). The JNK1 peptide chain F is included only if that is <= 10 A.

Writes data/external/fep/chain_decisions.json. CPU only. Usage: python -I tools/data/fep_chain_check.py
"""
import os
import json
from pathlib import Path

import gemmi
import numpy as np
from rdkit import Chem

BENCH = Path(os.environ.get("BOLTZ_MLX_EXT_DATA", Path(__file__).resolve().parents[2] / "data" / "external")).expanduser() / "fep_benchmark" / "fep_benchmark_inputs" / "structure_inputs" / "jacs_set"
ENT = Path(os.environ.get("BOLTZ_MLX_EXT_DATA", Path(__file__).resolve().parents[2] / "data" / "external")).expanduser() / "rcsb_entities"
OUT = Path(os.environ.get("BOLTZ_MLX_EXT_DATA", Path(__file__).resolve().parents[2] / "data" / "external")).expanduser() / "fep"

# target -> (file prefix, reference PDB from benchmark_metadata.csv / subset_metadata.csv)
TARGETS = {
    "cdk2": ("cdk2", "1H1Q"),
    "tyk2": ("tyk2", "4GIH"),
    "jnk1": ("jnk1_manual_flips", "2GMX"),
    "p38": ("p38", "3FLN"),
    "bace": ("bace", "4DJW"),
    "thrombin": ("thrombin_core", "2ZFF"),
}
CUTOFF = 10.0
THREE2ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "HIE": "H", "HID": "H", "HIP": "H", "CYX": "C", "CYM": "C", "ASH": "D", "GLH": "E", "LYN": "K", "MSE": "M", "TPO": "T",
    "SEP": "S", "PTR": "Y", "ACE": "", "NME": "", "NMA": "",
}


def local_align_stats(a: str, b: str):
    """Smith-Waterman local alignment (match +2, mismatch -1, gap -2). Returns (n_identical, aligned_len)."""
    n, m = len(a), len(b)
    H = [[0] * (m + 1) for _ in range(n + 1)]
    best, bi, bj = 0, 0, 0
    for i in range(1, n + 1):
        Hi, Hp, ai = H[i], H[i - 1], a[i - 1]
        for j in range(1, m + 1):
            v = max(0, Hp[j - 1] + (2 if ai == b[j - 1] else -1), Hp[j] - 2, Hi[j - 1] - 2)
            Hi[j] = v
            if v > best:
                best, bi, bj = v, i, j
    i, j, ident, alen = bi, bj, 0, 0
    while i > 0 and j > 0 and H[i][j] > 0:
        s = 2 if a[i - 1] == b[j - 1] else -1
        if H[i][j] == H[i - 1][j - 1] + s:
            ident += a[i - 1] == b[j - 1]
            i -= 1
            j -= 1
        elif H[i][j] == H[i - 1][j] - 2:
            i -= 1
        else:
            j -= 1
        alen += 1
    return ident, alen


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    result = {}
    for tgt, (prefix, pdb) in TARGETS.items():
        st = gemmi.read_structure(str(BENCH / f"{prefix}_protein.pdb"))
        model = st[0]
        sup = Chem.SDMolSupplier(str(BENCH / f"{prefix}_ligands.sdf"), removeHs=False)
        lig_xyz = []
        for m in sup:
            conf = m.GetConformer()
            for k, atom in enumerate(m.GetAtoms()):
                if atom.GetAtomicNum() > 1:
                    p = conf.GetAtomPosition(k)
                    lig_xyz.append((p.x, p.y, p.z))
        lig_xyz = np.array(lig_xyz)
        ents = json.loads((ENT / f"{pdb}.json").read_text())["entities"]
        chains = {}
        for ch in model:
            seq, xyz, nca = [], [], 0
            for r in ch:
                if r.het_flag == "H" and r.name not in THREE2ONE:
                    continue
                seq.append(THREE2ONE.get(r.name, "X"))
                if r.find_atom("CA", "*"):
                    nca += 1
                for at in r:
                    if at.element.name != "H":
                        xyz.append((at.pos.x, at.pos.y, at.pos.z))
            seq = "".join(seq)
            if not xyz:  # water / ion / ligand-only chain
                continue
            xyz = np.array(xyz)
            d = np.sqrt(((xyz[:, None, :] - lig_xyz[None, :, :]) ** 2).sum(-1))
            matches = []
            for e in ents:
                ident, alen = local_align_stats(seq, e["seq"])
                matches.append(
                    {"entity": e["entity_id"], "desc": e["description"], "identical": int(ident), "aligned_len": int(alen),
                     "frac_of_chain": round(ident / max(1, len(seq)), 3), "frac_of_entity": round(ident / len(e["seq"]), 3)}
                )
            chains[ch.name] = {
                "n_residues": len(seq), "n_ca": nca, "min_dist_to_any_ligand_A": round(float(d.min()), 2),
                "n_heavy_atoms_within_10A": int((d.min(axis=1) <= CUTOFF).sum()),
                "best_entity_match": max(matches, key=lambda x: x["identical"]), "all_matches": matches,
            }
        result[tgt] = {"prefix": prefix, "ref_pdb": pdb, "n_ligand_structures": len(sup), "chains": chains}
        print(tgt, pdb, {c: (v["n_ca"], v["min_dist_to_any_ligand_A"], v["best_entity_match"]["entity"], v["best_entity_match"]["frac_of_chain"]) for c, v in chains.items()})
    # pre-registered decision
    jn = result["jnk1"]["chains"]
    decisions = {
        "cdk2": "chains A (CDK2, entity 1) + B (cyclin A2, entity 2) as a multimer (pre-registered)",
        "jnk1_chain_F_min_dist_A": jn["F"]["min_dist_to_any_ligand_A"] if "F" in jn else None,
        "jnk1_include_chain_F": bool("F" in jn and jn["F"]["min_dist_to_any_ligand_A"] <= CUTOFF),
        "rule": f"JNK1 chain F included only if any atom is within {CUTOFF} A of any ligand atom in the prepared structure (all ligand SDF entries)",
    }
    result["decisions"] = decisions
    (OUT / "chain_decisions.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(decisions, indent=1))


if __name__ == "__main__":
    main()
