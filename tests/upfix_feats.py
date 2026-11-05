"""Subprocess helper of tests/test_upfix_parse.py: upstream preprocessing + featurisation of some normal YAMLs, with or
without boltz_mlx.upfix installed. Prints one JSON object: per case, the sha256 of every feature tensor, the atom names
of the processed structure and of the extra (SMILES-ligand) molecules.

  python tests/upfix_feats.py stock|patched <cache dir> <work dir>
"""

import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

PROT = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQ"
PROT2 = "GSHMLEDPVDAFKQRMDEQLK"
CASES = {
    "prot_smiles": [{"protein": {"id": "A", "sequence": PROT, "msa": "empty"}},
                    {"ligand": {"id": "L", "smiles": "CC(C)[C@]1(c2cccc(-c3cccc(Cl)c3)c2)[NH+]=C(N)N(C)C1=O"}}],
    "prot_ccd": [{"protein": {"id": "A", "sequence": PROT, "msa": "empty"}}, {"ligand": {"id": "L", "ccd": "SAH"}}],
    "multi_chain": [{"protein": {"id": ["A", "B"], "sequence": PROT, "msa": "empty"}},
                    {"protein": {"id": "C", "sequence": PROT2, "msa": "empty"}},
                    {"ligand": {"id": "D", "ccd": "ATP"}}, {"ligand": {"id": "E", "smiles": "CC(=O)Oc1ccccc1C(=O)O"}},
                    {"ligand": {"id": "F", "smiles": "Brc1ccc(Cl)cc1CN"}}],
}


def main(mode: str, cache: Path, work: Path) -> dict:
    from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
    from boltz.data.types import Manifest, StructureV2
    from boltz.main import process_inputs

    from boltz_mlx.seeding import install_conformer_seed

    install_conformer_seed()
    if mode == "patched":
        from boltz_mlx.upfix import install_upstream_fixes

        install_upstream_fixes()
    out = {}
    for name, seqs in CASES.items():
        d = work / name
        d.mkdir(parents=True, exist_ok=True)
        y = d / f"{name}.yaml"
        y.write_text(yaml.safe_dump({"version": 1, "sequences": seqs}))
        process_inputs(data=[y], out_dir=d, ccd_path=cache / "ccd.pkl", mol_dir=cache / "mols", msa_server_url="",
                       msa_pairing_strategy="greedy", use_msa_server=False, boltz2=True)
        proc = d / "processed"
        man = Manifest.load(proc / "manifest.json")
        dm = Boltz2InferenceDataModule(manifest=man, target_dir=proc / "structures", msa_dir=proc / "msa", mol_dir=cache / "mols",
                                       num_workers=0, constraints_dir=None, template_dir=None,
                                       extra_mols_dir=(proc / "mols") if (proc / "mols").exists() else None,
                                       override_method=None, affinity=False)
        torch.manual_seed(0)
        batch = next(iter(dm.predict_dataloader()))
        feats = {k: hashlib.sha256(np.ascontiguousarray(v.numpy()).tobytes()).hexdigest()
                 for k, v in sorted(batch.items()) if isinstance(v, torch.Tensor)}
        st = StructureV2.load(proc / "structures" / f"{name}.npz")
        extra = {}
        if (proc / "mols").exists():
            for p in sorted((proc / "mols").glob("*.pkl")):
                obj = pickle.load(p.open("rb"))
                for key, mol in sorted(obj.items()):
                    extra[f"{p.stem}:{key}"] = [a.GetProp("name") for a in mol.GetAtoms()]
        out[name] = {"features": feats, "atom_names": [str(n) for n in st.atoms["name"]], "extra_mol_names": extra,
                     "n_tensors": len(feats)}
    return out


if __name__ == "__main__":
    print("RESULT" + json.dumps(main(sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]))))
