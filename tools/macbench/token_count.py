"""Token counts of the macbench inputs with the featurizer boltz-mlx uses (upstream `boltz` package of the project venv; CPU only,
no model, no MSA: the YAMLs are the `_empty` arm, token counts do not depend on the MSA).

  python -I tools/macbench/token_count.py --out docs/macbench/token_counts.json [CASE_YAML ...]

Method: boltz.main.process_inputs (as boltz_mlx.cli.predict calls it), then the first item of Boltz2InferenceDataModule's dataset
(tools/prep_case.py's loader): tokens = sum(token_pad_mask), plus the count straight from the tokenised structure
(Boltz2Tokenizer) as a cross-check. Expected (ChimeraX help page, "Tokens is number of standard polymer residues plus ligand atoms"):
8rf4 129, 9moj 660, 9b3h 911. Exit 1 when an expected count does not match (nothing is adjusted).
"""
import argparse
import os
import json
import sys
import tempfile
from pathlib import Path

CACHE = Path(os.environ.get("BOLTZ_MLX_CACHE", Path.home() / ".boltz")).expanduser()
HERE = Path(__file__).resolve().parent
EXPECT = {"8rf4_empty": 129, "8rf4aff_empty": 129, "9moj_empty": 660, "9b3h_empty": 911}  # published (ChimeraX help)
EXPECT.update({k.replace("_empty", "_msa"): v for k, v in list(EXPECT.items())})  # the MSA arm has the same tokens
NOTE = {"ntca_aff_empty": "2 x 222 aa + 2 x 31 nt + AKG heavy atoms (10): 516 expected from the sequences (not published)",
        "prot117_empty": "117 aa (MoleculeDesk's page says 115; its YAML has 117)",
        "nim_8eil_empty": "NVIDIA table length 186 (one chain) + SIN heavy atoms (8) as tokens",
        "nim_7qsj_empty": "NVIDIA table length 375; RCSB entity 373 (one chain) + celecoxib heavy atoms (26)",
        "nim_8cpk_empty": "NVIDIA table length 384 (one chain) + celecoxib heavy atoms (26)",
        "nim_8gpp_empty": "NVIDIA table length 628; RCSB entity 208 x 3 chains (trimer) = 624 residues + BCT heavy atoms (4)"}


def count(yaml_path: Path, work: Path) -> dict:
    import numpy as np
    from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
    from boltz.data.tokenize.boltz2 import Boltz2Tokenizer
    from boltz.data.types import Manifest, StructureV2
    from boltz.main import check_inputs, process_inputs

    out = work / f"boltz_results_{yaml_path.stem}"
    process_inputs(data=check_inputs(yaml_path), out_dir=out, ccd_path=CACHE / "ccd.pkl", mol_dir=CACHE / "mols", msa_server_url="",
                   msa_pairing_strategy="greedy", use_msa_server=False, boltz2=True, preprocessing_threads=1, max_msa_seqs=8192)
    proc = out / "processed"
    man = Manifest.load(proc / "manifest.json")
    opt = lambda p: p if p.exists() else None  # noqa: E731
    dm = Boltz2InferenceDataModule(manifest=man, target_dir=proc / "structures", msa_dir=proc / "msa", mol_dir=CACHE / "mols",
                                   num_workers=0, constraints_dir=opt(proc / "constraints"), template_dir=opt(proc / "templates"),
                                   extra_mols_dir=opt(proc / "mols"), override_method=None, affinity=False)
    item = dm.predict_dataloader().dataset[0]
    n_feat = int(np.asarray(item["token_pad_mask"]).sum())
    rec = man.records[0]
    st = StructureV2.load(proc / "structures" / f"{rec.id}.npz")
    chains = [(c.chain_name, str(c.mol_type), int(c.num_residues)) for c in rec.chains]
    # tokenizer cross-check on the structure (Input needs the MSA dict; the tokenizer itself only reads the structure)
    from boltz.data.types import Input
    tok = Boltz2Tokenizer().tokenize(Input(st, {}, record=rec, residue_constraints=None, templates=None, extra_mols={}))
    msa = np.asarray(item["msa"])
    return {"case": yaml_path.stem, "msa_rows": int(msa.shape[0]),
            "tokens_features": n_feat, "tokens_tokenizer": int(len(tok.tokens)),
            "chains": chains, "n_atoms": int(len(st.atoms))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("yamls", nargs="*")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ys = [Path(y) for y in a.yamls] or sorted((HERE / "cases").glob("*_empty.yaml"))
    ys = [y for y in ys]
    rows, bad = [], 0
    with tempfile.TemporaryDirectory(prefix="macbench_tok_") as td:
        for y in ys:
            r = count(y, Path(td))
            exp = EXPECT.get(y.stem)
            r["expected_published"] = exp
            r["status"] = "n/a (no published count)" if exp is None else ("MATCH" if exp == r["tokens_features"] else "MISMATCH")
            if r["tokens_features"] != r["tokens_tokenizer"]:
                r["status"] += " + features/tokenizer DISAGREE"
                bad += 1
            if exp is not None and exp != r["tokens_features"]:
                bad += 1
            if y.stem in NOTE:
                r["note"] = NOTE[y.stem]
            rows.append(r)
            print(json.dumps(r))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rows, indent=1) + "\n")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
