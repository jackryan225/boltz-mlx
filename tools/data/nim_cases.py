"""NIM benchmark cases (8eil, 8c4d, 7qsj, 8cpk): download the mmCIFs once from files.rcsb.org, read the polymer entity
sequence(s) from the file itself (_entity_poly.pdbx_seq_one_letter_code_can) and write cases/nim_<id>.yaml
(single protein chain A, `msa: empty`, or the MSA file data/external/msa/<id>.a3m with --use-msa when it exists).

The mmCIFs go to data/external/nim/ (outside the repo); a JSON summary (lengths, entities, sha256 of each file) is
written to data/external/nim/summary.json. The cif is parsed as data (gemmi), no code from it is run.

Usage: python -I tools/data/nim_cases.py
"""
import os
import argparse
import hashlib
import json
from pathlib import Path

import gemmi
import requests

IDS = ["8eil", "8c4d", "7qsj", "8cpk"]
DATA = Path(os.environ.get("BOLTZ_MLX_EXT_DATA", Path(__file__).resolve().parents[2] / "data" / "external")).expanduser() / "nim"
CASES = Path(__file__).resolve().parents[2] / "cases"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-msa", action="store_true")
    a = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    summary = {}
    for pid in IDS:
        p = DATA / f"{pid}.cif"
        if not p.exists():
            r = requests.get(f"https://files.rcsb.org/download/{pid}.cif", timeout=120)
            r.raise_for_status()
            p.write_bytes(r.content)
        doc = gemmi.cif.read(str(p))
        blk = doc.sole_block()
        poly = blk.find("_entity_poly.", ["entity_id", "type", "pdbx_seq_one_letter_code_can", "pdbx_strand_id"])
        ents = []
        for row in poly:
            ents.append({"entity_id": row[0], "type": row[1].strip("'\""), "seq": gemmi.cif.as_string(row[2]).replace("\n", "").replace(";", ""), "chains": gemmi.cif.as_string(row[3])})
        prot = [e for e in ents if "polypeptide" in e["type"]]
        assert len(prot) == 1, (pid, ents)
        seq = prot[0]["seq"]
        msa = "empty"
        f = Path(os.environ.get("BOLTZ_MLX_EXT_DATA", Path(__file__).resolve().parents[2] / "data" / "external")).expanduser() / "msa" / f"{pid}.a3m"
        if a.use_msa and f.exists():
            msa = str(f)
        (CASES / f"nim_{pid}.yaml").write_text(f"version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: {seq}\n      msa: {msa}\n")
        summary[pid] = {"length": len(seq), "n_entities": len(ents), "entity_chains": prot[0]["chains"],
                        "title": blk.find_value("_struct.title"), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
        print(pid, summary[pid]["length"], prot[0]["chains"], summary[pid]["bytes"], summary[pid]["sha256"][:12])
    (DATA / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
