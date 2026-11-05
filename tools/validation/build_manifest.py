"""Build validation/manifest.json, Boltz YAML inputs and copy mmCIFs for the hand-picked final cases.

The final cases are chosen from the filtered pool (validation/pool.json, produced by filter_cases.py)
by hand for fold diversity and size spread (3 x ~80, 4 x ~150, 3 x ~230).
"""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parents[2] / "data" / "validation"

SELECTED = ["8J9R", "8FJ8", "8OXK", "8TI5", "8T98", "8TA1", "8JQU", "8TSZ", "8C0M", "8YWO"]

NOTES = (
    "Candidates: RCSB search (query below) -> 5156 entries; seeded random sample (seed 0, n=300) -> metadata via "
    "data.rcsb.org GraphQL + mmCIF via files.rcsb.org; post-filters on the mmCIF (gemmi): protein, no nonstandard "
    "residues, 60-300 modeled residues, >90% of SEQRES modeled, CA present for all modeled residues, no X/U in "
    "sequence (231/300 pass). Final 10 chosen by hand from the passing pool for fold/size diversity. "
    "Release cutoff 2023-07-01: the Boltz-2 repo docs (README/docs) state no training cutoff; the paper's cutoff "
    "(2023-06-01, from memory) is earlier, so 2023-07-01 is the later of the two."
)


def main():
    pool = json.loads((ROOT / "validation" / "pool.json").read_text())
    query = json.loads((ROOT / "validation" / "query.json").read_text())
    (DATA / "cif").mkdir(parents=True, exist_ok=True)
    (ROOT / "validation" / "yaml").mkdir(parents=True, exist_ok=True)
    cases = []
    for pid in SELECTED:
        r = pool[pid]
        assert r["pass"], pid
        cid = pid.lower()
        shutil.copy(DATA / "pool" / f"{pid}.cif", DATA / "cif" / f"{cid}.cif")
        (ROOT / "validation" / "yaml" / f"{cid}.yaml").write_text(
            "version: 1\nsequences:\n  - protein:\n      id: A\n"
            f"      sequence: {r['seq']}\n      msa: empty\n"
        )
        cases.append(
            {
                "id": cid,
                "pdb_id": pid,
                "chain": r["chain"],
                "release_date": r["release"],
                "resolution": r["resolution"],
                "method": r["method"],
                "length_seqres": r["seqres_len"],
                "length_modeled": r["n_modeled"],
                "frac_modeled": round(r["frac_modeled"], 3),
                "description": r["desc"],
                "cif": f"data/validation/cif/{cid}.cif",
                "yaml": f"validation/yaml/{cid}.yaml",
            }
        )
    cases.sort(key=lambda c: c["length_seqres"])
    manifest = {
        "release_cutoff": "initial release date >= 2023-07-01",
        "selection_notes": NOTES,
        "query_endpoint": "https://search.rcsb.org/rcsbsearch/v2/query",
        "query": query,
        "sample": {"seed": 0, "n": 300, "script": "tools/validation/filter_cases.py"},
        "cases": cases,
    }
    (ROOT / "validation" / "manifest.json").write_text(json.dumps(manifest, indent=1))
    for c in cases:
        print(c["id"], c["release_date"], c["resolution"], c["length_seqres"], c["length_modeled"], c["description"])


if __name__ == "__main__":
    main()
