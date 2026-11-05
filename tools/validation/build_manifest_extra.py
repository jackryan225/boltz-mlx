"""Build validation/manifest_extra.json (20 more cases), validation/manifest_all.json (10 + 20) and the YAML/mmCIF inputs.

Cases are hand-picked from the filtered pools (validation/pool.json for <=300 residues, validation/pool_300_400.json for
300-400; see extend_pool.py) for fold diversity and a size spread of 5 x 60-120, 7 x 120-200, 5 x 200-300, 3 x 300-400.
Hard constraints are re-checked here: pass flag, release >= 2023-07-01, resolution <= 2.5, no two cases (including the
original 10) share a UniProt accession or a description keyword. validation/manifest.json is never modified.
"""
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parents[2] / "data" / "validation"

# (pdb id, short fold note)
SELECTED = [
    # 60-120
    ("9EYB", "SH3 domain (all-beta), spectrin"),
    ("9N3A", "CRISPR-associated helicase domain"),
    ("9B26", "EF-hand calcium-binding (all-alpha)"),
    ("9Z0M", "Ig-like V domain"),
    ("7IRE", "viral nsp1 N-terminal domain (beta-barrel)"),
    # 120-200
    ("8K9N", "cupredoxin (type-1 copper, beta-sandwich), 0.86 A"),
    ("8CHR", "FKBP peptidyl-prolyl isomerase domain"),
    ("8RMT", "galectin carbohydrate-recognition domain (beta-sandwich)"),
    ("9DTI", "globin (all-alpha)"),
    ("9F9U", "transcription elongation factor (alpha/beta, S1/KH)"),
    ("9EQB", "ferritin 4-helix bundle"),
    ("8X5S", "P-loop NTPase (shikimate kinase)"),
    # 200-300
    ("9FCL", "SAM-dependent methyltransferase (Rossmann)"),
    ("7H67", "trypsin-like serine protease (chymase)"),
    ("9RFG", "metallo-beta-lactamase (alpha-beta/beta-alpha)"),
    ("9THS", "heme peroxidase (mostly alpha)"),
    ("9M24", "Kelch beta-propeller"),
    # 300-400
    ("8VCW", "radical SAM TIM-barrel (biotin synthase)"),
    ("9G8V", "subtilisin-like serine protease"),
    ("8JV3", "two-domain periplasmic solute-binding protein"),
]

NOTES = (
    "20 additional cases drawn from the same filtered pool as manifest.json (RCSB search query, X-ray <= 2.5 A, "
    "initial release >= 2023-07-01, one polymer entity / one protein chain, then mmCIF post-filters: protein, no "
    "nonstandard residues, > 90% of SEQRES modeled, CA present for every modeled residue, no X/U in sequence). "
    "Sizes <= 300 come from the original seeded sample (validation/pool.json, seed 0, n=300; 221 passing, 10 already "
    "used excluded). That pool has at most one 300-residue entry, so the 300-400 bucket re-runs the identical query "
    "with only the size range changed (validation/query_300_400.json -> 2800 entries), seeded sample (seed 0, n=200, "
    "tools/validation/extend_pool.py) with the same post-filters (142 pass; validation/pool_300_400.json). "
    "Selection by hand for fold/size diversity: 5 x 60-120, 7 x 120-200, 5 x 200-300, 3 x 300-400 residues "
    "(SEQRES length). Redundancy control: UniProt accessions (data.rcsb.org, validation/uniprot.json) are pairwise "
    "distinct across all 30 cases (incl. the original 10), no two cases share a description keyword, and from the many "
    "same-protein series in the pool (fragment-screen / group depositions of nsp1, Mpro, endothiapepsin, carbonic "
    "anhydrase II, lysozyme, ...) at most one entry was considered. The original 10 are unchanged. "
    "There is no torch reference for these cases (data/<id>/ref does not exist); score against the experimental mmCIF."
)

STOP = {"protein", "domain", "chain", "subunit", "family", "putative", "like", "type", "of", "and", "the", "ec", "complex", "associated", "binding", "alpha", "beta", "gamma", "protein"}


def keywords(desc: str) -> set[str]:
    return {w for w in re.split(r"[^a-z0-9]+", desc.lower()) if len(w) > 2 and w not in STOP}


def case_dict(r, cid):
    return {
        "id": cid,
        "pdb_id": r["id"],
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


def main():
    pool_small = json.loads((ROOT / "validation" / "pool.json").read_text())
    pool = {**pool_small, **json.loads((ROOT / "validation" / "pool_300_400.json").read_text())}
    up = json.loads((ROOT / "validation" / "uniprot.json").read_text())
    old = json.loads((ROOT / "validation" / "manifest.json").read_text())
    old_ids = {c["pdb_id"] for c in old["cases"]}
    (DATA / "cif").mkdir(parents=True, exist_ok=True)
    (ROOT / "validation" / "yaml").mkdir(parents=True, exist_ok=True)

    seen_up = {a: c["pdb_id"] for c in old["cases"] for a in up[c["pdb_id"]]}
    seen_kw = {w: c["pdb_id"] for c in old["cases"] for w in keywords(c["description"])}
    cases = []
    for pid, fold in SELECTED:
        r = pool[pid]
        assert r["pass"] and pid not in old_ids, pid
        assert r["release"] >= "2023-07-01" and r["resolution"] <= 2.5 and r["frac_modeled"] > 0.9, pid
        n = r["seqres_len"]
        bucket = "60-120" if n <= 120 else "120-200" if n <= 200 else "200-300" if n <= 300 else "300-400"
        accs = up.get(pid, [])
        for a in accs:
            assert a not in seen_up, (pid, "uniprot dup", a, seen_up[a])
            seen_up[a] = pid
        for w in keywords(r["desc"]):
            assert w not in seen_kw, (pid, "keyword dup", w, seen_kw[w])
            seen_kw[w] = pid
        cid = pid.lower()
        shutil.copy(DATA / "pool" / f"{pid}.cif", DATA / "cif" / f"{cid}.cif")
        (ROOT / "validation" / "yaml" / f"{cid}.yaml").write_text(
            "version: 1\nsequences:\n  - protein:\n      id: A\n"
            f"      sequence: {r['seq']}\n      msa: empty\n"
        )
        c = case_dict(r, cid)
        c.update(uniprot=accs, size_bucket=bucket, fold_note=fold, source_pool="pool.json" if pid in pool_small else "pool_300_400.json")
        cases.append(c)
    cases.sort(key=lambda c: c["length_seqres"])
    counts = {b: sum(c["size_bucket"] == b for c in cases) for b in ("60-120", "120-200", "200-300", "300-400")}
    assert counts == {"60-120": 5, "120-200": 7, "200-300": 5, "300-400": 3}, counts

    extra = {
        "release_cutoff": old["release_cutoff"],
        "selection_notes": NOTES,
        "size_bucket_counts": counts,
        "query_endpoint": old["query_endpoint"],
        "query": old["query"],
        "query_300_400": json.loads((ROOT / "validation" / "query_300_400.json").read_text()),
        "sample": {"seed": 0, "n": 300, "script": "tools/validation/filter_cases.py",
                   "n_300_400": 200, "script_300_400": "tools/validation/extend_pool.py"},
        "cases": cases,
    }
    (ROOT / "validation" / "manifest_extra.json").write_text(json.dumps(extra, indent=1))

    keep = list(old["cases"][0].keys())
    allc = sorted(old["cases"] + [{k: c[k] for k in keep} for c in cases], key=lambda c: c["length_seqres"])
    allm = dict(old)
    allm["selection_notes"] = old["selection_notes"] + " || EXTENSION (20 more cases, see manifest_extra.json): " + NOTES
    allm["query_300_400"] = extra["query_300_400"]
    allm["cases"] = allc
    (ROOT / "validation" / "manifest_all.json").write_text(json.dumps(allm, indent=1))
    assert len({c["id"] for c in allc}) == 30
    for c in cases:
        print(c["id"], c["release_date"], c["resolution"], c["length_seqres"], c["length_modeled"], ",".join(c["uniprot"]), c["description"])


if __name__ == "__main__":
    main()
