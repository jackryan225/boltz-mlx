"""Extend the filtered validation pool (step 1b): add a 300-400 residue bucket and UniProt accessions.

The original query (select_cases.py) is limited to 60-300 residues. This re-runs the *same* query with only the
size range changed to 300-400 (validation/query_300_400.json, validation/candidate_ids_300_400.json), samples
candidates with a seeded RNG, downloads the mmCIFs into the shared pool dir, applies the same post-filters as
filter_cases.py (modeled range widened to 60-400) and writes validation/pool_300_400.json.

It also fetches the UniProt accession(s) of every passing entry in validation/pool.json + the new pool + the existing
manifest cases from data.rcsb.org, and writes validation/uniprot.json (used to avoid near-duplicate proteins).

Public RCSB endpoints only (search.rcsb.org, data.rcsb.org, files.rcsb.org). CPU only.
Usage: python tools/validation/extend_pool.py [--n 200] [--seed 0]
"""
import argparse
import copy
import json
import random
import time
from pathlib import Path

import requests

import filter_cases as fc
import select_cases as sc

ROOT = Path(__file__).resolve().parents[2]
GQL_UNIPROT = """
query($ids: [String!]!) {
  entries(entry_ids: $ids) {
    rcsb_id
    polymer_entities {
      rcsb_polymer_entity_container_identifiers {
        reference_sequence_identifiers { database_name database_accession }
      }
      rcsb_polymer_entity { pdbx_description }
    }
  }
}
"""


def uniprot_for(ids):
    out = {}
    ids = list(ids)
    for i in range(0, len(ids), 100):
        r = requests.post(fc.GQL, json={"query": GQL_UNIPROT, "variables": {"ids": ids[i : i + 100]}}, timeout=120)
        r.raise_for_status()
        for e in r.json()["data"]["entries"]:
            if not e:
                continue
            accs = []
            for ent in e["polymer_entities"]:
                for ref in (ent["rcsb_polymer_entity_container_identifiers"] or {}).get("reference_sequence_identifiers") or []:
                    if ref["database_name"] == "UniProt":
                        accs.append(ref["database_accession"])
            out[e["rcsb_id"]] = sorted(set(accs))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pool", type=Path, default=Path(__file__).resolve().parents[2] / "data" / "validation" / "pool")
    ap.add_argument("--lo", type=int, default=300)
    ap.add_argument("--hi", type=int, default=400)
    a = ap.parse_args()
    a.pool.mkdir(parents=True, exist_ok=True)

    # 1. same query, size range changed
    q = copy.deepcopy(sc.QUERY)
    for node in q["query"]["nodes"]:
        if node["parameters"]["attribute"] == "rcsb_entry_info.deposited_polymer_monomer_count":
            node["parameters"]["value"] = {"from": a.lo, "to": a.hi, "include_lower": True, "include_upper": True}
    r = requests.post(sc.SEARCH, json=q, timeout=60)
    r.raise_for_status()
    d = r.json()
    ids = [x["identifier"] for x in d["result_set"]]
    print("total_count", d["total_count"], "returned", len(ids))
    (ROOT / "validation" / "query_300_400.json").write_text(json.dumps(q, indent=1))
    (ROOT / "validation" / "candidate_ids_300_400.json").write_text(json.dumps(ids))

    # 2. seeded sample -> metadata + mmCIF + post-filters (same as filter_cases.py, size range widened)
    sample = random.Random(a.seed).sample(ids, min(a.n, len(ids)))
    pool = {}
    for i in range(0, len(sample), 50):
        meta = fc.fetch_meta(sample[i : i + 50])
        for pid, e in meta.items():
            ent = e["polymer_entities"][0]
            ep = ent["entity_poly"]
            seq = ep["pdbx_seq_one_letter_code_can"].replace("\n", "")
            rec = {
                "id": pid,
                "release": e["rcsb_accession_info"]["initial_release_date"][:10],
                "resolution": e["rcsb_entry_info"]["resolution_combined"][0],
                "method": e["exptl"][0]["method"],
                "title": e["struct"]["title"],
                "desc": ent["rcsb_polymer_entity"]["pdbx_description"],
                "seq": seq,
                "poly_type": ep["rcsb_entity_polymer_type"],
                "nstd_monomer": ep["nstd_monomer"],
                "chain_ids": ep["pdbx_strand_id"],
            }
            cif = a.pool / f"{pid}.cif"
            if not cif.exists():
                rr = requests.get(fc.FILES.format(pid), timeout=60)
                rr.raise_for_status()
                cif.write_bytes(rr.content)
                time.sleep(0.1)
            try:
                rec.update(fc.analyze(cif, seq))
            except Exception as ex:  # noqa: BLE001
                rec["error"] = repr(ex)
            pool[pid] = rec
        print(f"{i + len(meta)}/{len(sample)}", flush=True)
    for rec in pool.values():
        why = []
        if "error" in rec:
            why.append("parse error")
        else:
            if rec["poly_type"] != "Protein":
                why.append("not protein")
            if rec["nstd_monomer"] == "yes" or rec["nonstd"]:
                why.append("nonstandard residues")
            if not (60 <= rec["n_modeled"] <= a.hi):
                why.append("modeled length out of range")
            if rec["frac_modeled"] <= 0.90:
                why.append("<=90% modeled")
            if rec["n_ca"] != rec["n_modeled"]:
                why.append("missing CA")
            if "X" in rec["seq"] or "U" in rec["seq"]:
                why.append("X/U in sequence")
        rec["pass"] = not why
        rec["fail_reasons"] = why
    (ROOT / "validation" / "pool_300_400.json").write_text(json.dumps(pool, indent=1))
    print("sampled", len(pool), "passing", sum(r["pass"] for r in pool.values()))

    # 3. UniProt accessions for everything that could be selected (+ the existing 10)
    old = json.loads((ROOT / "validation" / "pool.json").read_text())
    manifest = json.loads((ROOT / "validation" / "manifest.json").read_text())
    want = {k for k, v in old.items() if v["pass"]} | {k for k, v in pool.items() if v["pass"]} | {c["pdb_id"] for c in manifest["cases"]}
    up = uniprot_for(sorted(want))
    (ROOT / "validation" / "uniprot.json").write_text(json.dumps(up, indent=1, sort_keys=True))
    print("uniprot for", len(up), "entries;", sum(1 for v in up.values() if v), "with an accession")


if __name__ == "__main__":
    main()
