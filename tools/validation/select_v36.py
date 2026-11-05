"""Round 2: select the V36 extension (6 chains, 380-500 residues) and the V66 extension (30 further chains).

Same filters as round 1 (select_cases.py query + filter_cases.py post-filters): X-ray <= 2.5 A, initial release >= 2023-07-01,
one polymer entity / one protein chain / one deposited instance, protein, no nonstandard residues, > 90% of SEQRES modelled,
CA present for every modelled residue, no X/U in the sequence. Added for round 2: a UniProt accession must exist and be
distinct from every case already in the set (and from each other). (The RCSB 30%-identity cluster membership is null in the
data API for these entries, as in round 1, so only UniProt is enforced; the code would also enforce clusters if present.)

Deterministic procedure (no hand picking):
  1. Run the round-1 RCSB query with only the size range changed; save the candidate ids (RCSB order) in validation/.
  2. Permute the SORTED candidate ids with random.Random(seed).shuffle.
  3. Walk the permutation in chunks of 50 (GraphQL metadata + mmCIF into the pool dir + gemmi stats), accept an entry if it
     passes the post-filters, falls in a still-open size bucket, and has a UniProt accession not used yet.
     Stop when all buckets are full. Every examined entry is logged with its verdict.
  stage v36: seed 0, bucket 380-500 x 6, exclusion set = the 30 cases of validation/manifest_all.json.
  stage v66: seed 1, buckets matching V30's size spread (8 x 60-120, 11 x 120-200, 8 x 200-300, 3 x 300-500; the last bucket
             is widened from V30's 300-400 to 300-500 because V36 now covers the headline size), exclusion set = V36.
Writes validation/manifest_v36.json / manifest_v66_ext.json, YAMLs (validation/yaml/<id>.yaml, msa: empty) and mmCIFs
(<data>/validation/cif/<id>.cif). Public RCSB endpoints only (search/data/files.rcsb.org). CPU only.

Usage: python -I tools/validation/select_v36.py --stage v36|v66
"""
import argparse
import copy
import json
import random
import shutil
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import filter_cases as fc  # noqa: E402
import select_cases as sc  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
VAL = ROOT / "validation"
DATA = Path(__file__).resolve().parents[2] / "data" / "validation"
POOL = DATA / "pool"

GQL_Q = """
query($ids: [String!]!) {
  entries(entry_ids: $ids) {
    rcsb_id
    rcsb_accession_info { initial_release_date }
    rcsb_entry_info { resolution_combined deposited_polymer_monomer_count }
    exptl { method }
    struct { title }
    polymer_entities {
      entity_poly { pdbx_seq_one_letter_code_can pdbx_strand_id nstd_monomer rcsb_entity_polymer_type }
      rcsb_polymer_entity { pdbx_description }
      rcsb_cluster_membership { cluster_id identity }
      rcsb_polymer_entity_container_identifiers { reference_sequence_identifiers { database_name database_accession } }
    }
  }
}
"""

STAGES = {
    "v36": {"seed": 0, "lo": 380, "hi": 500, "buckets": {"380-500": (380, 500, 6)}, "out": "manifest_v36.json", "cand": "candidate_ids_380_500.json", "query": "query_380_500.json"},
    "v66": {"seed": 1, "lo": None, "hi": None,
            "buckets": {"60-120": (60, 120, 8), "120-200": (121, 200, 11), "200-300": (201, 300, 8), "300-500": (301, 500, 3)},
            "out": "manifest_v66_ext.json", "cand": None, "query": None},
}


def gql(ids):
    r = requests.post(fc.GQL, json={"query": GQL_Q, "variables": {"ids": ids}}, timeout=120)
    r.raise_for_status()
    return {e["rcsb_id"]: e for e in r.json()["data"]["entries"] if e}


def candidates(lo, hi, cand_name, query_name):
    q = copy.deepcopy(sc.QUERY)
    for node in q["query"]["nodes"]:
        if node["parameters"]["attribute"] == "rcsb_entry_info.deposited_polymer_monomer_count":
            node["parameters"]["value"] = {"from": lo, "to": hi, "include_lower": True, "include_upper": True}
    path = VAL / cand_name
    if path.exists():  # saved list is authoritative (pre-registration); do not re-query
        return json.loads(path.read_text()), json.loads((VAL / query_name).read_text())
    r = requests.post(sc.SEARCH, json=q, timeout=60)
    r.raise_for_status()
    d = r.json()
    ids = [x["identifier"] for x in d["result_set"]]
    print("query", lo, hi, "total_count", d["total_count"], "returned", len(ids))
    path.write_text(json.dumps(ids))
    (VAL / query_name).write_text(json.dumps(q, indent=1))
    return ids, q


def uniprot_of(e):
    accs = []
    for ent in e["polymer_entities"]:
        for ref in (ent["rcsb_polymer_entity_container_identifiers"] or {}).get("reference_sequence_identifiers") or []:
            if ref["database_name"] == "UniProt":
                accs.append(ref["database_accession"])
    return sorted(set(accs))


def record(pid, e):
    ent = e["polymer_entities"][0]
    ep = ent["entity_poly"]
    seq = ep["pdbx_seq_one_letter_code_can"].replace("\n", "")
    return {
        "id": pid, "release": e["rcsb_accession_info"]["initial_release_date"][:10], "resolution": e["rcsb_entry_info"]["resolution_combined"][0],
        "method": e["exptl"][0]["method"], "title": e["struct"]["title"], "desc": ent["rcsb_polymer_entity"]["pdbx_description"], "seq": seq,
        "poly_type": ep["rcsb_entity_polymer_type"], "nstd_monomer": ep["nstd_monomer"], "chain_ids": ep["pdbx_strand_id"],
        "cluster30": next((c["cluster_id"] for c in ent["rcsb_cluster_membership"] or [] if c["identity"] == 30), None),
        "uniprot": uniprot_of(e),
    }


def verdict(rec, hi):
    why = []
    if "error" in rec:
        return ["parse error"]
    if rec["release"] < "2023-07-01":
        why.append("release < 2023-07-01")
    if rec["resolution"] > 2.5:
        why.append("resolution > 2.5")
    if rec["method"] != "X-RAY DIFFRACTION":
        why.append("not X-ray")
    if rec["poly_type"] != "Protein":
        why.append("not protein")
    if rec["nstd_monomer"] == "yes" or rec["nonstd"]:
        why.append("nonstandard residues")
    if not (60 <= rec["n_modeled"] <= hi):
        why.append("modeled length out of range")
    if rec["frac_modeled"] <= 0.90:
        why.append("<=90% modeled")
    if rec["n_ca"] != rec["n_modeled"]:
        why.append("missing CA")
    if "X" in rec["seq"] or "U" in rec["seq"]:
        why.append("X/U in sequence")
    return why


def case_dict(r, cid):
    return {
        "id": cid, "pdb_id": r["id"], "chain": r["chain"], "release_date": r["release"], "resolution": r["resolution"], "method": r["method"],
        "length_seqres": r["seqres_len"], "length_modeled": r["n_modeled"], "frac_modeled": round(r["frac_modeled"], 3), "description": r["desc"],
        "cif": f"data/validation/cif/{cid}.cif", "yaml": f"validation/yaml/{cid}.yaml",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=list(STAGES), required=True)
    a = ap.parse_args()
    cfg = STAGES[a.stage]
    POOL.mkdir(parents=True, exist_ok=True)
    (DATA / "cif").mkdir(parents=True, exist_ok=True)
    (VAL / "yaml").mkdir(parents=True, exist_ok=True)

    base = json.loads((VAL / "manifest_all.json").read_text())
    existing = [c["pdb_id"] for c in base["cases"]]
    if a.stage == "v66":
        v36 = json.loads((VAL / "manifest_v36.json").read_text())
        excluded_cases = [c["pdb_id"] for c in v36["cases"]]  # V30 + 6
    else:
        excluded_cases = existing

    # UniProt + cluster30 of the exclusion set
    up_used, cl_used, owner = set(), set(), {}
    meta = {}
    for i in range(0, len(excluded_cases), 50):
        meta.update(gql(excluded_cases[i : i + 50]))
    for pid in excluded_cases:
        rec = record(pid, meta[pid])
        for u in rec["uniprot"]:
            up_used.add(u)
            owner[u] = pid
        if rec["cluster30"]:
            cl_used.add(rec["cluster30"])
    print("exclusion set:", len(excluded_cases), "cases;", len(up_used), "UniProt;", len(cl_used), "clusters")

    # candidates
    if a.stage == "v36":
        ids, query = candidates(cfg["lo"], cfg["hi"], cfg["cand"], cfg["query"])
        hi = 500
    else:
        ids_small = json.loads((VAL / "candidate_ids.json").read_text())  # 60-300, round 1 query
        ids_big, _ = candidates(300, 500, "candidate_ids_300_500.json", "query_300_500.json")
        ids = sorted(set(ids_small) | set(ids_big))
        query = {"60-300": json.loads((VAL / "query.json").read_text()), "300-500": json.loads((VAL / "query_300_500.json").read_text())}
        hi = 500
    ids = sorted(set(ids) - set(excluded_cases) - {x.upper() for x in excluded_cases})
    order = list(ids)
    random.Random(cfg["seed"]).shuffle(order)
    print("candidates:", len(ids), "seed", cfg["seed"])

    need = {b: n for b, (_, _, n) in cfg["buckets"].items()}
    got = {b: [] for b in need}
    log = []
    pos = 0
    while any(need[b] > len(got[b]) for b in need) and pos < len(order):
        chunk = order[pos : pos + 50]
        pos += len(chunk)
        m = gql(chunk)
        for pid in chunk:
            if all(len(got[b]) >= need[b] for b in need):
                break
            if pid not in m:
                log.append({"id": pid, "verdict": "no metadata"})
                continue
            e = m[pid]
            if len(e["polymer_entities"]) != 1:
                log.append({"id": pid, "verdict": "reject: polymer entities != 1"})
                continue
            rec = record(pid, e)
            # cheap pre-checks before downloading
            n = len(rec["seq"])
            bucket = next((b for b, (lo, hi_, _) in cfg["buckets"].items() if lo <= n <= hi_), None)
            if bucket is None or len(got[bucket]) >= need[bucket]:
                log.append({"id": pid, "verdict": "skip: bucket full/out of range", "seqres_len": n})
                continue
            pre = []
            if rec["release"] < "2023-07-01" or rec["resolution"] > 2.5 or rec["poly_type"] != "Protein" or rec["nstd_monomer"] == "yes" or "X" in rec["seq"] or "U" in rec["seq"]:
                pre.append("pre-filter")
            if not rec["uniprot"]:
                pre.append("no UniProt accession")
            if pre:
                log.append({"id": pid, "verdict": "reject: " + ",".join(pre), "seqres_len": n})
                continue
            if any(u in up_used for u in rec["uniprot"]) or (rec["cluster30"] and rec["cluster30"] in cl_used):
                log.append({"id": pid, "verdict": "reject: UniProt/cluster30 already used", "seqres_len": n, "uniprot": rec["uniprot"], "cluster30": rec["cluster30"]})
                continue
            cif = POOL / f"{pid}.cif"
            if not cif.exists():
                rr = requests.get(fc.FILES.format(pid), timeout=60)
                rr.raise_for_status()
                cif.write_bytes(rr.content)
                time.sleep(0.1)
            try:
                rec.update(fc.analyze(cif, rec["seq"]))
            except Exception as ex:  # noqa: BLE001
                rec["error"] = repr(ex)
            why = verdict(rec, hi)
            if why:
                log.append({"id": pid, "verdict": "reject: " + ",".join(why), "seqres_len": n})
                continue
            got[bucket].append(rec)
            up_used.update(rec["uniprot"])
            if rec["cluster30"]:
                cl_used.add(rec["cluster30"])
            log.append({"id": pid, "verdict": f"ACCEPT {bucket}", "seqres_len": n, "uniprot": rec["uniprot"], "cluster30": rec["cluster30"]})
            print("ACCEPT", bucket, pid, n, rec["uniprot"], rec["desc"][:50], flush=True)
        print(f"walked {pos}/{len(order)}", {b: f"{len(got[b])}/{need[b]}" for b in need}, flush=True)
    assert all(len(got[b]) == need[b] for b in need), {b: len(got[b]) for b in need}

    new_cases = []
    for b in need:
        for rec in got[b]:
            cid = rec["id"].lower()
            shutil.copy(POOL / f"{rec['id']}.cif", DATA / "cif" / f"{cid}.cif")
            (VAL / "yaml" / f"{cid}.yaml").write_text(f"version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: {rec['seq']}\n      msa: empty\n")
            c = case_dict(rec, cid)
            new_cases.append((c, rec, b))
    new_cases.sort(key=lambda t: t[0]["length_seqres"])
    ext = {
        "seed": cfg["seed"], "buckets": {b: n for b, (_, _, n) in cfg["buckets"].items()}, "n_candidates": len(ids), "n_examined": pos,
        "procedure": __doc__.split("Deterministic procedure")[1].split("Writes ")[0].strip(),
        "uniprot": {c["pdb_id"]: r["uniprot"] for c, r, _ in new_cases}, "cluster30": {c["pdb_id"]: r["cluster30"] for c, r, _ in new_cases},
        "size_bucket": {c["pdb_id"]: b for c, _, b in new_cases}, "selection_log": log,
    }
    if a.stage == "v36":
        out = dict(base)
        out["selection_notes"] = base["selection_notes"] + " || V36 EXTENSION (round 2): 6 more chains of 380-500 residues, same RCSB query/post-filters with the size range changed (validation/query_380_500.json), seed-0 permutation of the sorted candidate ids, distinct UniProt from the 30 existing cases and from each other. See v36_extension."
        out["query_380_500"] = query
        out["cases"] = sorted(base["cases"] + [c for c, _, _ in new_cases], key=lambda c: c["length_seqres"])
        out["v36_extension"] = ext
        out["v36_new_case_ids"] = [c["id"] for c, _, _ in new_cases]
        assert len(out["cases"]) == 36 and len({c["id"] for c in out["cases"]}) == 36
    else:
        out = {k: base[k] for k in ("release_cutoff", "query_endpoint")}
        out["selection_notes"] = ("V66 extension (round 2): 30 further chains, same filters as V30/V36, seed-1 permutation of the sorted union of the 60-300 (round 1) and 300-500 candidate ids, "
                                  "size spread matching V30 (8 x 60-120, 11 x 120-200, 8 x 200-300, 3 x 300-500), distinct UniProt from V36 and from each other. Not run unless the null calibration shows L3-P underpowered on V36.")
        out["query"] = query
        out["cases"] = [c for c, _, _ in new_cases]
        out["v66_extension"] = ext
        assert len(out["cases"]) == 30
    (VAL / cfg["out"]).write_text(json.dumps(out, indent=1))
    for c, r, b in new_cases:
        print(c["id"], c["release_date"], c["resolution"], c["length_seqres"], c["length_modeled"], b, ",".join(r["uniprot"]), c["description"])


if __name__ == "__main__":
    main()
