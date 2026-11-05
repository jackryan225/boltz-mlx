"""Filter candidate RCSB entries (validation/candidate_ids.json) using the data API + the mmCIF itself.

For a seeded random sample of candidates: fetch GraphQL metadata (release date, resolution, entity
sequence, 30% sequence cluster), download the mmCIF (files.rcsb.org) into a pool directory, and compute
modeled-residue statistics with gemmi. Writes validation/pool.json (all stats, pass/fail + reasons).

Usage: python tools/validation/filter_cases.py [--n 300] [--seed 0] [--pool DIR]
"""
import argparse
import json
import random
import time
from pathlib import Path

import gemmi
import requests

ROOT = Path(__file__).resolve().parents[2]
GQL = "https://data.rcsb.org/graphql"
FILES = "https://files.rcsb.org/download/{}.cif"
STD = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split())

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
    }
  }
}
"""


def fetch_meta(ids):
    r = requests.post(GQL, json={"query": GQL_Q, "variables": {"ids": ids}}, timeout=120)
    r.raise_for_status()
    return {e["rcsb_id"]: e for e in r.json()["data"]["entries"] if e}


def analyze(cif_path: Path, seq: str):
    st = gemmi.read_structure(str(cif_path))
    st.setup_entities()
    model = st[0]
    out = {"n_chains": len(model)}
    ch = model[0]
    # modeled polymer residues with CA
    names, nca, hetero = [], 0, 0
    for r in ch:
        if r.het_flag == "H":
            hetero += 1
            continue
        names.append(r.name)
        if r.find_atom("CA", "*"):
            nca += 1
    nonstd = sorted({n for n in names if n not in STD})
    out.update(chain=ch.name, n_modeled=len(names), n_ca=nca, nonstd=nonstd, n_hetero_in_chain=hetero)
    out["seqres_len"] = len(seq)
    out["frac_modeled"] = len(names) / max(1, len(seq))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pool", type=Path, default=Path(__file__).resolve().parents[2] / "data" / "validation" / "pool")
    args = ap.parse_args()
    args.pool.mkdir(parents=True, exist_ok=True)

    ids = json.loads((ROOT / "validation" / "candidate_ids.json").read_text())
    rng = random.Random(args.seed)
    sample = rng.sample(ids, min(args.n, len(ids)))
    pool = {}
    for i in range(0, len(sample), 50):
        meta = fetch_meta(sample[i : i + 50])
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
                "cluster30": next((c["cluster_id"] for c in ent["rcsb_cluster_membership"] or [] if c["identity"] == 30), None),
            }
            cif = args.pool / f"{pid}.cif"
            if not cif.exists():
                rr = requests.get(FILES.format(pid), timeout=60)
                rr.raise_for_status()
                cif.write_bytes(rr.content)
                time.sleep(0.1)
            try:
                rec.update(analyze(cif, seq))
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
            if not (60 <= rec["n_modeled"] <= 300):
                why.append("modeled length out of range")
            if rec["frac_modeled"] <= 0.90:
                why.append("<=90% modeled")
            if rec["n_ca"] != rec["n_modeled"]:
                why.append("missing CA")
            if "X" in rec["seq"] or "U" in rec["seq"]:
                why.append("X/U in sequence")
        rec["pass"] = not why
        rec["fail_reasons"] = why
    (ROOT / "validation" / "pool.json").write_text(json.dumps(pool, indent=1))
    ok = [r for r in pool.values() if r["pass"]]
    print("sampled", len(pool), "passing", len(ok))
    for r in sorted(ok, key=lambda r: r["n_modeled"]):
        print(r["id"], r["release"], r["resolution"], r["n_modeled"], r["seqres_len"], r["cluster30"], r["desc"][:50])


if __name__ == "__main__":
    main()
