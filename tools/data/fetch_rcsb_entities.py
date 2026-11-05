"""Fetch RCSB polymer-entity sequences (data.rcsb.org core REST API) for a list of PDB ids.

Writes one JSON per entry to --out (default data/external/rcsb_entities/<ID>.json) with, per entity: the
canonical one-letter sequence (pdbx_seq_one_letter_code_can), the author chain ids, the description and the
polymer type; plus the sha256 of each raw API response so it can be recorded.

Public RCSB endpoint only (data.rcsb.org). CPU only. Usage:
  python tools/data/fetch_rcsb_entities.py 1H1Q 4GIH 2GMX 3FLN 4DJW 2ZFF
"""
import os
import argparse
import hashlib
import json
import time
from pathlib import Path

import requests

BASE = "https://data.rcsb.org/rest/v1/core"


def fetch_entry(pid: str) -> dict:
    pid = pid.upper()
    r = requests.get(f"{BASE}/entry/{pid}", timeout=60)
    r.raise_for_status()
    entry = r.json()
    n_ent = entry["rcsb_entry_info"]["polymer_entity_count"]
    ents = []
    for i in range(1, n_ent + 1):
        rr = requests.get(f"{BASE}/polymer_entity/{pid}/{i}", timeout=60)
        rr.raise_for_status()
        e = rr.json()
        ep = e["entity_poly"]
        ents.append(
            {
                "entity_id": i,
                "description": e["rcsb_polymer_entity"]["pdbx_description"],
                "type": ep["rcsb_entity_polymer_type"],
                "seq": ep["pdbx_seq_one_letter_code_can"].replace("\n", ""),
                "n_res": ep["rcsb_sample_sequence_length"],
                "strand_ids": ep["pdbx_strand_id"],
                "nstd_monomer": ep["nstd_monomer"],
                "api_sha256": hashlib.sha256(rr.content).hexdigest(),
            }
        )
        time.sleep(0.1)
    return {
        "pdb_id": pid,
        "title": entry["struct"]["title"],
        "release": entry["rcsb_accession_info"]["initial_release_date"][:10],
        "method": entry["exptl"][0]["method"],
        "entry_api_sha256": hashlib.sha256(r.content).hexdigest(),
        "entities": ents,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="+")
    ap.add_argument("--out", type=Path, default=Path(os.environ.get("BOLTZ_MLX_EXT_DATA", Path(__file__).resolve().parents[2] / "data" / "external")).expanduser() / "rcsb_entities")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for pid in a.ids:
        d = fetch_entry(pid)
        (a.out / f"{pid.upper()}.json").write_text(json.dumps(d, indent=1))
        print(pid.upper(), d["title"][:60])
        for e in d["entities"]:
            print("  entity", e["entity_id"], e["type"], e["n_res"], e["strand_ids"], e["description"])


if __name__ == "__main__":
    main()
