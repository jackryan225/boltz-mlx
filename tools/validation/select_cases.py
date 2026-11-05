"""Select post-cutoff X-ray monomer protein entries from RCSB for the Boltz-2 MLX validation set.

Step 1 (this file): run the RCSB search query and write validation/query.json and
validation/candidate_ids.json. Step 2 (filter_cases.py) pulls metadata + coordinates and filters.
Public RCSB endpoints only (search.rcsb.org, data.rcsb.org, files.rcsb.org).
"""
import json
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
CUTOFF = "2023-07-01"
SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"


def text(attr, op, value):
    return {"type": "terminal", "service": "text", "parameters": {"attribute": attr, "operator": op, "value": value}}


QUERY = {
    "query": {
        "type": "group",
        "logical_operator": "and",
        "nodes": [
            text("rcsb_accession_info.initial_release_date", "greater_or_equal", f"{CUTOFF}T00:00:00Z"),
            text("exptl.method", "exact_match", "X-RAY DIFFRACTION"),
            text("rcsb_entry_info.resolution_combined", "less_or_equal", 2.5),
            text("rcsb_entry_info.polymer_entity_count", "equals", 1),
            text("rcsb_entry_info.polymer_entity_count_protein", "equals", 1),
            text("rcsb_entry_info.deposited_polymer_entity_instance_count", "equals", 1),
            text("rcsb_entry_info.deposited_polymer_monomer_count", "range", {"from": 60, "to": 300, "include_lower": True, "include_upper": True}),
        ],
    },
    "return_type": "entry",
    "request_options": {
        "paginate": {"start": 0, "rows": 10000},
        "sort": [{"sort_by": "rcsb_accession_info.initial_release_date", "direction": "asc"}],
    },
}

if __name__ == "__main__":
    r = requests.post(SEARCH, json=QUERY, timeout=60)
    r.raise_for_status()
    d = r.json()
    ids = [x["identifier"] for x in d["result_set"]]
    print("total_count", d["total_count"], "returned", len(ids))
    (ROOT / "validation").mkdir(exist_ok=True)
    (ROOT / "validation" / "query.json").write_text(json.dumps(QUERY, indent=1))
    (ROOT / "validation" / "candidate_ids.json").write_text(json.dumps(ids))
