"""Screen results: one json line per ligand (results.jsonl, append-only, fsynced) and the ranked results.csv.

Resume = every ligand id with a row in results.jsonl is done (status ok or invalid; status error rows are retried
only with --retry-errors). results.csv is rebuilt from the jsonl (last row per ligand wins), so it never has duplicates.

Ranking (Boltz docs): --rank-by prob (default; hit finding) sorts by affinity_probability_binary, high first;
--rank-by value (lead optimisation) sorts by affinity_pred_value (log10 IC50 in uM), low = stronger first.
Ties break on ligand_id. Rows without a prediction (invalid / error) are listed after the ranked rows, without a rank.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

CSV_COLUMNS = [
    "rank", "ligand_id", "smiles", "affinity_pred_value", "value_mw_corrected", "affinity_probability_binary",
    "iptm", "ligand_iptm", "ligand_plddt", "complex_plddt", "ptm", "n_atoms", "mw", "seconds", "mode", "msa_source", "seed",
    "fp16_fallbacks", "chip", "kernel_path", "status", "warnings", "error",
]
RANK_BY = {"prob": ("affinity_probability_binary", True), "value": ("affinity_pred_value", False)}


def _jsonable(o):
    import numpy as np

    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not json serialisable: {type(o)}")


def append_row(path: str | os.PathLike, row: dict) -> None:
    """Append one json line, flushed + fsynced (a killed run loses at most the ligand in flight)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, default=_jsonable)
    with open(p, "a+b") as f:
        # a killed writer can leave a torn last line without "\n": start a fresh line so the new row parses
        f.seek(0, os.SEEK_END)
        if f.tell() > 0:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                f.write(b"\n")
        f.write(line.encode() + b"\n")
        f.flush()
        os.fsync(f.fileno())


def read_rows(path: str | os.PathLike) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue  # torn last line of a killed run
        if isinstance(r, dict) and "ligand_id" in r:
            rows.append(r)
    return rows


def latest_rows(rows: list[dict]) -> dict[str, dict]:
    """ligand_id -> last row (a retried error row is superseded by its rerun)."""
    out: dict[str, dict] = {}
    for r in rows:
        out[str(r["ligand_id"])] = r
    return out


def done_ids(rows: list[dict], retry_errors: bool = False) -> set[str]:
    return {lid for lid, r in latest_rows(rows).items() if not (retry_errors and r.get("status") == "error")}


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def rank_rows(rows: list[dict], rank_by: str = "prob") -> list[dict]:
    """Rows (one per ligand) sorted best first, with 'rank' set (1-based) on ranked rows and None elsewhere."""
    if rank_by not in RANK_BY:
        raise ValueError(f"rank_by must be one of {sorted(RANK_BY)}")
    key, desc = RANK_BY[rank_by]
    rows = list(latest_rows(rows).values())
    ok = [r for r in rows if r.get("status") == "ok" and _num(r.get(key)) is not None]
    rest = [r for r in rows if r not in ok]
    ok.sort(key=lambda r: ((-1 if desc else 1) * _num(r[key]), str(r["ligand_id"])))
    rest.sort(key=lambda r: (str(r.get("status")), int(r.get("index", 0)), str(r["ligand_id"])))
    out = []
    for i, r in enumerate(ok):
        out.append({**r, "rank": i + 1})
    for r in rest:
        out.append({**r, "rank": None})
    return out


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6g}"
    if isinstance(v, (list, tuple)):
        return "; ".join(str(x) for x in v)
    if isinstance(v, dict):
        return json.dumps(v, sort_keys=True)
    return str(v)


def write_csv(path: str | os.PathLike, rows: list[dict], rank_by: str = "prob") -> list[dict]:
    """Atomically (tmp + rename) write the ranked results.csv. Returns the ranked rows."""
    ranked = rank_rows(rows, rank_by)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        for r in ranked:
            w.writerow([_fmt(r.get(c)) for c in CSV_COLUMNS])
    os.replace(tmp, p)
    return ranked
