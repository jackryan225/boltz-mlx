"""Tiny mmCIF reader for the GUI's pose summary: ligand atoms, pocket residues, mean pLDDT per chain.

Only the `_atom_site` loop of the mmCIF files that Boltz writes is read (standard library, no gemmi): enough to list the
residues around the screened ligand next to the 3D view. The 3D view itself (Mol*) parses the file in the browser."""

from __future__ import annotations

import math
import re
from pathlib import Path

_TOKEN = re.compile(r"""'(?:[^']|'(?=\S))*'(?=\s|$)|"(?:[^"]|"(?=\S))*"(?=\s|$)|\S+""")
MAX_BYTES = 40 * 1024 * 1024


def _tokens(line: str) -> list[str]:
    out = []
    for t in _TOKEN.findall(line):
        if len(t) >= 2 and t[0] == t[-1] and t[0] in "'\"":
            t = t[1:-1]
        out.append(t)
    return out


def read_atom_site(path: str | Path) -> list[dict]:
    """Rows of the `_atom_site` loop as dicts of the columns used here (None if the file has no such loop)."""
    p = Path(path)
    if p.stat().st_size > MAX_BYTES:
        raise ValueError("structure file too large to summarise")
    cols: list[str] = []
    rows: list[dict] = []
    state = "scan"
    for line in p.read_text(errors="replace").splitlines():
        if state == "scan":
            if line.startswith("_atom_site."):
                state = "header"
                cols = [line.split()[0]]
            continue
        if state == "header":
            if line.startswith("_atom_site."):
                cols.append(line.split()[0])
                continue
            state = "rows"
        if state == "rows":
            if not line or line.startswith(("#", "loop_", "_", "data_")):
                break
            tok = _tokens(line)
            if len(tok) != len(cols):
                continue
            d = dict(zip(cols, tok))
            try:
                x, y, z = float(d["_atom_site.Cartn_x"]), float(d["_atom_site.Cartn_y"]), float(d["_atom_site.Cartn_z"])
            except (KeyError, ValueError):
                continue
            rows.append({
                "group": d.get("_atom_site.group_PDB", "ATOM"),
                "chain": d.get("_atom_site.auth_asym_id") or d.get("_atom_site.label_asym_id", "?"),
                "label_chain": d.get("_atom_site.label_asym_id", "?"),
                "resname": d.get("_atom_site.label_comp_id", "?"),
                "resseq": d.get("_atom_site.auth_seq_id") or d.get("_atom_site.label_seq_id", "?"),
                "name": d.get("_atom_site.label_atom_id", "?"),
                "element": d.get("_atom_site.type_symbol", "?"),
                "b": _f(d.get("_atom_site.B_iso_or_equiv")),
                "x": x, "y": y, "z": z,
            })
    return rows


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def pose_summary(path: str | Path, ligand_chain: str, cutoff: float = 5.0, max_residues: int = 60) -> dict:
    """Pocket residues (any heavy atom within `cutoff` Å of a heavy atom of `ligand_chain`) and per-chain mean B-factor
    (Boltz stores pLDDT x 100 there). Returns {"ligand_atoms", "pocket": [{"chain", "resname", "resseq", "min_dist", "plddt"}]}."""
    atoms = read_atom_site(path)
    lig = [a for a in atoms if a["chain"] == ligand_chain and a["element"] != "H"]
    if not lig:  # the chain id is not in the file (renamed by a writer): take the non-polymer, non-water atoms instead
        lig = [a for a in atoms if a["group"] == "HETATM" and a["resname"] not in ("HOH", "WAT") and a["element"] != "H"]
        if lig:
            ligand_chain = lig[0]["chain"]
            lig = [a for a in lig if a["chain"] == ligand_chain]
    out: dict = {"ligand_chain": ligand_chain, "ligand_atoms": len(lig), "cutoff": cutoff, "pocket": [], "n_atoms": len(atoms)}
    if not lig:
        return out
    rec = [a for a in atoms if a["chain"] != ligand_chain and a["element"] != "H"]
    # bounding box of the ligand, padded: skips most receptor atoms before the pairwise loop
    lo = [min(a[k] for a in lig) - cutoff for k in "xyz"]
    hi = [max(a[k] for a in lig) + cutoff for k in "xyz"]
    near = [a for a in rec if all(lo[i] <= a[k] <= hi[i] for i, k in enumerate("xyz"))]
    c2 = cutoff * cutoff
    best: dict[tuple, dict] = {}
    for a in near:
        d2min = None
        for l in lig:
            d2 = (a["x"] - l["x"]) ** 2 + (a["y"] - l["y"]) ** 2 + (a["z"] - l["z"]) ** 2
            if d2 <= c2 and (d2min is None or d2 < d2min):
                d2min = d2
        if d2min is None:
            continue
        key = (a["chain"], a["resseq"], a["resname"])
        r = best.get(key)
        if r is None:
            best[key] = {"chain": a["chain"], "resname": a["resname"], "resseq": a["resseq"], "min_dist": math.sqrt(d2min),
                         "plddt": a["b"], "_n": 1 if a["b"] is not None else 0, "_sum": a["b"] or 0.0}
        else:
            r["min_dist"] = min(r["min_dist"], math.sqrt(d2min))
            if a["b"] is not None:
                r["_n"] += 1
                r["_sum"] += a["b"]
    res = []
    for r in best.values():
        r["plddt"] = round(r.pop("_sum") / r["_n"], 1) if r["_n"] else None
        r.pop("_n")
        r["min_dist"] = round(r["min_dist"], 2)
        res.append(r)

    def k(r):
        try:
            return (r["chain"], int(r["resseq"]))
        except ValueError:
            return (r["chain"], 0)

    res.sort(key=k)
    out["pocket_total"] = len(res)
    out["pocket"] = res[:max_residues]
    return out
