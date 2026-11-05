"""PAE matrices (pae_<record>_model_0.npz, written by Boltz) for the results page's heatmap.

The browser gets a compact JSON: the matrix block-averaged to at most MAX_SIDE x MAX_SIDE, quantised to 1/8 Angstrom in
one byte per cell (base64; 255 marks a cell with no value), and the token range of each chain (read from the structure file), so it can draw chain
boundaries. numpy is a dependency of the screen already; nothing here touches the GPU."""

from __future__ import annotations

import base64
import math
from pathlib import Path

from boltz_mlx.gui.common import ApiError

MAX_SIDE = 512
STEP = 8.0          # quantisation: value = byte / STEP Angstrom (0..31.75)
MISSING = 255       # byte of a cell that is NaN in the matrix (shown hatched, never as 0 A)
MAX_NPZ = 256 * 1024 * 1024


def chain_spans(structure: Path | None) -> list[dict]:
    """[{"id", "type", "start", "end", "n"}] in token order: one token per polymer residue, one per heavy atom of a
    non-polymer chain (Boltz's tokenisation for standard residues). Empty if the file cannot be read."""
    if structure is None or structure.suffix != ".cif":
        return []
    from boltz_mlx.gui.pdbx import read_atom_site

    try:
        atoms = read_atom_site(structure)
    except (OSError, ValueError):
        return []
    order: list[str] = []
    info: dict[str, dict] = {}
    for a in atoms:
        c = a["chain"]
        if c not in info:
            order.append(c)
            info[c] = {"res": set(), "het": 0, "poly": False}
        if a["group"] == "HETATM":
            if a["element"] not in ("H", "D"):
                info[c]["het"] += 1
        else:
            info[c]["poly"] = True
            info[c]["res"].add(a["resseq"])
    spans, pos = [], 0
    for c in order:
        d = info[c]
        n = len(d["res"]) if d["poly"] else d["het"]
        spans.append({"id": c, "type": "polymer" if d["poly"] else "ligand", "start": pos, "end": pos + n, "n": n})
        pos += n
    return spans


def pae_payload(npz: Path, structure: Path | None = None, max_side: int = MAX_SIDE) -> dict:
    import numpy as np

    if not npz.is_file():
        raise ApiError(404, "no PAE matrix for this prediction")
    if npz.stat().st_size > MAX_NPZ:
        raise ApiError(413, "PAE matrix too large to show")
    try:
        with np.load(npz, allow_pickle=False) as z:
            key = "pae" if "pae" in z.files else z.files[0]
            m = np.asarray(z[key], dtype=np.float32)
    except Exception as ex:  # noqa: BLE001
        raise ApiError(500, f"could not read the PAE matrix: {type(ex).__name__}") from None
    if m.ndim == 3:  # (samples, n, n): the first sample is model_0
        m = m[0]
    if m.ndim != 2 or m.shape[0] != m.shape[1] or m.shape[0] == 0:
        raise ApiError(500, "unexpected PAE matrix shape")
    n = int(m.shape[0])
    f = max(1, math.ceil(n / max_side))
    if f > 1:  # block mean (pad with NaN to a multiple of f)
        side = math.ceil(n / f)
        pad = np.full((side * f, side * f), np.nan, dtype=np.float32)
        pad[:n, :n] = m
        small = np.nanmean(pad.reshape(side, f, side, f), axis=(1, 3))
    else:
        small = m
    miss = np.isnan(small)
    q = np.clip(np.rint(np.nan_to_num(small, nan=0.0) * STEP), 0, MISSING - 1).astype(np.uint8)
    q[miss] = MISSING
    spans = chain_spans(structure)
    if spans and spans[-1]["end"] != n:
        spans = []  # tokenisation differs (modified residues, ...): no boundaries rather than wrong ones
    out = {"n": n, "side": int(q.shape[0]), "factor": f, "step": STEP, "missing": MISSING, "n_missing": int(np.isnan(m).sum()), "max": round(float(np.nanmax(m)), 2) if not np.isnan(m).all() else None,
           "mean": round(float(np.nanmean(m)), 2) if not np.isnan(m).all() else None, "data": base64.b64encode(q.tobytes()).decode("ascii"), "chains": spans}
    if len(spans) > 1:  # mean PAE between each pair of chains (the off-diagonal blocks)
        pairs = []
        for a in spans:
            for b in spans:
                if a is not b and a["n"] and b["n"]:
                    blk = m[a["start"]:a["end"], b["start"]:b["end"]]
                    pairs.append({"a": a["id"], "b": b["id"], "mean": round(float(np.nanmean(blk)), 2)})
        out["pairs"] = pairs
    return out
