"""Compare two Boltz prediction directories for format parity (file set, JSON keys, npz keys/shapes, CIF atom records).

Usage: python tools/compare_outputs.py <pred_dir_a> <pred_dir_b>   (each = .../predictions/<id>)
"""

import json
import sys
from pathlib import Path

import numpy as np


def cif_atoms(path):
    rows = [l.split() for l in Path(path).read_text().splitlines() if l.startswith(("ATOM", "HETATM"))]
    return rows


def main(a, b):
    a, b = Path(a), Path(b)
    fa = sorted(p.name for p in a.iterdir())
    fb = sorted(p.name for p in b.iterdir())
    ok = True
    print("files A:", fa)
    if fa != fb:
        print("FILE SET DIFFERS:", set(fa) ^ set(fb))
        ok = False
    for name in sorted(set(fa) & set(fb)):
        pa, pb = a / name, b / name
        if name.endswith(".json"):
            ja, jb = json.loads(pa.read_text()), json.loads(pb.read_text())
            same = set(ja) == set(jb)
            ok &= same
            diffs = {k: round(abs(ja[k] - jb[k]), 4) for k in ja if k in jb and isinstance(ja[k], (int, float))}
            print(f"{name}: keys {'equal' if same else 'DIFFER'}; |Δ| {diffs}")
        elif name.endswith(".npz"):
            za, zb = np.load(pa, allow_pickle=True), np.load(pb, allow_pickle=True)
            same = sorted(za.files) == sorted(zb.files) and all(za[k].shape == zb[k].shape and za[k].dtype == zb[k].dtype for k in za.files)
            ok &= same
            print(f"{name}: keys/shapes/dtypes {'equal' if same else 'DIFFER'} {[(k, za[k].shape, str(za[k].dtype)) for k in za.files][:4]}")
        elif name.endswith(".cif"):
            ra, rb = cif_atoms(pa), cif_atoms(pb)
            # same atoms in same order: compare all non-coordinate columns
            ident = lambda r: [c for i, c in enumerate(r) if i not in (10, 11, 12, 17)]  # noqa: E731
            same = len(ra) == len(rb) and all(ident(x) == ident(y) for x, y in zip(ra, rb))
            ok &= same
            print(f"{name}: {len(ra)} vs {len(rb)} atom records, identity columns {'equal' if same else 'DIFFER'}")
    print("FORMAT PARITY:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main(*sys.argv[1:3]) else 1)
