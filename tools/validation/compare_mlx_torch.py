"""Per-case MLX vs PyTorch agreement on the validation set (replayed noise => same frame).

Usage: python tools/validation/compare_mlx_torch.py [--run mlx_gpu_replay] [--out validation/mlx_vs_torch.json]
Reports all-atom RMSD without superposition, CA Kabsch RMSD, max atom deviation, |Δ complex_pLDDT|,
mean/max |Δ per-token pLDDT|, |Δ pTM|.
"""

import os
import argparse
import json
from pathlib import Path

import gemmi
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("BOLTZ_MLX_DATA", Path(__file__).resolve().parents[2] / "data")).expanduser()


def atoms(cif):
    st = gemmi.read_structure(str(cif))
    xyz, ca = [], []
    for ch in st[0]:
        for res in ch:
            for at in res:
                xyz.append(at.pos.tolist())
                ca.append(at.name == "CA")
    return np.array(xyz), np.array(ca)


def kabsch(x, y):
    xc, yc = x - x.mean(0), y - y.mean(0)
    u, _, vt = np.linalg.svd(xc.T @ yc)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1, 1, d]) @ vt
    return float(np.sqrt(((xc @ r - yc) ** 2).sum(1).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="mlx_gpu_replay")
    ap.add_argument("--out", default=str(ROOT / "validation" / "mlx_vs_torch.json"))
    a = ap.parse_args()
    ids = [c["id"] for c in json.loads((ROOT / "validation" / "manifest.json").read_text())["cases"]]
    rows = {}
    for cid in ids:
        mdir = next((DATA / "validation" / "runs" / a.run / cid).rglob(f"predictions/{cid}"))
        tdir = DATA / cid / "ref" / "predictions" / cid
        xm, ca = atoms(mdir / f"{cid}_model_0.cif")
        xt, _ = atoms(tdir / f"{cid}_model_0.cif")
        cm = json.loads((mdir / f"confidence_{cid}_model_0.json").read_text())
        ct = json.loads((tdir / f"confidence_{cid}_model_0.json").read_text())
        pm = np.load(mdir / f"plddt_{cid}_model_0.npz")["plddt"]
        pt = np.load(tdir / f"plddt_{cid}_model_0.npz")["plddt"]
        dev = np.linalg.norm(xm - xt, axis=1)
        rows[cid] = {
            "n_atoms": int(len(xm)),
            "rmsd_same_frame": float(np.sqrt((dev**2).mean())),
            "ca_kabsch_rmsd": kabsch(xm[ca], xt[ca]),
            "max_atom_dev": float(dev.max()),
            "frac_atoms_gt_0.1A": float((dev > 0.1).mean()),
            "d_complex_plddt": abs(cm["complex_plddt"] - ct["complex_plddt"]),
            "d_plddt_mean": float(np.abs(pm - pt).mean()),
            "d_plddt_max": float(np.abs(pm - pt).max()),
            "d_ptm": abs(cm["ptm"] - ct["ptm"]),
            "complex_plddt_torch": ct["complex_plddt"],
            "complex_plddt_mlx": cm["complex_plddt"],
        }
        r = rows[cid]
        print(f"{cid}: rmsd {r['rmsd_same_frame']:.4f} Å  CA-kabsch {r['ca_kabsch_rmsd']:.4f}  maxdev {r['max_atom_dev']:.3f}  "
              f">0.1Å {100*r['frac_atoms_gt_0.1A']:.1f}%  Δcplddt {r['d_complex_plddt']:.4f}  Δplddt mean/max {r['d_plddt_mean']:.4f}/{r['d_plddt_max']:.4f}  ΔpTM {r['d_ptm']:.4f}")
    pm = np.array([r["complex_plddt_mlx"] for r in rows.values()])
    pt = np.array([r["complex_plddt_torch"] for r in rows.values()])
    summary = {
        "median_rmsd_same_frame": float(np.median([r["rmsd_same_frame"] for r in rows.values()])),
        "n_rmsd_lt_0.1": int(sum(r["rmsd_same_frame"] < 0.1 for r in rows.values())),
        "max_d_complex_plddt": float(np.abs(pm - pt).max()),
        "pearson_complex_plddt": float(np.corrcoef(pm, pt)[0, 1]),
    }
    print(json.dumps(summary, indent=1))
    Path(a.out).write_text(json.dumps({"run": a.run, "summary": summary, "cases": rows}, indent=1))


if __name__ == "__main__":
    main()
