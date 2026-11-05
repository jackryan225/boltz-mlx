"""Build docs/viewer/: Mol* page overlaying MLX predictions (pLDDT-coloured) on experimental structures.

Usage: python tools/make_viewer.py [--run mlx_gpu_feats_replay]
Experimental chains are superposed onto the MLX prediction (CA Kabsch over matched residues) and
written as new mmCIFs, so the page needs no client-side alignment. Serve with
`python -m http.server -d docs/viewer` (Mol* loads from a CDN).
"""

import os
import argparse
import json
import shutil
from pathlib import Path

import gemmi
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("BOLTZ_MLX_DATA", Path(__file__).resolve().parents[1] / "data")).expanduser()
OUT = ROOT / "docs" / "viewer"


def ca_by_seq(st, chain_name=None):
    st.setup_entities()
    ch = st[0][chain_name] if chain_name else st[0][0]
    out = {}
    for res in ch:
        seq_id = res.label_seq if res.label_seq is not None else res.seqid.num
        at = res.find_atom("CA", "*")
        if at is not None:
            out[int(seq_id)] = np.array(at.pos.tolist())
    return out


def kabsch(P, Q):
    """R, t such that R @ P_i + t ≈ Q_i."""
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, qc - R @ pc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="mlx_gpu_feats_replay", help="run for cases with a torch reference (replayed noise)")
    ap.add_argument("--extra-run", default="v30_strict_seed0", help="run for cases without a torch reference")
    ap.add_argument("--manifest", default="validation/manifest_all.json")
    a = ap.parse_args()
    (OUT / "structures").mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / a.manifest).read_text())
    def table(path):
        t = json.loads(path.read_text())["cases"]
        return t if isinstance(t, dict) else {c["id"]: c for c in t}

    scores = table(ROOT / "validation" / f"{a.run}_scores.json")
    scores.update({k: v for k, v in table(ROOT / "validation" / "sweep" / f"{a.extra_run}.json").items() if k not in scores})
    torch_scores = {c["id"] if isinstance(c, dict) and "id" in c else k: c
                    for k, c in (lambda t: t.items() if isinstance(t, dict) else [(c["id"], c) for c in t])(
                        json.loads((ROOT / "validation" / "torch_cpu_scores.json").read_text())["cases"])}
    parity = json.loads((ROOT / "validation" / "mlx_vs_torch.json").read_text())["cases"]
    entries = []
    for c in manifest["cases"]:
        cid = c["id"]
        has_ref = cid in parity
        run = a.run if has_ref else a.extra_run
        pdir = DATA / "validation" / "runs" / run / cid / "predictions" / cid
        pred = pdir / f"{cid}_model_0.cif"
        conf = json.loads((pdir / f"confidence_{cid}_model_0.json").read_text())
        exp_path = DATA / "validation" / "cif" / Path(c["cif"]).name
        st_pred = gemmi.read_structure(str(pred))
        st_exp = gemmi.read_structure(str(exp_path))
        st_exp.remove_ligands_and_waters()
        st_exp.remove_hydrogens()
        # keep only the evaluated chain
        for ch in list(st_exp[0]):
            if ch.name != c["chain"]:
                st_exp[0].remove_chain(ch.name)
        p_ca, e_ca = ca_by_seq(st_pred), ca_by_seq(st_exp, c["chain"])
        common = sorted(set(p_ca) & set(e_ca))
        R, t = kabsch(np.array([e_ca[i] for i in common]), np.array([p_ca[i] for i in common]))
        dst_pred = OUT / "structures" / f"{cid}_mlx.cif"
        dst_exp = OUT / "structures" / f"{cid}_exp.cif"
        shutil.copy(pred, dst_pred)
        # keep the deposited mmCIF (entity_poly_seq etc. for Mol* cartoons); rewrite only coordinates
        doc = gemmi.cif.read(str(exp_path))
        block = doc.sole_block()
        comp = block.find("_atom_site.", ["label_comp_id"])
        for i in reversed(range(len(comp))):
            if comp[i][0] == "HOH":
                comp.remove_row(i)
        tab = block.find("_atom_site.", ["Cartn_x", "Cartn_y", "Cartn_z"])
        for row in tab:
            xyz = R @ np.array([float(row[0]), float(row[1]), float(row[2])]) + t
            row[0], row[1], row[2] = (f"{v:.3f}" for v in xyz)
        doc.write_file(str(dst_exp))
        s, ts = scores[cid], torch_scores.get(cid)
        entries.append({
            "id": cid,
            "title": c.get("description", ""),
            "length": c["length_seqres"],
            "release": c["release_date"],
            "resolution": c["resolution"],
            "pred": f"structures/{cid}_mlx.cif",
            "exp": f"structures/{cid}_exp.cif",
            "lddt_mlx": round(s.get("lddt_ca", s.get("lddt", 0)), 3),
            "lddt_torch": round(ts.get("lddt_ca", ts.get("lddt", 0)), 3) if ts else None,
            "tm_mlx": round(s["tm_score"], 3),
            "tm_torch": round(ts["tm_score"], 3) if ts else None,
            "rmsd_vs_torch": round(parity[cid]["rmsd_same_frame"], 4) if has_ref else None,
            "plddt": round(conf["complex_plddt"], 3),
            "ptm": round(conf["ptm"], 3),
        })
        print(cid, "aligned on", len(common), "CA")
    (OUT / "data.json").write_text(json.dumps(entries, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
