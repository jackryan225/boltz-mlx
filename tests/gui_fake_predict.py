"""Stand-in for `python -m boltz_mlx.cli predict` in the GUI tests (and for trying the GUI's structure jobs without a GPU).

  python tests/gui_fake_predict.py [predict] job.yaml --out_dir=... [--sampling_steps=...] ...

Writes the `boltz-mlx predict` output layout with made-up numbers (CPU only, no model, no MLX): processed/, then
predictions/<id>/{<id>_model_0.cif (one atom per polymer residue, the ligand's heavy atoms), confidence_<id>_model_0.json,
pae_<id>_model_0.npz, plddt_<id>_model_0.npz, affinity_<id>.json when the YAML asks for affinity}, then boltz_mlx_timings.json.

Environment:  GUI_FAKE_DELAY   seconds to sleep in the "structure" and "affinity" phases (default 0)
              GUI_FAKE_FAIL    exit 3 after featurization with an error message
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml


def tokens(doc: dict) -> list[tuple[str, str, int, str]]:
    """[(chain, kind, n_tokens, resname)] in YAML order."""
    out = []
    for s in doc.get("sequences") or []:
        for kind, v in s.items():
            if kind == "ligand":
                n = 6
                if v.get("smiles"):
                    from rdkit import Chem

                    m = Chem.MolFromSmiles(v["smiles"])
                    n = m.GetNumHeavyAtoms() if m is not None else 6
                out.append((str(v["id"]), "ligand", n, "LIG1"))
            else:
                out.append((str(v["id"]), kind, len(v.get("sequence") or ""), "ALA" if kind == "protein" else "DA"))
    return out


def write_cif(path: Path, toks) -> None:
    lines = ["data_model", "loop_", "_atom_site.group_PDB", "_atom_site.id", "_atom_site.type_symbol", "_atom_site.label_atom_id",
             "_atom_site.label_comp_id", "_atom_site.label_seq_id", "_atom_site.auth_seq_id", "_atom_site.label_asym_id",
             "_atom_site.Cartn_x", "_atom_site.Cartn_y", "_atom_site.Cartn_z", "_atom_site.auth_asym_id", "_atom_site.B_iso_or_equiv"]
    k = 0
    for ci, (chain, kind, n, res) in enumerate(toks):
        for i in range(n):
            k += 1
            x, y, z = 3.8 * i * math.cos(i / 5) + 20 * ci, 3.8 * i * math.sin(i / 5), 1.5 * i
            if kind == "ligand":
                x, y, z = 1.4 * (i % 3), 1.4 * (i // 3), 0.5
                lines.append(f"HETATM {k} C C{i + 1} {res} . 1 {chain} {x:.3f} {y:.3f} {z:.3f} {chain} {80 + i % 15:.2f}")
            else:
                lines.append(f"ATOM {k} C CA {res} {i + 1} {i + 1} {chain} {x:.3f} {y:.3f} {z:.3f} {chain} {60 + (i * 7) % 40:.2f}")
    lines.append("#")
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str]) -> int:
    if argv and argv[0] == "predict":
        argv = argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--out_dir", default="./")
    for a in ("--sampling_steps", "--recycling_steps", "--sampling_steps_affinity", "--diffusion_samples", "--diffusion_samples_affinity", "--seed", "--cache",
              "--output_format", "--precision"):
        ap.add_argument(a, default=None)
    ap.add_argument("--turbo", action="store_true")
    a = ap.parse_args(argv)
    delay = float(os.environ.get("GUI_FAKE_DELAY", "0") or 0)
    data = Path(a.data)
    doc = yaml.safe_load(data.read_text())
    rid = data.stem
    out = Path(a.out_dir) / f"boltz_results_{rid}"
    (out / "processed" / "structures").mkdir(parents=True, exist_ok=True)
    print(f"fake predict: {rid}, steps {a.sampling_steps}, precision {a.precision or ('turbo' if a.turbo else 'fp32')}", flush=True)
    if os.environ.get("GUI_FAKE_FAIL"):
        print("Traceback (most recent call last):\n  ...\nRuntimeError: fake failure for the GUI test", file=sys.stderr, flush=True)
        return 3
    t0 = time.time()
    time.sleep(delay)
    toks = tokens(doc)
    n = sum(t[2] for t in toks)
    pd = out / "predictions" / rid
    pd.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(a.seed or 0))
    idx = np.arange(n)
    chains = {str(i): 0.8 for i in range(len(toks))}
    for k in range(max(1, int(getattr(a, "diffusion_samples", None) or 1))):  # model 0 is the best-ranked sample
        pae = np.clip(2 + 0.08 * np.abs(idx[:, None] - idx[None, :]) + rng.random((n, n)) + 0.4 * k, 0.25, 31.0).astype(np.float32)
        np.savez_compressed(pd / f"pae_{rid}_model_{k}.npz", pae=pae)
        np.savez_compressed(pd / f"plddt_{rid}_model_{k}.npz", plddt=np.full(n, 0.8, np.float32))
        conf = {"confidence_score": round(0.81 - 0.04 * k, 3), "ptm": 0.78 - 0.03 * k, "iptm": 0.74 - 0.03 * k if len(toks) > 1 else 0.0,
                "ligand_iptm": 0.7, "protein_iptm": 0.0, "complex_plddt": 0.82 - 0.02 * k, "complex_iplddt": 0.8, "complex_pde": 0.5,
                "complex_ipde": 0.6, "chains_ptm": chains, "pair_chains_iptm": {i: dict(chains) for i in chains}}
        (pd / f"confidence_{rid}_model_{k}.json").write_text(json.dumps(conf, indent=4))
        write_cif(pd / f"{rid}_model_{k}.cif", toks)
    print(f"[fake] structure written ({n} tokens)", flush=True)
    t1 = time.time()
    timings = {"load_conf_s": 0.01, f"predict_{rid}_s": round(t1 - t0, 3)}
    if any("affinity" in p for p in doc.get("properties") or []):
        time.sleep(delay)
        aff = {"affinity_pred_value": 0.42, "affinity_probability_binary": 0.87, "affinity_pred_value1": 0.4,
               "affinity_probability_binary1": 0.85, "affinity_pred_value2": 0.44, "affinity_probability_binary2": 0.89}
        (pd / f"affinity_{rid}.json").write_text(json.dumps(aff, indent=4))
        timings[f"affinity_{rid}_s"] = round(time.time() - t1, 3)
    timings["total_wall_s"] = round(time.time() - t0, 3)
    (out / "boltz_mlx_timings.json").write_text(json.dumps(timings, indent=1))
    print(json.dumps(timings), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
