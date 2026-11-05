"""End-to-end parity: MLX Boltz-2 (trunk + diffusion) vs CPU fp32 PyTorch reference, replayed noise.

Usage: python tools/e2e_parity.py prot117 [--steps 200] [--recycling 3] [--device gpu|cpu]
"""

import os
import argparse
import json
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # this checkout's boltz_mlx first: the editable install maps boltz_mlx to the main checkout
import boltz_mlx  # noqa: F401  (TF32 off)
import mlx.core as mx
import numpy as np

from boltz_mlx.compare import kabsch_rmsd, metrics, rmsd
from boltz_mlx.model import PredictArgs, load_model
from boltz_mlx.sampler import ReplayNoise

DATA = Path(os.environ.get("BOLTZ_MLX_DATA", Path(__file__).resolve().parents[1] / "data")).expanduser()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--ref", default="ref")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--recycling", type=int, default=3)
    ap.add_argument("--device", default="gpu")
    ap.add_argument("--affinity", action="store_true", help="affinity checkpoint on feats_affinity.npz (ref_aff)")
    ap.add_argument("--samples", type=int, default=None)
    args = ap.parse_args()
    mx.set_default_device(mx.gpu if args.device == "gpu" else mx.cpu)

    t0 = time.perf_counter()
    if args.affinity:
        args.ref = "ref_aff" if args.ref == "ref" else args.ref
        args.recycling = 5
    samples = args.samples or (5 if args.affinity else 1)
    model = load_model("aff" if args.affinity else "conf")
    t_load = time.perf_counter() - t0
    z = np.load(DATA / args.case / ("feats_affinity.npz" if args.affinity else "feats.npz"))
    feats = {k: mx.array(z[k]) for k in z.files}
    ref_dir = DATA / args.case / args.ref
    ref = np.load(ref_dir / "outputs.npz")
    noise = ReplayNoise(ref_dir / "noise.npz")

    timings = {}
    t0 = time.perf_counter()
    out = model(feats, PredictArgs(recycling_steps=args.recycling, sampling_steps=args.steps, diffusion_samples=samples), noise, timings)
    t_pred = time.perf_counter() - t0

    mask = ref["masks"][0].astype(bool)
    coords = np.array(out["sample_atom_coords"])
    rc = ref["coords"]
    per_sample = [rmsd(coords[i][mask], rc[i][mask]) for i in range(rc.shape[0])]
    heads = {}
    for k in ["plddt", "pae", "pde", "ptm", "iptm", "complex_plddt", "complex_iplddt", "complex_pde", "complex_ipde",
              "ligand_iptm", "protein_iptm", "affinity_pred_value", "affinity_probability_binary",
              "affinity_pred_value1", "affinity_probability_binary1", "affinity_pred_value2", "affinity_probability_binary2"]:
        if k in ref.files and k in out:
            heads[k] = float(np.abs(np.array(out[k]).reshape(ref[k].shape) - ref[k]).max())
    res = {
        "command": "python " + " ".join(sys.argv),
        "case": args.case,
        "device": args.device,
        "load_s": round(t_load, 2),
        "predict_s": round(t_pred, 2),
        **{k: round(v, 2) for k, v in timings.items()},
        "peak_mem_gb": round(mx.get_peak_memory() / 1e9, 2),
        "max_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2),
        "noise_draws_used": f"{noise.i}/{len(noise.draws)}",
        "s": metrics(out["s"], ref["s"]),
        "z": metrics(out["z"], ref["z"]),
        "coords_rmsd": max(per_sample),
        "coords_rmsd_per_sample": per_sample,
        "heads_max_abs_diff": heads,
        "coords_kabsch_rmsd": kabsch_rmsd(coords[0][mask], ref["coords"][0][mask]),
        "coords_max_dev": float(np.abs(coords[0][mask] - ref["coords"][0][mask]).max()),
    }
    print(json.dumps(res, indent=1))
    out_dir = ref_dir.parent / ("mlx_aff" if args.affinity else "mlx")
    out_dir.mkdir(exist_ok=True)
    np.savez(out_dir / "outputs.npz", **{k: np.array(v) for k, v in out.items() if k != "pair_chains_iptm"})
    (out_dir / "e2e.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
