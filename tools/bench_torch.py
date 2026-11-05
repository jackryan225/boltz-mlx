"""Time upstream Boltz-2 (PyTorch) model predict on cpu or mps, fp32 or bf16-mixed.

Usage: python tools/bench_torch.py prot117 --device mps --precision bf16-mixed [--runs 2]
Times Boltz2.predict_step (trunk + diffusion + confidence) on featurized inputs, i.e. the same
work the Lightning CLI reports as model predict; first run includes MPS kernel warm-up.
"""

import argparse
import json
import resource
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_reference import CACHE, ROOT, load_feats  # noqa: E402

from boltz.main import Boltz2DiffusionParams, BoltzSteeringParams, MSAModuleArgs, PairformerArgsV2  # noqa: E402
from boltz.model.models.boltz2 import Boltz2  # noqa: E402


def sync(dev):
    if dev == "mps":
        torch.mps.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--precision", default="fp32", choices=["fp32", "bf16-mixed"])
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--recycling", type=int, default=3)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--affinity", action="store_true", help="affinity checkpoint on feats_affinity.npz (5 recycles, 5 samples)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    predict_args = {
        "recycling_steps": 5 if args.affinity else args.recycling,
        "sampling_steps": args.steps,
        "diffusion_samples": 5 if args.affinity else 1,
        "max_parallel_samples": 1 if args.affinity else None,
        "write_confidence_summary": True,
        "write_full_pae": True,
        "write_full_pde": True,
    }
    steering = BoltzSteeringParams()
    if args.affinity:
        steering.contact_guidance_update = False
    extra = {"affinity_mw_correction": False} if args.affinity else {"use_kernels": False}
    model = Boltz2.load_from_checkpoint(
        CACHE / ("boltz2_aff.ckpt" if args.affinity else "boltz2_conf.ckpt"),
        strict=True,
        predict_args=predict_args,
        map_location="cpu",
        diffusion_process_args=asdict(Boltz2DiffusionParams()),
        ema=False,
        pairformer_args=asdict(PairformerArgsV2()),
        msa_args=asdict(MSAModuleArgs(use_paired_feature=True)),
        steering_args=asdict(steering),
        **extra,
    )
    model.use_kernels = False
    model.eval().to(args.device)
    feats = load_feats(ROOT / "data" / args.case / ("feats_affinity.npz" if args.affinity else "feats.npz"))
    feats = {k: (v.to(args.device) if isinstance(v, torch.Tensor) else v) for k, v in feats.items()}

    times = []
    for _ in range(args.runs):
        torch.manual_seed(0)
        sync(args.device)
        t0 = time.perf_counter()
        with torch.no_grad(), torch.autocast(
            "cpu" if args.device == "cpu" else args.device,
            dtype=torch.bfloat16,
            enabled=args.precision == "bf16-mixed",
        ):
            pred = model.predict_step(feats, 0)
        sync(args.device)
        times.append(time.perf_counter() - t0)
    res = {
        "command": "python " + " ".join(sys.argv),
        "case": args.case,
        "device": args.device,
        "precision": args.precision,
        "predict_s_runs": [round(t, 2) for t in times],
        "max_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2),
        "mps_driver_alloc_gb": round(torch.mps.driver_allocated_memory() / 1e9, 2) if args.device == "mps" else None,
        "plddt_mean": float(pred["plddt"].float().mean()) if "plddt" in pred else None,
        "affinity_pred_value": float(pred["affinity_pred_value"]) if "affinity_pred_value" in pred else None,
        "affinity": args.affinity,
        "torch": torch.__version__,
    }
    print(json.dumps(res))
    with open(ROOT / "logs" / "bench_torch.jsonl", "a") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
