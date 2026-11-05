"""CPU fp32 PyTorch reference run of Boltz-2 with recorded noise and per-module dumps.

Usage:
  python tools/make_reference.py prot117                     # structure+confidence, conf ckpt
  python tools/make_reference.py lig_aff --affinity          # affinity ckpt on feats_affinity.npz
  python tools/make_reference.py prot117 --list-modules      # print module tree and exit
Options: --recycling N --steps N --samples N --no-dump --tag NAME

Outputs (data/<case>/ref[_<tag>]/):
  outputs.npz   all tensor outputs of Boltz2.forward
  noise.npz     every torch.randn/randn_like draw during forward, in call order (noise_00000, ...)
  modules/<qualified.name>.npz  first-call inputs (in_<i>, kw_<name>) and outputs (out_<i>) of dumped modules
  predictions/  Boltz writer output (.cif, confidence json, pae/pde npz)
  run.json      command, timings, config
"""

import os
import argparse
import json
import pickle
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from boltz.main import (
    Boltz2DiffusionParams,
    BoltzSteeringParams,
    MSAModuleArgs,
    PairformerArgsV2,
)
from boltz.model.models.boltz2 import Boltz2

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get("BOLTZ_MLX_CACHE", Path.home() / ".boltz")).expanduser()

# First call of each of these modules is dumped (regex on qualified name).
DEFAULT_DUMP = [
    r"^input_embedder$",
    r"^rel_pos$",
    r"^contact_conditioning$",
    r"^template_module$",
    r"^msa_module$",
    r"^msa_module\.layers\.0$",
    r"^msa_module\.layers\.0\.[a-z_]+$",
    r"^template_module\.pairformer\.layers\.0$",
    r"^pairformer_module$",
    r"^pairformer_module\.layers\.0$",
    r"^pairformer_module\.layers\.0\.[a-z_]+$",
    r"^distogram_module$",
    r"^diffusion_conditioning$",
    r"^structure_module\.score_model$",
    r"^structure_module\.score_model\.[a-z_]+$",
    r"^confidence_module$",
    r"^confidence_module\.[a-z_]+$",
    r"^affinity_module[12]?$",
    r"^affinity_module1?\.[a-z_]+$",
    r"^bfactor_module$",
]


def to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return None


def flatten(prefix, obj, out):
    if isinstance(obj, torch.Tensor):
        out[prefix] = to_np(obj)
    elif isinstance(obj, (list, tuple)):
        for i, o in enumerate(obj):
            flatten(f"{prefix}_{i}", o, out)
    elif isinstance(obj, dict):
        if "token_pad_mask" in obj and "atom_pad_mask" in obj:
            out[prefix] = np.asarray("FEATS")  # the feats dict: load data/<case>/feats*.npz instead
            return
        for k, o in obj.items():
            flatten(f"{prefix}_{k}", o, out)
    elif isinstance(obj, (bool, int, float)):
        out[prefix] = np.asarray(obj)


class Dumper:
    def __init__(self, model, patterns, out_dir: Path):
        self.out_dir = out_dir
        self.done = set()
        self.handles = []
        regs = [re.compile(p) for p in patterns]
        for name, mod in model.named_modules():
            if name and any(r.search(name) for r in regs):
                self.handles.append(
                    mod.register_forward_hook(self.hook(name), with_kwargs=True)
                )

    def hook(self, name):
        def fn(mod, args, kwargs, output):
            if name in self.done:
                return
            self.done.add(name)
            d = {}
            flatten("in", list(args), d)
            flatten("kw", kwargs, d)
            flatten("out", output, d)
            np.savez(self.out_dir / f"{name}.npz", **d)

        return fn

    def remove(self):
        for h in self.handles:
            h.remove()


class NoiseRecorder:
    """Wrap torch.randn / randn_like and record every draw in order."""

    def __init__(self):
        self.draws = []
        self._randn = torch.randn
        self._randn_like = torch.randn_like

    def __enter__(self):
        rec = self

        def randn(*a, **k):
            t = rec._randn(*a, **k)
            rec.draws.append(t.detach().cpu().numpy().copy())
            return t

        def randn_like(*a, **k):
            t = rec._randn_like(*a, **k)
            rec.draws.append(t.detach().cpu().numpy().copy())
            return t

        torch.randn = randn
        torch.randn_like = randn_like
        return self

    def __exit__(self, *exc):
        torch.randn = self._randn
        torch.randn_like = self._randn_like


def load_feats(path: Path) -> dict:
    z = np.load(path)
    feats = {k: torch.from_numpy(z[k]) for k in z.files}
    with open(path.with_suffix(".meta.pkl"), "rb") as f:
        feats.update(pickle.load(f))
    return feats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--affinity", action="store_true")
    ap.add_argument("--recycling", type=int, default=None)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--samples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-dump", action="store_true")
    ap.add_argument("--list-modules", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    case_dir = ROOT / "data" / args.case
    out_dir = case_dir / ("ref" + (f"_{args.tag}" if args.tag else ""))
    if args.affinity:
        out_dir = case_dir / ("ref_aff" + (f"_{args.tag}" if args.tag else ""))
    (out_dir / "modules").mkdir(parents=True, exist_ok=True)

    recycling = args.recycling if args.recycling is not None else (5 if args.affinity else 3)
    samples = args.samples if args.samples is not None else (5 if args.affinity else 1)
    predict_args = {
        "recycling_steps": recycling,
        "sampling_steps": args.steps,
        "diffusion_samples": samples,
        "max_parallel_samples": 1 if args.affinity else None,
        "write_confidence_summary": not args.affinity,
        "write_full_pae": not args.affinity,
        "write_full_pde": not args.affinity,
    }
    steering = BoltzSteeringParams()
    steering.fk_steering = False
    steering.physical_guidance_update = False
    if args.affinity:
        steering.contact_guidance_update = False
    ckpt = CACHE / ("boltz2_aff.ckpt" if args.affinity else "boltz2_conf.ckpt")
    extra = {"affinity_mw_correction": False} if args.affinity else {"use_kernels": False}

    t0 = time.time()
    model = Boltz2.load_from_checkpoint(
        ckpt,
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
    model.eval()
    t_load = time.time() - t0

    if args.list_modules:
        for name, mod in model.named_modules():
            if name.count(".") <= 3 and ".layers." not in name.replace("layers.0.", ""):
                print(name, type(mod).__name__)
        return

    feats = load_feats(case_dir / ("feats_affinity.npz" if args.affinity else "feats.npz"))
    dumper = None if args.no_dump else Dumper(model, DEFAULT_DUMP, out_dir / "modules")

    torch.manual_seed(args.seed)
    t0 = time.time()
    with torch.no_grad(), NoiseRecorder() as rec:
        pred = model.predict_step(feats, 0)
    t_pred = time.time() - t0
    if dumper:
        dumper.remove()

    np.savez(out_dir / "noise.npz", **{f"noise_{i:05d}": d for i, d in enumerate(rec.draws)})
    outs = {}
    for k, v in pred.items():
        if isinstance(v, torch.Tensor):
            outs[k] = to_np(v)
        elif isinstance(v, dict):
            for kk, vv in v.items():
                if isinstance(vv, torch.Tensor):
                    outs[f"{k}.{kk}"] = to_np(vv)
    np.savez(out_dir / "outputs.npz", **outs)

    # Upstream writer output (for CLI/format parity and affinity featurization)
    from boltz.data.write.writer import BoltzAffinityWriter, BoltzWriter

    if args.affinity:
        writer = BoltzAffinityWriter(
            data_dir=case_dir / "processed" / "structures",
            output_dir=out_dir / "predictions",
        )
    else:
        writer = BoltzWriter(
            data_dir=case_dir / "processed" / "structures",
            output_dir=out_dir / "predictions",
            output_format="mmcif",
            boltz2=True,
        )
    writer.write_on_batch_end(None, None, pred, None, feats, 0, 0)

    info = {
        "command": " ".join(sys.argv),
        "case": args.case,
        "ckpt": str(ckpt),
        "predict_args": predict_args,
        "seed": args.seed,
        "threads": args.threads,
        "load_s": round(t_load, 2),
        "predict_s": round(t_pred, 2),
        "n_noise_draws": len(rec.draws),
        "dumped_modules": sorted(dumper.done) if dumper else [],
        "torch": torch.__version__,
    }
    (out_dir / "run.json").write_text(json.dumps(info, indent=1))
    summary = {k: (v.tolist() if v.size <= 4 else list(v.shape)) for k, v in outs.items()}
    print(json.dumps({**info, "outputs": summary}, indent=1, default=str))


if __name__ == "__main__":
    main()
