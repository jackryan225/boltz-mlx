"""MLX predictions on the exact featurized inputs + recorded noise of the CPU torch references.

The Boltz featurizer randomly rotates reference conformers (ref_pos), so a fresh featurization is
not comparable sample-for-sample; this reuses data/<id>/feats.npz and data/<id>/ref/noise.npz and
writes Boltz-format outputs with the upstream BoltzWriter.

Usage: python tools/validation/mlx_feats_predict.py [--run mlx_gpu_feats_replay] [--manifest validation/manifest_all.json] [ids...]
  Replay mode (default) needs data/<id>/ref/noise.npz. With --seed N the MLX RNG is used instead and no torch reference
  is required (cases from manifest_extra.json have none); data/<id>/{feats.npz,feats.meta.pkl,processed/structures}
  from tools/prep_case.py are enough.
"""

import os
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # this checkout's boltz_mlx first: the editable install maps boltz_mlx to the main checkout
import boltz_mlx  # noqa: F401
import mlx.core as mx
import numpy as np
import torch

from boltz_mlx.cli import _structure_pred_dict
from boltz_mlx.model import PredictArgs, load_model
from boltz_mlx.sampler import ReplayNoise, record_noise

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import feature_cache  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("BOLTZ_MLX_DATA", Path(__file__).resolve().parents[2] / "data")).expanduser()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--run", default="mlx_gpu_feats_replay")
    ap.add_argument("--manifest", type=Path, default=ROOT / "validation" / "manifest.json",
                    help="case manifest used when no ids are given (default validation/manifest.json)")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--recycling", type=int, default=3)
    ap.add_argument("--seed", type=int, default=None, help="MLX RNG instead of replaying the torch noise")
    a = ap.parse_args()
    from boltz.data.write.writer import BoltzWriter

    ids = a.ids or [c["id"] for c in json.loads(a.manifest.read_text())["cases"]]
    if a.seed is None:  # replay mode: fail before loading the model if a case has no recorded torch noise
        missing = [c for c in ids if not (DATA / c / "ref" / "noise.npz").exists()]
        if missing:
            sys.exit(f"no data/<id>/ref/noise.npz for {missing}; pass --seed N to use the MLX RNG instead of replaying torch noise")
    for c in ids:  # inputs the writer needs (created by tools/prep_case.py)
        for need in ("feats.npz", "feats.meta.pkl", "processed/structures"):
            if not (DATA / c / need).exists():
                sys.exit(f"missing data/{c}/{need}; run python tools/prep_case.py validation/yaml/{c}.yaml")
    model = load_model("conf")
    timings = {}
    for cid in ids:
        if a.seed is not None:  # per-(record, seed) seeded features, cached and shared by every mode
            arrays, meta = feature_cache.load(cid, a.seed)
        else:  # replay: the torch reference's own features
            z = np.load(DATA / cid / "feats.npz")
            arrays = {k: z[k] for k in z.files}
            with open(DATA / cid / "feats.meta.pkl", "rb") as f:
                meta = pickle.load(f)
        feats = {k: mx.array(v) for k, v in arrays.items()}
        batch = {k: torch.from_numpy(v) for k, v in arrays.items()}
        batch.update(meta)
        t0 = time.perf_counter()
        # seeded runs: per-record noise + per-(record, seed) cached features (identical across modes);
        # replay runs: the torch reference's exact features and noise
        noise = (record_noise(a.seed, cid, "structure", 1) if a.seed is not None
                 else ReplayNoise(DATA / cid / "ref" / "noise.npz"))
        out = model(feats, PredictArgs(recycling_steps=a.recycling, sampling_steps=a.steps), noise)
        timings[cid] = round(time.perf_counter() - t0, 2)
        out_dir = DATA / "validation" / "runs" / a.run / cid / "predictions"
        out_dir.mkdir(parents=True, exist_ok=True)
        writer = BoltzWriter(data_dir=DATA / cid / "processed" / "structures", output_dir=out_dir,
                             output_format="mmcif", boltz2=True)
        writer.write_on_batch_end(None, None, _structure_pred_dict(out, batch), None, batch, 0, 0)
        print(cid, timings[cid], "s", flush=True)
    (DATA / "validation" / "runs" / a.run / "timings.json").write_text(json.dumps(timings, indent=1))


if __name__ == "__main__":
    main()
