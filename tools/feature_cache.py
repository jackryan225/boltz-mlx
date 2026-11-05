"""Per-(record, global seed) feature cache shared by every mode.

Features are produced by the upstream featurizer with the per-record featurizer seed
(boltz_mlx.seeding: torch seeded from sha256(global_seed|record|features|0) before featurizing), then cached at
data/cache/features/<case>/s<seed>[_aff].npz (+ .meta.pkl). Every mode reads the same file, so seed-matched
comparisons see bit-identical inputs.

Usage (CLI): python tools/feature_cache.py <case> --seeds 0 1 2 [--affinity]
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # this checkout's boltz_mlx first: the editable install maps boltz_mlx to the main checkout
DATA = Path(os.environ.get("BOLTZ_MLX_DATA", Path(__file__).resolve().parents[1] / "data")).expanduser()  # env: other Macs (tools/portable_check.py)
CACHE_DIR = DATA / "cache" / "features"
sys.path.insert(0, str(ROOT / "tools"))


def _path(case: str, seed: int, affinity: bool, tag: str = "") -> Path:
    # affinity features depend on the mode's own structure pass (pred_dir), so they carry a run tag
    return CACHE_DIR / case / f"s{int(seed)}{'_aff' if affinity else ''}{'_' + tag if tag else ''}.npz"


def build(case: str, seed: int, affinity: bool = False, pred_dir: Path | None = None, tag: str = "") -> Path:
    import torch

    from prep_case import batch_to_numpy, load_batch_dataset

    from boltz_mlx.seeding import SeededFeatureDataset

    if affinity and not tag:
        raise ValueError("affinity features need a run tag (they depend on that run's structure pass)")
    out = _path(case, seed, affinity, tag)
    if out.exists():
        return out
    ds = load_batch_dataset(DATA / case, affinity, pred_dir)
    sds = SeededFeatureDataset(ds, seed)
    from boltz.data.module.inferencev2 import collate

    batch = collate([sds[0]])
    arrays, other = batch_to_numpy(batch)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    with open(out.with_suffix(".meta.pkl"), "wb") as f:
        pickle.dump(other, f)
    tmp.rename(out)
    del torch
    return out


def load(case: str, seed: int, affinity: bool = False, build_if_missing: bool = True,
         pred_dir: Path | None = None, tag: str = ""):
    """Returns (arrays dict of numpy, meta dict) for (case, seed[, affinity run tag])."""
    p = _path(case, seed, affinity, tag)
    if not p.exists():
        if not build_if_missing:
            raise FileNotFoundError(p)
        build(case, seed, affinity, pred_dir, tag)
    z = np.load(p)
    with open(p.with_suffix(".meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    return {k: z[k] for k in z.files}, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="+")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--affinity", action="store_true")
    a = ap.parse_args()
    for c in a.cases:
        for s in a.seeds:
            print(build(c, s, a.affinity))


if __name__ == "__main__":
    main()
