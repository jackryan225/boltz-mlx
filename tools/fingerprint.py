"""Numerics fingerprint of a build: hashes of model outputs on fixed inputs, global seed 0.

Cases: prot117 (structure pass), lig_aff (affinity pass on the reference structure's affinity features),
and, once the FEP+ data exists, tyk2_<first ligand> (structure + affinity via the CLI flow).
A gate result carries to a new build only if its fingerprint is bit-identical.

Usage: python tools/fingerprint.py [--mode strict|fast|turbo|f16|f16-portable|f16-tensorops] [--runs 3] [--no-log]
Writes one jsonl row per run to logs/fingerprints.jsonl. Rows carry the device (chip, architecture, nax flag): MLX
picks its own GEMM kernels per chip, so fingerprints compare only within one chip; "f16" without NAX kernels (M1-M4,
or BOLTZ_MLX_KERNEL_PATH=portable) is recorded as "f16-portable" ("f16-tensorops": MPP kernels without NAX), with the
number of Metal kernels built and how many used MetalPerformancePrimitives.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # this checkout's boltz_mlx first: the editable install maps boltz_mlx to the main checkout
DATA = Path(os.environ.get("BOLTZ_MLX_DATA", Path(__file__).resolve().parents[1] / "data")).expanduser()  # env: other Macs (tools/portable_check.py)

MODES = {  # environment for each shipped mode (set before boltz_mlx/mlx import)
    "strict": {},
    "fast": {"BOLTZ_MLX_DIFF_DTYPE": "fp16"},
    "turbo": {"BOLTZ_MLX_DIFF_DTYPE": "fp16", "BOLTZ_MLX_TF32": "1"},
    "f16": {"BOLTZ_MLX_PRECISION": "f16"},  # fp16 pair-stack GEMMs (pre-scaled), fp16 diffusion, no TF32
    # the f16 preset on the other kernel paths (kernels.py "Hardware capability"): portable = no MPP kernels (M1-M4
    # default; on an M5 = NAX disabled), tensorops = the MPP kernels on shader cores (M1-M4, opt-in). Plain --mode f16
    # is labelled by the path it took (row "mode" = kernels.mode_label: f16 | f16-tensorops | f16-portable).
    "f16-portable": {"BOLTZ_MLX_PRECISION": "f16", "BOLTZ_MLX_KERNEL_PATH": "portable"},
    "f16-tensorops": {"BOLTZ_MLX_PRECISION": "f16", "BOLTZ_MLX_KERNEL_PATH": "tensorops"},
}
STEPS = {"strict": 200, "fast": 200, "turbo": 50, "f16": 200, "f16-portable": 200, "f16-tensorops": 200}


def _hash(arrays: dict) -> str:
    import numpy as np

    h = hashlib.sha256()
    for k in sorted(arrays):
        a = np.ascontiguousarray(np.asarray(arrays[k], dtype=np.float32))
        h.update(k.encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()[:16]


WEIGHTS_USED: dict = {}  # checkpoint kind per model (fp32 | mixed), reported next to the hashes (not hashed)


def run_once(mode: str, save: Path | None = None, timings: dict | None = None) -> dict:
    """Hashes of the two fingerprint cases. save: also write the hashed output arrays (save/<case>.npz, + the atom /
    token masks) for accuracy comparisons (tools/portable_check.py); timings: per-case predict seconds."""
    import mlx.core as mx
    import numpy as np

    sys.path.insert(0, str(ROOT / "tools"))
    import feature_cache

    from boltz_mlx.model import PredictArgs, load_model
    from boltz_mlx.sampler import record_noise

    out = {}
    keys = ["sample_atom_coords", "plddt", "pae", "pde", "ptm", "iptm", "complex_plddt"]
    # prot117 structure pass
    model = load_model("conf")
    WEIGHTS_USED["conf"] = getattr(model, "weights_kind", "fp32")
    arrays, _ = feature_cache.load("prot117", 0)
    feats = {k: mx.array(v) for k, v in arrays.items()}
    t0 = time.perf_counter()
    o = model(feats, PredictArgs(3, STEPS[mode], 1), record_noise(0, "prot117", "structure", 1))
    arrs = {k: np.array(o[k]) for k in keys if k in o}
    if timings is not None:
        timings["prot117_s"] = round(time.perf_counter() - t0, 3)
    out["prot117"] = _hash(arrs)
    if save is not None:
        np.savez(Path(save) / "prot117.npz", **arrs, atom_pad_mask=arrays["atom_pad_mask"],
                 token_pad_mask=arrays["token_pad_mask"])
    del model
    mx.clear_cache()
    # lig_aff affinity pass on the CPU reference's affinity features
    model = load_model("aff")
    WEIGHTS_USED["aff"] = getattr(model, "weights_kind", "fp32")
    z = np.load(DATA / "lig_aff" / "feats_affinity.npz")
    feats = {k: mx.array(z[k]) for k in z.files}
    t0 = time.perf_counter()
    o = model(feats, PredictArgs(5, 200, 5), record_noise(0, "lig_aff", "affinity", 5))
    akeys = ["sample_atom_coords", "iptm", "affinity_pred_value", "affinity_probability_binary",
             "affinity_pred_value1", "affinity_pred_value2"]
    arrs = {k: np.array(o[k]) for k in akeys if k in o}
    if timings is not None:
        timings["lig_aff_affinity_s"] = round(time.perf_counter() - t0, 3)
    out["lig_aff_affinity"] = _hash(arrs)
    if save is not None:
        np.savez(Path(save) / "lig_aff_affinity.npz", **arrs, atom_pad_mask=z["atom_pad_mask"],
                 token_pad_mask=z["token_pad_mask"])
    del model
    mx.clear_cache()
    return out


def pkg_tree() -> str:
    """Git tree hash of boltz_mlx/ at HEAD: numerics-relevant identity of a build (tools/doc commits keep it)."""
    return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD:boltz_mlx"]).decode().strip()


def git_hash() -> str:
    h = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"]).decode().strip()
    dirty = subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", "HEAD", "--", "boltz_mlx"]).returncode != 0
    return h + ("-dirty" if dirty else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="strict", choices=list(MODES))
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if not a._child:  # each run in a fresh process so the mode env applies before MLX initializes
        rows = []
        for r in range(a.runs):
            env = {**os.environ, **MODES[a.mode]}
            res = subprocess.run([sys.executable, __file__, "--mode", a.mode, "--_child"], env=env,
                                 capture_output=True, text=True, check=True)
            row = json.loads(res.stdout.strip().splitlines()[-1])
            row.update(run=r, git=git_hash(), pkg_tree=pkg_tree(), date=time.strftime("%Y-%m-%d %H:%M"))
            rows.append(row)
            print(json.dumps(row))
            if not a.no_log:
                with open(ROOT / "logs" / "fingerprints.jsonl", "a") as f:
                    f.write(json.dumps(row) + "\n")
        same = all(r["hashes"] == rows[0]["hashes"] for r in rows)
        print(f"{rows[0]['mode']} on {rows[0]['device']['chip']} (path {rows[0]['device']['path']}): {len(rows)} runs, "
              f"fingerprint {rows[0]['fingerprint']}, bit-identical across runs: {same}")
        return
    import boltz_mlx  # noqa: F401  (TF32 env handling)
    import mlx.core as mx

    from boltz_mlx import kernels as K

    K.KERNEL_LOG = []  # audit: every Metal kernel built (kernels.metal_kernel refuses NAX sources without NAX)
    hashes = run_once(a.mode)
    built = {name: nax for name, nax in K.KERNEL_LOG}
    print(json.dumps({"mode": K.mode_label("f16" if a.mode.startswith("f16") else a.mode), "mlx": mx.__version__,
                      "hashes": hashes, "weights": WEIGHTS_USED, "device": K.device_record(),
                      "kernels_built": len(built), "mpp_kernels_built": sum(built.values()),
                      "fingerprint": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:12]}))


if __name__ == "__main__":
    main()
