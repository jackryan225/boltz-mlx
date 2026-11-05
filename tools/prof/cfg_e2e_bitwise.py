"""End-to-end bit identity of per-device kernel configs: f16 predictions of every L1 case with this chip's config
entry vs a reference entry (default "default" = the M5 Pro tuning every untuned chip uses), each in fresh processes,
output hashes compared. GPU (correctness only: class 3).

  python tools/prof/cfg_e2e_bitwise.py [--ref default] [--steps 50] [--cases prot117,lig_aff_aff,...]

Entries are selected with BOLTZ_MLX_CFG_CHIP (kernels.cfg_chip): the candidate run leaves it unset (the chip's own
entry), the reference run sets it to --ref. Structure passes (conf checkpoint, 3 recycles, 1 sample) hash coordinates,
pLDDT / PAE / PDE / pTM / ipTM / complex pLDDT and the distogram; the lig_aff affinity pass (aff checkpoint, 5
recycles, 5 samples; its own process: never both checkpoints in one process) hashes coordinates, ipTM and the affinity
outputs. Row -> logs/cfg_e2e_bitwise.jsonl. Exit 1 if any case differs (or a run fails).
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # this checkout's boltz_mlx first: the editable install maps boltz_mlx to the main checkout
DATA = Path(os.environ.get("BOLTZ_MLX_DATA", Path(__file__).resolve().parents[2] / "data")).expanduser()
OUT = ROOT / "logs" / "cfg_e2e_bitwise.jsonl"
CASES = "prot117,tyk2_ejm_31,msa384,lig_aff,prot400,ex_ligand,cdk2_17,lig_aff_aff"
SKEYS = ["sample_atom_coords", "plddt", "pae", "pde", "ptm", "iptm", "complex_plddt", "pdistogram"]
AFF_CASES = {"lig_aff_aff": "lig_aff"}   # affinity passes: case label -> data directory (lig_aff itself is a structure case)
AKEYS = ["sample_atom_coords", "iptm", "affinity_pred_value", "affinity_probability_binary", "affinity_pred_value1",
         "affinity_pred_value2"]


def _hash(arrs: dict) -> str:
    import numpy as np

    h = hashlib.sha256()
    for k in sorted(arrs):
        a = np.ascontiguousarray(np.asarray(arrs[k], dtype=np.float32))
        h.update(k.encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()[:16]


def child(kind: str, cases: list, steps: int) -> None:
    sys.path.insert(0, str(ROOT))
    import boltz_mlx  # noqa: F401
    import mlx.core as mx
    import numpy as np

    from boltz_mlx import kernels as K
    from boltz_mlx.model import PredictArgs, load_model
    from boltz_mlx.sampler import record_noise

    assert K.f16() and K.mpp_enabled()
    model = load_model(kind)
    res = {"cfg_chip": K.cfg_chip(), "chip": K.chip_name(), "weights": getattr(model, "weights_kind", "?"),
           "boltz_mlx": boltz_mlx.__file__, "hashes": {}, "s": {}}
    for case in cases:
        aff = case in AFF_CASES
        name = AFF_CASES.get(case, case)
        z = np.load(DATA / name / ("feats_affinity.npz" if aff else "feats.npz"))
        feats = {k: mx.array(z[k]) for k in z.files}
        args = PredictArgs(5, steps, 5) if aff else PredictArgs(3, steps, 1)
        t = time.perf_counter()
        o = model(feats, args, record_noise(0, name, "affinity" if aff else "structure", args.diffusion_samples))
        res["s"][case] = round(time.perf_counter() - t, 3)
        res["hashes"][case] = _hash({k: np.array(o[k]) for k in (AKEYS if aff else SKEYS) if k in o})
        del o, feats
        mx.clear_cache()
    print(json.dumps(res), flush=True)


def run(entry: str | None, kind: str, cases: list, steps: int) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("BOLTZ_MLX_")}
    env.update(BOLTZ_MLX_PRECISION="f16", PYTHONPATH=str(ROOT))
    if entry:
        env["BOLTZ_MLX_CFG_CHIP"] = entry
    r = subprocess.run([sys.executable, __file__, "--_child", kind, "--steps", str(steps), "--cases", ",".join(cases)],
                       env=env, capture_output=True, text=True)
    if r.returncode != 0:
        return {"error": (r.stdout + r.stderr)[-2000:]}
    return json.loads(r.stdout.strip().splitlines()[-1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="default")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cases", default=CASES)
    ap.add_argument("--_child", default=None)
    a = ap.parse_args()
    cases = a.cases.split(",")
    if a._child:
        child(a._child, cases, a.steps)
        return 0
    groups = {"conf": [c for c in cases if c not in AFF_CASES], "aff": [c for c in cases if c in AFF_CASES]}
    git = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    pkg = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD:boltz_mlx"], capture_output=True,
                         text=True).stdout.strip()
    row = {"exp": "cfg_e2e_bitwise", "date": time.strftime("%Y-%m-%d %H:%M:%S"), "git": git, "pkg_tree": pkg,
           "ref": a.ref, "steps": a.steps, "cases": {}, "runs": {}}
    ok = True
    for kind, cs in groups.items():
        if not cs:
            continue
        cand, ref = run(None, kind, cs, a.steps), run(a.ref, kind, cs, a.steps)
        row["runs"][kind] = {"cand": {k: v for k, v in cand.items() if k != "hashes"},
                             "ref": {k: v for k, v in ref.items() if k != "hashes"}}
        if "error" in cand or "error" in ref:
            ok = False
            print(f"{kind}: run failed\n{cand.get('error', '')}\n{ref.get('error', '')}")
            continue
        if cand["cfg_chip"] == ref["cfg_chip"]:
            print(f"warning: both runs used the entry {cand['cfg_chip']!r} (no entry for this chip?)")
        for c in cs:
            same = cand["hashes"][c] == ref["hashes"][c]
            row["cases"][c] = {"cand": cand["hashes"][c], "ref": ref["hashes"][c], "identical": same,
                               "cand_s": cand["s"][c], "ref_s": ref["s"][c]}
            ok &= same
            print(f"  {c:12s} {cand['cfg_chip']!r} {cand['hashes'][c]}  vs  {ref['cfg_chip']!r} {ref['hashes'][c]}  "
                  f"{'IDENTICAL' if same else 'DIFFERENT'}")
    missing = [c for c in cases if c not in row["cases"]]
    if missing:   # every requested case must have been compared
        ok = False
        print(f"cases without a comparison: {missing}")
    row["all_identical"] = bool(ok)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"cfg_e2e_bitwise ({len(row['cases'])} cases, {a.steps} steps, ref {a.ref!r}): "
          f"{'ALL IDENTICAL' if ok else 'MISMATCH / FAILURE'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
