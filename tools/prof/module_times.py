"""Per-module GPU time on real recorded inputs (the L1 captures), for the current precision mode.

  python tools/prof/module_times.py prot400 [--mode f16|strict|turbo] [--reps 20]

Each captured module call (data/cache/l1/<case>/*.pkl) is re-run `reps` times pipelined (one eval at the end),
min of 3 rounds. Rows -> logs/module_times.jsonl; prints a table with the per-trunk-pass estimate
(time × calls per pass: pairformer 64 layers, MSA module 4 layers, confidence 4).
"""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # this checkout's boltz_mlx first: the editable install maps boltz_mlx to the main checkout
sys.path.insert(0, str(ROOT / "tools" / "validation"))
MODE_ENV = {"strict": {}, "f16": {"BOLTZ_MLX_PRECISION": "f16"},
            "turbo": {"BOLTZ_MLX_DIFF_DTYPE": "fp16", "BOLTZ_MLX_TF32": "1"}}
PER_PASS = {"trunk.pairformer": 64, "trunk.msa": 4, "confidence": 4}


def child(case: str, mode: str, reps: int):
    import boltz_mlx  # noqa: F401
    import mlx.core as mx

    import l1_ops
    from boltz_mlx import kernels as K

    model = l1_ops._model(False)
    mods = dict(model.named_modules())
    files = sorted((l1_ops.CACHE / case).glob("*__0.pkl"))
    git = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"]).decode().strip()
    msa_env = {k: v for k, v in os.environ.items() if k.startswith("BOLTZ_MLX_MSA_")}
    rows = []
    for f in files:
        rec = pickle.load(open(f, "rb"))
        if rec["phase"] not in PER_PASS:
            continue
        m = mods[rec["path"]]
        K.set_phase(rec["phase"])
        args, kwargs = l1_ops._from_np(rec["args"]), l1_ops._from_np(rec["kwargs"])
        mx.eval(args, kwargs)

        def run():
            o = type(m).__call__(m, *args, **kwargs)
            o = tuple(K.unscale(x) for x in o) if isinstance(o, tuple) else K.unscale(o)
            return o

        mx.eval(run())
        best = 1e9
        for _ in range(3):
            t = time.perf_counter()
            mx.eval([run() for _ in range(reps)])
            best = min(best, (time.perf_counter() - t) / reps)
        K.guard_take()
        rows.append({"case": case, "mode": mode, "phase": rec["phase"], "cls": rec["cls"], "ms": round(best * 1e3, 3),
                     "git": git, "date": time.strftime("%Y-%m-%d %H:%M:%S"), **({"env": msa_env} if msa_env else {})})
    with open(ROOT / "logs" / "module_times.jsonl", "a") as fo:
        for r in rows:
            fo.write(json.dumps(r) + "\n")
    for r in rows:
        print(f"{r['phase']:18s} {r['cls']:32s} {r['ms']:8.3f} ms")


def main():
    case = sys.argv[1]
    mode = sys.argv[sys.argv.index("--mode") + 1] if "--mode" in sys.argv else "f16"
    reps = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 20
    if os.environ.get("_MT_CHILD") != "1":
        env = {k: v for k, v in os.environ.items()  # BOLTZ_MLX_MSA_* (MSA-kernel flags) pass through
           if (not k.startswith("BOLTZ_MLX_") or k.startswith("BOLTZ_MLX_MSA_")) and k != "MLX_ENABLE_TF32"}
        env.update(MODE_ENV[mode], _MT_CHILD="1", PYTHONPATH=str(ROOT))
        sys.exit(subprocess.run([sys.executable, __file__, *sys.argv[1:]], env=env).returncode)
    child(case, mode, reps)


if __name__ == "__main__":
    main()
