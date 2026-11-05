"""A/B: Transition (pair rep, D=128/H=512) implementations mlx | fused, alternating rounds, min over rounds.

Times the full Transition module call (LN + GEMMs + SwiGLU, incl. the large-N row-chunk path of the mlx impl).
GPU only; one process at a time.

  python tools/prof/ab_transition_impl.py                  # N=117,400
  N=117,400,964 REPS=20 ROUNDS=3 python tools/prof/ab_transition_impl.py
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # use this checkout, not an editable install elsewhere

import mlx.core as mx

import boltz_mlx  # noqa: F401  (TF32 off)
from boltz_mlx import kernels as K
from boltz_mlx.layers.transition import Transition
from boltz_mlx.weights import load_module

mx.set_default_device(mx.gpu)
NS = [int(n) for n in os.environ.get("N", "117,400").split(",")]
IMPLS = os.environ.get("IMPLS", "mlx,fused").split(",")
REPS = int(os.environ.get("REPS", "20"))
ROUNDS = int(os.environ.get("ROUNDS", "3"))

mod = load_module(Transition(128, 512), "pairformer_module.layers.0.transition_z")


def bench(f, reps):
    mx.eval(f())
    t = time.perf_counter()
    mx.eval([f() for _ in range(reps)])
    return (time.perf_counter() - t) / reps * 1e3


flops_per_row = 2.0 * (128 * 1024 + 512 * 128)
for N in NS:
    mx.random.seed(0)
    x = mx.random.normal((1, N, N, 128))
    mx.eval(x)
    reps = REPS if N <= 200 else max(3, REPS // 4)
    ref = None
    res = {i: [] for i in IMPLS}
    for _ in range(ROUNDS):
        for impl in IMPLS:
            K.TRANSITION_IMPL = impl
            res[impl].append(bench(lambda: mod(x), reps))
    for impl in IMPLS:
        K.TRANSITION_IMPL = impl
        y = mod(x)
        mx.eval(y)
        if ref is None:
            ref = y
        err = float(mx.abs(y - ref).max() / mx.abs(ref).max())
        t = min(res[impl])
        print(f"N={N:4d} {impl:6s} {t:8.3f} ms  ({min(res[IMPLS[0]]) / t:.2f}x vs {IMPLS[0]})  "
              f"{flops_per_row * N * N / t / 1e9:.2f} TFLOP/s  rel diff vs {IMPLS[0]} {err:.1e}")
K.TRANSITION_IMPL = "mlx"
