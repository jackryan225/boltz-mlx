"""A/B: tri-attention kernel implementations (scalar | mma | mma_half), alternating rounds, min over rounds.

Times (a) the kernel alone (tri_attention_fused on a precomputed fused projection) and (b) the full
TriangleAttention call (LN + projection GEMM + kernel + out-proj). GPU only; one process at a time.

  python tools/prof/ab_triatt_impl.py                  # N=117,400; impls scalar,mma
  N=117,400 IMPLS=scalar,mma,mma_half REPS=20 python tools/prof/ab_triatt_impl.py
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # use this checkout, not an editable install elsewhere

import mlx.core as mx

import boltz_mlx  # noqa: F401  (TF32 off)
from boltz_mlx import kernels as K
from boltz_mlx.layers.triangular_attention.attention import TriangleAttentionStartingNode
from boltz_mlx.weights import load_module

mx.set_default_device(mx.gpu)
NS = [int(n) for n in os.environ.get("N", "117,400").split(",")]
IMPLS = os.environ.get("IMPLS", "scalar,mma").split(",")
REPS = int(os.environ.get("REPS", "20"))
ROUNDS = int(os.environ.get("ROUNDS", "3"))
H, C = 4, 32

mod = load_module(TriangleAttentionStartingNode(128, C, H, inf=1e9), "pairformer_module.layers.0.tri_att_start")


def bench(f, reps):
    mx.eval(f())
    t = time.perf_counter()
    mx.eval([f() for _ in range(reps)])
    return (time.perf_counter() - t) / reps * 1e3


for N in NS:
    mx.random.seed(0)
    x = mx.random.normal((1, N, N, 128))
    mask = (mx.random.uniform(shape=(1, N, N)) > 0.05).astype(mx.float32)
    w = mx.concatenate([mod.mha.linear_q.weight, mod.mha.linear_k.weight, mod.mha.linear_v.weight,
                        mod.mha.linear_g.weight, mod.linear.weight], 0)
    h = x @ w.T
    mx.eval(x, mask, h)
    reps = REPS * 2 if N <= 200 else max(3, REPS // 2)
    res = {(i, kind): [] for i in IMPLS for kind in ("kernel", "full")}
    for _ in range(ROUNDS):
        for impl in IMPLS:
            K.TRIATT_IMPL = impl
            res[(impl, "kernel")].append(bench(lambda: K.tri_attention_fused(h, H, C, mask, 1e9), reps))
            res[(impl, "full")].append(bench(lambda: mod(x, mask=mask), reps))
    flops = 4.0 * N * N * N * C * H  # QK^T + PV, 2 flops/FMA, per (row, head): 4*J*J*C
    for kind in ("kernel", "full"):
        base = min(res[(IMPLS[0], kind)])
        for impl in IMPLS:
            t = min(res[(impl, kind)])
            extra = f"  {flops / t / 1e9:.2f} TFLOP/s" if kind == "kernel" else ""
            print(f"N={N:4d} {kind:6s} {impl:9s} {t:8.3f} ms  ({base / t:.2f}x vs {IMPLS[0]}){extra}")
K.TRIATT_IMPL = "scalar"
