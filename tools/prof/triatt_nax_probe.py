"""Rung 2N, candidate (a) step 1: can mx.fast.metal_kernel reach the M5 neural accelerators (NAX)?

  python tools/prof/triatt_nax_probe.py          # GPU, ~1 min

Rows (JSON, also appended to logs/triatt_2n.jsonl):
  lang   : __METAL_VERSION__ and __HAVE_TENSOR__ as seen by metal_kernel source (MPP tensor ops need Metal 4.0)
  mma1   : one mpp::tensor_ops::matmul2d (16x32x16, execution_simdgroup, cooperative tensors) through metal_kernel
           with MLX's fragment layout, vs numpy (TB = right operand transposed; A fp16 or fp32)
  peak   : register-only MMA throughput from a custom kernel: NAX matmul2d vs simdgroup_matrix 8x8 (half / float)
Interpretation: NAX is reachable if `mma1` passes and `peak` nax TFLOP/s is well above sg_* (round 1: ~6.3 TFLOP/s
for simdgroup MMA in real kernels; MLX fp16 GEMMs 14 TFLOP/s at N=400 shapes).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from triatt_common import bench, emit, mx, np  # noqa: E402

from boltz_mlx import kernels_nax as KN  # noqa: E402

EXP = "nax_probe"

# 1. language version
try:
    emit({"exp": EXP, "probe": "lang", **KN.probe_lang()})
except Exception as e:
    emit({"exp": EXP, "probe": "lang", "error": f"{type(e).__name__}: {str(e)[:3000]}"})

# 2. one NAX MMA, layout check
rng = np.random.default_rng(0)
for tb in (True, False):
    for adt in (mx.float16, mx.float32):
        row = {"exp": EXP, "probe": "mma1", "tb": tb, "a_dtype": str(adt)}
        try:
            a = rng.standard_normal((16, 16)).astype(np.float16).astype(np.float32)
            b = rng.standard_normal((32, 16) if tb else (16, 32)).astype(np.float16).astype(np.float32)
            c = np.array(KN.nax_mma_once(mx.array(a), mx.array(b), tb, adt))
            ref = a.astype(np.float64) @ (b.T if tb else b).astype(np.float64)
            row["max_rel"] = float(np.abs(c - ref).max() / np.abs(ref).max())
            row["pass"] = row["max_rel"] < 1e-3
            # diagnose a layout mismatch: does it match a transposed / permuted product instead?
            if not row["pass"]:
                beff = b.T if tb else b
                alts = {"AT_B": a.T @ beff, "B_halves_swapped": np.concatenate([a @ beff[:, 16:], a @ beff[:, :16]], 1)}
                row["alt_match"] = {k: float(np.abs(c - v).max() / np.abs(v).max()) for k, v in alts.items()}
                row["c_sample"] = c[:2, :6].round(3).tolist()
                row["ref_sample"] = ref[:2, :6].round(3).tolist()
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {str(e)[:3000]}"
        emit(row)

# 3. peak throughput, register operands
NSG, ITERS = 8192, 256
cfgs = [("nax", n, mx.float16) for n in (1, 2, 4)] + [("nax", 2, mx.float32)] + \
       [("sg_half", n, None) for n in (4, 8)] + [("sg_float", n, None) for n in (4, 8)]
for kind, nacc, adt in cfgs:
    for spt in (4, 8):
        row = {"exp": EXP, "probe": "peak", "kind": kind, "nacc": nacc, "sg_per_tg": spt,
               "a_dtype": str(adt) if adt is not None else None}
        try:
            fn, flops = KN.peak_kernel(kind, NSG, ITERS, nacc, sg_per_tg=spt, a_dtype=adt or mx.float16)
            t = bench(fn, reps=5)
            row.update(ms=round(t, 3), tflops=round(flops / t / 1e9, 2), finite=bool(mx.all(mx.isfinite(fn())).item()))
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {str(e)[:3000]}"
        emit(row)
