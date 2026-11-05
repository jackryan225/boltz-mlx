"""A/B: channel-major fast tri-mult vs row-major path (alternating, same conditions). CASE=..."""
import os, time, numpy as np, mlx.core as mx
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # this checkout's boltz_mlx first: the editable install maps boltz_mlx to the main checkout
import boltz_mlx
from boltz_mlx.layers.triangular_mult import TriangleMultiplicationOutgoing, TriangleMultiplicationIncoming
from boltz_mlx.weights import load_module
N = int(os.environ.get("N", "117")); n = int(os.environ.get("REPS", "20"))
x = mx.random.normal((1, N, N, 128)); m = mx.ones((1, N, N))
for cls, nm in [(TriangleMultiplicationOutgoing, "tri_mul_out"), (TriangleMultiplicationIncoming, "tri_mul_in")]:
    L = load_module(cls(128), f"pairformer_module.layers.0.{nm}")
    fast = lambda: L(x, m)
    x2 = mx.concatenate([x, x], 0); m2 = mx.concatenate([m, m], 0)
    slow = lambda: L(x2, m2)  # B=2 forces row-major path; report per batch element
    res = {"fast": [], "slow(B=2)/2": []}
    for _ in range(3):
        for k, f, div in (("fast", fast, 1), ("slow(B=2)/2", slow, 2)):
            mx.eval(f()); t = time.perf_counter(); mx.eval([f() for _ in range(n)]); res[k].append((time.perf_counter() - t) / n / div * 1e3)
    print(N, nm, {k: round(min(v), 3) for k, v in res.items()}, "ms")
