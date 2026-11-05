import os, time, mlx.core as mx, mlx.nn as nn
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # this checkout's boltz_mlx first: the editable install maps boltz_mlx to the main checkout
import boltz_mlx
from boltz_mlx.layers.transition import Transition
from boltz_mlx.weights import load_module
N = int(os.environ.get("N", "117")); n = int(os.environ.get("REPS", "20"))
L = load_module(Transition(128, 512), "pairformer_module.layers.0.transition_z")
x = mx.random.normal((1, N, N, 128))
def old():
    h = L.norm(x) @ L._w12.T
    return L.fc3(nn.silu(h[..., :512]) * h[..., 512:])
new = lambda: L(x)
mx.eval(new())
r = {"new": [], "old": []}
for _ in range(3):
    for k, f in (("new", new), ("old", old)):
        mx.eval(f()); t = time.perf_counter(); mx.eval([f() for _ in range(n)]); r[k].append((time.perf_counter() - t) / n * 1e3)
print(N, {k: round(min(v), 3) for k, v in r.items()}, "ms", "maxdiff", float(mx.abs(new() - old()).max()))
