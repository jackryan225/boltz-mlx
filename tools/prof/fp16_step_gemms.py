"""fp16 diffusion step at prot117: GEMM-only time of the token transformer vs the full compiled step."""
import time, mlx.core as mx
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # this checkout's boltz_mlx first: the editable install maps boltz_mlx to the main checkout
import boltz_mlx
dt = mx.float16
M = 117
shapes = [(768, 3072), (768, 768), (768, 4608), (1536, 768)] * 24 + [(768, 110592)]
X = {k: mx.random.normal((M, k)).astype(dt) for k in {s[0] for s in shapes}}
W = [mx.random.normal((n, k)).astype(dt) for k, n in shapes]
mx.eval(list(X.values()), W)
def run():
    return [X[k] @ w.T for (k, n), w in zip(shapes, W)]
mx.eval(run())
t = time.perf_counter()
for _ in range(20):
    mx.eval(run())
print(f"token-transformer GEMMs only (fp16, {len(shapes)} GEMMs): {(time.perf_counter()-t)/20*1e3:.2f} ms/step")
f = sum(2 * M * k * n for k, n in shapes)
print(f"FLOPs {f/1e9:.1f} GF")
