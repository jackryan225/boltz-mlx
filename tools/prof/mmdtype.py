import time, mlx.core as mx
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # this checkout's boltz_mlx first: the editable install maps boltz_mlx to the main checkout
import boltz_mlx
def T(f,n=20):
    r=f(); mx.eval(r); t=time.perf_counter()
    for i in range(n): r=f(); mx.eval(r)
    return (time.perf_counter()-t)/n
for (M,K,N,reps) in [(117,768,3072,50),(117,768,73728,5),(13689,128,640,20),(160000,128,640,3)]:
    for dt in (mx.float32, mx.float16, mx.bfloat16):
        a=mx.random.normal((M,K)).astype(dt); b=mx.random.normal((N,K)).astype(dt)
        t=T(lambda: [a@b.T for _ in range(reps)])/reps
        print(f"{M}x{K}x{N} {str(dt):14s} {t*1e3:7.3f} ms  {2*M*K*N/t/1e12:5.2f} TFLOP/s")
