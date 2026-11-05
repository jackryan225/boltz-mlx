import time, numpy as np, mlx.core as mx
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # this checkout's boltz_mlx first: the editable install maps boltz_mlx to the main checkout
import boltz_mlx
def ref(q,k,v,bias,scale):
    a=(q*scale)@k.swapaxes(-1,-2)+bias; return mx.softmax(a,axis=-1)@v
def T(f,n=50):
    r=f(); mx.eval(r); t=time.perf_counter()
    for i in range(n): r=f(); mx.eval(r)
    return (time.perf_counter()-t)/n*1e3
for (B,H,L,K,D) in [(1,16,117,117,48),(117,4,117,117,32),(29,4,32,128,32),(1,16,400,400,48)]:
    q,k,v=[mx.random.normal((B,H,n,D)) for n in (L,K,K)]
    bias=mx.random.normal((B,H,L,K))*3; bias=mx.where(mx.random.uniform(shape=(B,1,1,K))<0.1,-1e9,bias)
    s=D**-0.5
    r=ref(q,k,v,bias,s); f=mx.fast.scaled_dot_product_attention(q,k,v,scale=s,mask=bias)
    err=float(mx.abs(r-f).max()/mx.abs(r).max())
    print((B,H,L,K,D),"rel",f"{err:.2e}","ref ms",f"{T(lambda: ref(q,k,v,bias,s)):.3f}","sdpa ms",f"{T(lambda: mx.fast.scaled_dot_product_attention(q,k,v,scale=s,mask=bias)):.3f}")
