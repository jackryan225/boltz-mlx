import time, numpy as np, mlx.core as mx
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # this checkout's boltz_mlx first: the editable install maps boltz_mlx to the main checkout
import boltz_mlx
from boltz_mlx.kernels import layer_norm
def T(name,f,n=100):
    r=f(); mx.eval(r); t=time.perf_counter()
    for i in range(n): r=f(); mx.eval(r)
    print(f"{name:34s} {(time.perf_counter()-t)/n*1e3:7.3f} ms")
for shape in [(1,117,117,128),(1,400,400,128),(1,117,768),(3,5,64),(1,13,384)]:
    x=mx.random.normal(shape)*3+1; w=mx.random.normal((shape[-1],)); b=mx.random.normal((shape[-1],))
    xn=np.array(x).astype(np.float64); m=xn.mean(-1,keepdims=True); v=xn.var(-1,keepdims=True)
    ref=(xn-m)/np.sqrt(v+1e-5)*np.array(w)+np.array(b)
    for name,args in [("wb",(w,b)),("w",(w,None)),("none",(None,None))]:
        refi=(xn-m)/np.sqrt(v+1e-5)*(np.array(args[0]) if args[0] is not None else 1)+(np.array(args[1]) if args[1] is not None else 0)
        y=layer_norm(x,*args,1e-5); f=mx.fast.layer_norm(x,*args,1e-5)
        print(shape,name,"ours err",float(np.abs(np.array(y)-refi).max()),"fast err",float(np.abs(np.array(f)-refi).max()))
    T(" ours",lambda: layer_norm(x,w,b,1e-5)); T(" fast",lambda: mx.fast.layer_norm(x,w,b,1e-5)); T(" x*2",lambda: x*2)
