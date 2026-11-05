import os, numpy as np, mlx.core as mx
rng=np.random.default_rng(0); a=rng.normal(size=(512,512)).astype(np.float32); b=rng.normal(size=(512,512)).astype(np.float32)
ref=a.astype(np.float64)@b.astype(np.float64)
for dev in (mx.gpu, mx.cpu):
    with mx.stream(dev):
        c=np.array(mx.array(a)@mx.array(b))
    print(os.environ.get("MLX_ENABLE_TF32"), dev, "max rel err", np.abs(c-ref).max()/np.abs(ref).max())
