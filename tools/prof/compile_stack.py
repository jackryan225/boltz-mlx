"""Pairformer stack: eager vs per-layer compiled vs one shared compiled fn with params as inputs."""
import os, time, numpy as np, mlx.core as mx
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # this checkout's boltz_mlx first: the editable install maps boltz_mlx to the main checkout
import boltz_mlx
from mlx.utils import tree_flatten, tree_unflatten
from boltz_mlx.model import load_model
DATA = os.environ.get("BOLTZ_MLX_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")) + "/" + os.environ.get("CASE", "prot117") + "/"
model = load_model("conf")
z = np.load(DATA + "feats.npz"); feats = {k: mx.array(z[k]) for k in z.files}
s_inputs, s0, z0, rpe = model.embed(feats); mx.eval(s0, z0)
mask = feats["token_pad_mask"].astype(mx.float32); pm = mask[:, :, None] * mask[:, None, :]
layers = model.pairformer_module.layers
def run(fns):
    s, zz = s0, z0
    for f in fns:
        s, zz = f(s, zz)
    mx.eval(s, zz); return s, zz
eager = [lambda s, z, l=l: l(s, z, mask, pm) for l in layers]
t = time.perf_counter(); ref = run(eager); print("eager first", round(time.perf_counter() - t, 3))
t = time.perf_counter(); run(eager); print("eager warm ", round(time.perf_counter() - t, 3))
comp = [mx.compile(lambda s, z, l=l: l(s, z, mask, pm)) for l in layers]
t = time.perf_counter(); out = run(comp); print("per-layer compiled first", round(time.perf_counter() - t, 3))
t = time.perf_counter(); out = run(comp); print("per-layer compiled warm ", round(time.perf_counter() - t, 3))
print("max |dz|", float(mx.abs(out[1] - ref[1]).max()))
tmpl = layers[0]
keys = [k for k, _ in tree_flatten(tmpl.parameters())]
@mx.compile
def shared(s, z, *params):
    tmpl.update(tree_unflatten(list(zip(keys, params))))
    return tmpl(s, z, mask, pm)
plist = [[v for _, v in tree_flatten(l.parameters())] for l in layers]
orig = [v for _, v in tree_flatten(tmpl.parameters())]
fns = [lambda s, z, p=p: shared(s, z, *p) for p in plist]
t = time.perf_counter(); out2 = run(fns); print("shared compiled first", round(time.perf_counter() - t, 3))
t = time.perf_counter(); out2 = run(fns); print("shared compiled warm ", round(time.perf_counter() - t, 3))
print("max |dz| shared", float(mx.abs(out2[1] - ref[1]).max()))
