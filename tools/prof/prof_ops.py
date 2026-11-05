"""Pipelined per-op GPU timing inside a pairformer layer (n calls, one eval). CASE=prot117|prot400."""
import os, time, numpy as np, mlx.core as mx
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # this checkout's boltz_mlx first: the editable install maps boltz_mlx to the main checkout
import boltz_mlx
from boltz_mlx.model import load_model
DATA = os.environ.get("BOLTZ_MLX_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")) + "/" + os.environ.get("CASE", "prot117") + "/"
model = load_model("conf")
z = np.load(DATA + "feats.npz"); feats = {k: mx.array(z[k]) for k in z.files}
s_inputs, s0, z0, rpe = model.embed(feats); mx.eval(s0, z0)
mask = feats["token_pad_mask"].astype(mx.float32); pm = mask[:, :, None] * mask[:, None, :]
L = model.pairformer_module.layers[0]
N = z0.shape[1]
n = int(os.environ.get("REPS", "20"))

def T(name, f):
    r = f(); mx.eval(r)
    t = time.perf_counter()
    outs = [f() for _ in range(n)]; mx.eval(outs)
    print(f"{name:34s} {(time.perf_counter()-t)/n*1e3:8.3f} ms", flush=True)

x = z0
T("full layer", lambda: L(s0, z0, mask, pm))
T("compiled layer", (lambda f=mx.compile(lambda s, z: L(s, z, mask, pm)): (lambda: f(s0, z0)))())
for nm in ["tri_mul_out", "tri_mul_in", "tri_att_start", "tri_att_end"]:
    m = getattr(L, nm)
    T(nm, lambda m=m: m(x, mask=pm))
T("transition_z", lambda: L.transition_z(x))
T("attention(s)", lambda: L.attention(s=L.pre_norm_s(s0), z=z0, mask=mask, k_in=L.pre_norm_s(s0)))
T("transition_s", lambda: L.transition_s(s0))
tm = L.tri_mul_out
T("  LN norm_in", lambda: tm.norm_in(x))
w = mx.concatenate([tm.p_in.weight, tm.g_in.weight, tm.g_out.weight], 0); mx.eval(w)
xn = tm.norm_in(x); mx.eval(xn)
T("  proj 128->640", lambda: xn @ w.T)
h = xn @ w.T; mx.eval(h)
T("  gate+mask", lambda: h[..., :256] * mx.sigmoid(h[..., 256:512]) * pm[..., None])
y = h[..., :256] * mx.sigmoid(h[..., 256:512]); mx.eval(y)
T("  bmm (incl transposes)", lambda: mx.matmul(y[..., :128].transpose(0, 3, 1, 2), y[..., 128:].transpose(0, 3, 2, 1)).transpose(0, 2, 3, 1))
o = mx.matmul(y[..., :128].transpose(0, 3, 1, 2), y[..., 128:].transpose(0, 3, 2, 1)).transpose(0, 2, 3, 1); mx.eval(o)
T("  norm_out+p_out*sig(g)", lambda: tm.p_out(tm.norm_out(o)) * mx.sigmoid(h[..., 512:]))
T("  elementwise x*2 ref", lambda: x * 2)
