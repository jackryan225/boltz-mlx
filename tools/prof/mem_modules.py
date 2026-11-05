"""Peak MLX memory per trunk component. CASE=prot800"""
import os, time, numpy as np, mlx.core as mx
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # this checkout's boltz_mlx first: the editable install maps boltz_mlx to the main checkout
import boltz_mlx
from boltz_mlx.model import load_model
DATA = os.environ.get("BOLTZ_MLX_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")) + "/" + os.environ.get("CASE", "prot800") + "/"
model = load_model("conf")
z = np.load(DATA + "feats.npz"); feats = {k: mx.array(z[k]) for k in z.files}
base = mx.get_active_memory()
def M(name, f):
    mx.reset_peak_memory(); t = time.perf_counter(); r = f(); mx.eval(r)
    print(f"{name:28s} peak {(mx.get_peak_memory()-base)/1e9:6.2f} GB above weights  {time.perf_counter()-t:6.2f} s", flush=True); return r
s_inputs, s_init, z_init, rpe = M("embed", lambda: model.embed(feats))
mask = feats["token_pad_mask"].astype(mx.float32); pm = mask[:, :, None] * mask[:, None, :]
zz = M("template", lambda: model.template_module(z_init, feats, pm))
zz = M("msa", lambda: model.msa_module(z_init, s_inputs, feats))
L = model.pairformer_module.layers[0]
for nm in ["tri_mul_out", "tri_mul_in", "tri_att_start", "tri_att_end"]:
    M(nm, lambda nm=nm: getattr(L, nm)(z_init, mask=pm))
M("transition_z", lambda: L.transition_z(z_init))
M("pairformer layer", lambda: L(s_init, z_init, mask, pm))
def _run8():
    s, zz = s_init, z_init
    for l in model.pairformer_module.layers[:8]:
        s, zz = l(s, zz, mask, pm)
    return s, zz
M("8 layers (lazy)", _run8)
M("distogram+diffcond", lambda: model.diffusion_conditioning(s_init, z_init, rpe, feats))
