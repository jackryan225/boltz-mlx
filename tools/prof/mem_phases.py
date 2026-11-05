"""Peak MLX memory per prediction phase. CASE=prot1000"""
import os, time, numpy as np, mlx.core as mx
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # this checkout's boltz_mlx first: the editable install maps boltz_mlx to the main checkout
import boltz_mlx
from boltz_mlx.model import load_model, PredictArgs
from boltz_mlx.sampler import MLXNoise
DATA = os.environ.get("BOLTZ_MLX_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")) + "/" + os.environ.get("CASE", "prot1000") + "/"
model = load_model("conf")
z = np.load(DATA + "feats.npz"); feats = {k: mx.array(z[k]) for k in z.files}
w = mx.get_active_memory() / 1e9
def phase(name, f):
    mx.reset_peak_memory(); t = time.perf_counter(); r = f(); mx.eval(r)
    print(f"{name:14s} peak {mx.get_peak_memory()/1e9:6.2f} GB (weights {w:.2f})  active after {mx.get_active_memory()/1e9:6.2f}  cache {mx.get_cache_memory()/1e9:5.2f}  {time.perf_counter()-t:6.1f} s", flush=True)
    return r
s_inputs, s, zz, rpe = phase("trunk", lambda: model.trunk(feats, 3))
pd = phase("distogram", lambda: model.distogram_module(zz))
coords = phase("structure", lambda: model.structure(s_inputs, s, zz, rpe, feats, PredictArgs(steps := 3, 20, 1), MLXNoise(0)))
conf = phase("confidence", lambda: [v for k, v in model.confidence_module(s_inputs=s_inputs, s=s, z=zz, x_pred=coords, feats=feats, pred_distogram_logits=pd[:, :, :, 0], multiplicity=1, run_sequentially=True).items() if k != "pair_chains_iptm"])
print("--- structure breakdown")
cond_t = phase("diff_cond", lambda: model.diffusion_conditioning(s, zz, rpe, feats))
q, c, tk, eb, db, tb = cond_t
cond = dict(q=q, c=c, to_keys=tk, atom_enc_bias=eb, atom_dec_bias=db, token_trans_bias=tb, shared_times=True)
sm = model.structure_module
prep = phase("prepare", lambda: sm.score_model.prepare(feats, cond, 1))
cond["prepared"] = prep
x = mx.random.normal((1, feats["atom_pad_mask"].shape[1], 3)) * 10
phase("denoise step", lambda: sm.preconditioned_network_forward(x, mx.array([5.0]), dict(s_inputs=s_inputs, s_trunk=s, feats=feats, diffusion_conditioning=cond, multiplicity=1)))
