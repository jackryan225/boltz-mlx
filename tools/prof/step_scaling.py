"""Diffusion step time vs batched samples (multiplicity), fp32/fp16, compiled. CASE=prot117"""
import os, time, numpy as np, mlx.core as mx
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # this checkout's boltz_mlx first: the editable install maps boltz_mlx to the main checkout
import boltz_mlx
from boltz_mlx import kernels as K
from boltz_mlx.model import load_model
DATA = os.environ.get("BOLTZ_MLX_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")) + "/" + os.environ.get("CASE", "prot117") + "/"
model = load_model("conf")
z = np.load(DATA + "feats.npz"); feats = {k: mx.array(z[k]) for k in z.files}
s_inputs, s, zz, rpe = model.trunk(feats, 0)
q, c, tk, eb, db, tb = model.diffusion_conditioning(s, zz, rpe, feats)
sm = model.structure_module
for dt in ("fp32", "fp16"):
    K.set_diff_dtype(dt)
    for mult in (1, 2, 5):
        cond = dict(q=q, c=c, to_keys=tk, atom_enc_bias=eb, atom_dec_bias=db, token_trans_bias=tb, shared_times=True)
        cond["prepared"] = sm.score_model.prepare(feats, cond, mult); mx.eval(cond["prepared"])
        f = mx.compile(lambda x, sig: sm.preconditioned_network_forward(x, sig, dict(s_inputs=s_inputs, s_trunk=s, feats=feats, diffusion_conditioning=cond, multiplicity=mult)))
        x = mx.random.normal((mult, feats["atom_pad_mask"].shape[1], 3)) * 10; sig = mx.full((mult,), 5.0)
        y = f(x, sig); mx.eval(y)
        t = time.perf_counter()
        for _ in range(20):
            x = f(x, sig) * 0.999; mx.eval(x)
        dtm = (time.perf_counter() - t) / 20 * 1e3
        print(f"{dt} mult={mult}: {dtm:6.2f} ms/step  ({dtm/mult:5.2f} ms/sample-step)", flush=True)
