"""A/B of a full compiled denoise step (preconditioned_network_forward) with the token transformer implementation
mlx | fused (BOLTZ_MLX_DIFFTX_IMPL), fp32 and fp16 diffusion matmuls, alternating rounds, min over rounds.
CASE=prot117. GPU only, one process at a time.

  python tools/prof/ab_difftx_step.py                       # DT=fp32,fp16 MULT=1,5 IMPLS=mlx,fused
  DT=fp32 MULT=1 ROUNDS=5 python tools/prof/ab_difftx_step.py
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # this checkout, not an editable install elsewhere

import mlx.core as mx
import numpy as np

import boltz_mlx
from boltz_mlx import kernels as K
from boltz_mlx import kernels_difftx as KD
from boltz_mlx.model import load_model

DATA = os.environ.get("BOLTZ_MLX_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")) + "/" + os.environ.get("CASE", "prot117") + "/"
DTS = os.environ.get("DT", "fp32,fp16").split(",")
MULTS = [int(m) for m in os.environ.get("MULT", "1,5").split(",")]
IMPLS = os.environ.get("IMPLS", "mlx,fused").split(",")
ROUNDS = int(os.environ.get("ROUNDS", "3"))
STEPS = int(os.environ.get("STEPS", "20"))

print("boltz_mlx:", boltz_mlx.__file__)
mx.set_default_device(mx.gpu)
model = load_model("conf")
z = np.load(DATA + "feats.npz")
feats = {k: mx.array(z[k]) for k in z.files}
s_inputs, s, zz, rpe = model.trunk(feats, 0)
q, c, tk, eb, db, tb = model.diffusion_conditioning(s, zz, rpe, feats)
mx.eval(q, c, eb, db, tb)
sm = model.structure_module
n_atoms = feats["atom_pad_mask"].shape[1]


def build(mult, impl):
    KD.DIFFTX_IMPL = impl
    cond = dict(q=q, c=c, to_keys=tk, atom_enc_bias=eb, atom_dec_bias=db, token_trans_bias=tb, shared_times=True)
    cond["prepared"] = sm.score_model.prepare(feats, cond, mult)
    mx.eval(cond["prepared"])
    f = mx.compile(lambda x, sig: sm.preconditioned_network_forward(
        x, sig, dict(s_inputs=s_inputs, s_trunk=s, feats=feats, diffusion_conditioning=cond, multiplicity=mult)))
    x0 = mx.random.normal((mult, n_atoms, 3), key=mx.random.key(0)) * 10
    sig = mx.full((mult,), 5.0)
    y = f(x0, sig)
    mx.eval(y)
    return f, x0, sig, y


def run(f, x0, sig):
    x = x0
    t = time.perf_counter()
    for _ in range(STEPS):
        x = f(x, sig) * 0.999
        mx.eval(x)
    return (time.perf_counter() - t) / STEPS * 1e3


for dt in DTS:
    K.set_diff_dtype(dt)
    for L in sm.score_model.token_transformer.layers:
        L._fw = None
    sm.score_model.token_transformer._cw = None
    for mult in MULTS:
        fns = {impl: build(mult, impl) for impl in IMPLS}
        res = {i: [] for i in IMPLS}
        for _ in range(ROUNDS):
            for impl in IMPLS:
                f, x0, sig, _ = fns[impl]
                res[impl].append(run(f, x0, sig))
        ref = fns[IMPLS[0]][3]
        for impl in IMPLS:
            y = fns[impl][3]
            err = float(mx.abs(y - ref).max() / mx.abs(ref).max())
            t = min(res[impl])
            print(f"{dt} mult={mult} {impl:6s} {t:7.2f} ms/step ({t / mult:6.2f} ms/sample-step)  "
                  f"{min(res[IMPLS[0]]) / t:.3f}x vs {IMPLS[0]}  rounds {['%.2f' % r for r in res[impl]]}  "
                  f"rel diff of output vs {IMPLS[0]} {err:.1e}", flush=True)
        del fns
KD.DIFFTX_IMPL = "mlx"
