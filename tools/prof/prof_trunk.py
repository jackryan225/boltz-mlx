import time, numpy as np, mlx.core as mx
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # this checkout's boltz_mlx first: the editable install maps boltz_mlx to the main checkout
import boltz_mlx
from boltz_mlx.model import load_model
import os; DATA=os.environ.get("BOLTZ_MLX_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")) + "/"+os.environ.get("CASE","prot117")+"/"
model=load_model("conf")
z=np.load(DATA+"feats.npz"); feats={k:mx.array(z[k]) for k in z.files}
s_inputs,s_init,z_init,rpe=model.embed(feats); mx.eval(s_init,z_init)
mask=feats["token_pad_mask"].astype(mx.float32); pm=mask[:,:,None]*mask[:,None,:]
L=model.pairformer_module.layers[0]
def T(name,f,n=10):
    r=f(); mx.eval(r); t=time.perf_counter()
    for i in range(n): r=f(); mx.eval(r)
    print(f"{name:28s} {(time.perf_counter()-t)/n*1e3:7.2f} ms"); return r
s=s_init; zz=z_init
T("layer eager",lambda: L(s,zz,mask,pm))
Lc=mx.compile(lambda s,z: L(s,z,mask,pm))
T("layer compiled",lambda: Lc(s,zz))
T("tri_mul_out",lambda: L.tri_mul_out(zz,mask=pm))
T("tri_mul_in",lambda: L.tri_mul_in(zz,mask=pm))
T("tri_att_start",lambda: L.tri_att_start(zz,mask=pm))
T("tri_att_end",lambda: L.tri_att_end(zz,mask=pm))
T("transition_z",lambda: L.transition_z(zz))
T("msa_module",lambda: model.msa_module(zz,s_inputs,feats))
T("template_module",lambda: model.template_module(zz,feats,pm))
