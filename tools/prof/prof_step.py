import os
import time, sys, numpy as np, mlx.core as mx
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # this checkout's boltz_mlx first: the editable install maps boltz_mlx to the main checkout
import boltz_mlx
from boltz_mlx import kernels as K
from boltz_mlx.model import load_model
DATA=os.environ.get("BOLTZ_MLX_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")) + "/prot117/"
model=load_model("conf")
z=np.load(DATA+"feats.npz"); feats={k:mx.array(z[k]) for k in z.files}
s_inputs,s,zz,rpe=model.trunk(feats,3)
q,c,to_keys,eb,db,tb=model.diffusion_conditioning(s,zz,rpe,feats); cond=dict(q=q,c=c,to_keys=to_keys,atom_enc_bias=eb,atom_dec_bias=db,token_trans_bias=tb); mx.eval(q,c,eb,db,tb)
sm=model.structure_module
x=mx.random.normal((1,928,3))*10
def den(x,sig): return sm.preconditioned_network_forward(x,sig,dict(s_inputs=s_inputs,s_trunk=s,feats=feats,diffusion_conditioning=cond,multiplicity=1))
sig=mx.array([5.0])
for dt in ("fp32","fp16"):
    K.set_diff_dtype(dt)
    for name,f in [("eager",den),("compiled",mx.compile(den))]:
        y=f(x,sig); mx.eval(y)
        t=time.perf_counter()
        for i in range(20): y=f(x,sig); mx.eval(y)
        print(dt,name,"per step ms",round((time.perf_counter()-t)/20*1e3,2))
sc=sm.score_model
def T(name,f,n=20):
    r=f(); mx.eval(r); t=time.perf_counter()
    for i in range(n): r=f(); mx.eval(r)
    print(f"{name:28s} {(time.perf_counter()-t)/n*1e3:7.2f} ms"); return r
K.set_diff_dtype("fp16")
times=mx.array([0.1])
scond=sc.single_conditioner(times,s,s_inputs)[0]
enc=T("atom_encoder",lambda: sc.atom_attention_encoder(feats=feats,q=q,c=c,atom_enc_bias=eb,to_keys=to_keys,r=x,multiplicity=1))
a=enc[0]; mask=feats["token_pad_mask"].astype(mx.float32)
tt=sc.token_transformer
a2=T("token_transformer",lambda: tt(a,mask=mask,s=scond,bias=tb,multiplicity=1))
T("token_transformer compiled",mx.compile(lambda a,sc_: tt(a,mask=mask,s=sc_,bias=tb,multiplicity=1)).__call__ if False else (lambda f=mx.compile(lambda a,sc_: tt(a,mask=mask,s=sc_,bias=tb,multiplicity=1)): f(a,scond)))
T("precompute_cond",lambda: tt.precompute_cond(scond))
cnd=tt.precompute_cond(scond); mx.eval(cnd)
B,N,M,D=tb.shape; bias=tb.reshape(B,N,M,24,D//24)
L0=tt.layers[0]
T("1 layer fast",lambda: L0(a,scond,bias[:,:,:,0],mask,None,1,cond=[x_[...,0,:] for x_ in cnd]))
T("dec",lambda: sc.atom_attention_decoder(a=a2,q=enc[1],c=enc[2],atom_dec_bias=db,feats=feats,multiplicity=1,to_keys=enc[3]))
T("to_keys(q) only",lambda: to_keys(enc[1]))
