import sys, time, numpy as np
from pathlib import Path
sys.path.insert(0,'tools')
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # this checkout's boltz_mlx first: the editable install maps boltz_mlx to the main checkout
import boltz_mlx, mlx.core as mx, feature_cache
from boltz_mlx import kernels as K
from boltz_mlx.model import PredictArgs, load_model
from boltz_mlx.sampler import record_noise
case = sys.argv[1] if len(sys.argv) > 1 else "prot117"
print("f16", K.f16(), "diff dtype", K.DIFF_DTYPE, "K", K.TRIMULT_K)
m = load_model("conf")
arr,_ = feature_cache.load(case, 0)
feats = {k: mx.array(v) for k,v in arr.items()}
for r in range(3):
    mx.reset_peak_memory()
    t=time.perf_counter(); tm={}
    o = m(feats, PredictArgs(3, 50, 1), record_noise(0,case,"structure",1), tm)
    mx.eval(o["sample_atom_coords"])
    print(f"run{r} {time.perf_counter()-t:.2f}s {({k:round(v,2) for k,v in tm.items()})} plddt={float(np.array(o['complex_plddt']).ravel()[0]):.4f} fallbacks={o.get('fp16_fallbacks')} guard_post={o.get('fp16_guard_post_trunk')} nonfinite={o.get('fp16_nonfinite_coords')} peak={mx.get_peak_memory()/1e9:.1f}GB")
