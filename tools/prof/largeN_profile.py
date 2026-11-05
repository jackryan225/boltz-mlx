"""Large-N trunk profile: where does the time (and the memory) go for 1600 / 2400 / 3200 tokens?  (GPU; no model code changed)

One recycle of the Boltz-2 trunk on the real featurized case (data/<case>/feats.npz , read-only):
embed -> recycle input -> the first --msa-layers MSA-module layers -> the first --layers pairformer layers, run through the
PRODUCTION layer code (PairformerLayer / MSALayer __call__, so the chunked unfused path above CHUNK_N and the fused
pair block below it are exactly what a prediction runs). Every module call is synchronised with mx.eval and timed:

  trimul_out / trimul_in   triangle multiplication (both chunk over CHUNK_CH channel blocks above CHUNK_N)
  triatt_start / triatt_end  triangle attention (chunk over CHUNK_ROWS rows above CHUNK_N)
  transition_z / transition_s / transition_msa  (pair transition chunks over CHUNK_ROWS rows above CHUNK_N)
  attn_s                   s-track attention with pair bias (pair bias = proj_z(LN(z)): an N^2 x 128 pass)
  opm, pwa                 MSA-module OuterProductMean / PairWeightedAveraging (S = 1 for the msa: empty cases)
  radd, add_ln, ln16, f16_* fused-block pieces   residual adds / fused add+LN / fp16 LayerNorm outside a module
Per module call it reports: wall s, process user / system CPU s (getrusage; system time is where Metal buffer
allocation, page zero-fill and wiring show up), the number of mx.eval syncs inside the call (= chunk launches in the
chunk loops), MLX peak memory during the call (reset before each call), active and cache memory after it, an analytic
FLOP count (-> TFLOP/s) and optionally max|x| / non-finite counts of the outputs. The layer's own wall time is also
taken so the unattributed remainder (graph build, s-track glue, syncs) is visible, plus --plain layers run exactly as
production does (one mx.eval at the end of the layer, no per-module syncs) to price the instrumentation.

Settings are the production knobs, read from the environment as the package reads them (the flags below only set
them before the import):
  BOLTZ_MLX_CHUNK_N (512)   BOLTZ_MLX_CHUNK_ROWS (128)   BOLTZ_MLX_CHUNK_CH (32)      chunking (kernels.py)
  BOLTZ_MLX_CACHE_GB (4)    mx.set_cache_limit as the CLI sets it (the bench-r2 sweep used 4)
  BOLTZ_MLX_TRIMULT_K (4)   tri-mult fp16 pre-scaling exponent (overflow headroom, kernels.py)
  BOLTZ_MLX_PRECISION (f16 here by default)
  --mem-limit-gb (100)      mx.set_memory_limit as the sweep set it

  python tools/prof/largeN_profile.py prot3200 --layers 2                  # ~2-4 min at 3200 (ESTIMATE); default chunking
  BOLTZ_MLX_CACHE_GB=32 python tools/prof/largeN_profile.py prot3200       # allocator cache raised
  BOLTZ_MLX_CHUNK_N=100000 python tools/prof/largeN_profile.py prot3200    # no chunking -> fused pair block (unvalidated above 512)
  python tools/prof/largeN_profile.py prot3200 --fp32-fallback --layers 2  # cost of the guard's fp32 recompute (mixed weights, PRECISION=fp32 as model.trunk sets it)
  python tools/prof/largeN_profile.py prot3200 --layers 64 --msa-layers 4 --passes 2 --amax   # overflow localiser: where does max|x| / the first non-finite appear

--amax and --sub add syncs and reductions inside modules (hook on kernels.record): use them to locate values, not to time.
Output: a table, one line `LARGEN_PROFILE_JSON {...}` on stdout, and the same row appended to --out (default
logs/largeN_profile.jsonl). Exit 0 = ran (non-finite values are reported, not an error), 2 = exception (partial table printed),
4 = imported the wrong boltz_mlx checkout.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("BOLTZ_MLX_DATA", Path(__file__).resolve().parents[2] / "data")).expanduser()


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("case", nargs="?", help="featurized case under data/ (prot1600, prot2400, prot3200, ...)")
    ap.add_argument("--tokens", type=int, help="shorthand for the case prot<tokens>")
    ap.add_argument("--layers", type=int, default=2, help="pairformer layers per pass (default 2; 64 = the full stack)")
    ap.add_argument("--msa-layers", type=int, default=1, help="MSA-module layers per pass (default 1; 4 = the full module)")
    ap.add_argument("--passes", type=int, default=1, help="recycle passes (pass p+1 takes pass p's s, z as recycle input)")
    ap.add_argument("--warm", type=int, default=1, help="warm-up MSA + pairformer layers on the first input, untimed (default 1; 0 = cold)")
    ap.add_argument("--plain", type=int, default=1, help="production-style pairformer layers (no per-module syncs) after the instrumented ones, from pass 0's input")
    ap.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    ap.add_argument("--fp32-fallback", action="store_true",
                    help="load in f16 mode (mixed weights), then set PRECISION=fp32 exactly as model.trunk does for a guard fallback")
    ap.add_argument("--mem-limit-gb", type=float, default=100.0)
    ap.add_argument("--cache-limit-gb", type=float, default=None, help="default: $BOLTZ_MLX_CACHE_GB, else 4")
    ap.add_argument("--chunk-n", type=int, default=None, help="sets BOLTZ_MLX_CHUNK_N")
    ap.add_argument("--chunk-rows", type=int, default=None, help="sets BOLTZ_MLX_CHUNK_ROWS")
    ap.add_argument("--chunk-ch", type=int, default=None, help="sets BOLTZ_MLX_CHUNK_CH")
    ap.add_argument("--trimult-k", type=int, default=None, help="sets BOLTZ_MLX_TRIMULT_K")
    ap.add_argument("--amax", action="store_true", help="max|x| and non-finite counts of module outputs and recorded intermediates")
    ap.add_argument("--sub", action="store_true", help="sub-op timing at kernels.record() points (syncs inside modules)")
    ap.add_argument("--max-s", type=float, default=540.0, help="wall budget for the instrumented layers; stops before the next layer (default 540)")
    ap.add_argument("--out", default="logs/largeN_profile.jsonl", help="JSONL appended to ('' = none)")
    ap.add_argument("--note", default="")
    return ap.parse_args(argv)


A = parse_args()
for flag, var in ((A.chunk_n, "BOLTZ_MLX_CHUNK_N"), (A.chunk_rows, "BOLTZ_MLX_CHUNK_ROWS"),
                  (A.chunk_ch, "BOLTZ_MLX_CHUNK_CH"), (A.trimult_k, "BOLTZ_MLX_TRIMULT_K")):
    if flag is not None:
        os.environ[var] = str(flag)
os.environ.setdefault("BOLTZ_MLX_PRECISION", "f16")  # read at import (kernels.PRECISION)
if A.fp32_fallback:
    os.environ["BOLTZ_MLX_PRECISION"] = "f16"
sys.path.insert(0, str(ROOT))  # this checkout's boltz_mlx first: the editable install maps boltz_mlx to the main checkout

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

import boltz_mlx  # noqa: E402,F401
from boltz_mlx import kernels as K  # noqa: E402

if not str(Path(boltz_mlx.__file__).resolve()).startswith(str(ROOT)):
    print(f"largeN_profile: imported boltz_mlx from {boltz_mlx.__file__}, not from {ROOT}", file=sys.stderr)
    sys.exit(4)
# after the import, so tests/test_tools_syspath.py (runs every tool with no arguments) sees which boltz_mlx it loads
if A.case is None and A.tokens is None:
    sys.exit("largeN_profile: give a case (prot3200) or --tokens N")
CASE = A.case or f"prot{A.tokens}"

GB = 1e9
_real_eval = mx.eval


def _sh(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=ROOT).stdout.strip()


def _gb(x: float) -> float:
    return round(x / GB, 3)


def _ru():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime, r.ru_stime


def _arrays(o):
    """Every mx.array inside a module output (array, tuple/list/dict, kernels.Scaled)."""
    if o is None:
        return []
    if isinstance(o, mx.array):
        return [o]
    if isinstance(o, K.Scaled):
        return [o.y]
    if isinstance(o, (list, tuple)):
        return [x for e in o for x in _arrays(e)]
    if isinstance(o, dict):
        return [x for e in o.values() for x in _arrays(e)]
    return []


def _rng(x: mx.array):
    """(max|x| as float, non-finite count as int) of one array; a second sync, never inside a timed region."""
    a = mx.max(mx.abs(x)).astype(mx.float32)
    n = mx.sum(~mx.isfinite(x)).astype(mx.int32)
    _real_eval(a, n)
    return float(a.item()), int(n.item())


# ---------------------------------------------------------------------------------------------------------------
# analytic FLOPs (2 x multiply-adds) of one module call; D = pair channels, tri-att 4 heads x 32
# ---------------------------------------------------------------------------------------------------------------
def _flops(role: str, a) -> float | None:
    try:
        if role in ("trimul_out", "trimul_in"):
            n, m, d = a.shape[1], a.shape[2], a.shape[3]
            return 2.0 * n * m * d * 4 * d + 2.0 * n * n * m * d + 2.0 * n * m * d * d * 2  # proj + contraction + g_out/p_out
        if role in ("triatt_start", "triatt_end"):
            n, m, d = a.shape[1], a.shape[2], a.shape[3]
            return 2.0 * n * m * d * (4 * 128 + 4) + 4.0 * 4 * 32 * n * n * m + 2.0 * n * m * d * d  # proj + core + linear_o
        if role.startswith("transition"):
            d = a.shape[-1]
            rows = a.size // d
            return 2.0 * rows * (d * 2 * (4 * d) + 4 * d * d)
        if role == "attn_s":
            return None
    except Exception:
        return None
    return None


class Prof:
    def __init__(self, sub: bool, amax: bool):
        self.on = False
        self.sub, self.amax = sub, amax
        self.depth = 0
        self.n_evals = 0
        self.ctx = {"pass": 0, "phase": "-", "layer": -1}
        self.rows: list[dict] = []
        self.cur: dict | None = None
        self.peak_gb = 0.0

    # -- wrappers -------------------------------------------------------------------------------------------
    def wrap(self, fn, role_of, x_of):
        prof = self

        def wrapper(*args, **kw):
            if not prof.on or prof.depth > 0:
                return fn(*args, **kw)
            prof.depth += 1
            try:
                role = role_of(args, kw)
                x = x_of(args, kw)
                mx.reset_peak_memory()
                act0 = mx.get_active_memory()
                row = {**prof.ctx, "role": role, "sub": {}, "amax": {}, "nonfinite": {}}
                prof.cur = row
                e0, (u0, s0) = prof.n_evals, _ru()
                t0 = time.perf_counter()
                row["_last"] = t0
                out = fn(*args, **kw)
                arrs = _arrays(out)
                if arrs:
                    _real_eval(*arrs)
                t1 = time.perf_counter()
                u1, s1 = _ru()
                row.update(s=t1 - t0, user_s=u1 - u0, sys_s=s1 - s0, evals=prof.n_evals - e0,
                           peak_gb=_gb(mx.get_peak_memory()), trans_gb=_gb(mx.get_peak_memory() - act0),
                           active_gb=_gb(mx.get_active_memory()), cache_gb=_gb(mx.get_cache_memory()),
                           flops=_flops(role, x) if x is not None else None)
                prof.peak_gb = max(prof.peak_gb, row["peak_gb"])
                if prof.amax and arrs:
                    am, nf = _rng(arrs[0])
                    row["amax"]["out"], row["nonfinite"]["out"] = am, nf
                row.pop("_last", None)
                prof.rows.append(row)
                prof.cur = None
                return out
            finally:
                prof.depth -= 1

        wrapper.__wrapped__ = fn
        return wrapper

    def hook(self, phase, name, x):
        """kernels.ACT_HOOK: called at every record(name, x) inside a module call."""
        row = self.cur
        if row is None or not isinstance(x, mx.array):
            return
        if self.sub:
            _real_eval(x)
            now = time.perf_counter()
            row["sub"][name] = row["sub"].get(name, 0.0) + (now - row["_last"])
        if self.amax:
            am, nf = _rng(x)
            row["amax"][name] = max(row["amax"].get(name, 0.0), am)
            row["nonfinite"][name] = row["nonfinite"].get(name, 0) + nf
        if self.sub:
            row["_last"] = time.perf_counter()


def install(prof: Prof):
    """Class-level wrappers around the production modules (so every layer instance is covered)."""
    from boltz_mlx.layers import triangular_mult as TM
    from boltz_mlx.layers.attentionv2 import AttentionPairBias
    from boltz_mlx.layers.norm import LayerNorm as FastLN
    from boltz_mlx.layers.outer_product_mean import OuterProductMean
    from boltz_mlx.layers.pair_averaging import PairWeightedAveraging
    from boltz_mlx.layers.transition import Transition
    from boltz_mlx.layers.triangular_attention.attention import TriangleAttention

    arg0 = lambda a, k: a[1] if len(a) > 1 else next(iter(k.values()), None)  # noqa: E731  (first array argument after self)

    def patch(cls, name, role_of, x_of=arg0):
        setattr(cls, name, prof.wrap(getattr(cls, name), role_of, x_of))

    tm_role = lambda a, k: "trimul_out" if a[0].outgoing else "trimul_in"  # noqa: E731
    ta_role = lambda a, k: "triatt_start" if a[0].starting else "triatt_end"  # noqa: E731

    def tr_role(a, k):
        d = a[0].norm.weight.shape[0]
        return "transition_z" if d == 128 else ("transition_s" if d >= 384 else "transition_msa")

    patch(TM._TriangleMultiplication, "__call__", tm_role)
    patch(TM._TriangleMultiplication, "f16_addln", lambda a, k: tm_role(a, k) + ".addln")
    patch(TriangleAttention, "__call__", ta_role)
    patch(TriangleAttention, "f16_addln", lambda a, k: ta_role(a, k) + ".addln")
    patch(Transition, "__call__", tr_role)
    patch(Transition, "f16_resid", lambda a, k: tr_role(a, k) + ".resid")
    patch(Transition, "f16_resid_lnx", lambda a, k: tr_role(a, k) + ".resid_lnx")
    patch(AttentionPairBias, "__call__", lambda a, k: "attn_s", lambda a, k: k.get("z"))
    patch(OuterProductMean, "__call__", lambda a, k: "opm")
    patch(PairWeightedAveraging, "__call__", lambda a, k: "pwa")
    patch(FastLN, "f16", lambda a, k: "ln16")
    K.radd = prof.wrap(K.radd, lambda a, k: "radd", lambda a, k: a[0])
    K.add_ln = prof.wrap(K.add_ln, lambda a, k: "add_ln", lambda a, k: a[0])
    if prof.sub or prof.amax:
        K.ACT_HOOK = prof.hook

    def counting_eval(*a, **k):
        prof.n_evals += 1
        return _real_eval(*a, **k)

    mx.eval = counting_eval  # the modules call mx.eval(...) through the module attribute (chunk loops): count them


# ---------------------------------------------------------------------------------------------------------------
def swap_used() -> str:
    """'used = 1759.88M' of `sysctl vm.swapusage`."""
    out = _sh("sysctl -n vm.swapusage")
    return out.split("used =")[1].split("free")[0].strip() if "used =" in out else out


def main():
    t_start = time.perf_counter()
    if A.device == "gpu":
        mx.set_default_device(mx.gpu)
    else:
        mx.set_default_device(mx.cpu)
    prev_mem = mx.set_memory_limit(int(A.mem_limit_gb * GB))
    cache_gb = A.cache_limit_gb if A.cache_limit_gb is not None else float(os.environ.get("BOLTZ_MLX_CACHE_GB", "4"))
    prev_cache = mx.set_cache_limit(int(cache_gb * GB))
    head = {
        "tool": "tools/prof/largeN_profile.py",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git": _sh("git rev-parse --short HEAD"), "pkg_tree": _sh("git rev-parse --short HEAD:boltz_mlx"),
        "tool_blob": _sh("git hash-object tools/prof/largeN_profile.py")[:12],
        "case": CASE, "layers": A.layers, "msa_layers": A.msa_layers, "passes": A.passes, "warm": A.warm, "plain_layers": A.plain,
        "device": A.device, "fp32_fallback": A.fp32_fallback, "note": A.note,
        "settings": {"chunk_n": K.CHUNK_N, "chunk_rows": K.CHUNK_ROWS, "chunk_ch": K.CHUNK_CH, "cache_limit_gb": cache_gb,
                     "mem_limit_gb": A.mem_limit_gb, "memory_limit_gb_reported": _gb(mx.get_memory_limit()),
                     "trimult_k": K.TRIMULT_K, "precision": K.PRECISION, "prev_memory_limit_gb": _gb(prev_mem),
                     "prev_cache_limit_gb": _gb(prev_cache)},
        "env": {k: v for k, v in os.environ.items() if k.startswith("BOLTZ_MLX") or k.startswith("MLX_")},
        "mlx": mx.__version__, "swap_start": swap_used(),
    }
    try:
        head["device_record"] = K.device_record() if A.device == "gpu" and K.f16() else None
    except Exception as e:  # noqa: BLE001
        head["device_record"] = f"unavailable: {e}"
    print("LARGEN_HEAD", json.dumps(head), flush=True)

    from boltz_mlx.layers.pairformer import _carry
    from boltz_mlx.model import load_model
    from boltz_mlx.modules.trunkv2 import NUM_TOKENS

    prof = Prof(A.sub, A.amax)
    install(prof)

    mem = {}
    t = time.perf_counter()
    model = load_model("conf")
    mem["load_s"] = round(time.perf_counter() - t, 2)
    mem["after_load_gb"] = _gb(mx.get_active_memory())
    if A.fp32_fallback:
        K.PRECISION = "fp32"  # as model.trunk does for a guard fallback: mixed (fp16-rounded) weights, fp32 activations / GEMMs
    path = DATA / CASE / "feats.npz"
    t = time.perf_counter()
    z_np = np.load(path)
    feats = {k: mx.array(z_np[k]) for k in z_np.files}
    _real_eval(*feats.values())
    del z_np
    mem["feats_s"] = round(time.perf_counter() - t, 2)
    mem["after_feats_gb"] = _gb(mx.get_active_memory())
    N = int(feats["token_pad_mask"].shape[1])
    head_info = {"tokens": N, "atoms": int(feats["atom_pad_mask"].shape[1]),
                 "msa_rows": int(feats["msa"].shape[1]), "templates_any": bool(mx.any(feats["template_mask"] != 0).item()),
                 "pairformer_layers": len(model.pairformer_module.layers), "msa_layers_total": len(model.msa_module.layers),
                 "chunking": K.chunking(N),
                 # kernels.use_pair_block_f16 without allocating an N^2 array to ask it (B = 1, square, fp32 z hold here)
                 "fused_pair_block": bool(K.FUSE_ADD_LN and K.f16() and not K.chunking(N))}
    print("LARGEN_CASE", json.dumps({**head_info, **mem}), flush=True)

    mx.reset_peak_memory()
    t = time.perf_counter()
    s_inputs, s_init, z_init, rpe = model.embed(feats)
    _real_eval(s_inputs, s_init, z_init, rpe)
    mem["embed_s"] = round(time.perf_counter() - t, 2)
    mem["after_embed_gb"], mem["embed_peak_gb"] = _gb(mx.get_active_memory()), _gb(mx.get_peak_memory())
    mask = feats["token_pad_mask"].astype(mx.float32)
    pair_mask = mask[:, :, None] * mask[:, None, :]
    token_mask = pair_mask  # MSAModule: token_mask[:, :, None] * token_mask[:, None, :]
    msa_mask = feats["msa_mask"]

    def recycle_in(s_prev, z_prev):  # model._recycle, first two lines
        s = s_init + model.s_recycle(model.s_norm(s_prev))
        z = z_init + model.z_recycle(model.z_norm(z_prev))
        return s, z

    def msa_start():  # MSAModule.__call__ up to the layer loop
        mm = model.msa_module
        msa = (feats["msa"][..., None] == mx.arange(NUM_TOKENS)).astype(mx.float32)
        hd = feats["has_deletion"].astype(mx.float32)[..., None]
        dv = feats["deletion_value"].astype(mx.float32)[..., None]
        ip = feats["msa_paired"].astype(mx.float32)[..., None]
        m = mx.concatenate([msa, hd, dv, ip] if mm.use_paired_feature else [msa, hd, dv], axis=-1)
        m = mm.msa_proj(m)
        return m + mx.expand_dims(mm.s_proj(s_inputs), 1)

    def sz_range(s, z):
        zm, zn = _rng(z)
        sm, sn = _rng(s)
        return {"z_amax": zm, "z_nonfinite": zn, "s_amax": sm, "s_nonfinite": sn}

    def emit_layer(lr):
        print(f"LARGEN_LAYER p{lr['pass']} {lr['phase']}{lr['layer']:>3} {lr['s']:8.3f}s modules {lr['modules_s']:8.3f}s sys {lr['sys_s']:7.3f}s "
              f"evals {lr['evals']:3d} peak {lr['peak_gb']:6.2f}GB cache {lr['cache_gb']:6.2f}GB z|max| {lr['z_amax']:.4g} "
              f"nonfinite z {lr['z_nonfinite']} s {lr.get('s_nonfinite', '-')}", flush=True)

    layers_pf = model.pairformer_module.layers
    layers_msa = model.msa_module.layers
    n_pf, n_msa = min(A.layers, len(layers_pf)), min(A.msa_layers, len(layers_msa))

    def run_pf_layer(i, s, z, xn):
        nxt = layers_pf[i + 1].tri_mul_out.norm_in if i + 1 < len(layers_pf) else None
        s, z = layers_pf[i](s, z, mask, pair_mask, xn_in=xn, next_ln=nxt)
        return s, z, _carry(layers_pf[i], z)

    # ---- warm-up (untimed): compiles the Metal kernels of this N and touches the allocator ----------------------
    s0, z0 = recycle_in(mx.zeros_like(s_init), mx.zeros_like(z_init))
    _real_eval(s0, z0)
    for _ in range(A.warm):
        if n_msa:
            t = time.perf_counter()
            m = msa_start()
            zw, m = layers_msa[0](z0, m, token_mask, msa_mask)
            _real_eval(zw, m)
            mem["warm_msa_layer_s"] = round(time.perf_counter() - t, 2)
            del zw, m
        if n_pf:
            t = time.perf_counter()
            sw, zw2, _ = run_pf_layer(0, s0, z0, None)
            _real_eval(sw, zw2)
            mem["warm_pf_layer_s"] = round(time.perf_counter() - t, 2)
            del zw2, sw
    mx.reset_peak_memory()

    # ---- instrumented passes ----------------------------------------------------------------------------------------
    layer_rows: list[dict] = []
    prof.on = True
    s_prev, z_prev = mx.zeros_like(s_init), mx.zeros_like(z_init)
    first_in = None
    stop = None
    t_loop = time.perf_counter()
    err = None
    try:
        for p in range(A.passes):
            s, z = recycle_in(s_prev, z_prev)
            _real_eval(s, z)
            prof.ctx.update({"pass": p, "phase": "msa", "layer": -1})
            m = msa_start()
            zm = z
            for i in range(n_msa):
                if time.perf_counter() - t_loop > A.max_s:
                    stop = f"max-s before pass {p} msa layer {i}"
                    break
                prof.ctx.update({"phase": "msa", "layer": i})
                r0, e0 = len(prof.rows), prof.n_evals
                u0, s0_ = _ru()
                mx.reset_peak_memory()
                t = time.perf_counter()
                zm, m = layers_msa[i](zm, m, token_mask, msa_mask)
                _real_eval(zm, m)
                dt = time.perf_counter() - t
                u1, s1_ = _ru()
                lr = {"pass": p, "phase": "msa", "layer": i, "s": round(dt, 4), "user_s": round(u1 - u0, 3),
                      "sys_s": round(s1_ - s0_, 3), "evals": prof.n_evals - e0,
                      "modules_s": round(sum(r["s"] for r in prof.rows[r0:]), 4),
                      "peak_gb": _gb(mx.get_peak_memory()), "active_gb": _gb(mx.get_active_memory()),
                      "cache_gb": _gb(mx.get_cache_memory())}
                zr = _rng(zm)
                lr.update({"z_amax": zr[0], "z_nonfinite": zr[1]})
                lr["peak_gb"] = max([lr["peak_gb"]] + [r["peak_gb"] for r in prof.rows[r0:]])
                layer_rows.append(lr)
                emit_layer(lr)
            if stop:
                break
            if n_msa:
                z = z + zm  # model._recycle: z = z + msa_module(z, ...)
                _real_eval(z)
            if p == 0:
                first_in = (s, z)
            xn = None
            for i in range(n_pf):
                if time.perf_counter() - t_loop > A.max_s:
                    stop = f"max-s before pass {p} pairformer layer {i}"
                    break
                prof.ctx.update({"phase": "pf", "layer": i})
                r0, e0 = len(prof.rows), prof.n_evals
                u0, s0_ = _ru()
                mx.reset_peak_memory()
                t = time.perf_counter()
                s, z, xn = run_pf_layer(i, s, z, xn)
                _real_eval(s, z)
                dt = time.perf_counter() - t
                u1, s1_ = _ru()
                lr = {"pass": p, "phase": "pf", "layer": i, "s": round(dt, 4), "user_s": round(u1 - u0, 3),
                      "sys_s": round(s1_ - s0_, 3), "evals": prof.n_evals - e0,
                      "modules_s": round(sum(r["s"] for r in prof.rows[r0:]), 4),
                      "peak_gb": _gb(mx.get_peak_memory()), "active_gb": _gb(mx.get_active_memory()),
                      "cache_gb": _gb(mx.get_cache_memory())}
                lr.update(sz_range(s, z))
                lr["peak_gb"] = max([lr["peak_gb"]] + [r["peak_gb"] for r in prof.rows[r0:]])  # modules reset the peak
                layer_rows.append(lr)
                emit_layer(lr)
            if stop:
                break
            s_prev, z_prev = s, z
    except Exception as e:  # noqa: BLE001  (OOM / kernel error: report what we have)
        err = f"{type(e).__name__}: {e}"
        print("LARGEN_ERROR", err, file=sys.stderr, flush=True)
    prof.on = False
    loop_s = time.perf_counter() - t_loop

    # ---- plain (production-style) layers: one sync at the end of the layer ---------------------------------------
    plain = []
    if err is None and first_in is not None and A.plain > 0 and n_pf > 0:
        mx.eval = _real_eval  # no counting needed; module-level syncs inside chunk loops still happen
        sp, zp = first_in
        for _ in range(A.plain):
            mx.reset_peak_memory()
            u0, s0_ = _ru()
            t = time.perf_counter()
            s2, z2, _ = run_pf_layer(0, sp, zp, None)
            _real_eval(*K.guard_first(s2, z2))
            dt = time.perf_counter() - t
            u1, s1_ = _ru()
            plain.append({"s": round(dt, 4), "user_s": round(u1 - u0, 3), "sys_s": round(s1_ - s0_, 3),
                          "peak_gb": _gb(mx.get_peak_memory()), "cache_gb": _gb(mx.get_cache_memory())})
            del s2, z2

    # ---- report -----------------------------------------------------------------------------------------------------
    agg: dict = {}
    for r in prof.rows:
        key = (r["phase"], r["role"])
        a = agg.setdefault(key, {"n": 0, "s": 0.0, "user_s": 0.0, "sys_s": 0.0, "evals": 0, "peak_gb": 0.0, "trans_gb": 0.0, "flops": 0.0,
                                 "has_flops": True, "cache_gb": 0.0, "amax": {}, "nonfinite": {}, "sub": {}})
        a["n"] += 1
        a["s"] += r["s"]
        a["user_s"] += r["user_s"]
        a["sys_s"] += r["sys_s"]
        a["evals"] += r["evals"]
        a["peak_gb"] = max(a["peak_gb"], r["peak_gb"])
        a["trans_gb"] = max(a["trans_gb"], r["trans_gb"])
        a["cache_gb"] = max(a["cache_gb"], r["cache_gb"])
        if r["flops"] is None:
            a["has_flops"] = False
        else:
            a["flops"] += r["flops"]
        for k, v in r["amax"].items():
            a["amax"][k] = max(a["amax"].get(k, 0.0), v)
        for k, v in r["nonfinite"].items():
            a["nonfinite"][k] = a["nonfinite"].get(k, 0) + v
        for k, v in r["sub"].items():
            a["sub"][k] = a["sub"].get(k, 0.0) + v

    def layer_mean(phase, skip_first_if_many=True):
        ls = [l for l in layer_rows if l["phase"] == phase]
        if len(ls) > 1 and skip_first_if_many and A.passes == 1 and A.warm == 0:
            ls = ls[1:]
        return (sum(l["s"] for l in ls) / len(ls)) if ls else None

    pf_mean, msa_mean = layer_mean("pf"), layer_mean("msa")
    plain_mean = (sum(x["s"] for x in plain) / len(plain)) if plain else None
    est = {}
    if pf_mean is not None:
        per_pass = pf_mean * head_info["pairformer_layers"] + (msa_mean or pf_mean * 0.9) * head_info["msa_layers_total"]
        est = {"pf_layer_s_instrumented": round(pf_mean, 3), "msa_layer_s_instrumented": round(msa_mean, 3) if msa_mean else None,
               "ESTIMATE_trunk_pass_s_instrumented": round(per_pass, 1),
               "ESTIMATE_trunk_4pass_s_instrumented": round(4 * per_pass, 1)}
        if plain_mean is not None:
            ratio = plain_mean / pf_mean  # instrumented -> production-style (no per-module syncs)
            pp = ratio * per_pass
            est.update({"pf_layer_s_plain": round(plain_mean, 3), "ESTIMATE_trunk_pass_s_plain": round(pp, 1),
                        "ESTIMATE_trunk_4pass_s_plain": round(4 * pp, 1),
                        "instrumentation_overhead_pct": round(100 * (pf_mean / plain_mean - 1), 1)})

    print("\n== largeN profile: %s  N=%d  chunking=%s (CHUNK_N %d, ROWS %d, CH %d)  cache %.1f GB  mem limit %.0f GB  precision %s%s  trimult K %d"
          % (CASE, N, head_info["chunking"], K.CHUNK_N, K.CHUNK_ROWS, K.CHUNK_CH, cache_gb, A.mem_limit_gb, K.PRECISION,
             " (fp32 fallback)" if A.fp32_fallback else "", K.TRIMULT_K))
    print("   fused pair block: %s;  msa rows %d;  templates: %s;  load %.1f s, feats %.1f s, embed %.1f s; active GB: load %.2f, feats %.2f, embed %.2f (embed peak %.2f)"
          % (head_info["fused_pair_block"], head_info["msa_rows"], head_info["templates_any"], mem["load_s"], mem["feats_s"],
             mem["embed_s"], mem["after_load_gb"], mem["after_feats_gb"], mem["after_embed_gb"], mem["embed_peak_gb"]))
    print("%-5s %-22s %4s %9s %8s %8s %6s %8s %8s %7s %8s %9s" % ("phase", "module", "n", "wall s", "user s", "sys s", "evals",
                                                                  "peak GB", "trans GB", "cacheGB", "TFLOP/s", "per call"))
    tot_mod = 0.0
    for (phase, role), a in agg.items():
        tf = (a["flops"] / a["s"] / 1e12) if (a["has_flops"] and a["flops"] and a["s"]) else None
        tot_mod += a["s"]
        print("%-5s %-22s %4d %9.3f %8.3f %8.3f %6d %8.2f %8.2f %7.2f %8s %9.4f"
              % (phase, role, a["n"], a["s"], a["user_s"], a["sys_s"], a["evals"], a["peak_gb"], a["trans_gb"], a["cache_gb"],
                 f"{tf:.1f}" if tf else "-", a["s"] / a["n"]))
        for k, v in sorted(a["sub"].items(), key=lambda kv: -kv[1]):
            print("        . %-40s %9.3f s" % (k, v))
        if a["amax"] or a["nonfinite"]:
            nf = {k: v for k, v in a["nonfinite"].items() if v}
            print("        max|x|: %s%s" % (", ".join(f"{k} {v:.4g}" for k, v in a["amax"].items()),
                                            ("   NON-FINITE: %s" % nf) if nf else ""))
    wall_layers = sum(l["s"] for l in layer_rows)
    print("   layers: wall %.3f s, in modules %.3f s (%.1f%%), unattributed %.3f s;  plain layer(s): %s"
          % (wall_layers, tot_mod, 100 * tot_mod / wall_layers if wall_layers else 0, wall_layers - tot_mod,
             ", ".join(f"{x['s']:.3f} s (sys {x['sys_s']:.2f})" for x in plain) or "-"))
    if est:
        print("   ESTIMATE (extrapolated from these layers x %d pairformer + %d MSA layers, no recycle-to-recycle effects): %s"
              % (head_info["pairformer_layers"], head_info["msa_layers_total"], json.dumps(est)))
    u, sy = _ru()
    print("   process: user %.1f s, system %.1f s (%.0f%% of %.1f s wall), max RSS %.2f GB, MLX peak (last reset) %.2f GB, swap %s -> %s"
          % (u, sy, 100 * sy / (time.perf_counter() - t_start), time.perf_counter() - t_start,
             resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / GB, _gb(mx.get_peak_memory()), head["swap_start"], swap_used()))
    if stop:
        print("   STOPPED:", stop)
    if err:
        print("   ERROR:", err)

    res = {**head, **head_info, **mem, "layer_rows": layer_rows, "plain": plain, "estimate": est, "stopped": stop, "error": err,
           "modules": [{"phase": ph, "role": ro, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in a.items() if k != "has_flops"},
                        "tflops": (round(a["flops"] / a["s"] / 1e12, 2) if (a["has_flops"] and a["flops"] and a["s"]) else None)}
                       for (ph, ro), a in agg.items()],
           "module_rows": prof.rows if (A.sub or A.amax) else None, "swap_end": swap_used(),
           "proc": {"user_s": round(u, 1), "sys_s": round(sy, 1), "wall_s": round(time.perf_counter() - t_start, 1)}}
    print("LARGEN_PROFILE_JSON", json.dumps(res), flush=True)
    if A.out:
        out = Path(A.out)
        out = out if out.is_absolute() else ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "a") as f:
            f.write(json.dumps(res) + "\n")
    return 2 if err else 0


if __name__ == "__main__":
    sys.exit(main())
