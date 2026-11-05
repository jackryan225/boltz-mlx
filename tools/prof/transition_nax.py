"""NAX follow-up, task 2: fused NAX transition (boltz_mlx/kernels_nax2.transition_nax) vs the current f16 MLX path
and the round-1 fused fp32 kernel, on the recorded L1 Transition inputs.

  python tools/prof/transition_nax.py            # GPU, ~3 min; rows -> logs/nax2.jsonl
  python tools/prof/transition_nax.py --quick    # default config only, prot400 + prot117

Inputs: data/cache/l1/<case>/<phase>__Transition__<call>.pkl with input dim 128 (pair transitions: pairformer,
MSA-module pairformer, confidence, affinity, diffusion pairwise conditioner; hidden 512 or 256) or 64 (MSA transition,
hidden 256, SwiGLU pre-scaled 2^-TRANS64_K like the f16 path). Captures: strict fp32 module outputs.
Per capture and candidate (numerics):
  module_vs_capture : f16 module output (LN fp16 -> candidate, unscaled) vs captured strict fp32 output (L1 gate
                      rel_fro <= 3e-3, cosine >= 0.99999; aim rel_fro <= 1e-3); max_abs_out (ACT_RANGES check)
  vs_f16_mlx        : candidate vs the current f16 MLX path on the same input
Timing (prot400 pairformer call 0: 160000 rows, D=128, H=512, 62.9 GFLOP; msa384 MSA transition call 0, D=64):
  core_ms (from the fp16 LN output, i.e. what pair_block_f16 calls), module_ms (LN + core), peak MB,
  resid_ms (core + residual add: radd(z, y) for the baselines, RESID=1 fused for NAX),
  resid_bit_identical: fused z + y == radd(z, y_nax) bitwise.
Baselines: f16_mlx (shipped f16 path: fp16 GEMM 128->1024, compiled SwiGLU, fp16 GEMM 512->128), its parts, fused
fp32 round-1 kernel (strict build's transition), MLX fp32.
"""

import math
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import triatt_common as tc  # noqa: E402
from triatt_common import L1, ROOT, bench, emit, metrics, mx, np, peak_mb  # noqa: E402

from boltz_mlx import kernels as K  # noqa: E402
from boltz_mlx import kernels_nax2 as KN2  # noqa: E402
from boltz_mlx.layers.transition import Transition  # noqa: E402
from boltz_mlx.weights import load_module, subtree  # noqa: E402

tc.OUT = ROOT / "logs" / "nax2.jsonl"
EXP = "nax_transition"
QUICK = "--quick" in sys.argv
GATE = {"rel_fro": 3e-3, "cosine": 0.99999}
K.PRECISION = "f16"  # K.f16() is read at call time: the module runs its shipped f16 path
CASES = ["prot400", "prot117"] if QUICK else ["prot400", "prot117", "msa384", "lig_aff", "lig_aff_aff", "ex_ligand",
                                             "cdk2_17", "tyk2_ejm_31"]
CFGS = [KN2.TRANS_DEFAULT] if QUICK else KN2.TRANS_CFGS
TIMED = {("prot400", "trunk.pairformer__Transition__0"), ("msa384", "trunk.msa__Transition__0"),
         ("prot400", "trunk.msa__Transition__3")}
if tc.CPU_SMOKE:  # --cpu-smoke: harness logic on CPU, kernels replaced by their MLX references, one small case
    KN2.transition_nax = KN2.transition_ref
    CASES, CFGS = ["prot117"], [KN2.TRANS_DEFAULT]
    TIMED = {("prot117", "trunk.pairformer__Transition__0"), ("prot117", "trunk.msa__Transition__0")}

_mods: dict = {}


def module_for(path, wname):
    key = (path, wname)
    if key not in _mods:
        w = subtree(path, wname)
        D, H = w["fc1.weight"].shape[1], w["fc1.weight"].shape[0]
        _mods[key] = load_module(Transition(D, H), path, wname)
    return _mods[key]


def weights16(m):
    w12 = mx.concatenate([m.fc1.weight, m.fc2.weight], 0)
    return K.w16(m, "_w12h", m.fc1.weight, m.fc2.weight), K.w16(m, "_w3h", m.fc3.weight), w12


def f16_mlx(m, x):
    return K.unscale(m(x))


def nax_module(m, x, cfg, xn=None):
    D = x.shape[-1]
    k = K.TRANSITION_DOWN_K.get(D, 0)
    w12h, w3h, _ = weights16(m)
    xn = m.norm.f16(x) if xn is None else xn
    y = KN2.transition_nax(xn, w12h, w3h, hidden_scale=2.0 ** -k, **cfg)
    return y, 2.0 ** k


def cfg_name(cfg):
    return "nax_tm{tm}_hb{hb}_nsg{nsg}_tgw{tgw}".format(**cfg)


for case in CASES:
    d = L1 / case
    wname = "aff" if case.endswith("_aff") else "conf"
    for f in sorted(d.glob("*__Transition__*.pkl")):
        rec = pickle.load(open(f, "rb"))
        x_np = rec["args"][1][0][1]
        if x_np.shape[-1] not in (64, 128):
            continue  # s-track / diffusion single transitions (D 384 / 768): not pair-shaped
        m = module_for(rec["path"], wname)
        x = mx.array(x_np)
        ref = rec["out"][1]
        D, H = m.fc1.weight.shape[1], m.fc1.weight.shape[0]
        base = {"exp": EXP, "input": f"{case}/{f.stem}", "path": rec["path"], "rows": int(x.size // D), "D": D, "H": H}
        timed = (case, f.stem) in TIMED
        y_mlx = f16_mlx(m, x)
        mx.eval(y_mlx)
        row = {**base, "cand": "f16_mlx", "module_vs_capture": metrics(y_mlx, ref)}
        mo = row["module_vs_capture"]
        row["l1_pass"] = bool(mo["nonfinite"] == 0 and mo["rel_fro"] <= GATE["rel_fro"] and mo["cosine"] >= GATE["cosine"])
        xn = m.norm.f16(x)
        mx.eval(xn)
        w12h, w3h, w12 = weights16(m)
        kk = K.TRANSITION_DOWN_K.get(D, 0)

        def mlx_core():
            h = xn @ w12h.T
            a = K._swiglu16_scale(h[..., :H], h[..., H:], mx.array(2.0 ** -kk, mx.float32)) if kk else K.swiglu16(h, H)
            return a @ w3h.T

        if timed:
            row["module_ms"] = round(bench(lambda: f16_mlx(m, x), 10), 3)
            row["module_peak_mb"] = round(peak_mb(lambda: f16_mlx(m, x)), 1)
            row["core_ms"] = round(bench(mlx_core, 10), 3)
            row["core_peak_mb"] = round(peak_mb(mlx_core), 1)
            hh = xn @ w12h.T
            aa = K.swiglu16(hh, H)
            mx.eval(hh, aa)
            row["parts_ms"] = {"ln_f16": round(bench(lambda: m.norm.f16(x), 10), 3),
                               "up_gemm": round(bench(lambda: xn @ w12h.T, 10), 3),
                               "swiglu": round(bench(lambda: K.swiglu16(hh, H), 10), 3),
                               "down_gemm": round(bench(lambda: aa @ w3h.T, 10), 3)}
            ymc = mlx_core()
            row["resid_ms"] = round(bench(lambda: K.radd(x, K.Scaled(mlx_core(), 2.0 ** kk) if kk else mlx_core()), 10), 3)
            row["tflops_core"] = round(2 * row["rows"] * (D * 2 * H + H * D) / row["core_ms"] / 1e9, 2)
            del hh, aa, ymc
        emit(row)

        for cfg in (CFGS if case in ("prot400", "prot117") else CFGS[:1]):  # all configs on the timed cases
            row = {**base, "cand": cfg_name(cfg), "cfg": cfg}
            try:
                y, s = nax_module(m, x, cfg)
                yt = y.astype(mx.float32) * s
                mx.eval(yt)
                row["module_vs_capture"] = metrics(yt, ref)
                row["vs_f16_mlx"] = metrics(yt, y_mlx)
                row["max_abs_out_fp16"] = float(mx.max(mx.abs(y.astype(mx.float32))).item())
                mo = row["module_vs_capture"]
                row["l1_pass"] = bool(mo["nonfinite"] == 0 and mo["rel_fro"] <= GATE["rel_fro"]
                                      and mo["cosine"] >= GATE["cosine"])
                # fused residual add: z + float(y) * s, must equal radd(z, Scaled(y, s)) bitwise
                zr = KN2.transition_nax(xn, w12h, w3h, hidden_scale=2.0 ** -kk, out_scale=s, resid=x, **cfg)
                zref = K.radd(x, K.Scaled(y, s) if kk else y)
                row["resid_bit_identical"] = bool(mx.array_equal(zr, zref).item())
                if timed:
                    core = lambda c=cfg: KN2.transition_nax(xn, w12h, w3h, hidden_scale=2.0 ** -kk, **c)  # noqa: E731
                    row["core_ms"] = round(bench(core, 10), 3)
                    row["tflops_core"] = round(2 * row["rows"] * (D * 2 * H + H * D) / row["core_ms"] / 1e9, 2)
                    row["module_ms"] = round(bench(lambda c=cfg: nax_module(m, x, c)[0], 10), 3)
                    row["core_peak_mb"] = round(peak_mb(core), 1)
                    row["module_peak_mb"] = round(peak_mb(lambda c=cfg: nax_module(m, x, c)[0]), 1)
                    row["resid_ms"] = round(bench(lambda c=cfg: KN2.transition_nax(
                        xn, w12h, w3h, hidden_scale=2.0 ** -kk, out_scale=s, resid=x, **c), 10), 3)
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {str(e)[:2000]}"
            emit(row)

        # round-1 fused fp32 kernel (strict build) and MLX fp32, timing rows only (pairformer 128/512 shape)
        if timed and K.transition_supported(x, w12, m.fc3.weight):
            packed = K.pack_transition_weights(w12, m.fc3.weight)
            f32 = lambda: K.transition_fused(x, m.norm.weight, m.norm.bias, w12, m.fc3.weight,  # noqa: E731
                                             eps=m.norm.eps, packed=packed)
            yf = f32()
            mx.eval(yf)
            emit({**base, "cand": "fused_fp32_round1", "module_vs_capture": metrics(yf, ref),
                  "module_ms": round(bench(f32, 10), 3), "module_peak_mb": round(peak_mb(f32), 1)})

            def mlx32():
                h = m.norm(x) @ w12.T
                return m.fc3(mx.sigmoid(h[..., :H]) * h[..., :H] * h[..., H:])

            emit({**base, "cand": "mlx_fp32", "module_ms": round(bench(mlx32, 10), 3),
                  "module_peak_mb": round(peak_mb(mlx32), 1)})
        del x, xn, y_mlx
        mx.clear_cache()

assert math  # noqa
