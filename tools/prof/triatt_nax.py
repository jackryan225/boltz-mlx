"""Rung 2N, candidates (a) + (d): flash triangle attention with NAX MMAs from a custom kernel
(boltz_mlx/kernels_nax.py: tri_attention_nax), against the shipped f16 core (fp32-operand simdgroup MMA, `mma`)
and simdgroup-MMA variants with half operands / other tilings.

  python tools/prof/triatt_nax.py              # GPU, ~3 min: all configs, prot400 calls 0/3/7 start+end, prot117
  python tools/prof/triatt_nax.py --quick      # default NAX config + baselines only

Numerics per capture: core vs fp32 MMA kernel on the same fp16 h (core_vs_same), vs the strict fp32 core
(core_vs_ref32), and the f16 module output vs the captured strict module output (module_vs_capture, L1 gate
rel_fro <= 3e-3, cosine >= 0.99999). Also a random-mask input (prot117 x, 30% masked keys, fully masked rows)
for the mask / tail paths. Timing (prot400 call 0, start and end): core_ms / core_tflops (33 GFLOP), module_ms.
Run triatt_nax_probe.py first: if it shows no NAX (have_tensor false / compile error), these rows will carry the
compile error.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from triatt_common import C, CPU_SMOKE, H, K, Ctx, baseline_candidates, captures, emit, metrics, mx, np, run  # noqa: E402

from boltz_mlx import kernels_nax as KN  # noqa: E402

EXP = "nax_flash"
QUICK = "--quick" in sys.argv


def nax_cand(name, **cfg):
    bias_dtype = cfg.get("bias_dtype", mx.float16)
    nkf = cfg.get("nkf", 2)

    def prep(ctx):
        p = KN.nax_tri_inputs(ctx.h16, H, C, ctx.mask, 1e9, kp_mult=max(64, 16 * nkf), bias_dtype=bias_dtype)
        mx.eval(p[0], p[1])
        return p

    def core(ctx, state):
        return KN.tri_attention_nax(ctx.h16, H, C, ctx.mask, 1e9, prepared=state, **cfg)

    def core_h(ctx, h, m):  # module path: bias / mask prep included
        return KN.tri_attention_nax(h, H, C, m, 1e9, **cfg)

    shown = {k: (str(v) if not isinstance(v, (int, float, bool)) else v) for k, v in cfg.items()}
    return {"name": name, "cfg": shown, "gpu_only": True, "prep": prep, "core": core, "core_h": core_h}


def nax_module_cand(name, trans=False, split=False, **cfg):
    """Module-level variants around the NAX core.
    trans: the ending node runs on h in the original orientation (no swapaxes of x before the LN, kernel reads /
           writes transposed, no swap of the output); identical to the default for the starting node.
    split: projection as a 512-column GEMM (q|k|v|g) + a 4-column GEMM (tri bias) instead of one 516-column GEMM."""
    def run_mod(ctx):
        m = ctx.mod
        tr = trans and not m.starting
        x16 = m.layer_norm.f16(ctx.x_in if tr else ctx.x)
        mask_att = ctx.mask  # attention orientation either way
        if split:
            h = x16 @ ctx.w16[:4 * H * C].T
            tb = x16 @ ctx.w16[4 * H * C:].T
            bias = tb.transpose(0, 3, 2, 1) if tr else tb.transpose(0, 3, 1, 2)
        else:
            h = x16 @ ctx.w16.T
            bias = None
        core = KN.tri_attention_nax(h, H, C, mask_att, 1e9, bias=bias, trans=tr, **cfg)
        o = K.lin16(core, m.mha.linear_o)
        return o if (tr or m.starting) else o.swapaxes(-2, -3)

    def prep(ctx):
        tr = trans and not ctx.mod.starting
        h = ctx.h16.swapaxes(1, 2) if tr else ctx.h16
        h = mx.contiguous(h) if hasattr(mx, "contiguous") else h * 1
        p = KN.nax_tri_inputs(h, H, C, ctx.mask, 1e9, trans=tr)
        mx.eval(h, p[0], p[1])
        return h, p, tr

    def core(ctx, st):
        h, p, tr = st
        o = KN.tri_attention_nax(h, H, C, ctx.mask, 1e9, prepared=p, trans=tr, **cfg)
        return o.swapaxes(1, 2) if tr else o

    shown = {k: (str(v) if not isinstance(v, (int, float, bool)) else v) for k, v in cfg.items()}
    return {"name": name, "cfg": {"trans": trans, "split": split, **shown}, "gpu_only": True, "prep": prep,
            "core": core, "module": run_mod}


def sgmma_cand(nsg, rq, bk, pad, half):
    """simdgroup-MMA kernel (shipped source) with another large-J tiling."""
    def core_h(ctx, h, m):
        old = dict(K._MMA_CFG_LARGE_J)
        K._MMA_CFG_LARGE_J["mma_half" if half else "mma"] = (nsg, rq, bk, pad, half)
        try:
            return K.tri_attention_fused(h, H, C, m, 1e9, impl="mma_half" if half else "mma", out_dtype=mx.float16)
        finally:
            K._MMA_CFG_LARGE_J.clear()
            K._MMA_CFG_LARGE_J.update(old)

    return {"name": f"sgmma_{'half' if half else 'f32'}_nsg{nsg}_rq{rq}_bk{bk}",
            "cfg": {"nsg": nsg, "rq": rq, "bk": bk, "pad": pad, "half": half}, "gpu_only": True,
            "core": lambda ctx, s: core_h(ctx, ctx.h16, ctx.mask), "core_h": core_h}


cands = baseline_candidates() + [nax_cand("nax_default"), nax_module_cand("nax_trans_split", trans=True, split=True)]
if not QUICK:
    cands += [
        nax_module_cand("nax_trans", trans=True), nax_module_cand("nax_split", split=True),
        nax_cand("nax_wq1", wq=1), nax_cand("nax_wq2", wq=2), nax_cand("nax_wq5", wq=5), nax_cand("nax_wq8", wq=8),
        nax_cand("nax_nkf4", nkf=4), nax_cand("nax_wq5_nkf4", wq=5, nkf=4),
        nax_cand("nax_hpt4_wq1", hpt=4, wq=1), nax_cand("nax_hpt4_wq2", hpt=4, wq=2),
        nax_cand("nax_order1", order=1), nax_cand("nax_order1_wq5", order=1, wq=5),
        nax_cand("nax_phalf", p_dtype=mx.float16), nax_cand("nax_bias32", bias_dtype=mx.float32),
        nax_cand("nax_norelax", relax=False, allow_unverified_layout=True),  # known wrong: see nax_layout probe
        sgmma_cand(4, 16, 16, 8, True), sgmma_cand(2, 16, 32, 8, True), sgmma_cand(4, 16, 16, 4, False),
    ]

if CPU_SMOKE:
    print("triatt_nax: all candidates are Metal kernels (GPU only); nothing to smoke-test on CPU")
    sys.exit(0)

run(EXP, cands, case="prot400", calls=(0, 3, 7), time_calls=(0,), reps=10, extra_cases=("prot117",))

# module time breakdown at prot400 (call 0, start and end): what the core is not
for f in captures("prot400", (0,)):
    ctx = Ctx(f)
    m = ctx.mod
    core = K.tri_attention_fused(ctx.h16, H, C, ctx.mask, 1e9, out_dtype=mx.float16)
    mx.eval(core)
    parts = {
        "ln_f16_module_orientation": lambda: m.layer_norm.f16(ctx.x),  # ending node: LN of a swapaxes view
        "ln_f16_contiguous": lambda: m.layer_norm.f16(ctx.x_in),
        "proj_516": lambda: ctx.x16 @ ctx.w16.T,
        "proj_512": lambda: ctx.x16 @ ctx.w16[:4 * H * C].T,
        "proj_bias_4": lambda: ctx.x16 @ ctx.w16[4 * H * C:].T,
        "bias_transpose_f32": lambda: ctx.h16[..., 4 * H * C:].transpose(0, 3, 2, 1).astype(mx.float32),
        "core_f16_mma": lambda: K.tri_attention_fused(ctx.h16, H, C, ctx.mask, 1e9, out_dtype=mx.float16),
        "out_proj_lin16": lambda: K.lin16(core, m.mha.linear_o),
        "module_f16_shipped": lambda: ctx.f16_module(lambda h, mm: K.tri_attention_fused(h, H, C, mm, 1e9,
                                                                                         out_dtype=mx.float16)),
    }
    from triatt_common import bench  # noqa: E402
    emit({"exp": EXP, "probe": "breakdown", "input": ctx.name,
          **{k: round(bench(fn, 10), 3) for k, fn in parts.items()}})

# random-mask coverage (masked keys, fully masked query rows, odd J): numerics vs the fp32 kernel on the same h
f = captures("prot117", (0,))[0]
ctx = Ctx(f)
rng = np.random.default_rng(1)
mnp = (rng.random((1, ctx.J, ctx.J)) > 0.3).astype(np.float32)
mnp[:, 3, :] = 0.0
mnp[:, :, 5] = 0.0
ctx.mask = mx.array(mnp)
ctx.ref_same = K.tri_attention_fused(ctx.h16.astype(mx.float32), H, C, ctx.mask, 1e9, impl="mma", out_dtype=mx.float32)
for cand in cands:
    row = {"exp": EXP, "input": ctx.name + "+randmask", "N": ctx.J, "cand": cand["name"], **cand.get("cfg", {})}
    try:
        st = cand["prep"](ctx) if "prep" in cand else None
        out = cand["core"](ctx, st)
        valid = mx.array([i for i in range(ctx.I) if i != 3])  # row 3 fully masked: padding, value not defined
        row["core_vs_same_valid_rows"] = metrics(out[:, valid], ctx.ref_same[:, valid])
        row["masked_row_finite"] = bool(mx.all(mx.isfinite(out[:, 3])).item())
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {str(e)[:2000]}"
    emit(row)
