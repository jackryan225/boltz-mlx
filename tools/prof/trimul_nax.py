"""NAX follow-up, task 3: triangle multiplication with NAX kernels (boltz_mlx/kernels_nax2.py) vs the f16 MLX path.

  python tools/prof/trimul_nax.py            # GPU, ~3 min; rows -> logs/nax2.jsonl
  python tools/prof/trimul_nax.py --quick    # default configs, prot400 + prot117

Candidates (all keep the f16 numerics policy: fp16 operands, fp32 accumulation / gates, operands pre-scaled 2^-K
(K = TRIMUL_K = 4) so the fp16 contraction output stays <= ~3.1e3, norm_out undoes the scale via eps * 2^-4K):
  f16_mlx          shipped path: fp16 GEMM W x^T [512, NM] -> compiled gate*mask*2^-K -> MLX fp16 bmm -> ...
  proj_<cfg>       trimul_proj_nax: projection + sigmoid gate + mask + 2^-K + channel-major fp16 store in one NAX
                   kernel (no [512, NM] intermediate); MLX contraction; g_out separate MLX GEMM
  proj_gout        + g_out = x Wgo^T fused in the same kernel (x read once)
  proj_gout_cnax   + NAX contraction kernel (trimul_contract_nax)
  contract_<cfg>   (contraction only, on the f16 path's operands) NAX contraction vs MLX fp16 bmm
Per capture (prot400 / prot117 / ... pairformer, MSA-module, confidence, affinity tri-mult calls 0/3/7):
  module_vs_capture (L1 gate rel_fro <= 3e-3, cosine >= 0.99999), vs_f16_mlx; ab_vs_ref (gated operands vs the MLX
  reference with identical rounding points), contraction max |o| (fp16 range, ACT_RANGES contraction <= 3088 scaled).
Timing at prot400 pairformer outgoing / incoming call 0: module_ms, parts (proj GEMM, gate, contraction, g_out,
norm_out + p_out, final gate), peak MB.
"""

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import triatt_common as tc  # noqa: E402
from triatt_common import L1, ROOT, bench, emit, metrics, mx, np, peak_mb  # noqa: E402

from boltz_mlx import kernels as K  # noqa: E402
from boltz_mlx import kernels_nax2 as KN2  # noqa: E402
from boltz_mlx.layers.triangular_mult import TriangleMultiplicationIncoming, TriangleMultiplicationOutgoing  # noqa
from boltz_mlx.weights import load_module  # noqa: E402

tc.OUT = ROOT / "logs" / "nax2.jsonl"
EXP = "nax_trimul"
QUICK = "--quick" in sys.argv
GATE = {"rel_fro": 3e-3, "cosine": 0.99999}
K.PRECISION = "f16"
CASES = ["prot400", "prot117"] if QUICK else ["prot400", "prot117", "msa384", "lig_aff", "lig_aff_aff", "ex_ligand",
                                             "cdk2_17", "tyk2_ejm_31"]
PROJ_CFGS = [KN2.TRIMUL_PROJ_CFGS[0]] if QUICK else KN2.TRIMUL_PROJ_CFGS
CON_CFGS = [KN2.TRIMUL_CFGS[0]] if QUICK else KN2.TRIMUL_CFGS
TIMED = {("prot400", "trunk.pairformer__TriangleMultiplicationOutgoing__0"),
         ("prot400", "trunk.pairformer__TriangleMultiplicationIncoming__0")}
if tc.CPU_SMOKE:  # --cpu-smoke: harness logic on CPU with the MLX references in place of the kernels
    KN2.trimul_proj_nax = KN2.trimul_proj_ref
    KN2.trimul_contract_nax = KN2.trimul_contract_ref
    CASES, PROJ_CFGS, CON_CFGS = ["prot117"], PROJ_CFGS[:1], CON_CFGS[:1]
    TIMED = {("prot117", "trunk.pairformer__TriangleMultiplicationOutgoing__0")}

_mods: dict = {}


def module_for(path, cls, wname):
    key = (path, wname)
    if key not in _mods:
        c = TriangleMultiplicationOutgoing if cls == "TriangleMultiplicationOutgoing" else TriangleMultiplicationIncoming
        _mods[key] = load_module(c(128), path, wname)
    return _mods[key]


def w16s(m):
    # [p_a | p_b | g_a | g_b], as the f16 layer
    return K.w16(m, "_w_th", m.p_in.weight, m.g_in.weight), K.w16(m, "_w_gouth", m.g_out.weight)


def nax_module(m, x, mask, proj_cfg, fuse_gout=False, con_cfg=None, parts=None):
    """f16 tri-mult with the NAX projection kernel (+ fused g_out, + NAX contraction)."""
    D = m.p_out.weight.shape[0]
    _, N, M, _ = x.shape
    kexp = K.TRIMULT_K
    wh, wgo = w16s(m)
    xn = m.norm_in.f16(x)
    xf = xn.reshape(N * M, D)
    mf = mask.reshape(N * M)
    if fuse_gout:
        ab, g_out = KN2.trimul_proj_nax(xf, wh, mf, 2.0 ** -kexp, wgo, **proj_cfg)
    else:
        ab = KN2.trimul_proj_nax(xf, wh, mf, 2.0 ** -kexp, **proj_cfg)
        g_out = xf @ wgo.T
    ab = ab.reshape(2 * D, N, M)
    a, b = ab[:D], ab[D:]
    if con_cfg is None:
        o = mx.matmul(a, b.transpose(0, 2, 1)) if m.outgoing else mx.matmul(a.transpose(0, 2, 1), b)
    else:
        o = KN2.trimul_contract_nax(a, b, m.outgoing, **con_cfg)
    if parts is not None:
        parts.update(ab=ab, o=o)
    o = o.reshape(D, N * M).T
    p_o = K.lin16(m.norm_out.f16(o, eps_scale=2.0 ** (-4 * kexp)), m.p_out)
    return K._gate16(p_o, g_out).reshape(1, N, M, D)


def l1(mo):
    return bool(mo["nonfinite"] == 0 and mo["rel_fro"] <= GATE["rel_fro"] and mo["cosine"] >= GATE["cosine"])


def pname(c):
    return "tp{tp}_tc{tc}_nsg{nsg}_x{xreg}".format(**c)


def cname(c):
    return "tm{tm}_tn{tn}_bk{bk}_sm{sm}_sn{sn}".format(**c)


for case in CASES:
    wname = "aff" if case.endswith("_aff") else "conf"
    for f in sorted((L1 / case).glob("*__TriangleMultiplication*__*.pkl")):
        rec = pickle.load(open(f, "rb"))
        m = module_for(rec["path"], rec["cls"], wname)
        x = mx.array(rec["args"][1][0][1])
        mask = mx.array(rec["kwargs"][1]["mask"][1])
        if x.shape[0] != 1:
            continue
        ref = rec["out"][1]
        N = x.shape[1]
        base = {"exp": EXP, "input": f"{case}/{f.stem}", "N": N, "outgoing": m.outgoing}
        timed = (case, f.stem) in TIMED
        y_mlx = m(x, mask=mask)
        mx.eval(y_mlx)
        row = {**base, "cand": "f16_mlx", "module_vs_capture": metrics(y_mlx, ref)}
        row["l1_pass"] = l1(row["module_vs_capture"])
        D = 128
        wh, wgo = w16s(m)
        xn = m.norm_in.f16(x)
        xf = xn.reshape(-1, D)
        mf = mask.reshape(1, -1).astype(mx.float32)
        sc = mx.array(2.0 ** -K.TRIMULT_K, mx.float32)
        h_t = wh @ xf.T
        ab_mlx = K._gate_mask_scale16(h_t[:2 * D], h_t[2 * D:], mf, sc).reshape(2 * D, N, N)
        a, b = ab_mlx[:D], ab_mlx[D:]
        mx.eval(h_t, ab_mlx)
        if timed:
            o = mx.matmul(a, b.transpose(0, 2, 1)) if m.outgoing else mx.matmul(a.transpose(0, 2, 1), b)
            og = xf @ wgo.T
            mx.eval(o, og)
            o_t = o.reshape(D, -1).T
            pn = m.norm_out.f16(o_t, eps_scale=2.0 ** (-4 * K.TRIMULT_K))
            po = K.lin16(pn, m.p_out)
            mx.eval(pn, po)
            row["module_ms"] = round(bench(lambda: m(x, mask=mask), 10), 3)
            row["module_peak_mb"] = round(peak_mb(lambda: m(x, mask=mask)), 1)
            row["parts_ms"] = {
                "ln_in": round(bench(lambda: m.norm_in.f16(x), 10), 3),
                "proj_gemm": round(bench(lambda: wh @ xf.T, 10), 3),
                "gate_mask_scale": round(bench(lambda: K._gate_mask_scale16(h_t[:2 * D], h_t[2 * D:], mf, sc), 10), 3),
                "contraction_mlx": round(bench(lambda: mx.matmul(a, b.transpose(0, 2, 1)) if m.outgoing
                                               else mx.matmul(a.transpose(0, 2, 1), b), 10), 3),
                "g_out_gemm": round(bench(lambda: xf @ wgo.T, 10), 3),
                "norm_out_T": round(bench(lambda: m.norm_out.f16(o_t, eps_scale=2.0 ** (-4 * K.TRIMULT_K)), 10), 3),
                "p_out": round(bench(lambda: K.lin16(pn, m.p_out), 10), 3),
                "gate_out": round(bench(lambda: K._gate16(po, og), 10), 3),
            }
            row["contraction_tflops_mlx"] = round(2 * D * N ** 3 / row["parts_ms"]["contraction_mlx"] / 1e9, 2)
        emit(row)

        # NAX projection kernel (+ g_out) on the same LN output: operand-level numerics vs the MLX path
        for pc in (PROJ_CFGS if case in ("prot400", "prot117") else PROJ_CFGS[:1]):
            for fuse in (False, True):
                if fuse and pc is not PROJ_CFGS[0]:
                    continue
                row = {**base, "cand": f"proj_{pname(pc)}" + ("_gout" if fuse else ""), "cfg": pc}
                try:
                    parts = {}
                    y = nax_module(m, x, mask, pc, fuse_gout=fuse, parts=parts)
                    mx.eval(y)
                    row["module_vs_capture"] = metrics(y, ref)
                    row["vs_f16_mlx"] = metrics(y, y_mlx)
                    row["ab_vs_mlx"] = metrics(parts["ab"], ab_mlx)
                    row["l1_pass"] = l1(row["module_vs_capture"])
                    if timed:
                        fn = lambda c=pc, fu=fuse: nax_module(m, x, mask, c, fuse_gout=fu)  # noqa: E731
                        row["module_ms"] = round(bench(fn, 10), 3)
                        row["module_peak_mb"] = round(peak_mb(fn), 1)
                        kern = (lambda c=pc: KN2.trimul_proj_nax(xf, wh, mask.reshape(-1), 2.0 ** -K.TRIMULT_K, wgo, **c)) \
                            if fuse else (lambda c=pc: KN2.trimul_proj_nax(xf, wh, mask.reshape(-1), 2.0 ** -K.TRIMULT_K, **c))
                        row["proj_kernel_ms"] = round(bench(kern, 10), 3)
                        gfl = 2 * N * N * D * (4 * D + (D if fuse else 0)) / 1e9
                        row["proj_kernel_tflops"] = round(gfl / row["proj_kernel_ms"], 2)
                except Exception as e:
                    row["error"] = f"{type(e).__name__}: {str(e)[:2000]}"
                emit(row)

        # NAX contraction on the f16 path's operands
        o_ref = (a.astype(mx.float32) @ b.astype(mx.float32).transpose(0, 2, 1)) if m.outgoing else \
            (a.astype(mx.float32).transpose(0, 2, 1) @ b.astype(mx.float32))
        o_mlx = mx.matmul(a, b.transpose(0, 2, 1)) if m.outgoing else mx.matmul(a.transpose(0, 2, 1), b)
        mx.eval(o_ref, o_mlx)
        emit({**base, "cand": "contract_mlx_f16", "o_vs_fp32": metrics(o_mlx, o_ref),
              "o_max_abs": float(mx.max(mx.abs(o_ref)).item())})
        for cc in (CON_CFGS if case in ("prot400", "prot117") else CON_CFGS[:1]):
            row = {**base, "cand": f"contract_{cname(cc)}", "cfg": cc}
            try:
                o = KN2.trimul_contract_nax(a, b, m.outgoing, **cc)
                mx.eval(o)
                row["o_vs_fp32"] = metrics(o, o_ref)
                row["o_vs_mlx"] = metrics(o, o_mlx)
                if timed:
                    row["ms"] = round(bench(lambda c=cc: KN2.trimul_contract_nax(a, b, m.outgoing, **c), 10), 3)
                    row["tflops"] = round(2 * D * N ** 3 / row["ms"] / 1e9, 2)
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {str(e)[:2000]}"
            emit(row)

        # full NAX module: fused projection + g_out + NAX contraction
        row = {**base, "cand": "proj_gout_cnax", "cfg": {"proj": PROJ_CFGS[0], "contract": CON_CFGS[0]}}
        try:
            y = nax_module(m, x, mask, PROJ_CFGS[0], fuse_gout=True, con_cfg=CON_CFGS[0])
            mx.eval(y)
            row["module_vs_capture"] = metrics(y, ref)
            row["vs_f16_mlx"] = metrics(y, y_mlx)
            row["l1_pass"] = l1(row["module_vs_capture"])
            if timed:
                fn = lambda: nax_module(m, x, mask, PROJ_CFGS[0], fuse_gout=True, con_cfg=CON_CFGS[0])  # noqa: E731
                row["module_ms"] = round(bench(fn, 10), 3)
                row["module_peak_mb"] = round(peak_mb(fn), 1)
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {str(e)[:2000]}"
        emit(row)
        del x, xn, h_t, ab_mlx, a, b, y_mlx, o_ref, o_mlx
        mx.clear_cache()

assert np  # noqa
