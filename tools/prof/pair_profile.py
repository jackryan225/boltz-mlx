"""Per-kernel profile of one f16 pairformer layer + the R3-pair A/B (GPU). Rows -> logs/pair_profile.jsonl.

  python tools/prof/pair_profile.py profile      # ~2 min: prot400 / msa384 / prot117 trunk layer 0
  python tools/prof/pair_profile.py gate         # ~3 min: R3-pair flags on vs off vs strict on 13 L1 layer captures
                                                 #   + a 3-layer PairformerModule chain (carry) on prot400
  python tools/prof/pair_profile.py r4           # ~5 min: R4-pair — fused-tail sub-step ablation (cut after o loads /
                                                 #   norm_out LN / p GEMM / g GEMM), split tail per config, transition
                                                 #   variants (staging, simdgroups, butterfly) incl. LNX, tri-mult proj
                                                 #   configs; whole z track / layer R4 off vs on; R4 gate on 13 captures
  python tools/prof/pair_profile.py r5           # ~4 min: R5-tail — why the fused tail's g GEMM is slow (stage cuts, g
                                                 #   with p skipped / p folded early), tail3 configs (bitwise vs shipped),
                                                 #   thread-bound diagnostics, whole z track / layer TAIL3 on vs off, gate
  python tools/prof/pair_profile.py r6           # ~4 min: R6-trans — pair transition implementations (register NAX
                                                 #   kernel configs, MPP configs, MLX GEMMs + SwiGLU) plain / + LNX vs the
                                                 #   shipped kernel, rel errors, MLX pieces, wholes, gate on 13 captures
  python tools/prof/pair_profile.py suggest --write   # choices of the last r4 / r5 / r6 summary -> nax3_cfg.json
  python tools/prof/pair_profile.py all

BOLTZ_MLX_PRECISION=f16 is set here (before importing boltz_mlx); the model loads as in production (weights auto =
mixed checkpoint). Inputs: the recorded L1 PairformerLayer captures (data/cache/l1/<case>/trunk.pairformer__...).

profile: every kernel of the layer timed alone on its real input (pipelined, min of 3 rounds), in pair_block_f16 order,
  with a category (gemm = projections / contraction / GEMM-epilogue kernels, attn = attention cores, glue = LayerNorm,
  copies, bias projections, s-track elementwise) and the whole z track / layer with the R3-pair flags off and on, so the
  sum of parts vs the whole shows dependency / launch overhead. Each R3-pair piece is timed against what it replaces.
gate: per capture the layer's z and s outputs vs the captured strict fp32 outputs with R3-pair off (= the shipped f16
  path) and on; gate as the owner brief: error <= the shipped f16 error, or <= 1.5x it and <= 3e-3 rel. Frobenius and
  cosine >= 0.99999. Plus bitwise checks: xn carried by the transition == norm_in recomputed, z unchanged by PAIRLNX.
"""

from __future__ import annotations

import gc
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("BOLTZ_MLX_PRECISION", "f16")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "validation"))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

import boltz_mlx  # noqa: E402,F401
from boltz_mlx import kernels as K  # noqa: E402
from boltz_mlx import kernels_nax2 as KN2  # noqa: E402
from boltz_mlx import kernels_nax3 as N3  # noqa: E402

mx.set_default_device(mx.gpu)
DATA = Path(os.environ.get("BOLTZ_MLX_DATA", Path(__file__).resolve().parents[2] / "data")).expanduser()
L1 = DATA / "cache" / "l1"
OUT = ROOT / "logs" / "pair_profile.jsonl"
GIT = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
R3 = ("PAIRLNX", "PAIRPROJ", "PAIRTB")


def emit(row: dict):
    row = {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "git": GIT, "chip": K.chip_name(), **row}
    print(json.dumps(row), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "a") as fo:
        fo.write(json.dumps(row) + "\n")


def bench(f, reps: int = 8, rounds: int = 3) -> float:
    mx.eval(f())
    best = 1e9
    for _ in range(rounds):
        t = time.perf_counter()
        mx.eval([f() for _ in range(reps)])
        best = min(best, (time.perf_counter() - t) / reps * 1e3)
    return round(best, 4)


def bench_sync(f, n: int = 8) -> float:
    mx.eval(f())
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        mx.eval(f())
        ts.append(time.perf_counter() - t)
    return round(float(np.median(ts)) * 1e3, 4)


def metrics(out, ref) -> dict:
    o = np.asarray(out.astype(mx.float32) if isinstance(out, mx.array) else out, np.float64).ravel()
    r = np.asarray(ref.astype(mx.float32) if isinstance(ref, mx.array) else ref, np.float64).ravel()
    nf = int((~np.isfinite(o)).sum())
    if nf:
        return {"rel_fro": float("nan"), "cosine": float("nan"), "nonfinite": nf}
    d = np.linalg.norm(r)
    return {"rel_fro": float(np.linalg.norm(o - r) / d), "cosine": float(o @ r / (np.linalg.norm(o) * d)),
            "max_abs": float(np.abs(o - r).max()), "nonfinite": 0}


def gate(c: dict, s: dict) -> bool:
    if c["rel_fro"] != c["rel_fro"]:
        return False
    return c["rel_fro"] <= s["rel_fro"] or (c["rel_fro"] <= 1.5 * s["rel_fro"] and c["rel_fro"] <= 3e-3
                                            and c["cosine"] >= 0.99999)


def set_r3(on: bool):
    for g in R3:
        K.NAX_FLAGS[g] = "1" if on else "0"


_MODS: dict = {}   # named modules of the last _model() (module kwargs of the L1 captures: l1_ops.restore_module_kwargs)


def _model():
    from boltz_mlx.model import load_model

    m = load_model("conf", compile_diffusion=False)
    _MODS.clear()
    _MODS.update(m.named_modules())
    return m


def _load(case: str, call: int, phase: str = "trunk.pairformer"):
    import l1_ops

    f = L1 / case / f"{phase}__PairformerLayer__{call}.pkl"
    rec = pickle.load(open(f, "rb"))
    args, kwargs = l1_ops._from_np(rec["args"]), l1_ops._from_np(rec["kwargs"])
    l1_ops.restore_module_kwargs(rec, kwargs, _MODS)   # next_ln comes back as a parameter dict
    ref = dict(l1_ops._flat_outputs(rec["out"]))
    return rec, args, kwargs, ref


# =============================================================================================== profile
def profile_layer(model, case: str):
    rec, args, kwargs, _ = _load(case, 0)
    layer = dict(model.named_modules())[rec["path"]]
    s, z, mask = args[0], args[1], args[2]
    pm = args[3] if len(args) > 3 else kwargs["pair_mask"]
    _, N, M, D = z.shape
    base = {"exp": "pair_profile", "case": case, "N": N}
    tmo, tmi, ts, te, tr, att = (layer.tri_mul_out, layer.tri_mul_in, layer.tri_att_start, layer.tri_att_end,
                                 layer.transition_z, layer.attention)
    rows = []

    def part(name, cat, fn, **kw):
        try:
            ms = bench(fn)
            rows.append((name, cat, ms))
            emit({**base, "part": name, "cat": cat, "ms": ms, **kw})
        except Exception as e:  # noqa: BLE001
            emit({**base, "part": name, "error": f"{type(e).__name__}: {str(e)[:300]}"})

    set_r3(False)
    kexp = K.TRIMULT_K
    # intermediate inputs along the shipped f16 path
    xn0 = tmo.norm_in.f16(z)
    z1, xn1 = tmo.f16_addln(z, pm, xn0, tmi.norm_in)
    z2, xn2 = tmi.f16_addln(z1, pm, xn1, ts.layer_norm)
    z3, xt3 = ts.f16_addln(z2, pm, xn2, te.layer_norm, trans_out=True)
    z4, xn4 = te.f16_addln(z3, pm, xt3, tr.norm)
    z5 = tr.f16_resid(z4, xn4)
    mx.eval(xn0, z1, xn1, z2, xn2, z3, xt3, z4, xn4, z5)

    part("norm_in LN (tri_mul_out)", "glue", lambda: tmo.norm_in.f16(z))
    for tag, mod, zz, xx, nxt in (("trimul_out", tmo, z, xn0, tmi.norm_in), ("trimul_in", tmi, z1, xn1, ts.layer_norm)):
        wh = K.w16(mod, "_w_th", mod.p_in.weight, mod.g_in.weight)
        xf = xx.reshape(N * M, -1)
        mf = pm.reshape(-1).astype(mx.float32)
        ab = KN2.trimul_proj_nax(xf, wh, mf, 2.0 ** -kexp, **K.NAX_TRIMUL_CFG).reshape(2 * D, N, M)
        a, b = ab[:D], ab[D:]
        mx.eval(ab)
        part(f"{tag} proj+gate (NAX)", "gemm", lambda: KN2.trimul_proj_nax(xf, wh, mf, 2.0 ** -kexp, **K.NAX_TRIMUL_CFG))
        cfn = (lambda: mx.matmul(a, b.transpose(0, 2, 1))) if mod.outgoing else (lambda: mx.matmul(a.transpose(0, 2, 1), b))
        part(f"{tag} contraction (MLX bmm)", "gemm", cfn)
        o = cfn()
        mx.eval(o)
        wpo, wgo = K.w16(mod.p_out, "_w16", mod.p_out.weight), K.w16(mod, "_w_gouth", mod.g_out.weight)
        part(f"{tag} tail: norm_out, p_out, g_out, gate, +z, LN (fused)", "gemm",
             lambda: N3.trimul_tail_addln(o.reshape(D, N * M), mod.norm_out, 2.0 ** (-4 * kexp), wpo, xf, wgo, zz, nxt,
                                          **N3.PAIR_SITE_CFG["tail"]))
        if tag == "trimul_in":
            wtb = ts.tb_weight16()
            part(f"{tag} tail + tri bias (R3 PAIRTB)", "gemm",
                 lambda: N3.trimul_tail_addln(o.reshape(D, N * M), mod.norm_out, 2.0 ** (-4 * kexp), wpo, xf, wgo, zz,
                                              nxt, w_tb=wtb, **N3.PAIR_SITE_CFG["tail"]))
    for tag, mod, zz, xx, nxt, tout in (("triatt_start", ts, z2, xn2, te.layer_norm, True),
                                        ("triatt_end", te, z3, xt3, tr.norm, False)):
        H, C = mod.no_heads, mod.mha.c_hidden
        HC = H * C
        wh = K.w16(mod, "_wh", mod.mha.linear_q.weight, mod.mha.linear_k.weight, mod.mha.linear_v.weight,
                   mod.mha.linear_g.weight, mod.linear.weight)
        part(f"{tag} q|k|v|g projection (MLX fp16 GEMM)", "gemm", lambda: xx @ wh[: 4 * HC].T)
        part(f"{tag} q|k|v|g projection (gemm16, R3 PAIRPROJ)", "gemm", lambda: N3.pair_proj16(xx, wh[: 4 * HC]))
        part(f"{tag} tri-bias projection + transposed copy", "glue",
             lambda: mx.contiguous((xx @ wh[4 * HC:].T).transpose(0, 3, 1, 2)))
        msk = pm if mod.starting else pm.swapaxes(-1, -2)
        h = xx @ wh[: 4 * HC].T
        tb = (xx @ wh[4 * HC:].T).transpose(0, 3, 1, 2)
        mx.eval(h, tb)
        from boltz_mlx import kernels_nax as KN

        part(f"{tag} core (NAX flash)", "attn", lambda: KN.tri_attention_nax(h, H, C, msk, mod.inf, bias=tb,
                                                                               **K.TRIATT_NAX_CFG))
        core = KN.tri_attention_nax(h, H, C, msk, mod.inf, bias=tb, **K.TRIATT_NAX_CFG)
        mx.eval(core)
        wo = K.w16(mod.mha.linear_o, "_w16", mod.mha.linear_o.weight)
        part(f"{tag} linear_o, +z, LN (fused)", "gemm",
             lambda: N3.linear_addln(core, wo, zz, nxt, trans_y=not mod.starting, trans_out=tout, **N3.PAIR_SITE_CFG["lin"]))
        if tag == "triatt_start":
            wtb = te.tb_weight16()
            part(f"{tag} linear_o, +z, LN + tri bias (R3 PAIRTB)", "gemm",
                 lambda: N3.linear_addln(core, wo, zz, nxt, trans_y=False, trans_out=True, w_tb=wtb,
                                         **N3.PAIR_SITE_CFG["lin"]))
    part("transition (NAX, fused +z)", "gemm", lambda: tr.f16_resid(z4, xn4))
    nxt_ln = tmo.norm_in  # stand-in for the next layer's norm_in (same shape / cost)
    part("transition + next LN + pair bias (R3 PAIRLNX)", "gemm", lambda: tr.f16_resid_lnx(z4, xn4, nxt_ln, att))
    # s track
    part("s-track pair bias: proj_z LN + projection (lin16)", "glue",
         lambda: K.lin16(att.proj_z[0].f16(z5), att.proj_z[1]))
    s_n = layer.pre_norm_s(s)
    mx.eval(s_n)
    part("s-track pre_norm_s", "glue", lambda: layer.pre_norm_s(s))
    part("s-track attention (pair bias recomputed)", "attn", lambda: att(s=s_n, z=z5, mask=mask, k_in=s_n))
    pb = mx.contiguous(K.lin16(att.proj_z[0].f16(z5), att.proj_z[1]).transpose(0, 3, 1, 2))
    mx.eval(pb)
    part("s-track attention (pair bias given)", "attn", lambda: att(s=s_n, z=z5, mask=mask, k_in=s_n, pair_bias=pb))
    part("s-track transition_s", "gemm", lambda: layer.transition_s(s))
    # wholes
    for on in (False, True):
        set_r3(on)
        tag = "R3 on" if on else "R3 off (shipped)"
        whole_z = bench_sync(lambda: K.pair_block_f16(layer, z, pm))
        whole_l = bench_sync(lambda: layer(*args, **kwargs))
        emit({**base, "part": f"whole z track ({tag})", "cat": "whole", "ms": whole_z})
        emit({**base, "part": f"whole layer ({tag})", "cat": "whole", "ms": whole_l})
    set_r3(False)
    cats: dict = {}
    for name, cat, ms in rows:
        if "R3" in name or "pair bias given" in name or name.startswith(("triatt_start q|k|v|g projection (gemm16",
                                                                          "triatt_end q|k|v|g projection (gemm16")):
            continue  # alternatives, not on the shipped path
        cats[cat] = round(cats.get(cat, 0.0) + ms, 4)
    emit({**base, "part": "sum of shipped-path parts by category", "cat": "summary", "by_cat": cats,
          "total_ms": round(sum(cats.values()), 4)})
    del rec, args, kwargs
    gc.collect()
    mx.clear_cache()


def part_profile():
    model = _model()
    for case in ("prot400", "msa384", "prot117"):
        try:
            profile_layer(model, case)
        except Exception as e:  # noqa: BLE001
            emit({"exp": "pair_profile", "case": case, "error": f"{type(e).__name__}: {str(e)[:1500]}"})


# =============================================================================================== gate
def part_gate():
    model = _model()
    mods = dict(model.named_modules())
    caps = [(c, "trunk.pairformer", i) for c in ("prot400", "prot117", "msa384", "lig_aff") for i in (0, 3, 7)]
    caps.append(("prot117", "confidence", 0))
    allpass = True
    for case, phase, call in caps:
        base = {"exp": "pair_r3_gate", "input": f"{case}/{phase}/{call}"}
        try:
            rec, args, kwargs, ref = _load(case, call, phase)
        except FileNotFoundError:
            emit({**base, "error": "missing capture"})
            continue
        layer = mods[rec["path"]]
        outs = {}
        for on in (False, True):
            set_r3(on)
            so, zo = layer(*args, **kwargs)
            mx.eval(so, zo)
            outs[on] = (so, zo)
        row = {**base, "z_off": metrics(outs[False][1], ref["z"]), "z_on": metrics(outs[True][1], ref["z"]),
               "s_off": metrics(outs[False][0], ref["s"]), "s_on": metrics(outs[True][0], ref["s"]),
               "z_on_vs_off": metrics(outs[True][1] - args[1], outs[False][1] - args[1])}
        row["pass"] = gate(row["z_on"], row["z_off"]) and gate(row["s_on"], row["s_off"])
        allpass &= row["pass"]
        emit(row)
        del rec, args, kwargs, ref, outs
        gc.collect()
        mx.clear_cache()
    # 3-layer chain with the carry (PairformerModule loop) on prot400 layer-0 input; bitwise pieces
    try:
        from boltz_mlx.layers.pairformer import PairformerModule

        rec, args, kwargs, _ = _load("prot400", 0)
        s, z, mask = args[0], args[1], args[2]
        pm = args[3] if len(args) > 3 else kwargs["pair_mask"]
        layers = model.pairformer_module.layers[:3]
        chain = SimpleNamespace(layers=layers)
        res = {}
        for on in (False, True):
            set_r3(on)
            so, zo = PairformerModule.__call__(chain, s, z, mask, pm)
            mx.eval(so, zo)
            res[on] = (so, zo)
            ms = bench_sync(lambda: PairformerModule.__call__(chain, s, z, mask, pm), 5)
            emit({"exp": "pair_r3_chain", "case": "prot400", "layers": 3, "r3": on, "ms": ms, "ms_per_layer": round(ms / 3, 4)})
        set_r3(True)
        l0 = layers[0]
        zt, xn_next, bias = K.pair_block_f16(l0, z, pm, lnx=(layers[1].tri_mul_out.norm_in, l0.attention))
        set_r3(False)
        z_ref = K.pair_block_f16(l0, z, pm)
        set_r3(True)
        xn_ref = layers[1].tri_mul_out.norm_in.f16(zt)
        b_ref = K.lin16(l0.attention.proj_z[0].f16(zt), l0.attention.proj_z[1]).transpose(0, 3, 1, 2)
        z_tb = K.pair_block_f16(l0, z, pm)  # PAIRTB + PAIRPROJ on, no LNX
        set_r3(False)
        emit({"exp": "pair_r3_chain_bitwise", "case": "prot400",
              "z_lnx_vs_f16_resid_bitwise": bool(mx.array_equal(zt, z_tb).item()),
              "xn_carry_vs_norm_in_bitwise": bool(mx.array_equal(xn_next, xn_ref).item()),
              "pair_bias_vs_lin16": metrics(bias, b_ref),
              "z_r3_vs_off_layer0": metrics(zt - z, z_ref - z),
              "chain_z_on_vs_off": metrics(res[True][1] - z, res[False][1] - z),
              "chain_s_on_vs_off": metrics(res[True][0] - s, res[False][0] - s)})
    except Exception as e:  # noqa: BLE001
        emit({"exp": "pair_r3_chain", "error": f"{type(e).__name__}: {str(e)[:1500]}"})
    for g in R3:
        K.NAX_FLAGS[g] = None
    emit({"exp": "pair_r3_gate_summary", "gate_pass": allpass})


# =============================================================================================== R4-pair
R4 = ("TAIL2",)


def set_flags(groups, on: bool):
    for g in groups:
        K.NAX_FLAGS[g] = "1" if on else "0"


def _layer_inputs(model, case):
    """Real inputs of each pair-stack site of trunk layer 0 along the shipped (R3) path."""
    rec, args, kwargs, ref = _load(case, 0)
    layer = dict(model.named_modules())[rec["path"]]
    z, pm = args[1], (args[3] if len(args) > 3 else kwargs["pair_mask"])
    _, N, M, D = z.shape
    tmo, tmi, ts, te, tr = layer.tri_mul_out, layer.tri_mul_in, layer.tri_att_start, layer.tri_att_end, layer.transition_z
    set_flags(R3 + R4, False)
    xn0 = tmo.norm_in.f16(z)
    z1, xn1 = tmo.f16_addln(z, pm, xn0, tmi.norm_in)
    z2, xn2 = tmi.f16_addln(z1, pm, xn1, ts.layer_norm)
    z3, xt3 = ts.f16_addln(z2, pm, xn2, te.layer_norm, trans_out=True)
    z4, xn4 = te.f16_addln(z3, pm, xt3, tr.norm)
    mx.eval(xn0, z1, xn1, z2, xn2, z3, xt3, z4, xn4)
    sites = {}
    for tag, mod, zz, xx, nxt in (("trimul_out", tmo, z, xn0, tmi.norm_in), ("trimul_in", tmi, z1, xn1, ts.layer_norm)):
        o, xf = mod._f16_contract(xx, pm, D, N, M)
        o = o.reshape(D, N * M)
        mx.eval(o)
        sites[tag] = dict(mod=mod, o=o, xf=xf, z=zz, nxt=nxt, wpo=K.w16(mod.p_out, "_w16", mod.p_out.weight),
                          wgo=K.w16(mod, "_w_gouth", mod.g_out.weight))
    sites["transition"] = dict(mod=tr, z=z4, xn=xn4, att=layer.attention, nxt=tmo.norm_in)
    return rec, layer, z, pm, sites, args, kwargs


def profile_r4(model, case: str) -> dict:
    """Sub-step timing of the fused tail, the split tail per config, the transition variants; returns timings for
    suggest."""
    rec, layer, z, pm, sites, args, kwargs = _layer_inputs(model, case)
    N = z.shape[1]
    base = {"exp": "pair_r4", "case": case, "N": N}
    es = 2.0 ** (-4 * K.TRIMULT_K)
    best = {}

    def t(row, fn, n=8):
        try:
            row["ms"] = bench(fn, n)
        except Exception as e:  # noqa: BLE001
            row["error"] = f"{type(e).__name__}: {str(e)[:400]}"
        emit(row)
        return row.get("ms")

    for tag in ("trimul_out", "trimul_in"):
        s = sites[tag]
        mod, o, xf, zz, nxt, wpo, wgo = s["mod"], s["o"], s["xf"], s["z"], s["nxt"], s["wpo"], s["wgo"]
        full = t({**base, "site": tag, "part": "fused tail (shipped)"},
                 lambda: N3.trimul_tail_addln(o, mod.norm_out, es, wpo, xf, wgo, zz, nxt, **N3.PAIR_SITE_CFG["tail"]))
        for st, what in ((1, "o loads"), (2, "+ norm_out LN"), (3, "+ p GEMM"), (4, "+ g GEMM")):
            t({**base, "site": tag, "part": f"fused tail cut after step {st} ({what})", "stage": st},
              lambda st=st: N3.trimul_tail_stage(o, mod.norm_out, es, wpo, xf, wgo, zz, nxt, st, **N3.PAIR_SITE_CFG["tail"]))
        ref = N3.trimul_tail_addln(o, mod.norm_out, es, wpo, xf, wgo, zz, nxt, **N3.PAIR_SITE_CFG["tail"])
        on_ref = K.layer_norm(o.T, mod.norm_out.get("weight"), mod.norm_out.get("bias"), mod.norm_out.eps * es,
                              out_dtype=mx.float16)
        tl = {}
        for i, cfg in enumerate(N3.TLN_CFGS):
            row = {**base, "site": tag, "part": "split: transpose_ln16", "cfg": cfg}
            try:
                on = N3.transpose_ln16(o, mod.norm_out, es, **cfg)
                row["bitwise_vs_layer_norm_of_transpose"] = bool(mx.array_equal(on, on_ref).item())
            except Exception as e:  # noqa: BLE001
                row["error"] = str(e)[:300]
            ms = t(row, lambda c=cfg: N3.transpose_ln16(o, mod.norm_out, es, **c))
            if ms is not None:
                tl[i] = ms
        on = N3.transpose_ln16(o, mod.norm_out, es)
        mx.eval(on)
        t2 = {}
        for i, cfg in enumerate(N3.TAIL2_CFGS):
            row = {**base, "site": tag, "part": "split: gemm2_gate_addln", "cfg": cfg}
            try:
                got = N3.gemm2_gate_addln(on, wpo, xf, wgo, zz, nxt, **cfg)
                row["z_vs_fused"] = metrics(got[0] - zz, ref[0] - zz)
                row["ln_vs_fused"] = metrics(got[1], ref[1])
            except Exception as e:  # noqa: BLE001
                row["error"] = str(e)[:300]
            ms = t(row, lambda c=cfg: N3.gemm2_gate_addln(on, wpo, xf, wgo, zz, nxt, **c))
            if ms is not None:
                t2[i] = ms
        if tl and t2:
            it, i2 = min(tl, key=tl.get), min(t2, key=t2.get)
            tot = t({**base, "site": tag, "part": "split total (best configs)", "tln": N3.TLN_CFGS[it],
                     "tail2": N3.TAIL2_CFGS[i2]},
                    lambda: N3.trimul_tail2_addln(o, mod.norm_out, es, wpo, xf, wgo, zz, nxt, tln=N3.TLN_CFGS[it],
                                                  cfg=N3.TAIL2_CFGS[i2]))
            best.setdefault("tail", []).append((full, tot, it, i2))
        if tag == "trimul_in":
            wtb = layer.tri_att_start.tb_weight16()
            t({**base, "site": tag, "part": "fused tail + tri bias (shipped R3)"},
              lambda: N3.trimul_tail_addln(o, mod.norm_out, es, wpo, xf, wgo, zz, nxt, w_tb=wtb, **N3.PAIR_SITE_CFG["tail"]))
            t({**base, "site": tag, "part": "split tail + tri bias"},
              lambda: N3.trimul_tail2_addln(o, mod.norm_out, es, wpo, xf, wgo, zz, nxt, w_tb=wtb))
    # secondary: tri-mult projection configs (NAX, kernels_nax2)
    so = sites["trimul_out"]
    whp = K.w16(so["mod"], "_w_th", so["mod"].p_in.weight, so["mod"].g_in.weight)
    mf = pm.reshape(-1).astype(mx.float32)
    for cfg in KN2.TRIMUL_PROJ_CFGS:
        t({**base, "site": "trimul_proj", "part": "NAX proj+gate config sweep", "cfg": cfg,
           "shipped": cfg == K.NAX_TRIMUL_CFG},
          lambda c=cfg: KN2.trimul_proj_nax(so["xf"], whp, mf, 2.0 ** -K.TRIMULT_K, **c))
    # wholes with the best configs
    for on in (False, True):
        set_flags(R3, True)
        set_flags(R4, on)
        emit({**base, "part": f"whole z track (R3 on, R4 {'on' if on else 'off'})", "cat": "whole",
              "ms": bench_sync(lambda: K.pair_block_f16(layer, z, pm))})
        emit({**base, "part": f"whole layer (R3 on, R4 {'on' if on else 'off'})", "cat": "whole",
              "ms": bench_sync(lambda: layer(*args, **kwargs))})
    set_flags(R3 + R4, False)
    del rec, args, kwargs, sites
    gc.collect()
    mx.clear_cache()
    return best


def part_r4():
    model = _model()
    agg: dict = {}
    for case in ("prot400", "msa384"):
        try:
            b = profile_r4(model, case)
            for k, v in b.items():
                agg.setdefault(k, []).append(v)
        except Exception as e:  # noqa: BLE001
            emit({"exp": "pair_r4", "case": case, "error": f"{type(e).__name__}: {str(e)[:1500]}"})
    # R4 gate on the L1 layer captures (R3 on in both arms)
    mods = dict(model.named_modules())
    allpass = True
    for case, phase, call in [(c, "trunk.pairformer", i) for c in ("prot400", "prot117", "msa384", "lig_aff")
                              for i in (0, 3, 7)] + [("prot117", "confidence", 0)]:
        base = {"exp": "pair_r4_gate", "input": f"{case}/{phase}/{call}"}
        try:
            rec, args, kwargs, ref = _load(case, call, phase)
        except FileNotFoundError:
            emit({**base, "error": "missing capture"})
            continue
        layer = mods[rec["path"]]
        outs = {}
        for on in (False, True):
            set_flags(R3, True)
            set_flags(R4, on)
            so, zo = layer(*args, **kwargs)
            mx.eval(so, zo)
            outs[on] = (so, zo)
        row = {**base, "z_off": metrics(outs[False][1], ref["z"]), "z_on": metrics(outs[True][1], ref["z"]),
               "s_off": metrics(outs[False][0], ref["s"]), "s_on": metrics(outs[True][0], ref["s"])}
        row["pass"] = gate(row["z_on"], row["z_off"]) and gate(row["s_on"], row["s_off"])
        allpass &= row["pass"]
        emit(row)
        del rec, args, kwargs, ref, outs
        gc.collect()
        mx.clear_cache()
    for g in R3 + R4:
        K.NAX_FLAGS[g] = None
    summ = {"exp": "pair_r4_summary", "gate_pass": allpass}
    tails = agg.get("tail", [])
    if tails:  # [(fused, split, tln_i, tail2_i)] over sites and cases
        fused = sum(x[0] for c in tails for x in c if x[0])
        split = sum(x[1] for c in tails for x in c if x[1])
        last = tails[0][-1]
        summ["tail_fused_ms_sum"], summ["tail_split_ms_sum"] = round(fused, 4), round(split, 4)
        summ["tail2"] = last[3] if split < fused else "off"
        summ["tln"] = last[2]
    if not allpass:
        summ["tail2"] = "off"
    emit(summ)


def suggest_write():
    """Merge the latest pair_r4_summary choices into boltz_mlx/nax3_cfg.json's "pair" string (other keys kept)."""
    rows = K.rows_for_chip([json.loads(x) for x in open(OUT)] if OUT.exists() else [])  # this Mac's rows only
    r = [x for x in rows if x.get("exp") in ("pair_r4_summary", "pair_r5_summary", "pair_r6_summary")]
    if not r:
        print("no pair_r4_summary / pair_r5_summary row")
        return
    r = r[-1]
    d = K.device_cfg_entry(N3.CFG_FILE) or {"gemm": "", "pair": ""}  # this chip's entry (else default) of the file
    items = dict(x.split("=") for x in filter(None, d.get("pair", "").split(",")))
    for k in ("tln", "tail2", "tail3", "trans3"):
        if k in r:
            items[k] = str(r[k])
    d["pair"] = ",".join(f"{k}={v}" for k, v in items.items())
    d["source_pair_" + r["exp"].split("_")[1]] = (f"tools/prof/pair_profile.py (logs/pair_profile.jsonl, {r['exp']}, "
                                                    f"git {r.get('git')}, {r.get('date')})")
    K.write_device_cfg(N3.CFG_FILE, d)
    print(json.dumps(d))
    print(f"wrote the {K.chip_name()!r} entry of {N3.CFG_FILE}")


# =============================================================================================== R5-tail
def profile_r5(model, case: str) -> dict:
    """Why the shipped fused tail's g-GEMM step is slow, and the low-register tail3 configs (bitwise the same outputs)."""
    rec, layer, z, pm, sites, args, kwargs = _layer_inputs(model, case)
    for g in R3 + R4:  # back to the build's defaults (R3 on in f16) for everything below
        K.NAX_FLAGS[g] = None
    N = z.shape[1]
    base = {"exp": "pair_r5", "case": case, "N": N}
    es = 2.0 ** (-4 * K.TRIMULT_K)
    out: dict = {}

    def t(row, fn, n=8):
        try:
            row["ms"] = bench(fn, n)
        except Exception as e:  # noqa: BLE001
            row["error"] = f"{type(e).__name__}: {str(e)[:400]}"
        emit(row)
        return row.get("ms")

    for tag in ("trimul_out", "trimul_in"):
        s_ = sites[tag]
        mod, o, xf, zz, nxt, wpo, wgo = s_["mod"], s_["o"], s_["xf"], s_["z"], s_["nxt"], s_["wpo"], s_["wgo"]
        args_ = (o, mod.norm_out, es, wpo, xf, wgo, zz, nxt)
        shipped = t({**base, "site": tag, "part": "shipped fused tail"},
                    lambda: N3.trimul_tail_addln(*args_, **N3.PAIR_SITE_CFG["tail"]))
        for st, what in ((3, "+ p GEMM"), (4, "+ g GEMM")):
            t({**base, "site": tag, "part": f"shipped tail cut after step {st} ({what})"},
              lambda st=st: N3.trimul_tail_stage(*args_, st))
        for kind, what in (("g_only", "g GEMM with p skipped (Ph not live)"),
                           ("p_dead", "p computed and folded before the g GEMM (Ph not live)")):
            t({**base, "site": tag, "part": f"diag {kind}: {what}"}, lambda k=kind: N3.trimul_tail_diag(*args_, k))
        ref = N3.trimul_tail_addln(*args_)
        mx.eval(ref)
        times = {}
        for i, cfg in enumerate(N3.TAIL3_CFGS):
            row = {**base, "site": tag, "part": "tail3", "cfg": cfg, "cfg_i": i}
            try:
                got = N3.trimul_tail3_addln(*args_, **cfg)
                row["bitwise_vs_shipped"] = (bool(mx.array_equal(got[0], ref[0]).item())
                                             and bool(mx.array_equal(got[1], ref[1]).item()))
                row["z_vs_shipped"] = metrics(got[0] - zz, ref[0] - zz)
            except Exception as e:  # noqa: BLE001
                row["error"] = str(e)[:300]
            ms = t(row, lambda c=cfg: N3.trimul_tail3_addln(*args_, **c))
            if ms is not None:
                times[i] = ms
        for st in (3, 4):
            t({**base, "site": tag, "part": f"tail3 (cfg 0) cut after phase {'P' if st == 3 else 'G'}"},
              lambda st=st: N3.trimul_tail3_addln(*args_, stage=st))
        if tag == "trimul_in":
            wtb = layer.tri_att_start.tb_weight16()
            t({**base, "site": tag, "part": "shipped tail + tri bias"},
              lambda: N3.trimul_tail_addln(*args_, w_tb=wtb, **N3.PAIR_SITE_CFG["tail"]))
            if times:
                bi = min(times, key=times.get)
                t({**base, "site": tag, "part": "tail3 + tri bias (best cfg)", "cfg_i": bi},
                  lambda: N3.trimul_tail3_addln(*args_, w_tb=wtb, **N3.TAIL3_CFGS[bi]))
        out.setdefault("tail", []).append((shipped, times))
    # thread-bound diagnostics on other pair kernels (same arithmetic)
    s_ = sites["transition"]
    tr, zz, xn = s_["mod"], s_["z"], s_["xn"]
    wh = K.w16(tr, "_w12h", tr.fc1.weight, tr.fc2.weight)
    w3 = K.w16(tr, "_w3h", tr.fc3.weight)
    t({**base, "site": "transition", "part": "shipped"}, lambda: KN2.transition_nax(xn, wh, w3, resid=zz,
                                                                                    **K.NAX_TRANSITION_CFG[128]))
    t({**base, "site": "transition", "part": "shipped + thread bound (diag)"}, lambda: N3.transition_maxt(xn, wh, w3, zz))
    core = mx.array(np.random.default_rng(0).normal(size=zz.shape).astype(np.float16))
    wo = K.w16(layer.tri_att_start.mha.linear_o, "_w16", layer.tri_att_start.mha.linear_o.weight)
    lnn = layer.tri_att_end.layer_norm
    t({**base, "site": "linear_addln", "part": "shipped"}, lambda: N3.linear_addln(core, wo, zz, lnn, trans_out=True))
    t({**base, "site": "linear_addln", "part": "shipped + thread bound (diag)"},
      lambda: N3.linear_addln_maxt(core, wo, zz, lnn, trans_out=True))
    # wholes
    for on in (False, True):
        K.NAX_FLAGS["TAIL3"] = "1" if on else "0"
        emit({**base, "part": f"whole z track (TAIL3 {'on' if on else 'off'})", "cat": "whole",
              "ms": bench_sync(lambda: K.pair_block_f16(layer, z, pm))})
        emit({**base, "part": f"whole layer (TAIL3 {'on' if on else 'off'})", "cat": "whole",
              "ms": bench_sync(lambda: layer(*args, **kwargs))})
    K.NAX_FLAGS["TAIL3"] = None
    del rec, args, kwargs, sites
    gc.collect()
    mx.clear_cache()
    return out


def part_r5():
    model = _model()
    agg = []
    for case in ("prot400", "msa384"):
        try:
            agg += profile_r5(model, case).get("tail", [])
        except Exception as e:  # noqa: BLE001
            emit({"exp": "pair_r5", "case": case, "error": f"{type(e).__name__}: {str(e)[:1500]}"})
    # gate on the L1 layer captures: TAIL3 on vs off (expected identical)
    mods = dict(model.named_modules())
    allpass, identical = True, True
    for case, phase, call in [(c, "trunk.pairformer", i) for c in ("prot400", "prot117", "msa384", "lig_aff")
                              for i in (0, 3, 7)] + [("prot117", "confidence", 0)]:
        base = {"exp": "pair_r5_gate", "input": f"{case}/{phase}/{call}"}
        try:
            rec, args, kwargs, ref = _load(case, call, phase)
        except FileNotFoundError:
            emit({**base, "error": "missing capture"})
            continue
        layer = mods[rec["path"]]
        outs = {}
        for on in (False, True):
            K.NAX_FLAGS["TAIL3"] = "1" if on else "0"
            so, zo = layer(*args, **kwargs)
            mx.eval(so, zo)
            outs[on] = (so, zo)
        row = {**base, "z_off": metrics(outs[False][1], ref["z"]), "z_on": metrics(outs[True][1], ref["z"]),
               "s_off": metrics(outs[False][0], ref["s"]), "s_on": metrics(outs[True][0], ref["s"]),
               "bitwise_on_vs_off": bool(mx.array_equal(outs[True][1], outs[False][1]).item())
               and bool(mx.array_equal(outs[True][0], outs[False][0]).item())}
        row["pass"] = gate(row["z_on"], row["z_off"]) and gate(row["s_on"], row["s_off"])
        allpass &= row["pass"]
        identical &= row["bitwise_on_vs_off"]
        emit(row)
        del rec, args, kwargs, ref, outs
        gc.collect()
        mx.clear_cache()
    K.NAX_FLAGS["TAIL3"] = None
    summ = {"exp": "pair_r5_summary", "gate_pass": allpass, "bitwise_on_vs_off_all": identical}
    if agg:  # per site: (shipped ms, {cfg_i: ms}); pick the cfg with the least total over sites / cases
        shipped = sum(x[0] for x in agg if x[0])
        cand = {i: sum(x[1].get(i, 1e9) for x in agg) for i in range(len(N3.TAIL3_CFGS))}
        best = min(cand, key=cand.get)
        summ.update(tail_shipped_ms_sum=round(shipped, 4), tail3_best_ms_sum=round(cand[best], 4), tail3_best=best)
        summ["tail3"] = best if (allpass and cand[best] < 0.97 * shipped) else "off"
    else:
        summ["tail3"] = "off"
    emit(summ)


# =============================================================================================== R6-trans
def profile_r6(model, case: str) -> dict:
    """The pair transition: shipped MPP kernel (+ LNX epilogue) vs every TRANS3 implementation, plain and with the
    next-LN / pair-bias outputs, accuracy vs shipped; MLX-variant pieces; whole z track / layer TRANS3 on vs off."""
    rec, layer, z, pm, sites, args, kwargs = _layer_inputs(model, case)
    for g in R3 + R4:
        K.NAX_FLAGS[g] = None
    N = z.shape[1]
    base = {"exp": "pair_r6", "case": case, "N": N}
    s_ = sites["transition"]
    tr, zz, xn, att, nxt = s_["mod"], s_["z"], s_["xn"], s_["att"], s_["nxt"]
    wh = K.w16(tr, "_w12h", tr.fc1.weight, tr.fc2.weight)
    w3 = K.w16(tr, "_w3h", tr.fc3.weight)
    ln_s, lin = att.proj_z
    w_s = K.w16(lin, "_w16", lin.weight)
    H = w3.shape[1]

    def t(row, fn, n=8):
        try:
            row["ms"] = bench(fn, n)
        except Exception as e:  # noqa: BLE001
            row["error"] = f"{type(e).__name__}: {str(e)[:400]}"
        emit(row)
        return row.get("ms")

    sc = K.NAX_TRANSITION_CFG[128]
    plain0 = t({**base, "part": "shipped MPP transition (+z)"}, lambda: KN2.transition_nax(xn, wh, w3, resid=zz, **sc))
    lnx0 = t({**base, "part": "shipped MPP transition + LNX epilogue (production)"},
             lambda: N3.transition_lnx(xn, wh, w3, zz, next_ln=nxt, ln_s=ln_s, w_s=w_s, **sc))
    zs, xs, bs = N3.transition_lnx(xn, wh, w3, zz, next_ln=nxt, ln_s=ln_s, w_s=w_s, **sc)
    mx.eval(zs, xs, bs)
    res = {"plain0": plain0, "lnx0": lnx0, "lnx": {}}
    for i, cfg in enumerate(N3.TRANS3_CFGS):
        row = {**base, "part": "trans3 + LNX", "cfg": cfg, "cfg_i": i}
        try:
            zo, xo, bo = N3.transition3(xn, wh, w3, zz, next_ln=nxt, ln_s=ln_s, w_s=w_s, cfg=cfg)
            row["z_update_vs_shipped"] = metrics(zo - zz, zs - zz)
            row["next_ln_vs_shipped"] = metrics(xo, xs)
            row["bias_vs_shipped"] = metrics(bo, bs)
        except Exception as e:  # noqa: BLE001
            row["error"] = str(e)[:300]
        ms = t(row, lambda c=cfg: N3.transition3(xn, wh, w3, zz, next_ln=nxt, ln_s=ln_s, w_s=w_s, cfg=c))
        if ms is not None and "error" not in row:
            res["lnx"][i] = ms
        t({**base, "part": "trans3 plain", "cfg": cfg, "cfg_i": i}, lambda c=cfg: N3.transition3(xn, wh, w3, zz, cfg=c))
    # where the MLX variant's time goes
    xr = xn.reshape(-1, xn.shape[-1])
    h = xr @ wh.T
    s16 = K.swiglu16(h, H)
    mx.eval(h, s16)
    t({**base, "part": "mlx pieces: GEMM 128 -> 1024 (fp16 out)"}, lambda: xr @ wh.T)
    t({**base, "part": "mlx pieces: SwiGLU (compiled, fp16 in / out)"}, lambda: K.swiglu16(h, H))
    t({**base, "part": "mlx pieces: GEMM 512 -> 128 + radd"}, lambda: K.radd(zz.reshape(-1, 128), s16 @ w3.T))
    t({**base, "part": "standalone next LN + pair bias rows (lnx_rows)"},
      lambda: N3.lnx_rows(zs, next_ln=nxt, ln_s=ln_s, w_s=w_s))
    # wholes with the best LNX config (if any)
    if res["lnx"]:
        best = min(res["lnx"], key=res["lnx"].get)
        prev = N3.PAIR_SITE_CFG.get("trans3")
        N3.PAIR_SITE_CFG["trans3"] = N3.TRANS3_CFGS[best]
        for on in (False, True):
            K.NAX_FLAGS["TRANS3"] = "1" if on else "0"
            emit({**base, "part": f"whole z track (TRANS3 {'on, cfg %d' % best if on else 'off'})", "cat": "whole",
                  "ms": bench_sync(lambda: K.pair_block_f16(layer, z, pm))})
            emit({**base, "part": f"whole layer (TRANS3 {'on, cfg %d' % best if on else 'off'})", "cat": "whole",
                  "ms": bench_sync(lambda: layer(*args, **kwargs))})
        K.NAX_FLAGS["TRANS3"] = None
        N3.PAIR_SITE_CFG["trans3"] = prev
    del rec, args, kwargs, sites
    gc.collect()
    mx.clear_cache()
    return res


def part_r6():
    model = _model()
    agg = []
    for case in ("prot400", "msa384"):
        try:
            agg.append(profile_r6(model, case))
        except Exception as e:  # noqa: BLE001
            emit({"exp": "pair_r6", "case": case, "error": f"{type(e).__name__}: {str(e)[:1500]}"})
    best = None
    if agg and all(a["lnx"] for a in agg):
        cand = {i: sum(a["lnx"].get(i, 1e9) for a in agg) for i in range(len(N3.TRANS3_CFGS))}
        best = min(cand, key=cand.get)
        shipped = sum(a["lnx0"] or 0 for a in agg)
    # gate on the L1 layer captures with the best config: TRANS3 on vs off, both vs strict
    mods = dict(model.named_modules())
    allpass = best is not None
    prev = N3.PAIR_SITE_CFG.get("trans3")
    if best is not None:
        N3.PAIR_SITE_CFG["trans3"] = N3.TRANS3_CFGS[best]
    for case, phase, call in [(c, "trunk.pairformer", i) for c in ("prot400", "prot117", "msa384", "lig_aff")
                              for i in (0, 3, 7)] + [("prot117", "confidence", 0)]:
        if best is None:
            break
        base = {"exp": "pair_r6_gate", "input": f"{case}/{phase}/{call}", "cfg_i": best}
        try:
            rec, args, kwargs, ref = _load(case, call, phase)
        except FileNotFoundError:
            emit({**base, "error": "missing capture"})
            continue
        layer = mods[rec["path"]]
        outs = {}
        for on in (False, True):
            K.NAX_FLAGS["TRANS3"] = "1" if on else "0"
            so, zo = layer(*args, **kwargs)
            mx.eval(so, zo)
            outs[on] = (so, zo)
        row = {**base, "z_off": metrics(outs[False][1], ref["z"]), "z_on": metrics(outs[True][1], ref["z"]),
               "s_off": metrics(outs[False][0], ref["s"]), "s_on": metrics(outs[True][0], ref["s"]),
               "z_update_on_vs_off": metrics(outs[True][1] - args[1], outs[False][1] - args[1])}
        row["pass"] = (gate(row["z_on"], row["z_off"]) and gate(row["s_on"], row["s_off"])
                       and row["z_on"]["rel_fro"] <= 3e-3 and row["s_on"]["rel_fro"] <= 3e-3)
        allpass &= row["pass"]
        emit(row)
        del rec, args, kwargs, ref, outs
        gc.collect()
        mx.clear_cache()
    K.NAX_FLAGS["TRANS3"] = None
    N3.PAIR_SITE_CFG["trans3"] = prev
    summ = {"exp": "pair_r6_summary", "gate_pass": bool(allpass)}
    if best is not None:
        summ.update(trans_lnx_shipped_ms_sum=round(shipped, 4), trans3_best=best,
                    trans3_best_ms_sum=round(cand[best], 4), trans3_best_cfg=N3.TRANS3_CFGS[best])
        summ["trans3"] = best if (allpass and cand[best] < 0.97 * shipped) else "off"
    else:
        summ["trans3"] = "off"
    emit(summ)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which == "suggest":  # suggest --write: R4 choices -> boltz_mlx/nax3_cfg.json
        if "--write" in sys.argv:
            suggest_write()
        sys.exit(0)
    if which in ("profile", "all"):
        part_profile()
    if which in ("gate", "all"):
        part_gate()
    if which == "r4":
        part_r4()
    if which == "r5":
        part_r5()
    if which in ("r6", "all"):
        part_r6()
