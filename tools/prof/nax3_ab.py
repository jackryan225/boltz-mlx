"""Round-3 NAX A/B (speed-first round): kernels_nax3 vs the shipped --precision f16 paths, timing + accuracy. GPU.

  python tools/prof/nax3_ab.py gemm   # ~2 min: gemm16 configs vs MLX fp16 GEMM (+ its elementwise epilogue) at the
                                      #   token-transformer / conditioning shapes, M = 400 (prot400) and 1065 (lig_aff 5x213)
  python tools/prof/nax3_ab.py pair   # ~4 min: pair-stack epilogues on the recorded L1 PairformerLayer captures
  python tools/prof/nax3_ab.py diff   # ~4 min: token transformer on inputs captured from f16 eager runs
  python tools/prof/nax3_ab.py all

BOLTZ_MLX_PRECISION=f16 is set by this script (before importing boltz_mlx). Models load with the fp32 checkpoint
(fp16 GEMM weights cast at first use = bitwise the mixed checkpoint's), so the fp32 reference paths see fp32 weights.
Rows -> logs/nax3.jsonl; the last line of each part prints the suggested per-site config
(BOLTZ_MLX_NAX_GEMM_CFG / BOLTZ_MLX_NAX_PAIR_CFG) and the accuracy gate verdict.

pair: per capture (trunk pairformer calls 0/3/7 of prot400, prot117, msa384, lig_aff; confidence prot117) the layer's
  z and s outputs vs the captured strict fp32 outputs, shipped f16 (PAIREPI=0) vs epilogue kernels (PAIREPI=1);
  timing of the z track (pair_block_f16) and of each fused site vs its unfused chain on prot400 call 0, per config.
diff: the token transformer (24 layers) and single layers 0 / 12 / 23 on (a, s) captured at diffusion steps 0 / 4 / 9
  of 10-step f16 eager runs (lig_aff affinity pass x5 samples, prot400 x1), each mode's update vs the fp32 (strict
  fused-kernel) update; timing eager and mx.compile'd of the whole transformer per mode / site config.
Gate (owner brief): candidate error vs fp32 <= the shipped f16 error, or <= 1.5x it and <= 3e-3 rel. Frobenius and
cosine >= 0.99999.
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

os.environ.setdefault("BOLTZ_MLX_PRECISION", "f16")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

import boltz_mlx  # noqa: E402,F401
from boltz_mlx import kernels as K  # noqa: E402
from boltz_mlx import kernels_nax3 as N3  # noqa: E402

mx.set_default_device(mx.gpu)
DATA = Path(os.environ.get("BOLTZ_MLX_DATA", Path(__file__).resolve().parents[2] / "data")).expanduser()
L1 = DATA / "cache" / "l1"
OUT = ROOT / "logs" / "nax3.jsonl"
GIT = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()


def emit(row: dict):
    row = {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "git": GIT, "chip": K.chip_name(), **row}
    print(json.dumps(row), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "a") as fo:
        fo.write(json.dumps(row) + "\n")


def bench(f, reps: int = 10, rounds: int = 3) -> float:
    """ms per call: min over rounds of (reps pipelined calls, one eval)."""
    mx.eval(f())
    best = 1e9
    for _ in range(rounds):
        t = time.perf_counter()
        mx.eval([f() for _ in range(reps)])
        best = min(best, (time.perf_counter() - t) / reps * 1e3)
    return round(best, 4)


def bench_sync(f, n: int = 7) -> float:
    """ms per call, each call evaluated on its own (big graphs: whole transformer / pair block); median."""
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


def gate(cand: dict, shipped: dict) -> bool:
    c, s = cand["rel_fro"], shipped["rel_fro"]
    if not (c == c):
        return False
    return c <= s or (c <= 1.5 * s and c <= 3e-3 and cand["cosine"] >= 0.99999)


def _err(row, e):
    row["error"] = f"{type(e).__name__}: {str(e)[:1500]}"
    emit(row)


# =============================================================================================== gemm
def part_gemm():
    rng = np.random.default_rng(0)
    h = lambda *s, sc=1.0: mx.array((rng.normal(size=s) * sc).astype(np.float16))  # noqa: E731
    best = {}
    sites = []
    for M, ntok, rep in ((400, 400, 1), (1065, 213, 5)):
        sites += [("qkvg", M, 768, 3072, N3.EPI_STORE, ntok, rep), ("o", M, 768, 768, N3.EPI_GRESID, ntok, rep),
                  ("ff", M, 768, 1536, N3.EPI_SWIGLU, ntok, rep), ("ba", M, 1536, 768, N3.EPI_GRESID, ntok, rep)]
    sites += [("cond", 213, 768, 73728, N3.EPI_STORE, 213, 1), ("cond", 400, 768, 73728, N3.EPI_STORE, 400, 1),
              ("cond2", 400, 768, 36864, N3.EPI_STORE, 400, 1),
              # pair stack at N = 400 (measurement only, not integrated): tri-att q|k|v|g projection (the 4 tri-bias
              # columns stay a separate GEMM) and a plain 128 -> 128 projection (what g_out / p_out / linear_o were)
              ("triatt_proj", 160000, 128, 512, N3.EPI_STORE, 160000, 1),
              ("pair_128", 160000, 128, 128, N3.EPI_STORE, 160000, 1)]
    for name, M, Kd, NOUT, epi, ntok, rep in sites:
        ngrp = 3 if epi == N3.EPI_SWIGLU else 1
        x, w = h(M, Kd), h(ngrp * NOUT, Kd, sc=Kd ** -0.5)
        kw = {}
        if epi == N3.EPI_GRESID:
            kw = dict(gate=h(ntok, 48 * 768), gate_bias=mx.array(rng.normal(size=48 * 768).astype(np.float32)),
                      gate_off=5 * 768, resid=mx.array(rng.normal(size=(M, NOUT)).astype(np.float32)), ntok=ntok,
                      rep=rep)
        flops = 2.0 * M * Kd * NOUT * ngrp
        ref = N3.gemm16_ref(x, w, epi=epi, ngrp=ngrp, **kw)
        reps = max(2, min(20, int(4e8 // (M * NOUT * ngrp * 2))))  # <= ~400 MB of pipelined outputs
        base = {"exp": "nax3_gemm", "site": name, "M": M, "K": Kd, "N": NOUT * ngrp, "epi": epi}
        site = "cond" if name.startswith("cond") else name
        cands = [("mlx_gemm_only", None), ("mlx", "mlx")]
        cands += [(f"nax {c}", c) for c in (N3.SWIGLU_CFGS if epi == N3.EPI_SWIGLU else N3.GEMM_CFGS)]
        for cname, cfg in cands:
            row = {**base, "cand": cname, "cfg": cfg}
            try:
                if cfg is None:
                    fn = lambda: x @ w.T  # noqa: E731
                elif cfg == "mlx":
                    fn = lambda: N3._gemm_site(site, x, w, cfg="mlx", epi=epi, ngrp=ngrp, **kw)  # noqa: E731
                else:
                    if Kd % (16 * cfg["ku"]) or NOUT % (cfg["wn"] * cfg["sn"]):
                        continue
                    fn = lambda c_=cfg: N3.gemm16(x, w, epi=epi, ngrp=ngrp, **kw, **c_)  # noqa: E731
                y = fn()
                if cfg is not None:
                    row["vs_fp32"] = metrics(y, ref)
                row["ms"] = bench(fn, reps)
                row["tflops"] = round(flops / row["ms"] / 1e9, 2)
                emit(row)
                if cfg is not None:
                    key = (site, M)
                    if key not in best or row["ms"] < best[key][1]:
                        best[key] = (cfg, row["ms"], cname)
            except Exception as e:  # noqa: BLE001
                _err(row, e)
        mx.clear_cache()
    for M in (400, 1065):
        sug = []
        for site, lst in (("qkvg", N3.GEMM_CFGS), ("o", N3.GEMM_CFGS), ("ff", N3.SWIGLU_CFGS), ("ba", N3.GEMM_CFGS)):
            if (site, M) in best:
                cfg = best[(site, M)][0]
                sug.append(f"{site}={'mlx' if cfg == 'mlx' else lst.index(cfg)}")
        emit({"exp": "nax3_gemm_suggest", "M": M, "BOLTZ_MLX_NAX_GEMM_CFG": ",".join(sug),
              "best": {f"{k[0]}@{k[1]}": [v[2], v[1]] for k, v in best.items() if k[1] == M}})
    for M in (213, 400):
        if ("cond", M) in best:
            emit({"exp": "nax3_gemm_suggest_cond", "M": M, "best": best[("cond", M)][2], "ms": best[("cond", M)][1]})
    # atom-attention keys (per atom layer and step: k|v of all atoms -> key windows): contraction vs gather
    from boltz_mlx.modules.encodersv2 import make_to_keys

    for n_atoms, B in ((1536, 5), (3072, 1)):  # lig_aff affinity pass x5, prot400
        tk = make_to_keys(n_atoms, 32, 128)
        kv = mx.array(rng.normal(size=(B, n_atoms, 256)).astype(np.float32))
        row = {"exp": "nax3_tokeys", "atoms": n_atoms, "B": B,
               "equal": bool(mx.array_equal(tk(kv), tk.gather(kv)).item()),
               "contraction_ms": bench(lambda: tk(kv), 20), "gather_ms": bench(lambda: tk.gather(kv), 20)}
        emit(row)


# =============================================================================================== pair
def _l1_helpers():
    sys.path.insert(0, str(ROOT / "tools" / "validation"))
    import l1_ops

    return l1_ops


def part_pair():
    from boltz_mlx.model import load_model

    l1 = _l1_helpers()
    model = load_model("conf", weights="fp32", compile_diffusion=False)
    mods = dict(model.named_modules())
    sug = {}
    files = []
    for case in ("prot400", "prot117", "msa384", "lig_aff"):
        files += [(case, L1 / case / f"trunk.pairformer__PairformerLayer__{i}.pkl") for i in (0, 3, 7)]
    files += [("prot117", L1 / "prot117" / f"confidence__PairformerLayer__{i}.pkl") for i in (0,)]
    gates = []
    for case, f in files:
        if not f.exists():
            emit({"exp": "nax3_pair", "input": str(f), "error": "missing capture"})
            continue
        rec = pickle.load(open(f, "rb"))
        layer = mods[rec["path"]]
        args, kwargs = l1._from_np(rec["args"]), l1._from_np(rec["kwargs"])
        ref = dict(l1._flat_outputs(rec["out"]))
        s_in, z = args[0], args[1]
        pair_mask = args[3] if len(args) > 3 else kwargs["pair_mask"]
        N = z.shape[1]
        base = {"exp": "nax3_pair", "input": f"{case}/{rec['phase']}/{rec['cls']}/{rec['call']}", "N": N}
        outs = {}
        for mode in ("0", "1"):
            K.NAX_FLAGS["PAIREPI"] = mode
            row = {**base, "cand": "f16_shipped" if mode == "0" else "nax3_pair_epi"}
            try:
                assert K.use_pair_block_f16(z) and (mode == "0" or K.pair_epi_ok(z))
                so, zo = layer(*args, **kwargs)
                mx.eval(so, zo)
                outs[mode] = (so, zo)
                row["z_vs_fp32"] = metrics(zo, ref["z"])
                row["s_vs_fp32"] = metrics(so, ref["s"])
                if case == "prot400" and rec["call"] == 0 or case == "prot117" and rec["call"] == 0 \
                        and rec["phase"].startswith("trunk"):
                    row["ztrack_ms"] = bench_sync(lambda: K.pair_block_f16(layer, z, pair_mask), 10)
                    row["layer_ms"] = bench_sync(lambda: layer(*args, **kwargs), 10)
                emit(row)
            except Exception as e:  # noqa: BLE001
                _err(row, e)
        if len(outs) == 2:
            g = {"input": base["input"], "z": gate(metrics(outs["1"][1], ref["z"]), metrics(outs["0"][1], ref["z"])),
                 "s": gate(metrics(outs["1"][0], ref["s"]), metrics(outs["0"][0], ref["s"])),
                 "z_vs_shipped": metrics(outs["1"][1] - z, outs["0"][1] - z)}
            gates.append(g)
            emit({**base, "exp": "nax3_pair_gate", **g})
        if case == "prot400" and rec["call"] == 0:
            try:
                sug = _pair_sites(layer, z, pair_mask, base)
            except Exception as e:  # noqa: BLE001
                _err({**base, "exp": "nax3_pair_site"}, e)
        del rec, args, kwargs, ref, outs
        gc.collect()
        mx.clear_cache()
    K.NAX_FLAGS["PAIREPI"] = None
    emit({"exp": "nax3_pair_summary", "gate_pass": all(g["z"] and g["s"] for g in gates), "n": len(gates),
          "BOLTZ_MLX_NAX_PAIR_CFG": ",".join(f"{k}={v}" for k, v in sug.items())})


def _pair_sites(layer, z, pair_mask, base):
    """Each fused site vs its unfused chain (same inputs), per config. Returns the suggested site configs."""
    tm_o, tm_i, ts, te = layer.tri_mul_out, layer.tri_mul_in, layer.tri_att_start, layer.tri_att_end
    _, N, M, D = z.shape
    xn = tm_o.norm_in.f16(z)
    mx.eval(xn)
    sug = {}

    # tri-mult (outgoing) tail: contraction common to both; unfused = _f16_b1 + add_ln, fused = f16_addln
    def unfused_tm():
        return K.add_ln(z, tm_o._f16_b1(xn, pair_mask, D, N, M), tm_i.norm_in)

    emit({**base, "exp": "nax3_pair_site", "site": "trimul_out", "cand": "unfused", "ms": bench_sync(unfused_tm, 10)})
    emit({**base, "exp": "nax3_pair_site", "site": "trimul_out", "cand": "contraction_only",
          "ms": bench_sync(lambda: tm_o._f16_contract(xn, pair_mask, D, N, M)[0], 10)})
    ref = unfused_tm()
    best = None
    for i, cfg in enumerate(N3.TRIMUL_TAIL_CFGS):
        N3.PAIR_SITE_CFG["tail"] = cfg
        row = {**base, "exp": "nax3_pair_site", "site": "trimul_out", "cand": f"fused {cfg}"}
        try:
            got = tm_o.f16_addln(z, pair_mask, xn, tm_i.norm_in)
            row["z_vs_unfused"] = metrics(got[0] - z, ref[0] - z)
            row["ln_vs_unfused"] = metrics(got[1], ref[1])
            row["z_bitwise_unfused"] = bool(mx.array_equal(got[0], ref[0]).item())
            row["ms"] = bench_sync(lambda: tm_o.f16_addln(z, pair_mask, xn, tm_i.norm_in), 10)
            emit(row)
            if best is None or row["ms"] < best[1]:
                best = (i, row["ms"])
        except Exception as e:  # noqa: BLE001
            _err(row, e)
    if best:
        sug["tail"] = best[0]
        N3.PAIR_SITE_CFG["tail"] = N3.TRIMUL_TAIL_CFGS[best[0]]

    # tri-att start (LN output transposed for the ending node) and end (transposed core, read back transposed)
    xs = ts.layer_norm.f16(z)
    zs, xt = K.add_ln(z, ts(z, mask=pair_mask, xn=xs), te.layer_norm, trans_out=True)
    mx.eval(xs, zs, xt)
    variants = [
        ("triatt_start", lambda: K.add_ln(z, ts(z, mask=pair_mask, xn=xs), te.layer_norm, trans_out=True),
         lambda: ts.f16_addln(z, pair_mask, xs, te.layer_norm, trans_out=True)),
        ("triatt_end", lambda: K.add_ln(zs, te(zs, mask=pair_mask, xn=xt, keep_t=True), layer.transition_z.norm,
                                        trans_y=True),
         lambda: te.f16_addln(zs, pair_mask, xt, layer.transition_z.norm)),
    ]
    times = {}
    for site, unf, fus in variants:
        emit({**base, "exp": "nax3_pair_site", "site": site, "cand": "unfused", "ms": bench_sync(unf, 10)})
        ref = unf()
        for i, cfg in enumerate(N3.PAIR_CFGS):
            N3.PAIR_SITE_CFG["lin"] = cfg
            row = {**base, "exp": "nax3_pair_site", "site": site, "cand": f"fused {cfg}"}
            try:
                got = fus()
                zref = z if site == "triatt_start" else zs
                row["z_vs_unfused"] = metrics(got[0] - zref, ref[0] - zref)
                row["ln_vs_unfused"] = metrics(got[1], ref[1])
                row["z_bitwise_unfused"] = bool(mx.array_equal(got[0], ref[0]).item())
                row["ms"] = bench_sync(fus, 10)
                emit(row)
                times[i] = times.get(i, 0.0) + row["ms"]
            except Exception as e:  # noqa: BLE001
                _err(row, e)
    if times:
        sug["lin"] = min(times, key=times.get)
        N3.PAIR_SITE_CFG["lin"] = N3.PAIR_CFGS[sug["lin"]]
    return sug


# =============================================================================================== diff
def _capture_diff(case: str, aff: bool, steps=(0, 4, 9), n_steps: int = 10):
    """(model, prepared conditioning, captures [(step, a, s)]) from a 10-step f16 eager run (shipped f16 path)."""
    from boltz_mlx.model import load_model
    from boltz_mlx.modules.diffusionv2 import weighted_rigid_align
    from boltz_mlx.modules.transformersv2 import DiffusionTransformer
    from boltz_mlx.sampler import DiffusionParams, record_noise, sample

    K.NAX_FLAGS["GEMM"] = "0"
    K.set_diff_dtype("fp16")
    model = load_model("aff" if aff else "conf", weights="fp32", compile_diffusion=False)
    z = np.load(DATA / case / ("feats_affinity.npz" if aff else "feats.npz"))
    feats = {k: mx.array(z[k]) for k in z.files}
    mult = 5 if aff else 1
    s_inputs, s, zz, rpe = model.trunk(feats, 1)
    q, c, to_keys, eb, db, tb = model.diffusion_conditioning(s, zz, rpe, feats)
    cond = dict(q=q, c=c, to_keys=to_keys, atom_enc_bias=eb, atom_dec_bias=db, token_trans_bias=tb)
    sm = model.structure_module
    sc = sm.score_model
    cond["prepared"] = sc.prepare(feats, cond, mult)
    cond["shared_times"] = True
    mx.eval(s_inputs, s, q, c, eb, db, tb, cond["prepared"])
    del zz
    tok = sc.token_transformer
    caps, counter = [], [0]
    orig = DiffusionTransformer.__call__

    def wrapped(self, *args, **kw):
        if self is tok:
            a = kw["a"] if "a" in kw else args[0]
            s_ = kw["s"] if "s" in kw else args[1]
            if counter[0] in steps:
                mx.eval(a, s_)
                caps.append((counter[0], a, s_))
            counter[0] += 1
        return orig(self, *args, **kw)

    DiffusionTransformer.__call__ = wrapped
    try:
        def denoise(x, t_hat):
            return sm.preconditioned_network_forward(x, mx.full((x.shape[0],), t_hat, dtype=mx.float32), dict(
                s_inputs=s_inputs, s_trunk=s, feats=feats, diffusion_conditioning=cond, multiplicity=mult))

        out = sample(denoise, feats["atom_pad_mask"].astype(mx.float32), n_steps, mult,
                     record_noise(0, case, "affinity" if aff else "structure", mult), weighted_rigid_align,
                     DiffusionParams())
        mx.eval(out)
    finally:
        DiffusionTransformer.__call__ = orig
    mask_rep = mx.repeat(feats["token_pad_mask"], mult, axis=0).astype(mx.float32)
    return model, tok, tb, mask_rep, mult, caps


def _set_mode(mode: str):
    """fp32 (strict fused kernels) | f16 (shipped) | nax3 (kernels_nax3, current DIFF_GEMM_CFG)."""
    K.set_diff_dtype("fp32" if mode == "fp32" else "fp16")
    K.NAX_FLAGS["GEMM"] = "1" if mode.startswith("nax3") else "0"


def _tok_prep(tok, tb, mask_rep, mult):
    """Step-invariant preparation (outside the step in production; depends on the mode's flags)."""
    prep = tok.prepare(None, tb, mask_rep, None, mult, with_cond=False)
    mx.eval(prep["attn_bias"])
    return prep


def _tok_call(tok, prep, a, s, mult):
    """Per-step work as DiffusionModule.__call__ runs it: conditioning (shared over samples) + the transformer."""
    cond = tok.precompute_cond_shared(s[:1], mult) if mult > 1 else None
    return tok(a, s, prepared={**prep, "cond": cond})


def _tok_run(tok, a, s, tb, mask_rep, mult):
    return _tok_call(tok, _tok_prep(tok, tb, mask_rep, mult), a, s, mult)


def _layer_runner(tok, s, tb, mask_rep, mult, mode):
    """f(a_l, l) -> output of layer l alone in `mode` (same conditioning / bias preparation as the full run)."""
    from boltz_mlx.modules.transformersv2 import _cond_layer

    prep = tok.prepare(None, tb, mask_rep, None, mult, with_cond=False)
    if mode.startswith("nax3"):
        rc = tok.precompute_cond_raw16(s[:1] if mult > 1 else s)
        rc.rep = mult if mult > 1 else 1
        return lambda a, l: N3.transformer_forward(tok, a, rc, prep["attn_bias"], prep.get("attn_brep", 1), layers=[l])
    if mult > 1:
        cond = tok.precompute_cond_shared(s[:1], mult)
    else:
        from boltz_mlx import kernels_difftx as DT

        cond = tok.precompute_cond(s, raw=DT.transformer_supported(tok, None))
    return lambda a, l: tok.layers[l](a, s, None, None, None, mult, cond=_cond_layer(cond, l),
                                      attn_bias=prep["attn_bias"][l])


DIFF_SITE_VARIANTS = {  # name -> DIFF_GEMM_CFG overrides (indices into GEMM_CFGS / SWIGLU_CFGS)
    "nax3": {},                                                              # the module default
    "nax3_mlxgemm": {"qkvg": "mlx", "o": "mlx", "ff": "mlx", "ba": "mlx"},  # fused cond/AdaLN/bias, MLX GEMMs
    "nax3_allnax": {"qkvg": 0, "o": 0, "ff": 0, "ba": 0},
    "nax3_wide": {"qkvg": 2, "o": 1, "ff": 1, "ba": 2},
}


def _apply_variant(v: str, default: dict):
    N3.DIFF_GEMM_CFG.update(default)
    for site, i in DIFF_SITE_VARIANTS.get(v, {}).items():
        N3.DIFF_GEMM_CFG[site] = "mlx" if i == "mlx" else (N3.SWIGLU_CFGS if site == "ff" else N3.GEMM_CFGS)[i]


def part_diff():
    default = dict(N3.DIFF_GEMM_CFG)
    sg = suggest("gemm")
    if sg:  # the gemm part's per-site choice as a variant of its own
        DIFF_SITE_VARIANTS["nax3_suggest"] = {k: (v if v == "mlx" else int(v))
                                              for k, v in (it.split("=") for it in sg.split(",")) if k != "cond"}
    verdict = []
    for case, aff in (("lig_aff", True), ("prot400", False)):
        base0 = {"exp": "nax3_diff", "case": case + ("_aff" if aff else "")}
        try:
            model, tok, tb, mask_rep, mult, caps = _capture_diff(case, aff)
        except Exception as e:  # noqa: BLE001
            _err({**base0, "stage": "capture"}, e)
            continue
        for ci, (step, a, s) in enumerate(caps):
            base = {**base0, "step": step, "B": a.shape[0], "N": a.shape[1]}
            _set_mode("fp32")
            out32 = _tok_run(tok, a, s, tb, mask_rep, mult)
            # fp32 per-layer inputs (layers 0, 12, 23)
            run32 = _layer_runner(tok, s, tb, mask_rep, mult, "fp32")
            a_l, ins, outs32 = a, {}, {}
            for l in range(len(tok.layers)):
                if l in (0, 12, 23):
                    ins[l] = a_l
                a_l = run32(a_l, l)
                if l in (0, 12, 23):
                    outs32[l] = a_l
                mx.eval(a_l)
            mx.eval(out32)
            res = {}
            for mode in ["f16"] + list(DIFF_SITE_VARIANTS):
                row = {**base, "cand": mode}
                try:
                    _set_mode(mode)
                    _apply_variant(mode, default)
                    out = _tok_run(tok, a, s, tb, mask_rep, mult)
                    row["update_vs_fp32"] = metrics(out - a, out32 - a)
                    runl = _layer_runner(tok, s, tb, mask_rep, mult, mode)
                    row["layers_vs_fp32"] = {l: metrics(runl(ins[l], l) - ins[l], outs32[l] - ins[l]) for l in ins}
                    if ci == 0:
                        prep = _tok_prep(tok, tb, mask_rep, mult)
                        row["eager_ms"] = bench_sync(lambda: _tok_call(tok, prep, a, s, mult))
                        cf = mx.compile(lambda a_, s_: _tok_call(tok, prep, a_, s_, mult))
                        row["compiled_ms"] = bench_sync(lambda: cf(a, s))
                        row["attn_bias_mb"] = round(sum(x.nbytes for x in prep["attn_bias"]) / 1e6, 1)
                        del prep, cf
                    res[mode] = row
                    emit(row)
                except Exception as e:  # noqa: BLE001
                    _err(row, e)
            if "f16" in res:
                for mode, row in res.items():
                    if mode == "f16":
                        continue
                    g = {"whole": gate(row["update_vs_fp32"], res["f16"]["update_vs_fp32"]),
                         "layers": all(gate(row["layers_vs_fp32"][l], res["f16"]["layers_vs_fp32"][l])
                                       for l in row["layers_vs_fp32"])}
                    verdict.append((base["case"], step, mode, g))
                    emit({**base, "exp": "nax3_diff_gate", "cand": mode, **g})
            mx.clear_cache()
        _set_mode("f16")
        N3.DIFF_GEMM_CFG.update(default)
        del model, tok, caps
        gc.collect()
        mx.clear_cache()
    emit({"exp": "nax3_diff_summary",
          "gate_pass_by_cand": {m: all(g["whole"] and g["layers"] for c, s_, mm, g in verdict if mm == m)
                                for m in DIFF_SITE_VARIANTS}})
    K.NAX_FLAGS["GEMM"] = None


def suggest(kind: str) -> str:
    """Per-site config string from the latest rows in logs/nax3.jsonl (rows of the newest build that ran the part):
    gemm -> BOLTZ_MLX_NAX_GEMM_CFG (per site the candidate with the least ms summed over M = 400 and M = 1065; cond
    over M = 213 and 400), pair -> BOLTZ_MLX_NAX_PAIR_CFG. Empty string = keep the module defaults."""
    rows = K.rows_for_chip([json.loads(x) for x in open(OUT)] if OUT.exists() else [])  # this Mac's rows only
    if kind == "pair":
        r = [x for x in rows if x.get("exp") == "nax3_pair_summary"]
        return r[-1].get("BOLTZ_MLX_NAX_PAIR_CFG", "") if r else ""
    g = [x for x in rows if x.get("exp") == "nax3_gemm" and "ms" in x and x.get("cand") != "mlx_gemm_only"]
    if not g:
        return ""
    g = [x for x in g if x["git"] == g[-1]["git"]]
    out = []
    for site, lst, Ms in (("qkvg", N3.GEMM_CFGS, (400, 1065)), ("o", N3.GEMM_CFGS, (400, 1065)),
                          ("ff", N3.SWIGLU_CFGS, (400, 1065)), ("ba", N3.GEMM_CFGS, (400, 1065)),
                          ("cond", N3.GEMM_CFGS, (213, 400))):
        tot: dict = {}
        for x in g:
            if x["site"] == site and x["M"] in Ms:
                k = json.dumps(x["cfg"], sort_keys=True)
                tot.setdefault(k, {})[x["M"]] = x["ms"]
        full = {k: sum(v.values()) for k, v in tot.items() if len(v) == len(Ms)}
        if not full:
            continue
        cfg = json.loads(min(full, key=full.get))
        out.append(f"{site}={'mlx' if cfg == 'mlx' else lst.index(cfg)}")
    return ",".join(out)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which == "suggest":
        # suggest gemm|pair -> the env-style string; suggest all --write -> boltz_mlx/nax3_cfg.json (commit it: bench_mlx /
        # run_l3 / screen strip BOLTZ_MLX_* variables, so measured site choices must be in the package)
        kind = sys.argv[2] if len(sys.argv) > 2 else "all"
        if kind != "all":
            print(suggest(kind))
            sys.exit(0)
        # this chip's entry of the per-device file (kernels.write_device_cfg): site choices merged item by item into
        # the entry in use (this chip's, else default), so pair choices written by pair_profile.py are kept
        d = K.device_cfg_entry(N3.CFG_FILE)
        for key in ("gemm", "pair"):
            items = dict(x.split("=") for x in filter(None, d.get(key, "").split(",")))
            items.update(dict(x.split("=") for x in filter(None, suggest(key).split(","))))
            d[key] = ",".join(f"{k}={v}" for k, v in items.items())
        d["source"] = f"tools/prof/nax3_ab.py suggest (logs/nax3.jsonl, git {GIT}, {time.strftime('%Y-%m-%d %H:%M')})"
        print(json.dumps(d))
        if "--write" in sys.argv:
            K.write_device_cfg(N3.CFG_FILE, d)
            print(f"wrote the {K.chip_name()!r} entry of {N3.CFG_FILE}")
        sys.exit(0)
    for part, fn in (("gemm", part_gemm), ("pair", part_pair), ("diff", part_diff)):
        if which in (part, "all"):
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                _err({"exp": f"nax3_{part}", "stage": "part"}, e)
