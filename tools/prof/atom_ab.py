"""Atom encoder / decoder A/B (speed-first atom round): kernels_atom vs the shipped --precision f16 path. GPU.

  python tools/prof/atom_ab.py prof     # ~3 min: per-sub-op times of one atom layer, shipped vs kernels_atom, at the
                                        #   lig_aff affinity-pass shape (5 samples x 1536 atoms) and prot400 (3072);
                                        #   every attention / GEMM config; glue ops; whole 3-layer transformer
  python tools/prof/atom_ab.py acc      # ~4 min: encoder / decoder on inputs captured from 10-step f16 eager runs
                                        #   (lig_aff affinity pass x5, prot400 x1) vs fp32; eager + compiled ms
  python tools/prof/atom_ab.py all
  python tools/prof/atom_ab.py suggest [--write] [--require-gate]
                                        # per-site configs from the prof rows -> boltz_mlx/atom_cfg.json; with
                                        #   --require-gate only if this build's acc gate passed for them (exit 1 if not)

BOLTZ_MLX_PRECISION=f16 is set by this script (before importing boltz_mlx); the flag BOLTZ_MLX_NAX_ATOM is switched
per candidate in-process. Models load with the fp32 checkpoint (fp16 weights cast at first use = bitwise the mixed
checkpoint's), so the fp32 reference path sees fp32 weights. Rows -> logs/atom_ab.jsonl.

acc: per capture (diffusion steps 0 / 4 / 9) the encoder's token output a and atom update q - q_in, and the decoder's
r_update, for fp32 (DIFF_DTYPE fp32, the strict build's atom path), the shipped f16 path (flag 0) and the kernels_atom
path (flag 1; module default configs and the prof part's suggestion). Gate (owner brief): candidate error vs fp32 <=
the shipped f16 error, or <= 1.5x it and <= 3e-3 rel. Frobenius and cosine >= 0.99999.
"""

from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("BOLTZ_MLX_PRECISION", "f16")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import numpy as np  # noqa: E402

import boltz_mlx  # noqa: E402,F401
from boltz_mlx import kernels as K  # noqa: E402
from boltz_mlx import kernels_atom as KA  # noqa: E402
from boltz_mlx import kernels_nax3 as N3  # noqa: E402

mx.set_default_device(mx.gpu)
DATA = Path(os.environ.get("BOLTZ_MLX_DATA", Path(__file__).resolve().parents[2] / "data")).expanduser()
OUT = ROOT / "logs" / "atom_ab.jsonl"
GIT = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
SHAPES = (("lig_aff_x5", 1536, 5), ("prot400", 3072, 1))


def emit(row: dict):
    row = {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "git": GIT, "chip": K.chip_name(), **row}
    print(json.dumps(row), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "a") as fo:
        fo.write(json.dumps(row) + "\n")


def bench(f, reps: int = 20, rounds: int = 3) -> float:
    """ms per call: min over rounds of (reps pipelined calls, one eval)."""
    mx.eval(f())
    best = 1e9
    for _ in range(rounds):
        t = time.perf_counter()
        mx.eval([f() for _ in range(reps)])
        best = min(best, (time.perf_counter() - t) / reps * 1e3)
    return round(best, 4)


def bench_sync(f, n: int = 9) -> float:
    """ms per call, each call evaluated on its own; median."""
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


def _flag(v: str | None):
    K.NAX_FLAGS["ATOM"] = v


# =============================================================================================== prof
def _rand_transformer(seed=0):
    from boltz_mlx.modules.transformersv2 import AtomTransformer

    mx.random.seed(seed)
    at = AtomTransformer(attn_window_queries=32, attn_window_keys=128, dim=128, dim_single_cond=128, depth=3, heads=4)
    rng = np.random.default_rng(seed)
    for L in at.diffusion_transformer.layers:  # non-trivial conditioning
        for lin in (L.adaln.s_scale, L.transition.adaln.s_scale, L.output_projection_linear,
                    L.transition.output_projection[0]):
            lin.weight = mx.array(rng.normal(size=lin.weight.shape).astype(np.float32) * 0.05)
    return at


def part_prof():
    """Per-sub-op ms of one atom layer (layer 0) and of the whole 3-layer AtomTransformer, shipped f16 vs kernels_atom."""
    from boltz_mlx.modules.encodersv2 import make_to_keys

    K.set_diff_dtype("fp16")
    at = _rand_transformer()
    dt = at.diffusion_transformer
    layer = dt.layers[0]
    rng = np.random.default_rng(0)
    best = {}
    for name, NA, S in SHAPES:
        M = S * NA
        base = {"exp": "atom_prof", "shape": name, "NA": NA, "S": S, "M": M}
        tk = make_to_keys(NA, 32, 128)
        c = mx.array(rng.normal(size=(1, NA, 128)).astype(np.float32))
        bias = mx.array(rng.normal(size=(1, NA // 32, 32, 128, 12)).astype(np.float32))
        mask = mx.array(np.r_[np.ones(NA - 3), np.zeros(3)].astype(np.float32)[None])
        q = mx.array(rng.normal(size=(S, NA, 128)).astype(np.float32))
        # ---- shipped f16 path (flag 0): prepared biases / 6-array cond over all samples
        _flag("0")
        c_rep, m_rep = mx.repeat(c, S, 0), mx.repeat(mask, S, 0)
        P0 = at.prepare((S, NA, 1), c_rep, bias, tk, m_rep, S)
        mx.eval(P0)
        rows = {}
        rows["shipped_transformer_eager"] = bench_sync(lambda: at(q, c_rep, bias, tk, m_rep, S, prepared=P0))
        f0 = mx.compile(lambda q_: at(q_, c_rep, bias, tk, m_rep, S, prepared=P0))
        rows["shipped_transformer_compiled"] = bench_sync(lambda: f0(q))
        # shipped layer-0 parts (same ops as DiffusionTransformerLayer.fast_attn / fast_transition with to_keys)
        _, _, _, tkn = at._shape(q, c_rep, bias, tk, m_rep, S)
        cond = [x[..., 0, :] for x in P0["cond"]]
        a = q.reshape(S * NA // 32, 32, 128)
        w_qkvg, w_ff, w_o, w_ba, b_q, _ = layer._fused()
        bln = cond[0] * K.layer_norm(a, None, None, 1e-5) + cond[1]
        hq = K.mm_t(bln, w_qkvg)
        kv = mx.concatenate([hq[..., 128:256], hq[..., 256:384]], -1)
        mx.eval(bln, hq, kv)
        rows["shipped_ln_affine"] = bench(lambda: cond[0] * K.layer_norm(a, None, None, 1e-5) + cond[1])
        rows["shipped_qkvg_mm_t"] = bench(lambda: K.mm_t(bln, w_qkvg))
        rows["shipped_tokeys_kv"] = bench(lambda: tkn(kv))
        kk = tkn(kv)
        qh = (hq[..., :128] + b_q).reshape(-1, 32, 4, 32).transpose(0, 2, 1, 3)
        kh = kk[..., :128].reshape(-1, 128, 4, 32).transpose(0, 2, 1, 3)
        vh = kk[..., 128:].reshape(-1, 128, 4, 32).transpose(0, 2, 1, 3)
        mx.eval(qh, kh, vh)
        rows["shipped_sdpa_fp32"] = bench(lambda: mx.fast.scaled_dot_product_attention(qh, kh, vh, scale=32 ** -0.5,
                                                                                         mask=P0["attn_bias"][0]))
        rows["shipped_layer_attn_half"] = bench_sync(lambda: layer.fast_attn(a, cond, tkn, P0["attn_bias"][0]))
        rows["shipped_layer_transition_half"] = bench_sync(lambda: layer.fast_transition(a, cond))
        for k, v in rows.items():
            emit({**base, "op": k, "ms": v})
        del P0, f0, kk, qh, kh, vh
        # ---- kernels_atom (flag 1)
        _flag("1")
        assert at.nax_ok(NA)
        P = at.prepare_nax(c, bias, tk, mask, S)
        rc = P["cond"]
        a2 = q.reshape(M, 128)
        emit({**base, "op": "nax_attn_bias_mb", "mb": round(sum(x.nbytes for x in P["attn_bias"]) / 1e6, 2),
              "shipped_mb": round(S * sum(x.nbytes for x in P["attn_bias"]) / 1e6, 2)})
        rows = {}
        rows["nax_transformer_eager"] = bench_sync(lambda: at(q, None, None, None, None, S, prepared=P))
        f1 = mx.compile(lambda q_: at(q_, None, None, None, None, S, prepared=P))
        rows["nax_transformer_compiled"] = bench_sync(lambda: f1(q))
        rows["nax_adaln16"] = bench(lambda: N3.adaln16(a2, rc, 0, 0))
        t16 = N3.adaln16(a2, rc, 0, 0)
        h = t16 @ w_qkvg.T
        o = KA.atom_attention(h, P["attn_bias"][0], NA=NA, qbias=b_q, brep=S, **KA.ATT_CFG)
        hs = N3.gemm16(t16, w_ff, epi=N3.EPI_SWIGLU, ngrp=3, **N3.SWIGLU_CFGS[2])
        mx.eval(t16, h, o, hs)
        for k, v in rows.items():
            emit({**base, "op": k, "ms": v})
        # attention configs
        for i, cfg in enumerate(KA.ATT_CFGS):
            row = {**base, "op": "nax_atom_attention", "cfg_i": i, "cfg": cfg}
            try:
                row["ms"] = bench(lambda: KA.atom_attention(h, P["attn_bias"][0], NA=NA, qbias=b_q, brep=S, **cfg))
                row["vs_ref"] = metrics(KA.atom_attention(h, P["attn_bias"][0], NA=NA, qbias=b_q, brep=S, **cfg),
                                        KA.atom_attention_ref(h, P["attn_bias"][0], NA=NA, qbias=b_q, brep=S))
                emit(row)
                best.setdefault(("att", name), []).append((row["ms"], i))
            except Exception as e:  # noqa: BLE001
                _err(row, e)
        # GEMM sites: every config + 'mlx' (MLX fp16 GEMM + compiled elementwise epilogue)
        gk = dict(gate=rc.h2, gate_bias=rc.b2, resid=a2, ntok=rc.ntok, rep=rc.rep)
        sites = {"qkvg": (t16, w_qkvg, dict()), "o": (o, w_o, dict(epi=N3.EPI_GRESID, gate_off=0, **gk)),
                 "ff": (t16, w_ff, dict(epi=N3.EPI_SWIGLU, ngrp=3)),
                 "ba": (hs, w_ba, dict(epi=N3.EPI_GRESID, gate_off=128, **gk))}
        for site, (x, w, kw) in sites.items():
            lst = KA._LISTS[site]
            for cand in ["mlx"] + list(range(len(lst))):
                cfg = "mlx" if cand == "mlx" else lst[cand]
                if cfg != "mlx" and (x.shape[1] % (16 * cfg["ku"]) or (w.shape[0] // kw.get("ngrp", 1))
                                     % (cfg["wn"] * cfg["sn"])):
                    continue
                row = {**base, "op": "nax_gemm", "site": site, "cand": cand, "cfg": cfg}
                try:
                    row["ms"] = bench(lambda: N3._gemm_site(site, x, w, cfg=cfg, **kw))
                    emit(row)
                    best.setdefault((site, name), []).append((row["ms"], cand))
                except Exception as e:  # noqa: BLE001
                    _err(row, e)
        # transition half: split kernels (adaln16 -> ff -> ba, module site configs) vs the fused kernel per config
        def split_trans():
            t = N3.adaln16(a2, rc, 0, 2)
            hs_ = N3._gemm_site("ff", t, w_ff, cfg=KA.GEMM_CFG["ff"], epi=N3.EPI_SWIGLU, ngrp=3)
            return N3._gemm_site("ba", hs_, w_ba, cfg=KA.GEMM_CFG["ba"], epi=N3.EPI_GRESID, gate_off=128, **gk)

        # attention-half projection: adaln16 + qkvg GEMM (split) vs the AdaLN-prologue GEMM per config
        hsplit = N3._gemm_site("qkvg", N3.adaln16(a2, rc, 0, 0), w_qkvg, cfg=KA.GEMM_CFG["qkvg"])
        for i, cfg in enumerate(KA.ADALN_GEMM_CFGS):
            row = {**base, "op": "nax_aqkvg", "cand": i, "cfg": cfg}
            try:
                hf = KA.adaln_gemm(a2, rc, 0, 0, w_qkvg, **cfg)
                row["vs_split"] = metrics(hf, hsplit)
                row["vs_ref"] = metrics(hf, KA.adaln_gemm_ref(a2, rc, 0, 0, w_qkvg))
                row["ms"] = bench(lambda: KA.adaln_gemm(a2, rc, 0, 0, w_qkvg, **cfg))
                emit(row)
            except Exception as e:  # noqa: BLE001
                _err(row, e)
        ys = split_trans()
        emit({**base, "op": "nax_trans", "cand": "split", "ms": bench(split_trans)})
        for i, cfg in enumerate(KA.TRANS_CFGS):
            row = {**base, "op": "nax_trans", "cand": i, "cfg": cfg}
            try:
                yf = KA.atom_transition(a2, rc, 0, w_ff, w_ba, **cfg)
                row["update_vs_split"] = metrics(yf - a2, ys - a2)
                row["update_vs_ref"] = metrics(yf - a2, KA.atom_transition_ref(a2, rc, 0, w_ff, w_ba) - a2)
                row["ms"] = bench(lambda: KA.atom_transition(a2, rc, 0, w_ff, w_ba, **cfg))
                emit(row)
            except Exception as e:  # noqa: BLE001
                _err(row, e)
        del P, f1, h, o, hs, t16, ys
        gc.collect()
        mx.clear_cache()
    _flag(None)


def part_glue():
    """Glue ops at the lig_aff affinity-pass shape (5 samples, 1536 atoms, 213 tokens) on the recorded features."""
    z = np.load(DATA / "lig_aff" / "feats_affinity.npz")
    att = z["atom_to_token"][0].astype(np.float32)
    NA, T = att.shape
    S = 5
    rng = np.random.default_rng(2)
    base = {"exp": "atom_glue", "NA": NA, "T": T, "S": S}
    a2t = mx.array(att)[None]
    q = mx.array(rng.normal(size=(S, NA, 128)).astype(np.float32))
    lin = nn.Linear(128, 768, bias=False)
    a_tok = mx.array(rng.normal(size=(S, T, 128)).astype(np.float32))

    def agg_shipped():
        y = nn.relu(lin(q))
        m = mx.repeat(a2t, S, axis=0)
        m = m / (m.sum(axis=1, keepdims=True) + 1e-6)
        return mx.swapaxes(m, 1, 2) @ y

    seg = tuple(mx.array(x) for x in KA.token_segments(att))
    tok, valid = (mx.array(x) for x in KA.atom_tokens(att))

    def agg_nax():
        return KA.segment_mean_relu(lin(q), seg)

    emit({**base, "op": "aggregate", "shipped_ms": bench(agg_shipped), "nax_ms": bench(agg_nax),
          "nax_vs_shipped": metrics(agg_nax(), agg_shipped()),
          "linear_only_ms": bench(lambda: lin(q))})
    bc_s = lambda: mx.repeat(a2t, S, axis=0) @ a_tok  # noqa: E731
    bc_n = lambda: mx.where(valid[None, :, None], mx.take(a_tok, tok, axis=1), 0.0)  # noqa: E731
    emit({**base, "op": "broadcast", "shipped_ms": bench(bc_s), "nax_ms": bench(bc_n),
          "bitwise": bool(mx.array_equal(bc_s(), bc_n()).item())})
    from boltz_mlx.layers.norm import LayerNorm

    ln, lin2 = LayerNorm(768), nn.Linear(768, 768, bias=False)
    s1 = mx.array(rng.normal(size=(1, T, 768)).astype(np.float32))
    s5 = mx.repeat(s1, S, 0)
    emit({**base, "op": "s_to_a", "shipped_ms": bench(lambda: lin2(ln(s5))), "nax_ms": bench(lambda: lin2(ln(s1)))})


# =============================================================================================== acc
def _capture(case: str, aff: bool, steps=(0, 4, 9), n_steps: int = 10):
    """(model, feats, cond, mult, captures {step: (enc kwargs, dec kwargs)}) from a 10-step f16 eager run of the
    shipped path (flag 0)."""
    from boltz_mlx.model import load_model
    from boltz_mlx.modules.diffusionv2 import weighted_rigid_align
    from boltz_mlx.modules.encodersv2 import AtomAttentionDecoder, AtomAttentionEncoder
    from boltz_mlx.sampler import DiffusionParams, record_noise, sample

    _flag("0")
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
    caps, cnt = {}, {"enc": 0, "dec": 0}
    oe, od = AtomAttentionEncoder.__call__, AtomAttentionDecoder.__call__

    def we(self, *a, **kw):
        if cnt["enc"] in steps:
            mx.eval(kw["r"])
            caps.setdefault(cnt["enc"], {})["enc"] = {"r": kw["r"]}
        cnt["enc"] += 1
        return oe(self, *a, **kw)

    def wd(self, *a, **kw):
        if cnt["dec"] in steps:
            mx.eval(kw["a"], kw["q"])
            caps.setdefault(cnt["dec"], {})["dec"] = {"a": kw["a"], "q": kw["q"]}
        cnt["dec"] += 1
        return od(self, *a, **kw)

    AtomAttentionEncoder.__call__, AtomAttentionDecoder.__call__ = we, wd
    try:
        def denoise(x, t_hat):
            return sm.preconditioned_network_forward(x, mx.full((x.shape[0],), t_hat, dtype=mx.float32), dict(
                s_inputs=s_inputs, s_trunk=s, feats=feats, diffusion_conditioning=cond, multiplicity=mult))

        out = sample(denoise, feats["atom_pad_mask"].astype(mx.float32), n_steps, mult,
                     record_noise(0, case, "affinity" if aff else "structure", mult), weighted_rigid_align,
                     DiffusionParams())
        mx.eval(out)
    finally:
        AtomAttentionEncoder.__call__, AtomAttentionDecoder.__call__ = oe, od
    return model, feats, cond, mult, caps


def _set_mode(mode: str, default_cfg: dict):
    """fp32 (DIFF_DTYPE fp32: the strict build's atom path) | f16 (shipped) | nax* (kernels_atom, cfg variant)."""
    K.set_diff_dtype("fp32" if mode == "fp32" else "fp16")
    _flag("1" if mode.startswith("nax") else "0")
    KA.GEMM_CFG.update(default_cfg["gemm"])
    KA.ATT_CFG.clear()
    KA.ATT_CFG.update(default_cfg["att"])
    if mode == "nax_suggest":
        sg = suggest()
        if sg:
            KA.apply_cfg(sg)


def part_acc():
    default = {"gemm": dict(KA.GEMM_CFG), "att": dict(KA.ATT_CFG)}
    modes = ["fp32", "f16", "nax"] + (["nax_suggest"] if suggest() else [])
    verdict = []
    for case, aff in (("lig_aff", True), ("prot400", False)):
        base0 = {"exp": "atom_acc", "case": case + ("_aff" if aff else "")}
        try:
            model, feats, cond, mult, caps = _capture(case, aff)
        except Exception as e:  # noqa: BLE001
            _err({**base0, "stage": "capture"}, e)
            continue
        sc = model.structure_module.score_model
        enc, dec = sc.atom_attention_encoder, sc.atom_attention_decoder
        q0, c, eb, db, tk = cond["q"], cond["c"], cond["atom_enc_bias"], cond["atom_dec_bias"], cond["to_keys"]
        for step in sorted(caps):
            cp = caps[step]
            base = {**base0, "step": step, "mult": mult, "n_atoms": int(q0.shape[1])}
            res = {}
            for mode in modes:
                row = {**base, "cand": mode}
                try:
                    _set_mode(mode, default)
                    Pe = enc.prepare(feats, c, eb, tk, mult)
                    Pd = dec.prepare(feats, c, db, tk, mult)
                    assert bool(Pe.get("nax_atom")) == mode.startswith("nax"), "path selection"
                    c_dec = c if mode.startswith("nax") else (mx.repeat(c, mult, 0) if mult > 1 else c)

                    def fe(r_, Pe=Pe):
                        return enc(feats=feats, q=q0, c=c, atom_enc_bias=eb, to_keys=tk, r=r_, multiplicity=mult,
                                   prepared=Pe)[:2]

                    def fd(a_, q_, Pd=Pd, c_dec=c_dec):
                        return dec(a=a_, q=q_, c=c_dec, atom_dec_bias=db, feats=feats, to_keys=tk, multiplicity=mult,
                                   prepared=Pd)

                    a_o, q_o = fe(cp["enc"]["r"])
                    r_o = fd(cp["dec"]["a"], cp["dec"]["q"])
                    mx.eval(a_o, q_o, r_o)
                    res[mode] = (a_o, q_o - q0, r_o)
                    if step == min(caps):
                        row["enc_eager_ms"] = bench_sync(lambda: fe(cp["enc"]["r"]))
                        row["dec_eager_ms"] = bench_sync(lambda: fd(cp["dec"]["a"], cp["dec"]["q"]))
                        cfe, cfd = mx.compile(fe), mx.compile(fd)
                        row["enc_compiled_ms"] = bench_sync(lambda: cfe(cp["enc"]["r"]))
                        row["dec_compiled_ms"] = bench_sync(lambda: cfd(cp["dec"]["a"], cp["dec"]["q"]))
                        if mode.startswith("nax"):
                            row["compiled_eq_eager"] = bool(mx.array_equal(cfe(cp["enc"]["r"])[0], a_o).item()
                                                            and mx.array_equal(cfd(cp["dec"]["a"], cp["dec"]["q"]),
                                                                               r_o).item())
                        del cfe, cfd
                    if mode != "fp32" and "fp32" in res:
                        ref = res["fp32"]
                        row["enc_a_vs_fp32"] = metrics(a_o, ref[0])
                        row["enc_q_update_vs_fp32"] = metrics(q_o - q0, ref[1])
                        row["dec_vs_fp32"] = metrics(r_o, ref[2])
                    emit(row)
                except Exception as e:  # noqa: BLE001
                    _err(row, e)
            if "f16" in res and "fp32" in res:
                ref = res["fp32"]
                ship = [metrics(res["f16"][i], ref[i]) for i in range(3)]
                for mode in res:
                    if not mode.startswith("nax"):
                        continue
                    cand = [metrics(res[mode][i], ref[i]) for i in range(3)]
                    g = {k: gate(cm, sm_) for k, cm, sm_ in zip(("enc_a", "enc_q", "dec"), cand, ship)}
                    verdict.append((base["case"], step, mode, g))
                    emit({**base, "exp": "atom_acc_gate", "cand": mode, **g,
                          "rel_fro": {k: [round(cm["rel_fro"], 7), round(sm_["rel_fro"], 7)]
                                      for k, cm, sm_ in zip(("enc_a", "enc_q", "dec"), cand, ship)}})
            mx.clear_cache()
        _set_mode("f16", default)
        _flag(None)
        del model, caps, cond, feats
        gc.collect()
        mx.clear_cache()
    emit({"exp": "atom_acc_summary", "n": len(verdict),
          "gate_pass_by_cand": {m: all(all(g.values()) for c, s_, mm, g in verdict if mm == m)
                                for m in sorted({v[2] for v in verdict})}})


# =============================================================================================== suggest
def suggest() -> str:
    """BOLTZ_MLX_ATOM_CFG string from the newest build's prof rows: per GEMM site the candidate with the least ms
    summed over both shapes, and the attention config likewise. Empty = keep the module defaults."""
    rows = K.rows_for_chip([json.loads(x) for x in open(OUT)] if OUT.exists() else [])  # this Mac's rows only
    rows = [r for r in rows if r.get("exp") == "atom_prof" and "ms" in r]
    if not rows:
        return ""
    rows = [r for r in rows if r["git"] == rows[-1]["git"]]
    out = []
    site_best = {}
    for site in ("qkvg", "o", "ff", "ba"):
        tot: dict = {}
        for r in rows:
            if r.get("op") == "nax_gemm" and r["site"] == site:
                tot.setdefault(str(r["cand"]), {})[r["shape"]] = r["ms"]
        full = {k: sum(v.values()) for k, v in tot.items() if len(v) == len(SHAPES)}
        if full:
            out.append(f"{site}={min(full, key=full.get)}")
            site_best[site] = min(full.values())
    # transition: fused kernel (accurate configs only) vs split = adaln16 + best ff + best ba (summed over shapes)
    ad = sum(r["ms"] for r in rows if r.get("op") == "nax_adaln16")
    tot = {}
    for r in rows:
        if r.get("op") == "nax_trans" and r.get("cand") != "split" and \
                r.get("update_vs_split", {}).get("rel_fro", 1) <= 2e-3:
            tot.setdefault(r["cand"], {})[r["shape"]] = r["ms"]
    full = {k: sum(v.values()) for k, v in tot.items() if len(v) == len(SHAPES)}
    if full and "ff" in site_best and "ba" in site_best:
        k = min(full, key=full.get)
        out.append(f"trans={k}" if full[k] < ad + site_best["ff"] + site_best["ba"] else "trans=split")
    # attention-half projection: AdaLN-prologue GEMM vs adaln16 + best qkvg
    tot = {}
    for r in rows:
        if r.get("op") == "nax_aqkvg" and r.get("vs_ref", {}).get("rel_fro", 1) <= 2e-3:
            tot.setdefault(r["cand"], {})[r["shape"]] = r["ms"]
    full = {k: sum(v.values()) for k, v in tot.items() if len(v) == len(SHAPES)}
    if full and "qkvg" in site_best:
        k = min(full, key=full.get)
        out.append(f"aqkvg={k}" if full[k] < ad + site_best["qkvg"] else "aqkvg=split")
    tot = {}
    for r in rows:
        if r.get("op") == "nax_atom_attention" and r.get("vs_ref", {}).get("rel_fro", 1) <= 2e-3:
            tot.setdefault(r["cfg_i"], {})[r["shape"]] = r["ms"]
    full = {k: sum(v.values()) for k, v in tot.items() if len(v) == len(SHAPES)}
    if full:
        out.append(f"att={min(full, key=full.get)}")
    return ",".join(out)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which == "suggest":
        sg = suggest()
        d = {"atom": sg, "source": f"tools/prof/atom_ab.py suggest (logs/atom_ab.jsonl, git {GIT}, "
                                   f"{time.strftime('%Y-%m-%d %H:%M')})"}
        print(json.dumps(d))
        if "--require-gate" in sys.argv:
            # write only if the acc part of this build passed the gate for the suggested configs (cand nax_suggest)
            rows = [json.loads(x) for x in open(OUT)] if OUT.exists() else []
            summ = [r for r in K.rows_for_chip(rows) if r.get("exp") == "atom_acc_summary" and r.get("git") == GIT]
            ok = bool(summ) and summ[-1]["gate_pass_by_cand"].get("nax_suggest", False)
            print(f"acc gate for the suggested configs: {'PASS' if ok else 'FAIL / missing'}")
            if not ok:
                sys.exit(1)
        if "--write" in sys.argv and sg:
            K.write_device_cfg(KA.CFG_FILE, d)  # this chip's entry of the per-device file
            print(f"wrote the {K.chip_name()!r} entry of {KA.CFG_FILE}")
        sys.exit(0)
    for part, fn in (("prof", part_prof), ("glue", part_glue), ("acc", part_acc)):
        if which in (part, "all") or (which == "prof" and part == "glue"):
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                _err({"exp": f"atom_{part}", "stage": "part"}, e)
