"""Per-chip retune sweep (boltz-opt: O2 per-chip retune, O3 small sizes). GPU (timed: class 1).

  python tools/prof/chip_sweep.py [gemm] [attn] [pair] [msa] [--quick]   # default: all parts; --quick = smoke
                                                                         #   (exit 1 on any error row)
  python tools/prof/chip_sweep.py report [--git G]                       # CPU: tables from logs/chip_sweep.jsonl
  python tools/prof/chip_sweep.py suggest [--git G] [--chip C] [--json]  # CPU: per site and size the fastest
                                                                         #   bitwise-equal candidate (or keep shipped)
  python tools/prof/chip_sweep.py recheck [gemm] [attn] [pair]           # GPU: this chip's per-device entry vs the
                                                                         #   "default" entry at every swept site /
                                                                         #   size where they differ, interleaved
  python tools/prof/chip_sweep.py recheck_report [--probe]               # CPU: table of the latest recheck (--probe:
                                                                         #   of the latest probe recheck)
  CHIP_SWEEP_PROBE='{"pair": "tail3@300=9"}' python tools/prof/chip_sweep.py recheck pair
      # info only: B = this chip's entry + the probe's items appended (gemm / pair / kern / atom spec strings), to time
      # a bucket candidate at sizes the entry does not give it; rows carry "probe": true. The config files are unchanged.
Recheck mode adds sizes the full sweep does not have (RECHECK_EXTRA: single-sample 213 / 256 token GEMMs, pproj and the
pair captures at N 213 / 323 / 397 / 586, T400's 3008 padded atoms) and exits 1 on any error row.

For every per-chip kernel site, every candidate launch configuration that keeps the per-element arithmetic of the
shipped one (same k order, same reductions; only the work split across simdgroups / threadgroups / the grid changes)
is run at the shapes of the real cases (token counts of prot117, tyk2_ejm_31, msa384, lig_aff, prot400, ex_ligand,
cdk2_17 and the lig_aff affinity pass) and its output is compared BITWISE with the shipped configuration's output on
the same input (`bitwise`). Only bitwise-equal candidates are eligible for the per-chip config files; a candidate that
is not bitwise equal is reported, never chosen.

Timings per candidate (ms per call):
  tp     pipelined throughput: reps independent calls, one eval, min of 3 rounds (the tools/prof/* convention)
  chain  dependent chain: each call reads the previous call's output (GEMM sites: the residual of the gated-residual
         epilogue, else the output reinterpreted as the next input), one eval of 16 calls, median of 5 -> latency of
         a kernel whose predecessor must finish first, as inside the model (low-occupancy launches show up here)
  sync   one call per eval, median of 25 (includes a constant host round trip; for kernels without a chain)
Parts:
  gemm  kernels_nax3.gemm16 sites: token transformer o / ff / ba (M = B * N) and conditioning (M = tokens), atom
        transformer o / ff / ba (M = samples * padded atoms), tri-att q|k|v|g projection pproj (M = N^2); candidates
        = a grid over (sm, sn, wm, wn, ku) (k ascending in 16-steps for every one); the shipped config is the site's
        entry of this chip's config (else default).
  attn  diffusion token attention (kernels_nax2.attention_nax: wq, order), triangle attention core
        (kernels_nax.tri_attention_nax: wq, hpt, order), atom window attention (kernels_atom.atom_attention: hpt,
        order) — query blocks / heads per threadgroup and grid order; nkf (key blocking of the online softmax) fixed.
  pair  on the recorded L1 PairformerLayer captures (layer 0 of prot117, prot400, msa384): tri-mult tail
        (TAIL3_CFGS), pair transition (register kernel nsg / keepa / unrk / maxt), tri-att linear_o + add + LN
        (linear_addln tm / nsg), tri-mult projection (trimul_proj_nax tp / tc / nsg / xreg).
  msa   on the msa384 / ex_ligand L1 MSA captures: OPM with the NAX proj_o epilogue (OPM_FUSED_CFGS), PWA average
        (PWA_CFGS), MSA transition (kernels_nax2.TRANS_CFGS) — module outputs bitwise vs shipped, core ms.
BOLTZ_MLX_PRECISION=f16 is set here before boltz_mlx is imported. Rows -> logs/chip_sweep.jsonl.
"""

from __future__ import annotations

import gc
import itertools
import json
import os
import pickle
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # this checkout's boltz_mlx first: the editable install maps boltz_mlx to the main checkout
OUT = ROOT / "logs" / "chip_sweep.jsonl"


def sh(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


GIT = sh(f"git -C {ROOT} rev-parse --short HEAD")
PKG = sh(f"git -C {ROOT} rev-parse --short HEAD:boltz_mlx")
QUICK = "--quick" in sys.argv

# real shapes on this chip (tokens; padded atoms = ceil(atoms / 32) * 32; affinity pass: 5 samples x 213 tokens)
TOKEN_M = {"prot117": 117, "tyk2": 323, "msa384": 384, "lig_aff": 397, "prot400": 400, "ex_ligand": 423,
           "cdk2": 586, "aff": 1065}
COND_M = {"prot117": 117, "aff": 213, "tyk2": 323, "lig_aff": 397, "prot400": 400, "cdk2": 586}
ATOM_M = {"prot117": (928, 1), "msa384": (2976, 1), "prot400": (3072, 1), "cdk2": (4544, 1), "aff": (1536, 5)}
PPROJ_N = {"prot117": 117, "msa384": 384, "prot400": 400}
DIFFATT = {"prot117": (1, 117), "tyk2": (1, 323), "msa384": (1, 384), "prot400": (1, 400), "cdk2": (1, 586),
           "aff": (5, 213)}
TRIATT_N = {"prot117": 117, "aff": 213, "tyk2": 323, "msa384": 384, "prot400": 400, "cdk2": 586}
if QUICK:
    TOKEN_M = {"prot117": 117, "aff": 1065}
    COND_M = {"prot117": 117}
    ATOM_M = {"prot117": (928, 1)}
    PPROJ_N = {"prot117": 117}
    DIFFATT = {"prot117": (1, 117), "aff": (5, 213)}
    TRIATT_N = {"prot117": 117}
PAIR_CASES = ["prot117"] if QUICK else ["prot117", "prot400", "msa384"]
# recheck only: more sizes (L1 pair captures lig_aff_aff = the lig_aff affinity trunk N 213, tyk2_ejm_31 323, lig_aff
# 397, cdk2_17 586; T400 complexes ~2985 atoms -> 3008 padded)
RECHECK_EXTRA = {"TOKEN_M": {"m213": 213, "m256": 256}, "PPROJ_N": {"aff": 213, "tyk2": 323, "lig_aff": 397, "cdk2": 586},
                 "ATOM_M": {"t400": (3008, 1)}, "PAIR_CASES": ["lig_aff_aff", "tyk2_ejm_31", "lig_aff", "cdk2_17"]}
PROBE: dict = {}   # recheck: spec items appended to this chip's entry for arm B (CHIP_SWEEP_PROBE; info only)


ERRORS = []


def emit(row: dict) -> None:
    import boltz_mlx.kernels as K

    if "error" in row:
        ERRORS.append(row.get("site", row.get("part")))

    row = {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "git": GIT, "pkg_tree": PKG, "chip": K.chip_name(), **row}
    print(json.dumps(row), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "a") as fo:
        fo.write(json.dumps(row) + "\n")


# ------------------------------------------------------------------------------------------------ timing helpers
def _mx():
    import mlx.core as mx

    return mx


def t_tp(f, reps: int = 10, rounds: int = 3) -> float:
    mx = _mx()
    mx.eval(f())
    best = 1e9
    for _ in range(rounds):
        t = time.perf_counter()
        mx.eval([f() for _ in range(reps)])
        best = min(best, (time.perf_counter() - t) / reps * 1e3)
    return round(best, 5)


def t_chain(step, x0, n: int = 16, rounds: int = 5) -> float:
    """step(x) -> next x (depends on x); ms per step: median over rounds of one eval of n chained steps."""
    mx = _mx()

    def run():
        x = x0
        for _ in range(n):
            x = step(x)
        return x

    mx.eval(run())
    ts = []
    for _ in range(rounds):
        t = time.perf_counter()
        mx.eval(run())
        ts.append((time.perf_counter() - t) / n * 1e3)
    return round(statistics.median(ts), 5)


def t_sync(f, n: int = 25) -> float:
    mx = _mx()
    mx.eval(f())
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        mx.eval(f())
        ts.append((time.perf_counter() - t) * 1e3)
    return round(statistics.median(ts), 5)


def equal(a, b) -> bool:
    mx = _mx()
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    if a is None or b is None:
        return a is None and b is None
    return a.shape == b.shape and a.dtype == b.dtype and bool(mx.array_equal(a, b).item())


def cfg_key(c) -> str:
    return json.dumps(c, sort_keys=True)


RECHECK = None   # `recheck` mode: (part, site, size) -> (default entry's cfg, this chip's cfg) | None (_recheck_pick)


def _recheck(base: dict, fn, chain, sync: bool, tp: bool, reps: int, rounds: int = 6) -> None:
    """The default entry's config (A) vs this chip's (B) at one site / size, timed interleaved A B B A A B ... (each
    sample a full t_* measurement): the sweep times candidates one after another, so a slow drift (clocks, power,
    other load) can favour whichever ran first; here both arms see the same drift. Rows exp chip_sweep_recheck."""
    site = SPLIT_SITES.get((base["site"], base.get("sub", "")), base["site"])
    size = _size(base)
    picks = RECHECK(base["part"], site, size)
    if picks is None or cfg_key(picks[0]) == cfg_key(picks[1]):
        return
    mx = _mx()
    a, b = picks
    ref, out = fn(a), fn(b)
    mx.eval(ref, out)
    bw = equal(out, ref)
    del ref, out
    samples: dict = {"A": {}, "B": {}}
    for i in range(rounds):
        for arm, c in ((("A", a), ("B", b)) if i % 2 == 0 else (("B", b), ("A", a))):
            sm = samples[arm]
            if tp:
                sm.setdefault("tp_ms", []).append(t_tp(lambda: fn(c), reps))
            if chain is not None:
                sm.setdefault("chain_ms", []).append(t_chain(chain[0](c), chain[1]))
            if sync:
                sm.setdefault("sync_ms", []).append(t_sync(lambda: fn(c)))
    for arm, c in (("A", a), ("B", b)):
        row = {**base, "exp": "chip_sweep_recheck", "arm": arm, "cand": c, "bitwise": bw, "site_key": site,
               "size": size, "probe": bool(PROBE)}
        for m, v in samples[arm].items():
            row[m] = round(statistics.median(v), 5)
            row[m + "_all"] = v
        emit(row)
    mx.clear_cache()


def _recheck_pick():
    """(part, site, size) -> (default cfg, chip cfg) from the per-device files (kernels_nax3 / kernels_atom /
    kernels.resolve_*), for the chip entry this process selects (kernels.cfg_chip)."""
    from boltz_mlx import kernels as K
    from boltz_mlx import kernels_atom as KA
    from boltz_mlx import kernels_nax3 as N3

    nd, ad = json.load(open(N3.CFG_FILE)), json.load(open(KA.CFG_FILE))
    chip = K.cfg_chip()
    assert chip in nd or chip in ad, f"recheck: no per-device entry for {chip!r}"
    ents = ("default", chip)

    def spec(d: dict, e: str, key: str) -> str:   # entry e's spec string for key; arm B (the chip) + the probe items
        s = d.get(e, d["default"]).get(key, "")
        return ",".join(filter(None, (s, PROBE.get(key, "")))) if e == chip else s

    n3 = [N3.resolve_entry({k: spec(nd, e, k) for k in ("gemm", "pair")}) for e in ents]
    ka = [KA.resolve_entry(spec(ad, e, "atom")) for e in ents]
    kk = [K.resolve_kern(spec(nd, e, "kern")) for e in ents]

    def pick(part, site, size):
        if size is None:
            return None
        if part in ("gemm_tok", "gemm_cond", "gemm_pair") or (part == "pair" and site in ("tail3", "trans3", "lin",
                                                                                            "lin_end")):
            f = n3
        elif part == "gemm_atom":
            f = ka
        elif part == "attn" and site in ("triatt", "diffattn"):
            f = kk
        elif part == "attn" and site == "atomattn":
            f, site = ka, "att"
        else:
            return None
        return f[0](site, size), f[1](site, size)

    return pick


def run_cands(base: dict, shipped, cands, fn, chain=None, sync: bool = False, tp: bool = True, reps: int = 10):
    """fn(cfg) -> output(s); chain: (step_factory(cfg) -> step, x0) or None. Emits one row per candidate."""
    if RECHECK is not None:
        return _recheck(base, fn, chain, sync, tp, reps)
    mx = _mx()
    ref = fn(shipped)
    mx.eval(ref)
    seen = set()
    for c in [shipped] + [c for c in cands if cfg_key(c) != cfg_key(shipped)]:
        k = cfg_key(c)
        if k in seen:
            continue
        seen.add(k)
        row = {**base, "cand": c, "shipped": k == cfg_key(shipped)}
        try:
            out = fn(c)
            mx.eval(out)
            row["bitwise"] = equal(out, ref)
            del out
            if tp:
                row["tp_ms"] = t_tp(lambda: fn(c), reps)
            if chain is not None:
                factory, x0 = chain
                row["chain_ms"] = t_chain(factory(c), x0)
            if sync:
                row["sync_ms"] = t_sync(lambda: fn(c))
        except Exception as e:  # noqa: BLE001  a config that fails to build / run is reported, the sweep goes on
            row["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        emit(row)
    del ref
    mx.clear_cache()


# ------------------------------------------------------------------------------------------------ gemm
def gemm_grid(K: int, NOUT: int, ngrp: int, M: int) -> list:
    out = []
    shapes = [(16, 32), (32, 32)] if ngrp == 3 else [(16, 32), (16, 64), (32, 32), (32, 64), (64, 32)]
    for (sm, sn), (wm, wn), ku in itertools.product(
            shapes, [(1, 1), (1, 2), (2, 1), (2, 2), (1, 4), (4, 1), (2, 4), (4, 2)], (1, 2, 4)):
        if ku == 4 and sm != 16:
            continue
        if ngrp == 3 and ku == 4:
            continue
        if K % (16 * ku) or NOUT % (wn * sn) or wm * sm > 4 * max(M, 16):
            continue
        out.append(dict(sm=sm, sn=sn, wm=wm, wn=wn, ku=ku))
    if QUICK:
        out = out[:3]
    return out


def part_gemm():
    import numpy as np

    mx = _mx()
    from boltz_mlx import kernels_atom as KA
    from boltz_mlx import kernels_nax3 as N3

    rng = np.random.default_rng(0)

    def h(*s, sc=1.0):
        return mx.array((rng.normal(size=s) * sc).astype(np.float16))

    def f32(*s):
        return mx.array(rng.normal(size=s).astype(np.float32))

    def gemm_site(part, site, case, M, Kd, NOUT, epi, ngrp, shipped, gate_w, ntok, rep, gate_off, chain_ok=True):
        x, w = h(M, Kd), h(ngrp * NOUT, Kd, sc=Kd ** -0.5)
        kw = {}
        if epi == N3.EPI_GRESID:
            nc = -(-M // (ntok * rep)) * ntok
            kw = dict(gate=h(nc, gate_w), gate_bias=f32(gate_w), gate_off=gate_off, resid=f32(M, NOUT), ntok=ntok,
                      rep=rep)
        mx.eval(x, w, *[v for v in kw.values() if isinstance(v, mx.array)])
        base = {"exp": "chip_sweep", "part": part, "site": site, "case": case, "M": M, "K": Kd, "N": NOUT * ngrp,
                "epi": epi}

        def fn(c):
            return N3.gemm16(x, w, epi=epi, ngrp=ngrp, **kw, **c)

        if epi == N3.EPI_GRESID:
            def factory(c):
                return lambda r: N3.gemm16(x, w, epi=epi, ngrp=ngrp, **{**kw, "resid": r}, **c)
            chain = (factory, kw["resid"])
        else:
            rows_per = (ngrp * NOUT if epi == N3.EPI_STORE else NOUT) // Kd  # output reinterpreted as the next input

            def factory(c):
                def step(xx):
                    y = N3.gemm16(xx, w, epi=epi, ngrp=ngrp, **c)
                    return y.reshape(-1, Kd)[:M] if rows_per >= 1 else xx
                return step
            chain = (factory, x) if rows_per >= 1 and chain_ok else None
        reps = max(2, min(20, int(4e8 // (M * NOUT * ngrp * 2))))
        run_cands(base, shipped, gemm_grid(Kd, NOUT, ngrp, M), fn, chain=chain, reps=reps)
        del x, w, kw
        mx.clear_cache()

    D = 768
    tok = N3.DIFF_GEMM_CFG
    for case, M in TOKEN_M.items():
        ntok, rep = (213, 5) if case == "aff" else (M, 1)
        for site, Kd, NOUT, epi, ngrp in (("o", D, D, N3.EPI_GRESID, 1), ("ff", D, 2 * D, N3.EPI_SWIGLU, 3),
                                          ("ba", 2 * D, D, N3.EPI_GRESID, 1)):
            if tok[site] == "mlx":
                continue  # 'mlx' vs a NAX config is not bitwise: nothing to tune within
            gemm_site("gemm_tok", site, case, M, Kd, NOUT, epi, ngrp, tok[site], 48 * D, ntok, rep, 5 * D)
    if tok["cond"] != "mlx":
        for case, M in COND_M.items():
            for NOUT in (24 * 4 * D, 24 * 2 * D):
                gemm_site("gemm_cond", "cond", case, M, D, NOUT, N3.EPI_STORE, 1, tok["cond"], 0, M, 1, 0,
                          chain_ok=False)  # >= 576 threadgroups: throughput is the latency
    D = 128
    for case, (NA, S) in ATOM_M.items():
        M = NA * S
        for site, Kd, NOUT, epi, ngrp in (("o", D, D, N3.EPI_GRESID, 1), ("ff", D, 2 * D, N3.EPI_SWIGLU, 3),
                                          ("ba", 2 * D, D, N3.EPI_GRESID, 1)):
            sc = KA.GEMM_CFG[site]
            if sc == "mlx" or (site in ("ff", "ba") and KA.GEMM_CFG.get("trans", "split") != "split"):
                continue
            gemm_site("gemm_atom", site, case, M, Kd, NOUT, epi, ngrp, sc, 6 * D, NA, S, 1 * D)
    # atom AdaLN-prologue q|k|v|g GEMM (kernels_atom.adaln_gemm: TM rows per simdgroup, NSG simdgroups)
    aq = KA.GEMM_CFG.get("aqkvg", "split")
    if aq != "split":
        for case, (NA, S) in ATOM_M.items():
            M = NA * S
            rc = N3.RawCond16(h(NA, 3 * 4 * 128, sc=0.5), f32(3 * 4 * 128), h(NA, 3 * 2 * 128, sc=0.5), f32(768),
                              ntok=NA, L=3, D=128, rep=S)
            a2, wq = f32(M, 128), h(512, 128, sc=128 ** -0.5)
            mx.eval(rc.h1, rc.b1, rc.h2, rc.b2, a2, wq)
            cands = [dict(tm=tm, nsg=ns) for tm in (16, 32) for ns in (1, 2, 4, 8, 16) if tm * ns <= 512]
            run_cands({"exp": "chip_sweep", "part": "gemm_atom", "site": "aqkvg", "case": case, "M": M}, aq,
                      cands[:3] if QUICK else cands, lambda c: KA.adaln_gemm(a2, rc, 1, 0, wq, **c), sync=True)
            del rc, a2, wq
    pp = N3.PAIR_SITE_CFG["pproj"]
    for case, N in PPROJ_N.items():
        gemm_site("gemm_pair", "pproj", case, N * N, 128, 512, N3.EPI_STORE, 1, pp, 0, N * N, 1, 0, chain_ok=False)


# ------------------------------------------------------------------------------------------------ attn
def part_attn():
    import numpy as np

    mx = _mx()
    from boltz_mlx import kernels as K
    from boltz_mlx import kernels_atom as KA
    from boltz_mlx import kernels_nax as KN
    from boltz_mlx import kernels_nax2 as KN2

    rng = np.random.default_rng(1)
    # diffusion token attention: 16 heads x 48, h = q|k|v|g [B*N, 3072] fp16, bias [B / brep, 16, N, N] fp32
    sh0 = dict(K.NAX_DIFFATTN_CFG)
    for case, (B, N) in DIFFATT.items():
        brep = B
        hh = mx.array((rng.normal(size=(B * N, 4 * 768)) * 0.5).astype(np.float16))
        bias = mx.array(rng.normal(size=(B // brep, 16, N, N)).astype(np.float32))
        qb = mx.array(rng.normal(size=768).astype(np.float32) * 0.1)
        mx.eval(hh, bias, qb)
        base = {"exp": "chip_sweep", "part": "attn", "site": "diffattn", "case": case, "B": B, "N": N}
        cands = [dict(sh0, wq=wq, order=o) for wq in (1, 2, 3, 4, 5, 6, 8) for o in (0, 1)]
        run_cands(base, sh0, cands[:3] if QUICK else cands,
                  lambda c: KN2.attention_nax(hh, bias, B=B, N=N, heads=16, c=48, qbias=qb, brep=brep, **c),
                  sync=True)
        del hh, bias
    # triangle attention core: 4 heads x 32, h [1, N, N, 512+] fp16, bias [1, 4, N, N] fp16
    sh1 = dict(K.TRIATT_NAX_CFG)
    for case, N in TRIATT_N.items():
        hh = mx.array((rng.normal(size=(1, N, N, 512)) * 0.5).astype(np.float16))
        mask = mx.array((rng.random(size=(1, N, N)) > 0.02).astype(np.float32))
        tb = mx.array((rng.normal(size=(1, 4, N, N))).astype(np.float16))
        mx.eval(hh, mask, tb)
        prep = KN.nax_tri_inputs(hh, 4, 32, mask, 1e9, tb, kp_mult=64, bias_dtype=mx.float16)
        mx.eval(prep)
        base = {"exp": "chip_sweep", "part": "attn", "site": "triatt", "case": case, "N": N}
        cands = [dict(sh1, wq=wq, hpt=hp, order=o) for wq in (1, 2, 3, 4, 5, 6, 8) for hp in (1, 2, 4)
                 for o in (0, 1) if wq * hp <= 16]
        run_cands(base, dict(sh1), cands[:3] if QUICK else cands,
                  lambda c: KN.tri_attention_nax(hh, 4, 32, mask, 1e9, bias=tb, prepared=prep, **c), sync=True,
                  reps=4)
        del hh, mask, tb, prep
        mx.clear_cache()
    # atom window attention: 4 heads x 32, windows of 32 queries x 128 keys, h [M, 512] fp16, bias per (sample) window
    sh2 = dict(KA.ATT_CFG)
    for case, (NA, S) in ATOM_M.items():
        M = NA * S
        hh = mx.array((rng.normal(size=(M, 512)) * 0.5).astype(np.float16))
        bias = mx.array(rng.normal(size=(NA // 32, 4, 32, 128)).astype(np.float32))
        qb = mx.array(rng.normal(size=128).astype(np.float32) * 0.1)
        mx.eval(hh, bias, qb)
        base = {"exp": "chip_sweep", "part": "attn", "site": "atomattn", "case": case, "M": M}
        cands = [dict(sh2, hpt=hp, order=o) for hp in (1, 2, 4) for o in (0, 1)]
        run_cands(base, sh2, cands[:3] if QUICK else cands,
                  lambda c: KA.atom_attention(hh, bias, NA=NA, qbias=qb, brep=S, **c), sync=True)
        del hh, bias
        mx.clear_cache()


# ------------------------------------------------------------------------------------------------ pair
def _l1():
    sys.path.insert(0, str(ROOT / "tools" / "validation"))
    import l1_ops

    return l1_ops


L1DIR = Path(os.environ.get("BOLTZ_MLX_DATA", Path(__file__).resolve().parents[2] / "data")).expanduser() / "cache" / "l1"


def _load(case: str, phase: str, cls: str, call: int):
    l1 = _l1()
    f = L1DIR / case / f"{phase}__{cls}__{call}.pkl"
    rec = pickle.load(open(f, "rb"))
    return rec, l1._from_np(rec["args"]), l1._from_np(rec["kwargs"])


def part_pair(model):
    mx = _mx()
    from boltz_mlx import kernels as K
    from boltz_mlx import kernels_nax2 as KN2
    from boltz_mlx import kernels_nax3 as N3

    mods = dict(model.named_modules())
    for case in PAIR_CASES:
        try:
            rec, args, kwargs = _load(case, "trunk.pairformer", "PairformerLayer", 0)
        except FileNotFoundError:
            emit({"exp": "chip_sweep", "part": "pair", "case": case, "error": "missing capture"})
            continue
        layer = mods[rec["path"]]
        z, pm = args[1], (args[3] if len(args) > 3 else kwargs["pair_mask"])
        _, N, _, D = z.shape
        tmo, tmi, ts, te, tr = (layer.tri_mul_out, layer.tri_mul_in, layer.tri_att_start, layer.tri_att_end,
                                layer.transition_z)
        # inputs of each site along the shipped path
        xn0 = tmo.norm_in.f16(z)
        z1, xn1 = tmo.f16_addln(z, pm, xn0, tmi.norm_in)
        mx.eval(xn0, z1, xn1)
        base0 = {"exp": "chip_sweep", "part": "pair", "case": case, "N": N}
        # tri-mult projection (both tri-mults share the kernel; outgoing layer-0 input)
        xf = xn0.reshape(N * N, -1)
        mf = pm.reshape(-1).astype(mx.float32)
        wh = K.w16(tmo, "_w_th", tmo.p_in.weight, tmo.g_in.weight)
        cands = [dict(tp=tp, tc=tc, nsg=ns, xreg=xr) for tp in (32, 64) for tc in (16, 32) for ns in (2, 4, 8)
                 for xr in (0, 1)]
        run_cands({**base0, "site": "trimul_proj"}, dict(K.NAX_TRIMUL_CFG), cands[:3] if QUICK else cands,
                  lambda c: KN2.trimul_proj_nax(xf, wh, mf, 2.0 ** -K.TRIMULT_K, **c), reps=8)
        # tri-mult tails (outgoing: next LN = tri_mul_in.norm_in; incoming: next = tri-att start LN + its tri bias)
        es = 2.0 ** (-4 * K.TRIMULT_K)
        for tag, mod, zz, xx, nxt, wtb in (("trimul_out", tmo, z, xn0, tmi.norm_in, None),
                                           ("trimul_in", tmi, z1, xn1, ts.layer_norm, ts.tb_weight16())):
            o, xfo = mod._f16_contract(xx, pm, D, N, N)
            o = o.reshape(D, N * N)
            mx.eval(o)
            a_ = (o, mod.norm_out, es, K.w16(mod.p_out, "_w16", mod.p_out.weight), xfo,
                  K.w16(mod, "_w_gouth", mod.g_out.weight), zz, nxt)
            cands = list(N3.TAIL3_CFGS) + [dict(tm=16, nsg=ns, ch=ch, maxt=mt, unrk=ur) for ns in (2, 4, 8, 16)
                                           for ch in (1, 2, 4) for ur in (0, 1) for mt in (0, 1)]
            run_cands({**base0, "site": "tail3", "sub": tag}, N3.PAIR_SITE_CFG["tail3"], cands[:3] if QUICK else cands,
                      lambda c: N3.trimul_tail3_addln(*a_, w_tb=wtb, **c), reps=8)
            del o, a_
        # tri-att start: q|k|v|g projection + tri bias from the incoming tail, core, linear_o + add + next LN
        o1, xf1 = tmi._f16_contract(xn1, pm, D, N, N)
        z2, xn2, tb2 = N3.trimul_tail3_addln(o1.reshape(D, N * N), tmi.norm_out, es,
                                             K.w16(tmi.p_out, "_w16", tmi.p_out.weight), xf1,
                                             K.w16(tmi, "_w_gouth", tmi.g_out.weight), z1, ts.layer_norm,
                                             w_tb=ts.tb_weight16(), **N3.PAIR_SITE_CFG["tail3"])
        del o1, xf1
        mx.eval(z2, xn2, tb2)
        mha = ts.mha
        whs = K.w16(ts, "_wh", mha.linear_q.weight, mha.linear_k.weight, mha.linear_v.weight, mha.linear_g.weight,
                    ts.linear.weight)
        core = ts._f16_core(xn2, pm, whs, True, tb=tb2)
        mx.eval(core)
        w_o = K.w16(mha.linear_o, "_w16", mha.linear_o.weight)
        b_o = mha.linear_o.get("bias") if "bias" in mha.linear_o else None
        cands = [dict(tm=tm, nsg=ns) for tm in (16, 32) for ns in (1, 2, 4, 8, 16) if tm * ns <= 256]
        run_cands({**base0, "site": "lin", "sub": "start"}, N3.PAIR_SITE_CFG["lin"], cands[:3] if QUICK else cands,
                  lambda c: N3.linear_addln(core, w_o, z2, te.layer_norm, bias=b_o, trans_y=False, trans_out=True,
                                            w_tb=te.tb_weight16(), **c), reps=8)
        # ending node variant (rows of the core read transposed, next LN = the transition's, no tri bias)
        run_cands({**base0, "site": "lin", "sub": "end"}, N3.PAIR_SITE_CFG["lin"], cands[:3] if QUICK else cands,
                  lambda c: N3.linear_addln(core, w_o, z2, tr.norm, bias=b_o, trans_y=True, trans_out=False, **c),
                  reps=8)
        # pair transition (register kernel): inputs from the layer-0 transition site
        z4 = z2  # any fp32 residual of the right shape (bitwise comparison is per input)
        xn4 = tr.norm.f16(z4)
        mx.eval(xn4)
        w12 = K.w16(tr, "_w12h", tr.fc1.weight, tr.fc2.weight)
        w3 = K.w16(tr, "_w3h", tr.fc3.weight)
        ln_s, lin = layer.attention.proj_z
        w_s = K.w16(lin, "_w16", lin.weight)
        sh = N3.PAIR_SITE_CFG["trans3"]
        if sh.get("impl") == "reg":
            cands = [dict(impl="reg", nsg=ns, keepa=ka, unrk=ur, maxt=mt) for ns in (2, 4, 8, 16) for ka in (0, 1)
                     for ur in (0, 1) for mt in (0, 1)]
            run_cands({**base0, "site": "trans3"}, sh, cands[:3] if QUICK else cands,
                      lambda c: N3.transition3(xn4, w12, w3, z4, next_ln=tmo.norm_in, ln_s=ln_s, w_s=w_s, cfg=c),
                      reps=8)
        del rec, args, kwargs, z, z1, xn0, xn1, z2, xn2, tb2, core, xn4
        gc.collect()
        mx.clear_cache()


# ------------------------------------------------------------------------------------------------ msa
def part_msa():
    mx = _mx()
    from boltz_mlx import kernels as K
    from boltz_mlx import kernels_msa as KM
    from boltz_mlx import kernels_nax2 as KN2
    from boltz_mlx.layers.outer_product_mean import OuterProductMean
    from boltz_mlx.layers.pair_averaging import PairWeightedAveraging
    from boltz_mlx.weights import load_module

    K.PRECISION = "f16"
    for case in (("msa384",) if QUICK else ("msa384", "ex_ligand")):
        base0 = {"exp": "chip_sweep", "part": "msa", "case": case}
        try:
            rec, args, kwargs = _load(case, "trunk.msa", "OuterProductMean", 0)
        except FileNotFoundError:
            emit({**base0, "error": "missing capture"})
            continue
        mod = load_module(OuterProductMean(64, 32, 128), rec["path"])
        K.MSA_FLAGS["OPM"] = "1"
        K.MSA_OPM_NAX = True
        sh = dict(K.MSA_OPM_NAX_CFG)

        def opm(c):
            K.MSA_OPM_NAX_CFG.clear()
            K.MSA_OPM_NAX_CFG.update(c)
            return K.unscale(type(mod).__call__(mod, *args, **kwargs))

        run_cands({**base0, "site": "opm_nax", "N": args[0].shape[2], "S": args[0].shape[1]}, sh,
                  KM.OPM_FUSED_CFGS[:2] if QUICK else KM.OPM_FUSED_CFGS, opm, reps=5)
        K.MSA_OPM_NAX_CFG.clear()
        K.MSA_OPM_NAX_CFG.update(sh)
        rec, args, kwargs = _load(case, "trunk.msa", "PairWeightedAveraging", 0)
        mod = load_module(PairWeightedAveraging(64, 128, 32, 8), rec["path"])
        K.MSA_FLAGS["PWA"] = "1"
        sh = dict(K.MSA_PWA_CFG)

        def pwa(c):
            K.MSA_PWA_CFG.clear()
            K.MSA_PWA_CFG.update(c)
            return K.unscale(type(mod).__call__(mod, *args, **kwargs))

        run_cands({**base0, "site": "pwa", "N": args[0].shape[2], "S": args[0].shape[1]}, sh,
                  KM.PWA_CFGS[:2] if QUICK else KM.PWA_CFGS, pwa, reps=5)
        K.MSA_PWA_CFG.clear()
        K.MSA_PWA_CFG.update(sh)
        # MSA transition (D = 64, hidden 256): transition_nax configs on the msa384 m input
        import numpy as np

        m = args[0]
        rng = np.random.default_rng(2)
        xn = mx.array((rng.normal(size=m.shape)).astype(np.float16))
        w12 = mx.array((rng.normal(size=(512, 64)) * 0.125).astype(np.float16))
        w3 = mx.array((rng.normal(size=(64, 256)) * 0.06).astype(np.float16))
        resid = mx.array(rng.normal(size=m.shape).astype(np.float32))
        mx.eval(xn, w12, w3, resid)
        sh = K.NAX_TRANSITION_CFG[64]
        run_cands({**base0, "site": "msa_transition", "rows": int(m.size // 64)}, sh,
                  KN2.TRANS_CFGS[:2] if QUICK else KN2.TRANS_CFGS,
                  lambda c: KN2.transition_nax(xn, w12, w3, resid=resid, **c), reps=5)
        del rec, args, kwargs, mod, xn, resid
        gc.collect()
        mx.clear_cache()
    K.MSA_FLAGS["OPM"] = K.MSA_FLAGS["PWA"] = None


# ------------------------------------------------------------------------------------------------ report (CPU)
def _rows() -> list:
    """Candidate rows of the latest full (non --quick) sweep per (git, chip): rows written after its start row (a smoke
    run of the same commit would otherwise mix its few-candidate timings in)."""
    rows, cur, last_full = [], None, {}
    for x in open(OUT) if OUT.exists() else []:
        r = json.loads(x)
        if r.get("exp") == "chip_sweep_start":
            cur = None if (r.get("quick") or r.get("recheck")) else (r["git"], r["chip"], r["date"])
            if cur:
                last_full[cur[:2]] = cur[2]
            continue
        if r.get("exp") == "chip_sweep" and "cand" in r:
            rows.append((cur, r))
    return [r for c, r in rows if c and last_full.get(c[:2]) == c[2]]


def report(git: str | None):
    rows = _rows()
    if git:
        rows = [r for r in rows if r["git"] == git]
    elif rows:
        rows = [r for r in rows if r["git"] == rows[-1]["git"] and r["chip"] == rows[-1]["chip"]]
    groups: dict = {}
    for r in rows:
        k = (r["part"], r["site"], r.get("sub", ""), r["case"])
        groups.setdefault(k, []).append(r)
    print("| part | site | case | metric | shipped ms | best bitwise ms (cfg) | gain % | non-bitwise / errors |")
    print("|---|---|---|---|---|---|---|---|")
    for k, rs in groups.items():
        sh = [r for r in rs if r.get("shipped")]
        if not sh:
            continue
        sh = sh[0]
        metric = "chain_ms" if "chain_ms" in sh else ("sync_ms" if "sync_ms" in sh else "tp_ms")
        if metric not in sh:
            print(f"| {k[0]} | {k[1]}{'/' + k[2] if k[2] else ''} | {k[3]} | - | error {sh.get('error')} | | | |")
            continue
        ok = [r for r in rs if r.get("bitwise") and metric in r]
        best = min(ok, key=lambda r: r[metric])
        nb = sum(1 for r in rs if r.get("bitwise") is False)
        ne = sum(1 for r in rs if "error" in r)
        print(f"| {k[0]} | {k[1]}{'/' + k[2] if k[2] else ''} | {k[3]} | {metric} | {sh[metric]:.4f} | "
              f"{best[metric]:.4f} ({cfg_key(best['cand'])}) | {100 * (1 - best[metric] / sh[metric]):.1f} | "
              f"{nb} / {ne} |")


# ------------------------------------------------------------------------------------------------ suggest (CPU)
# metric per site: the dependent-chain latency where measured (GEMM sites), else the one-call-per-eval latency
# (attention), else pipelined throughput (pair / MSA kernels: >= 200 threadgroups at every case size)
MIN_GAIN = {"chain_ms": 0.03, "sync_ms": 0.05, "tp_ms": 0.03}   # a non-shipped choice must win by this fraction
SIZE_KEY = {"gemm_tok": "M", "gemm_cond": "M", "gemm_atom": "M", "gemm_pair": "N", "pair": "N", "msa": None}
ATT_DEFAULTS = {"pt": "float", "order": 0}


def _size(r) -> int | None:
    if r["part"] == "gemm_pair":
        return int(round(r["M"] ** 0.5))
    if r["part"] == "attn":
        return {"diffattn": r.get("B", 1) * r.get("N", 0), "triatt": r.get("N"), "atomattn": r.get("M")}[r["site"]]
    k = SIZE_KEY.get(r["part"])
    return r.get(k) if k else None


def _metric(rs) -> str:
    sh = [r for r in rs if r.get("shipped")]
    for m in ("chain_ms", "sync_ms", "tp_ms"):
        if sh and m in sh[0]:
            return m
    return "tp_ms"


SPLIT_SITES = {("lin", "end"): "lin_end"}   # sub-sites with their own config entry (kernels_nax3._SITE_PARENT)


def choices(rows) -> dict:
    """{(part, site): [(size, best_cfg | None (= shipped), gain, shipped_ms, best_ms, metric)]}, bitwise-equal
    candidates only; sub-sites sharing one config (tail3 out / in) have their metric summed; split sites
    (SPLIT_SITES: lin start -> lin, lin end -> lin_end) are chosen separately."""
    groups: dict = {}
    for r in rows:
        if "cand" not in r or "error" in r:
            continue
        if (r["site"], r.get("sub", "")) in SPLIT_SITES:
            r = {**r, "site": SPLIT_SITES[(r["site"], r["sub"])], "sub": ""}
        elif r["site"] == "lin":
            r = {**r, "sub": ""}
        groups.setdefault((r["part"], r["site"], _size(r), r["case"]), []).append(r)
    out: dict = {}
    for (part, site, size, case), rs in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1], x[0][2] or 0)):
        m = _metric(rs)
        tot: dict = {}
        cfg_of: dict = {}
        subs = {r.get("sub", "") for r in rs}
        for r in rs:
            if not r.get("bitwise") or m not in r:
                continue
            k = cfg_key(r["cand"])
            tot.setdefault(k, {})[r.get("sub", "")] = r[m]
            cfg_of[k] = r["cand"]
        full = {k: sum(v.values()) for k, v in tot.items() if set(v) == subs}
        shk = [cfg_key(r["cand"]) for r in rs if r.get("shipped")]
        if not shk or shk[0] not in full:
            continue
        sh_ms = full[shk[0]]
        bk = min(full, key=full.get)
        gain = 1.0 - full[bk] / sh_ms
        best = cfg_of[bk] if (bk != shk[0] and gain >= MIN_GAIN[m]) else None
        out.setdefault((part, site), []).append((size, best, round(gain, 4), round(sh_ms, 5), round(full[bk], 5), m,
                                                 case))
    return out


def suggest(git: str | None, chip: str | None) -> dict:
    rows = _rows()
    chip = chip or (rows[-1]["chip"] if rows else None)
    rows = [r for r in rows if r["chip"] == chip]
    if git:
        rows = [r for r in rows if r["git"] == git]
    elif rows:
        rows = [r for r in rows if r["git"] == rows[-1]["git"]]
    ch = choices(rows)
    for (part, site), lst in ch.items():
        print(f"{part}/{site}:")
        for size, best, gain, sms, bms, m, case in lst:
            print(f"   size {str(size):>7} ({case:9s}) {m} shipped {sms:.4f} best {bms:.4f} ({100 * gain:+.1f}%) -> "
                  f"{'keep shipped' if best is None else cfg_key(best)}")
    return {"chip": chip, "git": rows[-1]["git"] if rows else None, "choices": {f"{p}/{s}": v for (p, s), v in ch.items()}}


def recheck_report(probe: bool = False) -> int:
    """Table of the latest recheck run (rows after its start row; probe: the latest probe run, else the latest plain
    one): per site / case A (default entry) vs B (chip entry [+ probe]) medians of the interleaved samples, gain,
    bitwise. Exit 1 if any pair is not bitwise equal."""
    rows, cur, on = [], None, False
    for x in open(OUT) if OUT.exists() else []:
        r = json.loads(x)
        if r.get("exp") == "chip_sweep_start":
            on = bool(r.get("recheck")) and bool(r.get("probe")) == probe
            if on:
                rows, cur = [], r
        elif on and r.get("exp") == "chip_sweep_recheck":
            rows.append(r)
    if not cur:
        print("no", "probe" if probe else "plain", "recheck run in", OUT)
        return 1
    print(f"recheck {cur['date']} git {cur['git']} chip {cur['chip']}: A = default entry, B = {cur.get('cfg_chip')}"
          + (f" + PROBE {json.dumps(cur['probe'])} (info only: not the config files)" if probe else "")
          + "; median of 6 interleaved samples (A B B A A B ...)")
    print("| part | site | case | size | metric | A ms | B ms | B vs A % | bitwise | B cfg |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    pairs: dict = {}
    for r in rows:
        pairs.setdefault((r["part"], r["site_key"], r.get("sub", ""), r["case"]), {})[r["arm"]] = r
    bad = 0
    for (part, site, sub, case), d in pairs.items():
        if "A" not in d or "B" not in d:
            continue
        a, b = d["A"], d["B"]
        m = next(k for k in ("chain_ms", "sync_ms", "tp_ms") if k in a)
        bad += not b["bitwise"]
        print(f"| {part} | {site}{'/' + sub if sub else ''} | {case} | {b['size']} | {m} | {a[m]:.4f} | {b[m]:.4f} | "
              f"{100 * (b[m] / a[m] - 1):+.1f} | {b['bitwise']} | {cfg_key(b['cand'])} |")
    return 1 if bad else 0


def main():
    global RECHECK
    if len(sys.argv) > 1 and sys.argv[1] == "recheck_report":
        sys.exit(recheck_report("--probe" in sys.argv))
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        return report(sys.argv[sys.argv.index("--git") + 1] if "--git" in sys.argv else None)
    if len(sys.argv) > 1 and sys.argv[1] == "suggest":
        res = suggest(sys.argv[sys.argv.index("--git") + 1] if "--git" in sys.argv else None,
                      sys.argv[sys.argv.index("--chip") + 1] if "--chip" in sys.argv else None)
        if "--json" in sys.argv:
            print(json.dumps(res))
        return None
    os.environ.setdefault("BOLTZ_MLX_PRECISION", "f16")
    sys.path.insert(0, str(ROOT))
    import boltz_mlx  # noqa: F401
    import mlx.core as mx

    from boltz_mlx import kernels as K

    mx.set_default_device(mx.gpu)
    assert K.f16() and K.mpp_enabled(), "chip_sweep: needs --precision f16 on the NAX path"
    parts = [p for p in sys.argv[1:] if not p.startswith("--")] or ["gemm", "attn", "pair", "msa"]
    recheck = parts[0] == "recheck"
    if recheck:   # this chip's per-device entry vs the default entry, interleaved, at every swept site / size
        PROBE.update(json.loads(os.environ.get("CHIP_SWEEP_PROBE", "") or "{}"))
        assert set(PROBE) <= {"gemm", "pair", "kern", "atom"}, f"CHIP_SWEEP_PROBE keys: {sorted(PROBE)}"
        RECHECK = _recheck_pick()
        parts = parts[1:] or ["gemm", "attn", "pair"]
        if not QUICK:
            for name, extra in RECHECK_EXTRA.items():
                g = globals()[name]
                g.extend(extra) if isinstance(g, list) else g.update(extra)
    emit({"exp": "chip_sweep_start", "parts": parts, "quick": QUICK, "recheck": recheck, "cfg_chip": K.cfg_chip(),
          "probe": PROBE or None, "device": K.device_record(), "boltz_mlx": boltz_mlx.__file__})
    t0 = time.perf_counter()
    model = None
    for p in parts:
        t = time.perf_counter()
        try:
            if p == "pair":
                from boltz_mlx.model import load_model

                model = model or load_model("conf", compile_diffusion=False)
                part_pair(model)
            else:
                {"gemm": part_gemm, "attn": part_attn, "msa": part_msa}[p]()
        except Exception as e:  # noqa: BLE001
            import traceback

            emit({"exp": "chip_sweep", "part": p, "error": f"{type(e).__name__}: {str(e)[:800]}",
                  "trace": traceback.format_exc()[-1500:]})
        emit({"exp": "chip_sweep_part_done", "part": p, "wall_s": round(time.perf_counter() - t, 1)})
    emit({"exp": "chip_sweep_done", "wall_s": round(time.perf_counter() - t0, 1), "error_rows": len(ERRORS)})
    if (QUICK or recheck) and ERRORS:  # smoke / recheck: only valid configs, so any error is a wiring bug
        print(f"chip_sweep {'--quick' if QUICK else 'recheck'}: {len(ERRORS)} error rows ({sorted(set(map(str, ERRORS)))})")
        sys.exit(1)


if __name__ == "__main__":
    main()
