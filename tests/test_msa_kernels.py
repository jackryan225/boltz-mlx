"""MSA-module kernels (boltz_mlx/kernels_msa.py) and their f16 integration (OuterProductMean / PairWeightedAveraging).

CPU part (default device): the wrapper math of the fused paths (MLX-op references composed exactly like the module
paths) against the fp32 modules, and numpy emulations that follow the kernels' index math line by line (tile loads,
double-buffered K loop, fragment MMAs, epilogue addressing via the simdgroup 8x8 lane layout) for every listed config
on ragged shapes, so tiling / addressing errors show up without a GPU.

GPU part (BOLTZ_MLX_DEVICE=gpu python -m pytest -q tests/test_msa_kernels.py): kernels vs references (every config,
ragged shapes, masks, row chunks) and the module-level f16 paths with the flags on vs off vs fp32.
"""

import math

import mlx.core as mx
import numpy as np
import pytest

from boltz_mlx import kernels as K
from boltz_mlx import kernels_msa as KM

GPU = mx.default_device() == mx.gpu
gpu_only = pytest.mark.skipif(not GPU, reason="custom kernels need the GPU device")
cpu_only = pytest.mark.skipif(GPU, reason="CPU emulation / reference checks")


def _rel(out, ref):
    o = np.asarray(out, np.float64)
    r = np.asarray(ref, np.float64)
    return float(np.linalg.norm(o - r) / np.linalg.norm(r)), float(np.abs(o - r).max() / np.abs(r).max())


def _np(x):
    return np.array(x.astype(mx.float32) if x.dtype != mx.float32 else x)


def _frag_lanes():
    lane = np.arange(32)
    qid = lane // 4
    fm = (qid & 4) + ((lane // 2) % 4)
    fn = (qid & 2) * 2 + (lane % 2) * 2
    return fm, fn


def _opm_inputs(rng, S, N, T=64, zero_frac=0.2):
    ab = rng.normal(size=(1, S, N, T)).astype(np.float32)
    mask = (rng.random((S, N)) > zero_frac).astype(np.float32)
    mask[0] = 1.0
    num = np.maximum(mask.T @ mask, 1.0).astype(np.float32)
    return ab, mask, num


# ------------------------------------------------------------------------------------------- numpy emulations
def emulate_opm(ab, mask, num, i_base, n_i, ti, tj, si, sj, bk, swz, at=0, pad=4, a_off=0, b_off=32):
    """Follows _OPM_SRC: returns the fp32 values the epilogue would round to fp16 (NaN where nothing is written)."""
    C = 32
    _, S, N, TOK = ab.shape
    BM, BN, WN, RM, RN = ti * C, tj * C, tj // sj, si * 4, sj * 4
    LDA, LDB = (bk + pad if at else BM + pad), BN + pad
    ASZ = BM * LDA if at else bk * LDA
    nsg = (ti // si) * (tj // sj)
    NT, NA4, NB4 = nsg * 32, bk * BM // 4, bk * BN // 4
    LA, LB = -(-NA4 // NT), -(-NB4 // NT)
    abf = ab.reshape(-1)
    out = np.full(n_i * N * C * C, np.nan, np.float64)
    tiles_i, tiles_j = -(-n_i // ti), -(-N // tj)
    gx, gy = tiles_j << swz, -(-tiles_i // (1 << swz))
    fm, fn = _frag_lanes()
    tid = np.arange(NT)
    seen = set()
    for gyi in range(gy):
        for gxi in range(gx):
            t_i, t_j = gyi, gxi
            if swz > 0:
                t_i, t_j = (gyi << swz) + (gxi & ((1 << swz) - 1)), gxi >> swz
            if t_i >= tiles_i or t_j >= tiles_j:
                continue
            assert (t_i, t_j) not in seen
            seen.add((t_i, t_j))
            i0, j0, iend = i_base + t_i * ti, t_j * tj, i_base + n_i
            As = np.zeros(2 * ASZ)
            Bs = np.zeros(2 * bk * LDB)
            acc = np.zeros((nsg, RM * 8, RN * 8))
            ntk = -(-S // bk)
            for it in range(ntk + 1):
                if it < ntk:
                    s0 = it * bk
                    ra = np.zeros((LA, NT, 4))
                    rb = np.zeros((LB, NT, 4))
                    for u in range(LA):
                        e = tid + u * NT
                        kk, r4 = e // (BM // 4), e % (BM // 4)
                        s, i = s0 + kk, i0 + r4 // (C // 4)
                        for t in np.nonzero((e < NA4) & (s < S) & (i < iend))[0]:
                            off = s[t] * N * TOK + i[t] * TOK + a_off + (r4[t] % (C // 4)) * 4
                            ra[u, t] = abf[off:off + 4] * mask[s[t], i[t]]
                    for u in range(LB):
                        e = tid + u * NT
                        kk, r4 = e // (BN // 4), e % (BN // 4)
                        s, j = s0 + kk, j0 + r4 // (C // 4)
                        for t in np.nonzero((e < NB4) & (s < S) & (j < N))[0]:
                            off = s[t] * N * TOK + j[t] * TOK + b_off + (r4[t] % (C // 4)) * 4
                            rb[u, t] = abf[off:off + 4] * mask[s[t], j[t]]
                if it > 0:
                    a_s = As[((it - 1) & 1) * ASZ:][:ASZ]
                    b_s = Bs[((it - 1) & 1) * bk * LDB:][: bk * LDB].reshape(bk, LDB)
                    for sg in range(nsg):
                        wm, wn = sg // WN, sg % WN
                        for kk in range(0, bk, 8):
                            for r in range(RM):
                                c0 = wm * RM * 8 + r * 8
                                if at:  # plain load of rows (i, c) c0.., cols k kk..
                                    af = a_s.reshape(BM, LDA)[c0:c0 + 8, kk:kk + 8]
                                else:  # transposed simdgroup_load
                                    af = a_s.reshape(bk, LDA)[kk:kk + 8, c0:c0 + 8].T
                                for c in range(RN):
                                    n0 = wn * RN * 8 + c * 8
                                    acc[sg, r * 8:r * 8 + 8, c * 8:c * 8 + 8] += af @ b_s[kk:kk + 8, n0:n0 + 8]
                if it < ntk:
                    a_buf = As[(it & 1) * ASZ:]
                    b_buf = Bs[(it & 1) * bk * LDB:]
                    for u in range(LA):
                        for t in range(NT):
                            e = t + u * NT
                            if e < NA4:
                                kk, m = e // (BM // 4), (e % (BM // 4)) * 4
                                if at:
                                    for q in range(4):
                                        a_buf[(m + q) * LDA + kk] = ra[u, t, q]
                                else:
                                    a_buf[kk * LDA + m:kk * LDA + m + 4] = ra[u, t]
                    for u in range(LB):
                        for t in range(NT):
                            e = t + u * NT
                            if e < NB4:
                                o = (e // (BN // 4)) * LDB + (e % (BN // 4)) * 4
                                b_buf[o:o + 4] = rb[u, t]
            for sg in range(nsg):
                wm, wn = sg // WN, sg % WN
                for s_i in range(si):
                    i = i0 + wm * si + s_i
                    for s_j in range(sj):
                        j = j0 + wn * sj + s_j
                        if not (i < iend and j < N):
                            continue
                        nm = num[i, j]
                        base = ((i - i_base) * N + j) * C * C
                        for r in range(4):
                            for c in range(4):
                                F = acc[sg, (s_i * 4 + r) * 8:(s_i * 4 + r) * 8 + 8,
                                        (s_j * 4 + c) * 8:(s_j * 4 + c) * 8 + 8]
                                addr = base + (r * 8 + fm) * C + c * 8 + fn
                                assert np.isnan(out[addr]).all() and np.isnan(out[addr + 1]).all()
                                out[addr] = F[fm, fn] / nm
                                out[addr + 1] = F[fm, fn + 1] / nm
    return out.reshape(n_i * N, C * C)


def emulate_opm_fused_epilogue(z, wo, bias, N, i_base, n_i, ti, tj, si, sj, kc, **_):
    """Follows _OPM_EPI_NAX given the fp32 z / num values of the K loop (pair-major, from emulate_opm): half-tile
    staging rows p = (i - i0) * TJ + (j - j0), the NAX column / K-chunk offsets, bias, rounding and the row scatter."""
    C, CO = 32, 128
    WN, nsg = tj // sj, (ti // si) * (tj // sj)
    fm, fn = _frag_lanes()
    out = np.full((n_i * N, CO), np.nan)
    for t_i in range(-(-n_i // ti)):
        for t_j in range(-(-N // tj)):
            i0, j0, iend = i_base + t_i * ti, t_j * tj, i_base + n_i
            acc = np.zeros((16, CO))
            for hh in range(2):
                zh = np.full((16, 16 * C), np.nan)
                for sg in range(nsg):
                    wm, wn = sg // WN, sg % WN
                    for s_i in range(si):
                        il = wm * si + s_i
                        i = i0 + il
                        for s_j in range(sj):
                            jl = wn * sj + s_j
                            j = j0 + jl
                            src = z[(i - i_base) * N + j] if (i < iend and j < N) else np.zeros(C * C)
                            for rr in range(2):
                                for c in range(4):
                                    r = 2 * hh + rr
                                    for e in range(2):  # fragment (r, c) of the pair block, element (fm, fn + e)
                                        zh[il * tj + jl, (rr * 8 + fm) * C + c * 8 + fn + e] = \
                                            np.float16(src[(r * 8 + fm) * C + c * 8 + fn + e])
                assert not np.isnan(zh).any()
                for q in range(4):
                    for k0 in range(0, 16 * C, kc):
                        acc[:, 32 * q:32 * q + 32] += zh[:, k0:k0 + kc] @ wo[32 * q:32 * q + 32,
                                                                            hh * 16 * C + k0:hh * 16 * C + k0 + kc].T
            ost = (acc + bias).astype(np.float16)
            for e in range(16 * (CO // 8)):
                p, o8 = e // (CO // 8), (e % (CO // 8)) * 8
                i, j = i0 + p // tj, j0 + p % tj
                if i < iend and j < N:
                    row = (i - i_base) * N + j
                    assert np.isnan(out[row, o8:o8 + 8]).all()
                    out[row, o8:o8 + 8] = ost[p, o8:o8 + 8]
    return out


def emulate_pwa_avg(w, v, g, bm, ts, wm_, wn_, bk, pad=4):
    """Follows _PWAA_SRC (fp64 arithmetic)."""
    C = 32
    H, N, NP = w.shape
    _, S, _, HC = v.shape
    BN = ts * C
    SM, SN = bm // wm_, BN // wn_
    RM, RN = SM // 8, SN // 8
    LDA, LDB = bk + pad, BN + pad
    nsg = wm_ * wn_
    NT, NA4, NB4 = nsg * 32, bm * bk // 4, bk * BN // 4
    LA, LB = -(-NA4 // NT), -(-NB4 // NT)
    wf, vf, gf = w.reshape(-1), v.reshape(-1), g.reshape(-1)
    og = np.full(S * N * HC, np.nan)
    fm, fn = _frag_lanes()
    tid = np.arange(NT)
    for h in range(H):
        for ty in range(-(-S // ts)):
            for tx in range(-(-N // bm)):
                i0, s0 = tx * bm, ty * ts
                As = np.zeros(2 * bm * LDA)
                Bs = np.zeros(2 * bk * LDB)
                acc = np.zeros((nsg, RM * 8, RN * 8))
                ntk = NP // bk
                for it in range(ntk + 1):
                    if it < ntk:
                        k0 = it * bk
                        ra = np.zeros((LA, NT, 4))
                        rb = np.zeros((LB, NT, 4))
                        for u in range(LA):
                            e = tid + u * NT
                            r, k4 = e // (bk // 4), e % (bk // 4)
                            i = i0 + r
                            for t in np.nonzero((e < NA4) & (i < N))[0]:
                                off = h * N * NP + i[t] * NP + k0 + k4[t] * 4
                                ra[u, t] = wf[off:off + 4]
                        for u in range(LB):
                            e = tid + u * NT
                            kk, n4 = e // (BN // 4), e % (BN // 4)
                            j, s = k0 + kk, s0 + n4 // (C // 4)
                            for t in np.nonzero((e < NB4) & (j < N) & (s < S))[0]:
                                off = (s[t] * N + j[t]) * HC + h * C + (n4[t] % (C // 4)) * 4
                                rb[u, t] = vf[off:off + 4]
                    if it > 0:
                        a_s = As[((it - 1) & 1) * bm * LDA:][: bm * LDA].reshape(bm, LDA)
                        b_s = Bs[((it - 1) & 1) * bk * LDB:][: bk * LDB].reshape(bk, LDB)
                        for sg in range(nsg):
                            wm, wn = sg // wn_, sg % wn_
                            for kk in range(0, bk, 8):
                                for r in range(RM):
                                    r0 = wm * SM + r * 8
                                    af = a_s[r0:r0 + 8, kk:kk + 8]
                                    for c in range(RN):
                                        n0 = wn * SN + c * 8
                                        acc[sg, r * 8:r * 8 + 8, c * 8:c * 8 + 8] += af @ b_s[kk:kk + 8, n0:n0 + 8]
                    if it < ntk:
                        a_buf = As[(it & 1) * bm * LDA:]
                        b_buf = Bs[(it & 1) * bk * LDB:]
                        for u in range(LA):
                            for t in range(NT):
                                e = t + u * NT
                                if e < NA4:
                                    o = (e // (bk // 4)) * LDA + (e % (bk // 4)) * 4
                                    a_buf[o:o + 4] = ra[u, t]
                        for u in range(LB):
                            for t in range(NT):
                                e = t + u * NT
                                if e < NB4:
                                    o = (e // (BN // 4)) * LDB + (e % (BN // 4)) * 4
                                    b_buf[o:o + 4] = rb[u, t]
                for sg in range(nsg):
                    wm, wn = sg // wn_, sg % wn_
                    for r in range(RM):
                        i = i0 + wm * SM + r * 8 + fm
                        for c in range(RN):
                            n = wn * SN + c * 8 + fn
                            s, d = s0 + n // C, n % C
                            F = acc[sg, r * 8:r * 8 + 8, c * 8:c * 8 + 8]
                            for L in np.nonzero((i < N) & (s < S))[0]:
                                row = s[L] * N + i[L]
                                for e in range(2):
                                    gv = gf[row * HC + h * C + d[L] + e]
                                    addr = row * HC + h * C + d[L] + e
                                    assert np.isnan(og[addr])
                                    og[addr] = F[fm[L], fn[L] + e] / (1.0 + math.exp(-gv))
    return og.reshape(1, S, N, HC)


# ------------------------------------------------------------------------------------------- CPU tests
@cpu_only
@pytest.mark.parametrize("cfg_i", range(len(KM.OPM_CFGS)))
def test_opm_emulated_index_math(cfg_i):
    cfg = KM.OPM_CFGS[cfg_i]
    rng = np.random.default_rng(cfg_i)
    for S, N, i_base, n_i in ((37, 11, 0, 11), (19, 13, 3, 7)):  # K / i / j tails, row chunk
        ab, mask, num = _opm_inputs(rng, S, N)
        got = emulate_opm(ab, mask, num, i_base, n_i, **cfg)
        assert not np.isnan(got).any(), "unwritten outputs"
        ref = _np(KM.opm_outer_ref(mx.array(ab), mx.array(mask), mx.array(num), i_base, n_i, dtype=mx.float32))
        rf, mr = _rel(got, ref)
        assert rf < 1e-6 and mr < 1e-5, (cfg, S, N, rf, mr)


@cpu_only
@pytest.mark.parametrize("cfg_i", range(len(KM.OPM_FUSED_CFGS)))
def test_opm_fused_emulated_index_math(cfg_i):
    cfg = dict(KM.OPM_FUSED_CFGS[cfg_i])
    rng = np.random.default_rng(20 + cfg_i)
    S, N, i_base, n_i = 21, 11, 2, 7  # pair tails in i and j, row chunk
    ab, mask, num = _opm_inputs(rng, S, N)
    wo = (rng.normal(size=(128, 1024)) * 1024 ** -0.5).astype(np.float16).astype(np.float64)
    bias = rng.normal(0, 0.1, 128).astype(np.float32)
    z = emulate_opm(ab, mask, num, i_base, n_i, swz=0, **{k: cfg[k] for k in ("ti", "tj", "si", "sj", "bk", "at")})
    got = emulate_opm_fused_epilogue(z, wo, bias, N, i_base, n_i, **cfg)
    assert not np.isnan(got).any(), "unwritten outputs"
    ref = _np(KM.opm_outer_proj_ref(mx.array(ab), mx.array(mask), mx.array(num), mx.array(wo.astype(np.float16)),
                                    mx.array(bias), i_base, n_i))
    rf, mr = _rel(got, ref)
    assert rf < 1e-3 and np.abs(got - ref).max() <= np.abs(ref).max() * 2.0 ** -9, (cfg, rf, mr)


@cpu_only
@pytest.mark.parametrize("cfg_i", range(len(KM.PWA_CFGS)))
def test_pwa_avg_emulated_index_math(cfg_i):
    cfg = KM.PWA_CFGS[cfg_i]
    rng = np.random.default_rng(10 + cfg_i)
    H, N, S = 2, 37, 5  # N tail (NP = 64), S tail of TS
    w = np.zeros((H, N, KM.pwa_np(N)), np.float32)
    w[:, :, :N] = rng.random((H, N, N)).astype(np.float32)
    v = rng.normal(size=(1, S, N, H * 32)).astype(np.float32)
    g = rng.normal(size=(1, S, N, H * 32)).astype(np.float32)
    got = emulate_pwa_avg(w, v, g, cfg["bm"], cfg["ts"], cfg["wm"], cfg["wn"], cfg["bk"])
    assert not np.isnan(got).any(), "unwritten outputs"
    ref = _np(KM.pwa_avg_ref(mx.array(w), mx.array(v), mx.array(g)))
    rf, mr = _rel(got, ref)
    assert rf < 1e-6 and mr < 1e-5, (cfg, rf, mr)


def _opm_module(rng, S, N):
    from boltz_mlx.layers.outer_product_mean import OuterProductMean

    mod = OuterProductMean(64, 32, 128)
    mod.update({"norm": {"weight": mx.array(rng.normal(1, 0.1, 64).astype(np.float32)),
                         "bias": mx.array(rng.normal(0, 0.1, 64).astype(np.float32))},
                "proj_o": {"bias": mx.array(rng.normal(0, 0.1, 128).astype(np.float32))}}, strict=False)
    m = mx.array(rng.normal(size=(1, S, N, 64)).astype(np.float32))
    mask = (rng.random((1, S, N)) > 0.2).astype(np.int64)
    mask[:, 0] = 1
    return mod, m, mx.array(mask)


@cpu_only
def test_opm_fused_math_cpu():
    """The fused path's composition (concatenated proj_a/proj_b GEMM, in-kernel mask, mask^T mask counts, z / num,
    proj_o) equals the fp32 module; with the fp16 rounding of z / num it stays at the f16 path's error level."""
    rng = np.random.default_rng(0)
    S, N = 40, 9
    mod, m, mask = _opm_module(rng, S, N)
    ref = mod(m, mask)
    maskf = mask.astype(mx.float32).reshape(S, N)
    wab = mx.concatenate([mod.proj_a.weight, mod.proj_b.weight], 0)
    ab = mod.norm(m) @ wab.T
    num = mx.maximum(maskf.T @ maskf, 1.0)
    for dt, tol in ((mx.float32, 1e-5), (mx.float16, 3e-3)):
        parts = [KM.opm_outer_ref(ab, maskf, num, i0, min(N, i0 + 4) - i0, dtype=dt) for i0 in range(0, N, 4)]
        z = mx.concatenate(parts, 0).astype(mx.float32).reshape(1, N, N, 1024)
        out = mod.proj_o(z)
        rf, _ = _rel(_np(out), _np(ref))
        assert rf < tol, (dt, rf)


def _pwa_module(rng):
    from boltz_mlx.layers.pair_averaging import PairWeightedAveraging

    mod = PairWeightedAveraging(64, 128, 32, 8)
    mod.update({"norm_z": {"weight": mx.array(rng.normal(1, 0.1, 128).astype(np.float32)),
                           "bias": mx.array(rng.normal(0, 0.1, 128).astype(np.float32))}}, strict=False)
    return mod


@cpu_only
def test_pwa_fused_math_cpu():
    """pwa_weights_ref + pwa_avg_ref composed like PairWeightedAveraging._f16_fused equal the fp32 module (fp32 proj_z
    weight); with the fp16 weight copy the error stays small."""
    rng = np.random.default_rng(1)
    S, N = 5, 11
    mod = _pwa_module(rng)
    m = mx.array(rng.normal(size=(1, S, N, 64)).astype(np.float32))
    z = mx.array(rng.normal(size=(1, N, N, 128)).astype(np.float32))
    pm = (rng.random((1, N, N)) > 0.15).astype(np.float32)
    pm[:, :, 0] = 1
    pm = mx.array(pm)
    ref = mod(m, z, pm)
    mn = mod.norm_m(m)
    v, g = mod.proj_m(mn), mod.proj_g(mn)
    for wz, tol in ((mod.proj_z.weight, 1e-5), (mod.proj_z.weight.astype(mx.float16), 2e-3)):
        w = KM.pwa_weights_ref(z, pm, mod.norm_z.weight, mod.norm_z.bias, mod.norm_z.eps, wz, mod.inf)
        assert w.shape == (8, N, KM.pwa_np(N))
        out = mod.proj_o(KM.pwa_avg_ref(w, v, g))
        rf, _ = _rel(_np(out), _np(ref))
        assert rf < tol, (wz.dtype, rf)


@cpu_only
def test_msa_flags_never_on_cpu():
    from boltz_mlx import mixed as MX

    with MX.cpu_f16():
        assert K.f16()
        assert not K.msa_kernel("OPM") and not K.msa_kernel("PWA")


def test_tg_budgets():
    for cfg in KM.OPM_CFGS:
        assert KM.opm_tg_bytes(cfg["ti"], cfg["tj"], cfg["bk"], at=cfg["at"]) <= 32768
    for cfg in KM.PWA_CFGS:
        assert KM.pwa_tg_bytes(cfg["bm"], cfg["ts"], cfg["bk"]) <= 32768
    for cfg in KM.OPM_FUSED_CFGS:
        assert cfg["ti"] * cfg["tj"] == 16 and KM.opm_fused_tg_bytes(cfg["ti"], cfg["tj"], cfg["bk"], cfg["at"]) <= 32768
    assert K.MSA_OPM_CFG in KM.OPM_CFGS and K.MSA_PWA_CFG in KM.PWA_CFGS and K.MSA_OPM_NAX_CFG in KM.OPM_FUSED_CFGS


# ------------------------------------------------------------------------------------------- GPU tests
@gpu_only
@pytest.mark.parametrize("cfg_i", range(len(KM.OPM_CFGS)))
@pytest.mark.parametrize("S,N,i_base,n_i", [(249, 37, 0, 37), (40, 13, 3, 7), (33, 5, 0, 5), (64, 70, 64, 6)])
def test_opm_outer_gpu(cfg_i, S, N, i_base, n_i):
    cfg = KM.OPM_CFGS[cfg_i]
    rng = np.random.default_rng(S + N + cfg_i)
    ab, mask, num = (mx.array(x) for x in _opm_inputs(rng, S, N))
    got = KM.opm_outer(ab, mask, num, i_base, n_i, **cfg)
    ref32 = KM.opm_outer_ref(ab, mask, num, i_base, n_i, dtype=mx.float32)
    g, r = _np(got), _np(ref32)
    assert np.isfinite(g).all()
    rf, mr = _rel(g, r)
    # fp32 sums in a different order, then one fp16 rounding: <= ~1 fp16 ulp per element
    assert rf < 1e-3 and mr < 2e-3, (cfg, rf, mr)
    assert np.abs(g - r).max() <= np.abs(r).max() * 2.0 ** -10


@gpu_only
@pytest.mark.parametrize("cfg_i", range(len(KM.OPM_FUSED_CFGS)))
@pytest.mark.parametrize("S,N,i_base,n_i", [(249, 37, 0, 37), (40, 13, 3, 7), (64, 70, 64, 6)])
def test_opm_outer_proj_gpu(cfg_i, S, N, i_base, n_i):
    cfg = KM.OPM_FUSED_CFGS[cfg_i]
    rng = np.random.default_rng(S + N + 100 * cfg_i)
    ab, mask, num = (mx.array(x) for x in _opm_inputs(rng, S, N))
    wo = mx.array((rng.normal(size=(128, 1024)) * 1024 ** -0.5).astype(np.float16))
    bias = mx.array(rng.normal(0, 0.1, 128).astype(np.float32))
    got = KM.opm_outer_proj(ab, mask, num, wo, bias, i_base, n_i, **cfg)
    ref = KM.opm_outer_proj_ref(ab, mask, num, wo, bias, i_base, n_i)
    g, r = _np(got), _np(ref)
    assert np.isfinite(g).all()
    rf, mr = _rel(g, r)
    assert rf < 1e-3 and mr < 4e-3, (cfg, rf, mr)


@gpu_only
@pytest.mark.parametrize("N", [5, 37, 130])
@pytest.mark.parametrize("wdt", [mx.float16, mx.float32])
def test_pwa_weights_gpu(N, wdt):
    rng = np.random.default_rng(N)
    z = mx.array((rng.normal(size=(1, N, N, 128)) * 3 + 1).astype(np.float32))
    pm = (rng.random((1, N, N)) > 0.2).astype(np.float32)
    pm[:, :, 0] = 1
    pm = mx.array(pm)
    lw = mx.array(rng.normal(1, 0.1, 128).astype(np.float32))
    lb = mx.array(rng.normal(0, 0.1, 128).astype(np.float32))
    wz = mx.array((rng.normal(size=(8, 128)) * 128 ** -0.5).astype(np.float32)).astype(wdt)
    got = KM.pwa_weights(z, pm, lw, lb, 1e-5, wz, 1e6)
    ref = KM.pwa_weights_ref(z, pm, lw, lb, 1e-5, wz, 1e6)
    g, r = _np(got), _np(ref)
    assert np.isfinite(g).all() and (g[:, :, N:] == 0).all()
    np.testing.assert_allclose(g.sum(-1), 1.0, atol=1e-5)
    rf, mr = _rel(g, r)
    assert rf < 1e-5 and mr < 1e-5, (rf, mr)


@gpu_only
@pytest.mark.parametrize("cfg_i", range(len(KM.PWA_CFGS)))
@pytest.mark.parametrize("S,N", [(249, 37), (1, 130), (7, 64)])
def test_pwa_avg_gpu(cfg_i, S, N):
    cfg = KM.PWA_CFGS[cfg_i]
    rng = np.random.default_rng(S * N + cfg_i)
    H = 8
    w = np.zeros((H, N, KM.pwa_np(N)), np.float32)
    x = rng.random((H, N, N)).astype(np.float32)
    w[:, :, :N] = x / x.sum(-1, keepdims=True)
    w = mx.array(w)
    v = mx.array(rng.normal(size=(1, S, N, H * 32)).astype(np.float32))
    g = mx.array(rng.normal(size=(1, S, N, H * 32)).astype(np.float32))
    got = KM.pwa_avg(w, v, g, **cfg)
    ref = KM.pwa_avg_ref(w, v, g)
    rf, mr = _rel(_np(got), _np(ref))
    assert rf < 1e-5 and mr < 1e-4, (cfg, rf, mr)


def _with_precision(mode, flag_kind, flag, fn):
    saved = (K.PRECISION, dict(K.MSA_FLAGS))
    K.PRECISION = mode
    K.MSA_FLAGS[flag_kind] = flag
    try:
        out = fn()
        out = K.unscale(out)
        mx.eval(out)
        return out
    finally:
        K.PRECISION = saved[0]
        K.MSA_FLAGS.clear()
        K.MSA_FLAGS.update(saved[1])


@gpu_only
@pytest.mark.parametrize("S,N", [(249, 37), (64, 130)])
def test_opm_module_f16_fused(S, N):
    rng = np.random.default_rng(S + N)
    mod, m, mask = _opm_module(rng, S, N)
    ref = _with_precision("fp32", "OPM", "0", lambda: mod(m, mask))
    fused = _with_precision("f16", "OPM", "1", lambda: mod(m, mask))
    mlx16 = _with_precision("f16", "OPM", "0", lambda: mod(m, mask))
    assert fused.dtype == mlx16.dtype == mx.float16
    e_fused, _ = _rel(_np(fused), _np(ref))
    e_mlx, _ = _rel(_np(mlx16), _np(ref))
    assert e_fused < 3e-3 and e_fused <= 1.25 * e_mlx + 1e-6, (e_fused, e_mlx)
    rf, _ = _rel(_np(fused), _np(mlx16))
    assert rf < 2e-3, rf
    saved = K.MSA_OPM_NAX
    K.MSA_OPM_NAX = True
    try:
        nax = _with_precision("f16", "OPM", "1", lambda: mod(m, mask))
    finally:
        K.MSA_OPM_NAX = saved
    e_nax, _ = _rel(_np(nax), _np(ref))
    assert e_nax < 3e-3 and e_nax <= 1.25 * e_mlx + 1e-6, (e_nax, e_mlx)


@gpu_only
@pytest.mark.parametrize("S,N", [(249, 37), (1, 130), (17, 64)])
def test_pwa_module_f16_fused(S, N):
    rng = np.random.default_rng(S * 7 + N)
    mod = _pwa_module(rng)
    m = mx.array(rng.normal(size=(1, S, N, 64)).astype(np.float32))
    z = mx.array((rng.normal(size=(1, N, N, 128)) * 3).astype(np.float32))
    pm = (rng.random((1, N, N)) > 0.15).astype(np.float32)
    pm[:, :, 0] = 1
    pm = mx.array(pm)
    ref = _with_precision("fp32", "PWA", "0", lambda: mod(m, z, pm))
    fused = _with_precision("f16", "PWA", "1", lambda: mod(m, z, pm))
    mlx16 = _with_precision("f16", "PWA", "0", lambda: mod(m, z, pm))
    e_fused, _ = _rel(_np(fused), _np(ref))
    e_mlx, _ = _rel(_np(mlx16), _np(ref))
    assert e_fused < 3e-3 and e_fused <= 1.25 * e_mlx + 1e-6, (e_fused, e_mlx)
