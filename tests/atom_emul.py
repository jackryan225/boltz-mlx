"""Lane-level numpy emulation of the kernels_atom Metal kernels (CPU; used by tests/test_atom_cpu.py).

Re-executes each kernel's index arithmetic lane by lane (threadgroup -> simdgroup -> lane, fragment rows / columns,
key-window atom rows with clamping, the shared-bias window, xor-1 / xor-8 row groups, online softmax across key
blocks, P as the A operand of P V, the gated fp16 store), with the NAX 16x32x16 multiply emulated on fragments as the
layout contract states (tests/nax3_emul.py, pinned on the device by tests/test_nax_layout.py). Index logic only — not
Metal semantics or NAX arithmetic (GPU tests: tests/test_atom_kernels.py).
"""

from __future__ import annotations

import numpy as np

from nax3_emul import FM, FN, LANES, f16, f32, mlx_sigmoid, nax_mma

LOG2E = 1.4426950408889634


def _ld4(flat, off):
    """Per-lane 4 consecutive elements at flat[off[l]] -> [32, 4]."""
    return flat[np.asarray(off)[:, None] + np.arange(4)[None, :]]


def atom_attention(h, bias, *, NA, W=32, H=128, heads=4, c=32, qbias=None, brep=1, hpt=4, nkf=4, pt="float",
                   order=0):
    """h [M, F] fp16, bias [nwin / brep, heads, W, H] fp32 -> out [M, heads * c] fp16 (NaN where never written)."""
    M, F = h.shape
    HC, TD, KOFF = heads * c, c // 16, (H - W) // 2
    NWS, nwin = NA // W, M // W
    hf = np.asarray(h, f16).reshape(-1)
    bf = np.asarray(bias, f32).reshape(-1)
    qb = np.zeros(HC, f32) if qbias is None else np.asarray(qbias, f32)
    out = np.full(M * HC, np.nan, f16)
    scale2 = f32(c ** -0.5 * LOG2E)
    nsg = (W // 16) * hpt
    NS = M // NA
    for tgx in range(nwin):                                   # threadgroup_position_in_grid.x
        win = tgx if order == 0 else (tgx % NS) * NWS + tgx // NS
        for hg in range(heads // hpt):                        # threadgroup_position_in_grid.y
            for sg in range(nsg):
                hh = hg * hpt + sg % hpt
                q0 = (sg // hpt) * 16
                smp, kw = win // NWS, win % NWS
                hq = win * W * F
                hk = smp * NA * F
                brow = (((smp // brep) * NWS + kw) * heads + hh) * (W * H)
                kbase = kw * W - KOFF
                Qf = []
                for db in range(TD):
                    cc = hh * c + db * 16 + FN
                    va = _ld4(hf, hq + (q0 + FM) * F + cc).astype(f32)
                    vb = _ld4(hf, hq + (q0 + FM + 8) * F + cc).astype(f32)
                    qbl = qb[cc[:, None] + np.arange(4)[None, :]]
                    Qf.append(np.concatenate([(va + qbl).astype(f16), (vb + qbl).astype(f16)], 1))
                O = [np.zeros((32, 8), f32) for _ in range(TD)]
                m_r = np.full((2, 32), -3.0e38, f32)
                l_r = np.zeros((2, 32), f32)
                for k0 in range(0, H, 16 * nkf):
                    ka = [np.clip(kbase + k0 + f * 16 + FM, 0, NA - 1) * F for f in range(nkf)]
                    kb = [np.clip(kbase + k0 + f * 16 + FM + 8, 0, NA - 1) * F for f in range(nkf)]
                    bl = []
                    for f in range(nkf):
                        kc = k0 + f * 16 + FN
                        ba = _ld4(bf, brow + (q0 + FM) * H + kc)
                        bb = _ld4(bf, brow + (q0 + FM + 8) * H + kc)
                        bl.append(np.concatenate([ba, bb], 1).astype(f32) * f32(1.44269504089))
                    S = [np.zeros((32, 8), f32) for _ in range(nkf)]
                    for db in range(TD):
                        cc = HC + hh * c + db * 16 + FN
                        Kf = [np.concatenate([_ld4(hf, hk + ka[f] + cc), _ld4(hf, hk + kb[f] + cc)], 1)
                              for f in range(nkf)]
                        for f in range(0, nkf, 2):
                            S[f], S[f + 1] = nax_mma(S[f], S[f + 1], Qf[db], Kf[f], Kf[f + 1], tb=True)
                    for f in range(nkf):
                        S[f] = (S[f] * scale2 + bl[f]).astype(f32)
                    for r in range(2):
                        tmax = np.full(32, -3.0e38, f32)
                        for f in range(nkf):
                            tmax = np.maximum(tmax, S[f][:, r * 4: r * 4 + 4].max(1))
                        tmax = np.maximum(tmax, tmax[LANES ^ 1])
                        tmax = np.maximum(tmax, tmax[LANES ^ 8])
                        m_new = np.maximum(m_r[r], tmax)
                        corr = np.exp2(m_r[r] - m_new).astype(f32)
                        ls = np.zeros(32, f32)
                        for f in range(nkf):
                            p = np.exp2(S[f][:, r * 4: r * 4 + 4] - m_new[:, None]).astype(f32)
                            S[f][:, r * 4: r * 4 + 4] = p
                            ls = ls + p.sum(1)
                        l_r[r] = l_r[r] * corr + ls
                        m_r[r] = m_new
                        for d in range(TD):
                            O[d][:, r * 4: r * 4 + 4] *= corr[:, None]
                    for f in range(nkf):
                        P = S[f].astype(f16) if pt == "half" else S[f]
                        for dp in range(0, TD, 2):
                            cc = 2 * HC + hh * c + dp * 16 + FN
                            V0 = np.concatenate([_ld4(hf, hk + ka[f] + cc), _ld4(hf, hk + kb[f] + cc)], 1)
                            V1 = np.concatenate([_ld4(hf, hk + ka[f] + cc + 16), _ld4(hf, hk + kb[f] + cc + 16)], 1)
                            O[dp], O[dp + 1] = nax_mma(O[dp], O[dp + 1], P, V0, V1, tb=False)
                for r in range(2):
                    lsum = l_r[r] + l_r[r][LANES ^ 1]
                    lsum = lsum + lsum[LANES ^ 8]
                    inv = (f32(1.0) / lsum).astype(f32)
                    qi = q0 + FM + 8 * r
                    for d in range(TD):
                        cc = hh * c + d * 16 + FN
                        g = _ld4(hf, hq + qi * F + 3 * HC + cc).astype(f32)
                        o = (O[d][:, r * 4: r * 4 + 4] * inv[:, None] * mlx_sigmoid(g)).astype(f16)
                        dst = (win * W + qi) * HC + cc
                        for j in range(4):
                            out[dst + j] = o[:, j]
    return out.reshape(M, HC)


def segment_mean_relu(y, off, idx, wt):
    """_SEG_SRC thread by thread: y [S, NA, D] fp32."""
    y = np.asarray(y, f32)
    S, NA, D = y.shape
    T = wt.shape[0]
    xs = y.reshape(-1)
    out = np.full(S * T * D, np.nan, f32)
    for s in range(S):
        for t in range(T):
            for c4 in range(D // 4):
                acc = np.zeros(4, f32)
                for p in range(off[t], off[t + 1]):
                    base = ((s * NA + idx[p]) * (D // 4) + c4) * 4
                    acc = acc + np.maximum(xs[base: base + 4], 0) * wt[t]
                o = ((s * T + t) * (D // 4) + c4) * 4
                out[o: o + 4] = acc
    return out.reshape(S, T, D)


def frag2(flat, off_a, off_b):
    j = np.arange(4)[None, :]
    return np.concatenate([flat[np.asarray(off_a)[:, None] + j], flat[np.asarray(off_b)[:, None] + j]], 1).astype(f16)


def _tiles(M, TM, NSG):
    ntiles = (M + TM - 1) // TM
    for tg in range((ntiles + NSG - 1) // NSG):
        for sg in range(NSG):
            tile = tg * NSG + sg
            if tile < ntiles:
                yield min(tile * TM, M - TM)


def atom_transition(a2, h1, b1, h2, b2, wff, wba, *, ntok, rep, L, layer, tm=16, nsg=4):
    """_TRANS_SRC lane by lane: a2 [M, D] fp32, h1 [Bc * ntok, L * 4 * D] fp16, h2 [Bc * ntok, L * 2 * D] fp16,
    wff [3 HID, D] fp16, wba [D, HID] fp16 -> [M, D] fp32 (NaN where never written)."""
    a2 = np.asarray(a2, f32)
    M, D = a2.shape
    HID = wba.shape[1]
    KF, TMF, OF = D // 16, tm // 16, D // 16
    H1LD, GLD = L * 4 * D, L * 2 * D
    off1, off2 = (layer * 4 + 2) * D, (layer * 2 + 1) * D
    af = a2.reshape(-1)
    h1f, h2f = np.asarray(h1, f16).reshape(-1), np.asarray(h2, f16).reshape(-1)
    wf, wb = np.asarray(wff, f16).reshape(-1), np.asarray(wba, f16).reshape(-1)
    out = np.full(M * D, np.nan, f32)
    j4 = np.arange(4)[None, :]
    for r0 in _tiles(M, tm, nsg):
        Af = [[np.zeros((32, 8), f16) for _ in range(KF)] for _ in range(TMF)]
        for t in range(TMF):
            for r in range(2):
                row = r0 + t * 16 + FM + 8 * r
                cr = ((row // ntok) // rep) * ntok + row % ntok
                v = [af[(row * D + kf * 16 + FN)[:, None] + j4] for kf in range(KF)]
                s = sum(x.sum(1) for x in v).astype(f32)
                s = s + s[LANES ^ 1]
                s = s + s[LANES ^ 8]
                mean = (s / D).astype(f32)
                ss = sum(((x - mean[:, None]) ** 2).sum(1) for x in v).astype(f32)
                ss = ss + ss[LANES ^ 1]
                ss = ss + ss[LANES ^ 8]
                rstd = (1.0 / np.sqrt(ss / D + 1e-5)).astype(f32)
                for kf in range(KF):
                    c = kf * 16 + FN
                    sc = h1f[(cr * H1LD + off1 + c)[:, None] + j4].astype(f32) + b1[off1 + c[:, None] + j4]
                    sh = h1f[(cr * H1LD + off1 + D + c)[:, None] + j4].astype(f32) + b1[off1 + D + c[:, None] + j4]
                    Af[t][kf][:, r * 4: r * 4 + 4] = ((v[kf] - mean[:, None]) * rstd[:, None] * mlx_sigmoid(sc)
                                                      + sh).astype(f16)
        O = [[np.zeros((32, 8), f32) for _ in range(OF)] for _ in range(TMF)]
        for hb in range(0, HID, 32):
            X = [[np.zeros((32, 8), f32) for _ in range(2)] for _ in range(TMF)]
            G = [[np.zeros((32, 8), f32) for _ in range(2)] for _ in range(TMF)]
            AB = [[np.zeros((32, 8), f32) for _ in range(2)] for _ in range(TMF)]
            for kf in range(KF):
                px = (hb + FM) * D + kf * 16 + FN
                pg = px + HID * D
                pb = pg + HID * D
                x0, x1 = frag2(wf, px, px + 8 * D), frag2(wf, px + 16 * D, px + 24 * D)
                g0, g1 = frag2(wf, pg, pg + 8 * D), frag2(wf, pg + 16 * D, pg + 24 * D)
                e0, e1 = frag2(wf, pb, pb + 8 * D), frag2(wf, pb + 16 * D, pb + 24 * D)
                for t in range(TMF):
                    X[t][0], X[t][1] = nax_mma(X[t][0], X[t][1], Af[t][kf], x0, x1)
                    G[t][0], G[t][1] = nax_mma(G[t][0], G[t][1], Af[t][kf], g0, g1)
                    AB[t][0], AB[t][1] = nax_mma(AB[t][0], AB[t][1], Af[t][kf], e0, e1)
            Hs = [[(G[t][f] * mlx_sigmoid(G[t][f]) * X[t][f] * AB[t][f]).astype(f16) for f in range(2)]
                  for t in range(TMF)]
            for kf2 in range(2):
                for of in range(0, OF, 2):
                    pw = (of * 16 + FM) * HID + hb + kf2 * 16 + FN
                    w0, w1 = frag2(wb, pw, pw + 8 * HID), frag2(wb, pw + 16 * HID, pw + 24 * HID)
                    for t in range(TMF):
                        O[t][of], O[t][of + 1] = nax_mma(O[t][of], O[t][of + 1], Hs[t][kf2], w0, w1)
        for t in range(TMF):
            for r in range(2):
                row = r0 + t * 16 + FM + 8 * r
                cr = ((row // ntok) // rep) * ntok + row % ntok
                for f in range(OF):
                    col = f * 16 + FN
                    gh = h2f[(cr * GLD + off2 + col)[:, None] + j4].astype(f32)
                    gb = b2[off2 + col[:, None] + j4]
                    rs = af[(row * D + col)[:, None] + j4]
                    o = rs + mlx_sigmoid(gh + gb) * O[t][f][:, r * 4: r * 4 + 4]
                    out[(row * D + col)[:, None] + j4] = o
    return out.reshape(M, D)


def _adaln_frags(af, h1f, b1, r0, *, D, TMF, ntok, rep, H1LD, off1):
    """The _TRANS_SRC / _ADALN_GEMM_SRC prologue: Af[t][kf] [32, 8] fp16 for rows r0 .. r0 + 16 TMF."""
    KF = D // 16
    j4 = np.arange(4)[None, :]
    Af = [[np.zeros((32, 8), f16) for _ in range(KF)] for _ in range(TMF)]
    for t in range(TMF):
        for r in range(2):
            row = r0 + t * 16 + FM + 8 * r
            cr = ((row // ntok) // rep) * ntok + row % ntok
            v = [af[(row * D + kf * 16 + FN)[:, None] + j4] for kf in range(KF)]
            s = sum(x.sum(1) for x in v).astype(f32)
            s = s + s[LANES ^ 1]
            s = s + s[LANES ^ 8]
            mean = (s / D).astype(f32)
            ss = sum(((x - mean[:, None]) ** 2).sum(1) for x in v).astype(f32)
            ss = ss + ss[LANES ^ 1]
            ss = ss + ss[LANES ^ 8]
            rstd = (1.0 / np.sqrt(ss / D + 1e-5)).astype(f32)
            for kf in range(KF):
                c = kf * 16 + FN
                sc = h1f[(cr * H1LD + off1 + c)[:, None] + j4].astype(f32) + b1[off1 + c[:, None] + j4]
                sh = h1f[(cr * H1LD + off1 + D + c)[:, None] + j4].astype(f32) + b1[off1 + D + c[:, None] + j4]
                Af[t][kf][:, r * 4: r * 4 + 4] = ((v[kf] - mean[:, None]) * rstd[:, None] * mlx_sigmoid(sc)
                                                  + sh).astype(f16)
    return Af


def adaln_gemm(a2, h1, b1, w, *, ntok, rep, L, layer, slot, tm=16, nsg=4):
    """_ADALN_GEMM_SRC lane by lane -> [M, NOUT] fp16 (NaN where never written)."""
    a2 = np.asarray(a2, f32)
    M, D = a2.shape
    NOUT = w.shape[0]
    KF, TMF = D // 16, tm // 16
    af = a2.reshape(-1)
    h1f, wf = np.asarray(h1, f16).reshape(-1), np.asarray(w, f16).reshape(-1)
    out = np.full(M * NOUT, np.nan, f16)
    j4 = np.arange(4)[None, :]
    for r0 in _tiles(M, tm, nsg):
        Af = _adaln_frags(af, h1f, b1, r0, D=D, TMF=TMF, ntok=ntok, rep=rep, H1LD=L * 4 * D,
                          off1=(layer * 4 + slot) * D)
        for nb in range(0, NOUT, 32):
            C = [[np.zeros((32, 8), f32) for _ in range(2)] for _ in range(TMF)]
            for kf in range(KF):
                pw = (nb + FM) * D + kf * 16 + FN
                w0, w1 = frag2(wf, pw, pw + 8 * D), frag2(wf, pw + 16 * D, pw + 24 * D)
                for t in range(TMF):
                    C[t][0], C[t][1] = nax_mma(C[t][0], C[t][1], Af[t][kf], w0, w1)
            for t in range(TMF):
                for r in range(2):
                    row = r0 + t * 16 + FM + 8 * r
                    for f in range(2):
                        out[(row * NOUT + nb + f * 16 + FN)[:, None] + j4] = C[t][f][:, r * 4: r * 4 + 4].astype(f16)
    return out.reshape(M, NOUT)
