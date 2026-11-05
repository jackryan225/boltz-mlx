"""Lane-level numpy emulation of the kernels_nax3 Metal kernels (CPU; used by tests/test_nax3_cpu.py).

The register-fragment kernels are blind-written for the GPU; their failure modes are index bugs (fragment rows /
columns, B^T fragment rows, f / f + 1 pairing, the xor-1 / xor-8 row groups, transposed rows, the last tile moved back,
clamped rows, conditioning rows). This module re-executes each kernel's index arithmetic lane by lane, with the NAX
16x32x16 multiply emulated on the fragments exactly as the layout contract states (tests/test_nax_layout.py):
    element r * 4 + j of a 16x16 fragment  <->  row fm + 8 r, column fn + j
    nax_mma(c0, c1, a, b0, b1): C[16 x 32] += A[16 x 16] B[16 x 32], with TB the B fragments are rows 0-15 / 16-31
    of B^T, without TB columns 0-15 / 16-31 of B; c0 / c1 = columns 0-15 / 16-31 of C.
It checks the index logic, not Metal semantics or NAX arithmetic (GPU tests in tests/test_nax3_kernels.py).
"""

from __future__ import annotations

import numpy as np

LANES = np.arange(32)
FM = ((LANES >> 2) & 4) | ((LANES >> 1) & 3)
FN = (((LANES >> 2) & 2) | (LANES & 1)) * 4
f16 = np.float16
f32 = np.float32


def mlx_sigmoid(x):
    x = np.asarray(x, f32)
    y = (f32(1) / (f32(1) + np.exp(np.abs(x)).astype(f32))).astype(f32)
    return np.where(x < 0, y, f32(1) - y).astype(f32)


_EI = np.arange(8)
ROWS = FM[:, None] + 8 * (_EI[None, :] // 4)     # [32, 8] row of element i of lane l
COLS = FN[:, None] + (_EI[None, :] % 4)          # [32, 8] column
assert len(set(zip(ROWS.ravel(), COLS.ravel()))) == 256  # the 32 x 8 elements tile the 16 x 16 fragment


def frag_to_mat(fr):
    """[32, 8] lane fragments -> 16 x 16 matrix."""
    m = np.zeros((16, 16), fr.dtype)
    m[ROWS, COLS] = fr
    return m


def mat_to_frag(m):
    return m[ROWS, COLS].copy()


def nax_mma(c0, c1, a, b0, b1, tb=True):
    A = frag_to_mat(a).astype(f32)
    if tb:
        B = np.concatenate([frag_to_mat(b0), frag_to_mat(b1)], 0).astype(f32).T  # B^T rows -> B [16, 32]
    else:
        B = np.concatenate([frag_to_mat(b0), frag_to_mat(b1)], 1).astype(f32)
    C = np.concatenate([frag_to_mat(c0), frag_to_mat(c1)], 1) + A @ B
    return mat_to_frag(C[:, :16].astype(f32)), mat_to_frag(C[:, 16:].astype(f32))


def frag2(flat, off_a, off_b):
    """Per-lane frag2(pa, pb): 4 consecutive halves at flat[off_a[l]] and flat[off_b[l]]."""
    j = np.arange(4)[None, :]
    return np.concatenate([flat[np.asarray(off_a)[:, None] + j], flat[np.asarray(off_b)[:, None] + j]], 1).astype(f16)


def shuf_sum(v):
    """s += simd_shuffle_xor(s, 1); s += simd_shuffle_xor(s, 8) over the 32 lanes."""
    v = v + v[LANES ^ 1]
    return v + v[LANES ^ 8]


def _ln_rows_epi(V, zrows, NN, D, lnw, lnb, eps, trans_o, ln_out, TMF, NF, wtb=None, tb_out=None):
    """_LN_EPI (_LN_EPI_TB with wtb [NTB, D] / tb_out [NTB * NN * NN]). V[t][f]: [32, 8] fp32; zrows[t][r]: [32]
    z-layout rows."""
    for t in range(TMF):
        for r in range(2):
            s = np.zeros(32, f32)
            for f in range(NF):
                for j in range(4):
                    s = s + V[t][f][:, r * 4 + j]
            mean = shuf_sum(s) / D
            ss = np.zeros(32, f32)
            for f in range(NF):
                for j in range(4):
                    d = V[t][f][:, r * 4 + j] - mean
                    ss = ss + d * d
            ss = shuf_sum(ss)
            rstd = (1.0 / np.sqrt(ss / D + eps)).astype(f32)
            ntb = 0 if wtb is None else wtb.shape[0]
            tbp = np.zeros((32, max(ntb, 1)), f32)
            orows = np.zeros(32, np.int64)
            for l in range(32):
                zr = zrows[t][r][l]
                orow = (zr % NN) * NN + zr // NN if trans_o else zr
                orows[l] = orow
                for f in range(NF):
                    c = f * 16 + FN[l]
                    for j in range(4):
                        o = (V[t][f][l, r * 4 + j] - mean[l]) * rstd[l]
                        if lnw is not None:
                            o = o * lnw[c + j]
                        if lnb is not None:
                            o = o + lnb[c + j]
                        ln_out[orow * D + c + j] = f16(o)
                        for q in range(ntb):
                            tbp[l, q] += f32(f16(o)) * f32(wtb[q, c + j])
            for q in range(ntb):
                tv = shuf_sum(tbp[:, q])
                for l in range(32):
                    if FN[l] == 0:
                        tb_out[q * NN * NN + orows[l]] = f16(tv[l])


def _tiles(R, TM, NSG):
    ntiles = (R + TM - 1) // TM
    ntg = (ntiles + NSG - 1) // NSG
    for tg in range(ntg):
        for sg in range(NSG):
            tile = tg * NSG + sg
            if tile < ntiles:
                yield min(tile * TM, R - TM)


def linear_addln(x, w, z, lnw, lnb, eps, *, bias=None, trans_y=False, trans_out=False, tm=16, nsg=4, wtb=None):
    """x [R, D] fp16, w [D, D] fp16, z [R, D] fp32 (numpy); returns (z_out, ln_out)."""
    R, D = z.shape
    NN = int(round(np.sqrt(R)))
    NF, TMF = D // 16, tm // 16
    xf, wf, zf = x.reshape(-1).astype(f16), w.reshape(-1).astype(f16), z.reshape(-1).astype(f32)
    z_out = np.full(R * D, np.nan, f32)
    ln_out = np.full(R * D, np.nan, f16)
    tb_out = np.full((0 if wtb is None else wtb.shape[0]) * R, np.nan, f16)
    for r0 in _tiles(R, tm, nsg):
        xrow = lambda t, r: r0 + t * 16 + FM + 8 * r  # noqa: E731
        zrow = (lambda t, r: (xrow(t, r) % NN) * NN + xrow(t, r) // NN) if trans_y else xrow
        acc = [[np.zeros((32, 8), f32) for _ in range(NF)] for _ in range(TMF)]
        for kf in range(NF):
            Af = []
            for t in range(TMF):
                pa = xrow(t, 0) * D + kf * 16 + FN
                Af.append(frag2(xf, pa, pa + 8 * D))
            for f in range(0, NF, 2):
                pw = (f * 16 + FM) * D + kf * 16 + FN
                w0, w1 = frag2(wf, pw, pw + 8 * D), frag2(wf, pw + 16 * D, pw + 24 * D)
                for t in range(TMF):
                    acc[t][f], acc[t][f + 1] = nax_mma(acc[t][f], acc[t][f + 1], Af[t], w0, w1)
        V = [[None] * NF for _ in range(TMF)]
        for t in range(TMF):
            for f in range(NF):
                y = acc[t][f].astype(f16)
                if bias is not None:
                    y = (y.astype(f32) + np.array([[bias[f * 16 + FN[l] + (i % 4)] for i in range(8)]
                                                   for l in range(32)], f32)).astype(f16)
                V[t][f] = np.zeros((32, 8), f32)
                for r in range(2):
                    zr = zrow(t, r)
                    for l in range(32):
                        for j in range(4):
                            c = f * 16 + FN[l] + j
                            v = zf[zr[l] * D + c] + f32(y[l, r * 4 + j])
                            V[t][f][l, r * 4 + j] = v
                            z_out[zr[l] * D + c] = v
        _ln_rows_epi(V, [[zrow(t, r) for r in range(2)] for t in range(TMF)], NN, D, lnw, lnb, eps, trans_out, ln_out,
                     TMF, NF, wtb, tb_out)
    if wtb is not None:
        return z_out.reshape(R, D), ln_out.reshape(R, D), tb_out.reshape(-1, R)
    return z_out.reshape(R, D), ln_out.reshape(R, D)


def trimul_tail_addln(o, wo, bo, eps_o, w_po, xn, w_go, z, lnw, lnb, eps, *, tm=16, nsg=4, wtb=None, ch=None):
    """o [D, R] fp16 channel-major; returns (z_out, ln_out). ch: the R5 tail3 order (p then g in `ch` column chunks of
    NF / ch fragments, the fp16 p overwritten in place by y16) instead of the shipped one (all p, then all g)."""
    R, D = z.shape
    NN = int(round(np.sqrt(R)))
    NF, TMF = D // 16, tm // 16
    of, xf = o.reshape(-1).astype(f16), xn.reshape(-1).astype(f16)
    wpf, wgf, zf = w_po.reshape(-1).astype(f16), w_go.reshape(-1).astype(f16), z.reshape(-1).astype(f32)
    z_out = np.full(R * D, np.nan, f32)
    ln_out = np.full(R * D, np.nan, f16)
    tb_out = np.full((0 if wtb is None else wtb.shape[0]) * R, np.nan, f16)
    for r0 in _tiles(R, tm, nsg):
        xrow = lambda t, r: r0 + t * 16 + FM + 8 * r  # noqa: E731
        Of = [[np.zeros((32, 8), f16) for _ in range(NF)] for _ in range(TMF)]
        for t in range(TMF):
            for kf in range(NF):
                for r in range(2):
                    for j in range(4):
                        Of[t][kf][:, r * 4 + j] = of[(kf * 16 + FN + j) * R + xrow(t, r)]
        for t in range(TMF):
            for r in range(2):
                s = np.zeros(32, f32)
                for kf in range(NF):
                    for j in range(4):
                        s = s + Of[t][kf][:, r * 4 + j].astype(f32)
                mean = shuf_sum(s) / D
                ss = np.zeros(32, f32)
                for kf in range(NF):
                    for j in range(4):
                        d = Of[t][kf][:, r * 4 + j].astype(f32) - mean
                        ss = ss + d * d
                rstd = (1.0 / np.sqrt(shuf_sum(ss) / D + eps_o)).astype(f32)
                for kf in range(NF):
                    for j in range(4):
                        c = kf * 16 + FN + j
                        y = (Of[t][kf][:, r * 4 + j].astype(f32) - mean) * rstd
                        if wo is not None:
                            y = y * wo[c]
                        if bo is not None:
                            y = y + bo[c]
                        Of[t][kf][:, r * 4 + j] = y.astype(f16)
        nch = ch or 1
        CF = NF // nch
        Ph = [[None] * NF for _ in range(TMF)]
        for c in range(nch):  # phase P (chunked when ch): fp16 p per fragment
            P = [[np.zeros((32, 8), f32) for _ in range(CF)] for _ in range(TMF)]
            for kf in range(NF):
                for f in range(0, CF, 2):
                    pw = ((c * CF + f) * 16 + FM) * D + kf * 16 + FN
                    w0, w1 = frag2(wpf, pw, pw + 8 * D), frag2(wpf, pw + 16 * D, pw + 24 * D)
                    for t in range(TMF):
                        P[t][f], P[t][f + 1] = nax_mma(P[t][f], P[t][f + 1], Of[t][kf], w0, w1)
            for t in range(TMF):
                for f in range(CF):
                    Ph[t][c * CF + f] = P[t][f].astype(f16)
        Y = [[None] * NF for _ in range(TMF)]
        for c in range(nch):  # phase G: y16 in place of p
            G = [[np.zeros((32, 8), f32) for _ in range(CF)] for _ in range(TMF)]
            for kf in range(NF):
                Af = []
                for t in range(TMF):
                    pa = xrow(t, 0) * D + kf * 16 + FN
                    Af.append(frag2(xf, pa, pa + 8 * D))
                for f in range(0, CF, 2):
                    pw = ((c * CF + f) * 16 + FM) * D + kf * 16 + FN
                    w0, w1 = frag2(wgf, pw, pw + 8 * D), frag2(wgf, pw + 16 * D, pw + 24 * D)
                    for t in range(TMF):
                        G[t][f], G[t][f + 1] = nax_mma(G[t][f], G[t][f + 1], Af[t], w0, w1)
            for t in range(TMF):
                for f in range(CF):
                    Y[t][c * CF + f] = (Ph[t][c * CF + f].astype(f32)
                                        * mlx_sigmoid(G[t][f].astype(f16).astype(f32))).astype(f16)
        V = [[None] * NF for _ in range(TMF)]
        for t in range(TMF):
            for f in range(NF):
                y = Y[t][f]
                V[t][f] = np.zeros((32, 8), f32)
                for r in range(2):
                    zr = xrow(t, r)
                    for l in range(32):
                        for j in range(4):
                            c = f * 16 + FN[l] + j
                            v = zf[zr[l] * D + c] + f32(y[l, r * 4 + j])
                            V[t][f][l, r * 4 + j] = v
                            z_out[zr[l] * D + c] = v
        _ln_rows_epi(V, [[xrow(t, r) for r in range(2)] for t in range(TMF)], NN, D, lnw, lnb, eps, False, ln_out,
                     TMF, NF, wtb, tb_out)
    if wtb is not None:
        return z_out.reshape(R, D), ln_out.reshape(R, D), tb_out.reshape(-1, R)
    return z_out.reshape(R, D), ln_out.reshape(R, D)


def gemm16(x, w, *, epi=0, ngrp=1, out_dtype=f16, bias=None, gate=None, gate_bias=None, gate_off=0, resid=None,
           ntok=None, rep=1, sm=32, sn=32, wm=2, wn=2, ku=2):
    M, K = x.shape
    NOUT = w.shape[0] // ngrp
    ntok = ntok or M
    TMF, TNF = sm // 16, sn // 16
    xf, wf = x.reshape(-1).astype(f16), w.reshape(-1).astype(f16)
    out = np.full((M, NOUT), np.nan, f32 if epi == 1 else out_dtype)
    gld = gate.shape[-1] if gate is not None else 0
    gflat = gate.reshape(-1) if gate is not None else None
    for ty in range((M + wm * sm - 1) // (wm * sm)):
        for tx in range(NOUT // (wn * sn)):
            for sg in range(wm * wn):
                r0 = (ty * wm + sg // wn) * sm
                c0 = (tx * wn + sg % wn) * sn
                if r0 >= M:
                    continue
                Ap = [[np.minimum(r0 + t * 16 + FM + 8 * r, M - 1) * K + FN for r in range(2)] for t in range(TMF)]
                Wp = (c0 + FM) * K + FN
                acc = [[[np.zeros((32, 8), f32) for _ in range(TNF)] for _ in range(TMF)] for _ in range(ngrp)]
                for k0 in range(0, K, 16 * ku):
                    for u in range(ku):
                        k = k0 + 16 * u
                        Af = [frag2(xf, Ap[t][0] + k, Ap[t][1] + k) for t in range(TMF)]
                        Bf = [[frag2(wf, Wp + (g * NOUT + f * 16) * K + k, Wp + (g * NOUT + f * 16) * K + k + 8 * K)
                               for f in range(TNF)] for g in range(ngrp)]
                        for g in range(ngrp):
                            for t in range(TMF):
                                for f in range(0, TNF, 2):
                                    acc[g][t][f], acc[g][t][f + 1] = nax_mma(acc[g][t][f], acc[g][t][f + 1], Af[t],
                                                                             Bf[g][f], Bf[g][f + 1])
                for t in range(TMF):
                    for r in range(2):
                        for l in range(32):
                            row = r0 + t * 16 + FM[l] + 8 * r
                            if row >= M:
                                continue
                            cr = ((row // ntok) // rep) * ntok + row % ntok
                            for f in range(TNF):
                                col = c0 + f * 16 + FN[l]
                                for j in range(4):
                                    a0 = acc[0][t][f][l, r * 4 + j]
                                    if epi == 0:
                                        out[row, col + j] = a0 + (bias[col + j] if bias is not None else 0.0)
                                    elif epi == 1:
                                        g = mlx_sigmoid(f32(gflat[cr * gld + gate_off + col + j])
                                                        + gate_bias[gate_off + col + j])
                                        out[row, col + j] = resid[row, col + j] + g * a0
                                    else:
                                        gt = acc[1][t][f][l, r * 4 + j]
                                        out[row, col + j] = f16(gt * mlx_sigmoid(gt) * a0 * acc[2][t][f][l, r * 4 + j])
    return out


def adaln16(a2, h1, b1, *, ntok, rep, L, layer, slot):
    M, D = a2.shape
    out = np.zeros((M, D), f16)
    off = (layer * 4 + slot) * D
    h1f = h1.reshape(-1)
    for row in range(M):
        cr = ((row // ntok) // rep) * ntok + row % ntok
        v = a2[row].astype(f32)
        mean = v.mean()
        rstd = 1.0 / np.sqrt(((v - mean) ** 2).mean() + 1e-5)
        base = cr * (L * 4 * D) + off
        sc = h1f[base: base + D].astype(f32) + b1[off: off + D]
        sh = h1f[base + D: base + 2 * D].astype(f32) + b1[off + D: off + 2 * D]
        out[row] = ((v - mean) * rstd * mlx_sigmoid(sc) + sh).astype(f16)
    return out


def transition_lnx_epilogue(z_new, ln1, ln2, wz, *, tm=16, nsg=8):
    """_LNX_EPI over z_new [R, D] (the transition's fp32 output), ln1 / ln2 = (w, b, eps) of the next-layer / s-track
    LayerNorms (None: not requested), wz [NHB, D]: per simdgroup tile (rows r0 .. r0 + TM - 1, the last
    tile moved back, duplicate clamped tiles included), per row lane l holds columns l + 32 i. Returns (xn [R, D] fp16
    or None, pb [NHB, R] fp16 or None)."""
    R, D = z_new.shape
    NV = D // 32
    (w1, b1, e1), (w2, b2, e2) = ln1 or (None, None, None), ln2 or (None, None, None)
    xn = np.full((R, D), np.nan, f16) if ln1 is not None else None
    nhb = 0 if wz is None else wz.shape[0]
    pb = np.full((nhb, R), np.nan, f16) if nhb else None
    ntiles = (R + tm - 1) // tm
    ntg = (ntiles + nsg - 1) // nsg
    cols = LANES[:, None] + 32 * np.arange(NV)[None, :]          # [32, NV]
    for tg in range(ntg):
        for sg in range(nsg):
            tile = min(tg * nsg + sg, ntiles - 1)                # clamped (LNX: no early return)
            r0 = min(tile * tm, R - tm)
            for rr in range(tm):
                row = r0 + rr
                v = z_new[row][cols].astype(f32)                 # [32, NV]
                s = v[:, 0].copy()
                for i in range(1, NV):
                    s = s + v[:, i]
                mean = f32(s.sum(dtype=np.float64)) / D           # simd_sum (order immaterial for the index check)
                d = v - mean
                ssum = f32((d * d).sum(dtype=np.float64))
                if xn is not None:
                    rstd = f32(1.0 / np.sqrt(ssum / D + e1))
                    y = d * rstd
                    if w1 is not None:
                        y = y * w1[cols]
                    if b1 is not None:
                        y = y + b1[cols]
                    xn[row][cols] = y.astype(f16)
                if nhb:
                    rstd2 = f32(1.0 / np.sqrt(ssum / D + e2))
                    y = d * rstd2
                    if w2 is not None:
                        y = y * w2[cols]
                    if b2 is not None:
                        y = y + b2[cols]
                    yh = y.astype(f16).astype(f32)
                    for h in range(nhb):
                        p = (yh * wz[h][cols].astype(f32)).sum(axis=1)   # per-lane partial
                        pb[h, row] = f16(p.sum())                         # simd_sum, lane h stores
    return xn, pb


# ----------------------------------------------------------------------------------------------- R4-pair
def transpose_ln16(o, w, b, eps, *, tp=32, nsg=4):
    """_TLN_SRC: o [D, R] fp16 -> [R, D] fp16 through the padded threadgroup tile, per position one simdgroup with
    lane l holding channels l + 32 i."""
    D, R = o.shape
    tps = tp + 2
    of = o.reshape(-1)
    out = np.full((R, D), np.nan, f16)
    nthreads = nsg * 32
    for tg in range((R + tp - 1) // tp):
        p0 = tg * tp
        tile = np.full(D * tps, np.nan, f16)
        for tid in range(nthreads):
            for e in range(tid, D * tp, nthreads):
                c, p = e // tp, e % tp
                pos = p0 + p
                tile[c * tps + p] = of[c * R + pos] if pos < R else f16(0)
        for sg in range(nsg):
            for p in range(sg, tp, nsg):
                pos = p0 + p
                if pos >= R:
                    break
                cols = LANES[:, None] + 32 * np.arange(D // 32)[None, :]
                v = tile[cols * tps + p].astype(f32)                     # [32, NV]
                assert np.isfinite(v).all()
                mean = f32(v.sum(dtype=np.float64)) / D
                d = v - mean
                rstd = f32(1.0 / np.sqrt(f32((d * d).sum(dtype=np.float64)) / D + eps))
                y = d * rstd
                if w is not None:
                    y = y * w[cols]
                if b is not None:
                    y = y + b[cols]
                out[pos][cols] = y.astype(f16)
    return out


def gemm2_gate_addln(on, w_po, xn, w_go, z, lnw, lnb, eps, *, tm=16, nsg=4, wtb=None):
    """_TAIL2_SRC: both GEMMs with vector fragment loads from row-major on / xn, _gate16, +z, LN_next [+ tb]."""
    R, D = z.shape
    NN = int(round(np.sqrt(R)))
    NF, TMF = D // 16, tm // 16
    onf, xf = on.reshape(-1).astype(f16), xn.reshape(-1).astype(f16)
    wpf, wgf, zf = w_po.reshape(-1).astype(f16), w_go.reshape(-1).astype(f16), z.reshape(-1).astype(f32)
    z_out = np.full(R * D, np.nan, f32)
    ln_out = np.full(R * D, np.nan, f16)
    tb_out = np.full((0 if wtb is None else wtb.shape[0]) * R, np.nan, f16)
    for r0 in _tiles(R, tm, nsg):
        xrow = lambda t, r: r0 + t * 16 + FM + 8 * r  # noqa: E731
        acc = {}
        for name, a_flat, w_flat in (("P", onf, wpf), ("G", xf, wgf)):
            A = [[np.zeros((32, 8), f32) for _ in range(NF)] for _ in range(TMF)]
            for kf in range(NF):
                Af = []
                for t in range(TMF):
                    pa = xrow(t, 0) * D + kf * 16 + FN
                    Af.append(frag2(a_flat, pa, pa + 8 * D))
                for f in range(0, NF, 2):
                    pw = (f * 16 + FM) * D + kf * 16 + FN
                    w0, w1 = frag2(w_flat, pw, pw + 8 * D), frag2(w_flat, pw + 16 * D, pw + 24 * D)
                    for t in range(TMF):
                        A[t][f], A[t][f + 1] = nax_mma(A[t][f], A[t][f + 1], Af[t], w0, w1)
            acc[name] = A
        V = [[None] * NF for _ in range(TMF)]
        for t in range(TMF):
            for f in range(NF):
                ph = acc["P"][t][f].astype(f16).astype(f32)
                y = (ph * mlx_sigmoid(acc["G"][t][f].astype(f16).astype(f32))).astype(f16)
                V[t][f] = np.zeros((32, 8), f32)
                for r in range(2):
                    zr = xrow(t, r)
                    for l in range(32):
                        for j in range(4):
                            c = f * 16 + FN[l] + j
                            v = zf[zr[l] * D + c] + f32(y[l, r * 4 + j])
                            V[t][f][l, r * 4 + j] = v
                            z_out[zr[l] * D + c] = v
        _ln_rows_epi(V, [[xrow(t, r) for r in range(2)] for t in range(TMF)], NN, D, lnw, lnb, eps, False, ln_out,
                     TMF, NF, wtb, tb_out)
    if wtb is not None:
        return z_out.reshape(R, D), ln_out.reshape(R, D), tb_out.reshape(-1, R)
    return z_out.reshape(R, D), ln_out.reshape(R, D)




# ----------------------------------------------------------------------------------------------- R6-trans
def transition_reg(xn, w12, w3, z, ln1=None, ln2=None, wz=None, *, nsg=4, hb_size=32):
    """_RTRANS_SRC + _RT_LN_EPI: per 16-row tile (last tile moved back), O[16 x D] over hidden blocks, Sh as A
    fragments, z_new = z + float(half(O)); next LN / s-track LN per fragment; pair bias as NAX MMAs of the LN_s
    fragments with Wz (second B^T fragment zero), stored head-major. ln1 / ln2 = (w, b, eps) or None."""
    R, D = z.shape
    H = w3.shape[1]
    NF, HF = D // 16, hb_size // 16
    xf, w12f, w3f, zf = (xn.reshape(-1).astype(f16), w12.reshape(-1).astype(f16), w3.reshape(-1).astype(f16),
                         z.reshape(-1).astype(f32))
    z_out = np.full(R * D, np.nan, f32)
    xo = np.full(R * D, np.nan, f16) if ln1 is not None else None
    nhb = 0 if wz is None else wz.shape[0]
    pb = np.full(nhb * R, np.nan, f16) if nhb else None
    wzf = wz.reshape(-1).astype(f16) if nhb else None
    zero = np.zeros((32, 8), f16)
    for r0 in _tiles(R, 16, nsg):
        row = lambda r: r0 + FM + 8 * r  # noqa: E731
        O = [np.zeros((32, 8), f32) for _ in range(NF)]
        for hb in range(0, H, hb_size):
            Sa = [np.zeros((32, 8), f32) for _ in range(HF)]
            Sb = [np.zeros((32, 8), f32) for _ in range(HF)]
            for kf in range(NF):
                pa = row(0) * D + kf * 16 + FN
                Af = frag2(xf, pa, pa + 8 * D)
                for f in range(0, HF, 2):
                    p1 = (hb + f * 16 + FM) * D + kf * 16 + FN
                    p2 = p1 + H * D
                    a0, a1 = frag2(w12f, p1, p1 + 8 * D), frag2(w12f, p1 + 16 * D, p1 + 24 * D)
                    b0, b1 = frag2(w12f, p2, p2 + 8 * D), frag2(w12f, p2 + 16 * D, p2 + 24 * D)
                    Sa[f], Sa[f + 1] = nax_mma(Sa[f], Sa[f + 1], Af, a0, a1)
                    Sb[f], Sb[f + 1] = nax_mma(Sb[f], Sb[f + 1], Af, b0, b1)
            Sh = []
            for f in range(HF):
                a = Sa[f]
                Sh.append((a * (f32(1) / (f32(1) + np.exp(-a).astype(f32))) * Sb[f]).astype(f16))
            for kh in range(HF):
                for f in range(0, NF, 2):
                    pw = (f * 16 + FM) * H + hb + kh * 16 + FN
                    w0, w1 = frag2(w3f, pw, pw + 8 * H), frag2(w3f, pw + 16 * H, pw + 24 * H)
                    O[f], O[f + 1] = nax_mma(O[f], O[f + 1], Sh[kh], w0, w1)
        V = [np.zeros((32, 8), f32) for _ in range(NF)]
        for f in range(NF):
            y = O[f].astype(f16)
            for r in range(2):
                for l in range(32):
                    for j in range(4):
                        c = f * 16 + FN[l] + j
                        v = zf[row(r)[l] * D + c] + f32(y[l, r * 4 + j])
                        V[f][l, r * 4 + j] = v
                        z_out[row(r)[l] * D + c] = v
        if ln1 is None and not nhb:
            continue
        mean, rn, rs = [None, None], [None, None], [None, None]
        for r in range(2):
            s = np.zeros(32, f32)
            for f in range(NF):
                for j in range(4):
                    s = s + V[f][:, r * 4 + j]
            mean[r] = shuf_sum(s) / D
            ss = np.zeros(32, f32)
            for f in range(NF):
                for j in range(4):
                    d = V[f][:, r * 4 + j] - mean[r]
                    ss = ss + d * d
            ss = shuf_sum(ss)
            rn[r] = (1.0 / np.sqrt(ss / D + (ln1[2] if ln1 else 1e-5))).astype(f32)
            rs[r] = (1.0 / np.sqrt(ss / D + (ln2[2] if ln2 else 1e-5))).astype(f32)
        B0, B1 = np.zeros((32, 8), f32), np.zeros((32, 8), f32)
        for f in range(NF):
            ls = np.zeros((32, 8), f16)
            for r in range(2):
                for l in range(32):
                    for j in range(4):
                        c = f * 16 + FN[l] + j
                        d = V[f][l, r * 4 + j] - mean[r][l]
                        if ln1 is not None:
                            o = d * rn[r][l]
                            if ln1[0] is not None:
                                o = o * ln1[0][c]
                            if ln1[1] is not None:
                                o = o + ln1[1][c]
                            xo[row(r)[l] * D + c] = f16(o)
                        if nhb:
                            yv = d * rs[r][l]
                            if ln2[0] is not None:
                                yv = yv * ln2[0][c]
                            if ln2[1] is not None:
                                yv = yv + ln2[1][c]
                            ls[l, r * 4 + j] = f16(yv)
            if nhb:
                pz = FM * D + f * 16 + FN
                B0, B1 = nax_mma(B0, B1, ls, frag2(wzf, pz, pz + 8 * D), zero)
        if nhb:
            for r in range(2):
                for l in range(32):
                    for j in range(4):
                        pb[(FN[l] + j) * R + row(r)[l]] = f16(B0[l, r * 4 + j])
    return z_out.reshape(R, D), (xo.reshape(R, D) if xo is not None else None), (pb.reshape(nhb, R) if nhb else None)
