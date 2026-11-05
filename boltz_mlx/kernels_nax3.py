"""NAX round 3 (speed-first round, --precision f16 only): GEMMs with fused prologues / epilogues. GPU only.

All matmuls here use the *register-fragment* NAX path of kernels_nax.py: `nax_mma<float, half, half, false, TB, true>`
(the relaxed 16x32x16 descriptor, left operand not transposed, fp16 operands, fp32 accumulate). That is the one
descriptor family whose cooperative-tensor layout is pinned to MLX's BaseNAXFrag layout on the device
(tests/test_nax_layout.py: relaxed, TA = false, TB = true / false, AT = half). Because the layout is known, a lane
knows the (row, column) of every accumulator element:
    element r * 4 + j of a 16x16 fragment  <->  row fm + 8 r, column fn + j      (NAX_COORD)
and the 4 lanes that share a row are lane ^ 1 and lane ^ 8, so row reductions (LayerNorm statistics of an output
row) are two simd shuffles. Fragments are loaded straight from device memory (L1-cached, as MLX's own NAX GEMM does);
no threadgroup memory, no barriers.

Kernels
  linear_addln       pair stack, width D = 128 (one simdgroup owns whole 128-channel rows):
                       y = half(x W^T) [+ bias] ; z_new = z + float(y) (fp32, = kernels.add_ln / radd arithmetic) ;
                       ln = LN_next(z_new) fp16, optionally transposed (ending-node layout); x rows optionally read
                       transposed. Replaces lin16(core, linear_o) + add_ln of the two triangle attentions.
  trimul_tail_addln  pair stack, tri-mult output: on = LN_out(o^T) (o channel-major, 2^-2K-scaled, eps * 2^-4K),
                       p = half(on Wpo^T), g = half(xn Wgo^T), y = half(p * sigmoid(g)) ; z_new = z + y ; LN_next.
                       Replaces norm_out (+ its transposed copy), p_out, g_out, the gate and add_ln.
  adaln16            diffusion token transformer: t = sigmoid(sc) * LN(a) + sh in fp16, sc / sh read from the raw
                       fp16 conditioning GEMM outputs (+ fp32 bias) in place (RawCond16).
  gemm16             y = epilogue(x W^T), x [M, K] fp16, W [NGRP * NOUT, K] fp16, fp32 accumulate. Epilogues:
                       EPI_STORE  : OT(acc [+ bias])
                       EPI_GRESID : resid + sigmoid(float(gate16) + gate_bias) * acc   (fp32; gated residual add)
                       EPI_SWIGLU : half(silu(acc_g) * acc_x * acc_ab)                  (NGRP = 3 weight groups)
  Rounding vs the MLX f16 path: the pair kernels keep every fp16 rounding point of lin16 / _gate16 / add_ln (only
  the LN statistics are summed in a different order); gemm16 epilogues use the fp32 accumulator directly where the
  MLX path rounded the GEMM output to fp16 first (more accurate, never less).

Every Python wrapper falls back by raising ValueError below its minimum size; callers check `*_ok()` first.
sigmoid is MLX's formula (unary_ops.h Sigmoid) so gated values match mx.sigmoid.
"""

from __future__ import annotations

import os

import mlx.core as mx

from boltz_mlx import kernels as _K  # metal_kernel(): the MPP guard (portability)

from boltz_mlx.kernels_nax import NAX_HEADER

_kernels: dict = {}

HEADER = NAX_HEADER + r"""
// MLX's sigmoid (mlx/backend/metal/kernels/unary_ops.h), so fused gates equal mx.sigmoid bit for bit
inline float mlx_sigmoid(float x) {
    const float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(x)));
    return (x < 0.0f) ? y : 1.0f - y;
}
// A / B^T fragment of a row-major fp16 matrix: 4 consecutive halves at pa (row fm) and pb (row fm + 8)
template <typename P> inline metal::vec<half, 8> frag2(P pa, P pb) {
    metal::vec<half, 8> f;
    const auto a = ld4(pa);
    const auto b = ld4(pb);
    UNR for (short j = 0; j < 4; j++) { f[j] = (half)a[j]; f[4 + j] = (half)b[j]; }
    return f;
}
"""

# ---------------------------------------------------------------------------------------------------------------
# Pair stack: LayerNorm of the new residual rows in the epilogue (rows of D = 128 channels, one simdgroup owns TM
# whole rows). V[t][f] holds z_new (fp32) for the lane's elements; ZROW(t, r) is the z-layout row of fragment row
# fm + 8 r of row block t. Same arithmetic as kernels._ADD_LN_SRC (two-pass mean / variance in fp32, precise rsqrt,
# (v - mean) * rstd * w + b, fp16 store); the statistics are summed lane-partial -> xor 1 -> xor 8.
# ---------------------------------------------------------------------------------------------------------------
_LN_EPI = """
    const float eps_n = epsn[0];
    UNR for (int t = 0; t < TMF; t++) UNR for (short r = 0; r < 2; r++) {
        float s = 0.0f;
        UNR for (int f = 0; f < NF; f++) UNR for (short j = 0; j < 4; j++) s += V[t][f][r * 4 + j];
        s += simd_shuffle_xor(s, (ushort)1);
        s += simd_shuffle_xor(s, (ushort)8);
        const float mean = s / D;
        float ss = 0.0f;
        UNR for (int f = 0; f < NF; f++) UNR for (short j = 0; j < 4; j++) {
            const float d = V[t][f][r * 4 + j] - mean;
            ss += d * d;
        }
        ss += simd_shuffle_xor(ss, (ushort)1);
        ss += simd_shuffle_xor(ss, (ushort)8);
        const float rstd = metal::precise::rsqrt(ss / D + eps_n);
        const int zr = ZROW(t, r);
        const int orow = TRANS_O ? (zr % NN) * NN + zr / NN : zr;
        UNR for (int f = 0; f < NF; f++) {
            const int c = f * 16 + fn;
            metal::vec<half, 4> h;
            UNR for (short j = 0; j < 4; j++) {
                float o = (V[t][f][r * 4 + j] - mean) * rstd;
                if (HAS_W) o = o * lnw[c + j];
                if (HAS_B) o = o + lnb[c + j];
                h[j] = half(o);
            }
            *(device metal::vec<half, 4>*)(ln_out + (size_t)orow * D + c) = h;
        }
    }
"""

# _LN_EPI plus a small projection of the fp16 LN output rows (R3-pair: the triangle attention's tri bias, NTB = 4
# heads): tb_out[q, orow] = half(sum_c half(LN)[c] * wtb[q, c]), head-major [NTB, N * N] in the LN output's layout (=
# (ln @ wtb^T).transpose(0, 3, 1, 2), what the NAX tri-att kernel reads). Row sums: lane partial -> xor 1 -> xor 8; the
# lane with fn == 0 of each row's 4-lane group stores.
_LN_EPI_TB = """
    const float eps_n = epsn[0];
    UNR for (int t = 0; t < TMF; t++) UNR for (short r = 0; r < 2; r++) {
        float s = 0.0f;
        UNR for (int f = 0; f < NF; f++) UNR for (short j = 0; j < 4; j++) s += V[t][f][r * 4 + j];
        s += simd_shuffle_xor(s, (ushort)1);
        s += simd_shuffle_xor(s, (ushort)8);
        const float mean = s / D;
        float ss = 0.0f;
        UNR for (int f = 0; f < NF; f++) UNR for (short j = 0; j < 4; j++) {
            const float d = V[t][f][r * 4 + j] - mean;
            ss += d * d;
        }
        ss += simd_shuffle_xor(ss, (ushort)1);
        ss += simd_shuffle_xor(ss, (ushort)8);
        const float rstd = metal::precise::rsqrt(ss / D + eps_n);
        const int zr = ZROW(t, r);
        const int orow = TRANS_O ? (zr % NN) * NN + zr / NN : zr;
        float tbp[NTB];
        UNR for (int q = 0; q < NTB; q++) tbp[q] = 0.0f;
        UNR for (int f = 0; f < NF; f++) {
            const int c = f * 16 + fn;
            metal::vec<half, 4> h;
            UNR for (short j = 0; j < 4; j++) {
                float o = (V[t][f][r * 4 + j] - mean) * rstd;
                if (HAS_W) o = o * lnw[c + j];
                if (HAS_B) o = o + lnb[c + j];
                h[j] = half(o);
                UNR for (int q = 0; q < NTB; q++) tbp[q] += float(h[j]) * float(wtb[q * D + c + j]);
            }
            *(device metal::vec<half, 4>*)(ln_out + (size_t)orow * D + c) = h;
        }
        UNR for (int q = 0; q < NTB; q++) {
            float tv = tbp[q];
            tv += simd_shuffle_xor(tv, (ushort)1);
            tv += simd_shuffle_xor(tv, (ushort)8);
            if (fn == 0) tb_out[(size_t)q * NN * NN + orow] = half(tv);
        }
    }
"""

_TILE_PROLOGUE = """
    constexpr int NF = D / 16;                        // 16-wide channel fragments of a row
    constexpr int TMF = TM / 16;                      // 16-row blocks per simdgroup
    const int R = (int)nrows[0];                      // runtime (no recompile per N)
    const uint lane = thread_index_in_simdgroup;
    const int tile = (int)(threadgroup_position_in_grid.x * NSG + simdgroup_index_in_threadgroup);
    const int ntiles = (R + TM - 1) / TM;
    if (tile >= ntiles) return;
    const int r0 = min(tile * TM, R - TM);            // last tile moved back inside (rewrites identical values)
    NAX_COORD(lane, fm, fn)
"""

# z_new = z + float(y16) for the accumulators Y (y16 = half(acc) [+ bias, as _bias16]), stored, kept in V
_RESID_EPI = """
    metal::vec<float, 8> V[TMF][NF];
    UNR for (int t = 0; t < TMF; t++) UNR for (short r = 0; r < 2; r++) {
        const int zr = ZROW(t, r);
        UNR for (int f = 0; f < NF; f++) {
            const int c = f * 16 + fn;
            const float4 zz = *(const device float4*)(z + (size_t)zr * D + c);
            float4 o;
            UNR for (short j = 0; j < 4; j++) {
                o[j] = zz[j] + float(YVAL(t, f, r * 4 + j, c + j));
                V[t][f][r * 4 + j] = o[j];
            }
            *(device float4*)(z_out + (size_t)zr * D + c) = o;
        }
    }
"""

_LIN_ADDLN_SRC = _TILE_PROLOGUE + """
    // ZROW: x rows are in the producer's layout; TRANS_Y: x row (j, i) is z row (i, j)
    #define XROW(t, r) (r0 + (t) * 16 + fm + 8 * (r))
    #define ZROW(t, r) (TRANS_Y ? (XROW(t, r) % NN) * NN + XROW(t, r) / NN : XROW(t, r))
    metal::vec<float, 8> acc[TMF][NF];
    UNR for (int t = 0; t < TMF; t++) UNR for (int f = 0; f < NF; f++) acc[t][f] = 0.0f;
    UNR for (int kf = 0; kf < NF; kf++) {             // K = D
        metal::vec<half, 8> Af[TMF];
        UNR for (int t = 0; t < TMF; t++) {
            auto pa = x + (size_t)XROW(t, 0) * D + kf * 16 + fn;
            Af[t] = frag2(pa, pa + 8 * D);
        }
        UNR for (int f = 0; f < NF; f += 2) {
            auto pw = w + (size_t)(f * 16 + fm) * D + kf * 16 + fn;
            const metal::vec<half, 8> w0 = frag2(pw, pw + 8 * D);
            const metal::vec<half, 8> w1 = frag2(pw + 16 * D, pw + 24 * D);
            UNR for (int t = 0; t < TMF; t++)
                nax_mma<float, half, half, false, true, true>(acc[t][f], acc[t][f + 1], Af[t], w0, w1);
        }
    }
    // lin16: y = half(acc) (MLX fp16 GEMM output), + bias in fp32 -> half (_bias16)
    #define YVAL(t, f, i, c) (HAS_BIAS ? half(float(half(acc[t][f][i])) + bias[c]) : half(acc[t][f][i]))
""" + _RESID_EPI + _LN_EPI

_TRIMUL_TAIL_SRC = _TILE_PROLOGUE + """
    #define XROW(t, r) (r0 + (t) * 16 + fm + 8 * (r))
    #define ZROW(t, r) XROW(t, r)
    // 1. o^T tile (positions x channels) as A fragments; o is channel-major [D][R] (the contraction output)
    metal::vec<half, 8> Of[TMF][NF];
    UNR for (int t = 0; t < TMF; t++) UNR for (int kf = 0; kf < NF; kf++) UNR for (short r = 0; r < 2; r++)
        UNR for (short j = 0; j < 4; j++)
            Of[t][kf][r * 4 + j] = o[(size_t)(kf * 16 + fn + j) * R + XROW(t, r)];
    // 2. norm_out (LayerNorm over the D channels of each position; eps already scaled by 2^-4K), in place, fp16
    {
        const float eps_o = epso[0];
        UNR for (int t = 0; t < TMF; t++) UNR for (short r = 0; r < 2; r++) {
            float s = 0.0f;
            UNR for (int kf = 0; kf < NF; kf++) UNR for (short j = 0; j < 4; j++) s += float(Of[t][kf][r * 4 + j]);
            s += simd_shuffle_xor(s, (ushort)1);
            s += simd_shuffle_xor(s, (ushort)8);
            const float mean = s / D;
            float ss = 0.0f;
            UNR for (int kf = 0; kf < NF; kf++) UNR for (short j = 0; j < 4; j++) {
                const float d = float(Of[t][kf][r * 4 + j]) - mean;
                ss += d * d;
            }
            ss += simd_shuffle_xor(ss, (ushort)1);
            ss += simd_shuffle_xor(ss, (ushort)8);
            const float rstd = metal::precise::rsqrt(ss / D + eps_o);
            UNR for (int kf = 0; kf < NF; kf++) UNR for (short j = 0; j < 4; j++) {
                const int c = kf * 16 + fn + j;
                float y = (float(Of[t][kf][r * 4 + j]) - mean) * rstd;
                if (HAS_WO) y = y * wo[c];
                if (HAS_BO) y = y + bo[c];
                Of[t][kf][r * 4 + j] = half(y);
            }
        }
    }
    // 3. p = half(LN(o^T) Wpo^T)
    metal::vec<half, 8> Ph[TMF][NF];
    {
        metal::vec<float, 8> P[TMF][NF];
        UNR for (int t = 0; t < TMF; t++) UNR for (int f = 0; f < NF; f++) P[t][f] = 0.0f;
        UNR for (int kf = 0; kf < NF; kf++) UNR for (int f = 0; f < NF; f += 2) {
            auto pw = wpo + (size_t)(f * 16 + fm) * D + kf * 16 + fn;
            const metal::vec<half, 8> w0 = frag2(pw, pw + 8 * D);
            const metal::vec<half, 8> w1 = frag2(pw + 16 * D, pw + 24 * D);
            UNR for (int t = 0; t < TMF; t++)
                nax_mma<float, half, half, false, true, true>(P[t][f], P[t][f + 1], Of[t][kf], w0, w1);
        }
        UNR for (int t = 0; t < TMF; t++) UNR for (int f = 0; f < NF; f++) UNR for (short i = 0; i < 8; i++)
            Ph[t][f][i] = half(P[t][f][i]);
    }
    // 4. g = xn Wgo^T (accumulators; rounded to fp16 in the epilogue)
    metal::vec<float, 8> G[TMF][NF];
    UNR for (int t = 0; t < TMF; t++) UNR for (int f = 0; f < NF; f++) G[t][f] = 0.0f;
    UNR for (int kf = 0; kf < NF; kf++) {
        metal::vec<half, 8> Af[TMF];
        UNR for (int t = 0; t < TMF; t++) {
            auto pa = xn + (size_t)XROW(t, 0) * D + kf * 16 + fn;
            Af[t] = frag2(pa, pa + 8 * D);
        }
        UNR for (int f = 0; f < NF; f += 2) {
            auto pw = wgo + (size_t)(f * 16 + fm) * D + kf * 16 + fn;
            const metal::vec<half, 8> w0 = frag2(pw, pw + 8 * D);
            const metal::vec<half, 8> w1 = frag2(pw + 16 * D, pw + 24 * D);
            UNR for (int t = 0; t < TMF; t++)
                nax_mma<float, half, half, false, true, true>(G[t][f], G[t][f + 1], Af[t], w0, w1);
        }
    }
    // 5. _gate16: y = half(float(p) * sigmoid(float(half(g))))
    #define YVAL(t, f, i, c) half(float(Ph[t][f][i]) * mlx_sigmoid(float(half(G[t][f][i]))))
""" + _RESID_EPI + _LN_EPI

# (TM rows / simdgroup, NSG simdgroups / threadgroup)
PAIR_CFGS = [dict(tm=16, nsg=4), dict(tm=16, nsg=8), dict(tm=32, nsg=4), dict(tm=16, nsg=2)]
TRIMUL_TAIL_CFGS = [dict(tm=16, nsg=4), dict(tm=16, nsg=8), dict(tm=16, nsg=2)]
PAIR_SITE_CFG = {"lin": PAIR_CFGS[0], "tail": TRIMUL_TAIL_CFGS[0]}   # BOLTZ_MLX_NAX_PAIR_CFG="lin=1,tail=0"


def _ln_params(ln):
    w, b = ln.get("weight"), ln.get("bias")
    return w, b, float(ln.eps)


def _dummy(dt=mx.float32):
    return mx.zeros((8,), dt)


def pair_ok(z: mx.array, tm: int = 16) -> bool:
    """Shapes the pair kernels handle: z [1, N, N, 128] fp32 with N * N >= tm."""
    return (z.ndim == 4 and z.shape[0] == 1 and z.shape[1] == z.shape[2] and z.shape[-1] == 128
            and z.dtype == mx.float32 and z.shape[1] * z.shape[2] >= tm)


def _tb_variant(src: str) -> str:
    assert src.count(_LN_EPI) == 1
    return src.replace(_LN_EPI, _LN_EPI_TB)


# R8-coldN (kernels.NAX_FLAGS DYNN): the token count NN read at run time from nrows[1] (a `const int` local with the
# template parameter's name and type: same expressions, same arithmetic) instead of a template constant, so a new N
# does not instantiate (= compile) new Metal libraries. Off: the shipped sources / templates / names, untouched.
_DYN_NN = "    const int NN = (int)nrows[1];                     // R8-coldN DYNN: run-time N\n"


def _dynn() -> bool:
    from boltz_mlx import kernels as _K

    return _K.nax("DYNN")


def _dsrc(src: str, dyn: bool) -> str:
    return _DYN_NN + src if dyn else src


def _nrows(R: int, N: int, dyn: bool) -> mx.array:
    return mx.array([R, N] if dyn else [R], mx.int32)


def _nn(N: int, dyn: bool) -> list:
    return [] if dyn else [("NN", N)]


def linear_addln(x: mx.array, w: mx.array, z: mx.array, ln, *, bias: mx.array | None = None, trans_y: bool = False,
                 trans_out: bool = False, tm: int = 16, nsg: int = 4, w_tb: mx.array | None = None):
    """(z + y, LN(z + y) fp16) with y = lin16(x) = half(x W^T) [+ bias -> half]. x: [1, N, N, D] fp16 (trans_y: row
    (j, i) of x belongs to z row (i, j)); w [D, D] fp16 ([out, in]); z [1, N, N, D] fp32; ln: LayerNorm module of the
    next block; trans_out writes the LN output at [j, i] (ending-node layout). Same contract as kernels.add_ln.
    w_tb [H, D] fp16: also returns tb = half(LN_out @ w_tb^T) as [1, H, N, N] (in the LN output's layout)."""
    B, N, M, D = z.shape
    R = N * M
    assert pair_ok(z, tm) and x.dtype == mx.float16 and x.size == z.size and w.shape == (D, D) and w.dtype == mx.float16
    assert tm % 16 == 0
    lw, lb, eps = _ln_params(ln)
    ntb = 0 if w_tb is None else w_tb.shape[0]
    dyn = _dynn()
    key = ("lin_addln",) + (("tb",) if ntb else ()) + (("dyn",) if dyn else ())
    if key not in _kernels:
        _kernels[key] = _K.metal_kernel(
            name="boltz_nax3_lin_addln" + ("_tb" if ntb else "") + ("_dyn" if dyn else ""),
            input_names=["x", "w", "bias", "z", "lnw", "lnb", "epsn", "nrows"] + (["wtb"] if ntb else []),
            output_names=["z_out", "ln_out"] + (["tb_out"] if ntb else []),
            source=_dsrc(_tb_variant(_LIN_ADDLN_SRC) if ntb else _LIN_ADDLN_SRC, dyn), header=HEADER)
    ntg = ((R + tm - 1) // tm + nsg - 1) // nsg
    outs = _kernels[key](
        inputs=[x, w, bias if bias is not None else _dummy(), z, lw if lw is not None else _dummy(),
                lb if lb is not None else _dummy(), mx.array([eps], mx.float32), _nrows(R, N, dyn)]
        + ([w_tb] if ntb else []),
        template=[("D", D), ("TM", tm), ("NSG", nsg)] + _nn(N, dyn) + [
            ("TRANS_Y", int(trans_y)), ("TRANS_O", int(trans_out)), ("HAS_BIAS", int(bias is not None)),
            ("HAS_W", int(lw is not None)), ("HAS_B", int(lb is not None))]
        + ([("NTB", ntb)] if ntb else []),
        grid=(ntg * nsg * 32, 1, 1), threadgroup=(nsg * 32, 1, 1),
        output_shapes=[z.shape, z.shape] + ([(ntb, R)] if ntb else []),
        output_dtypes=[mx.float32, mx.float16] + ([mx.float16] if ntb else []))
    if ntb:
        return outs[0], outs[1], outs[2].reshape(1, ntb, N, M)
    return outs[0], outs[1]


def trimul_tail_addln(o: mx.array, norm_out, eps_scale: float, w_po: mx.array, xn: mx.array, w_go: mx.array,
                      z: mx.array, ln, *, tm: int = 16, nsg: int = 4, w_tb: mx.array | None = None):
    """Tri-mult output + residual + next LN: o [D, N*N] fp16 channel-major contraction output (2^-2K scaled),
    norm_out its LayerNorm (eps * eps_scale), w_po / w_go [D, D] fp16, xn [.., D] fp16 the block's norm_in output,
    z [1, N, N, D] fp32. Returns (z + gate16(p_out(norm_out(o^T)), g_out(xn)), LN_next(.) fp16) [+ tb as in
    linear_addln when w_tb is given]."""
    B, N, M, D = z.shape
    R = N * M
    assert pair_ok(z, tm) and o.dtype == mx.float16 and o.size == R * D and xn.dtype == mx.float16
    assert w_po.shape == (D, D) and w_go.shape == (D, D) and w_po.dtype == mx.float16 and w_go.dtype == mx.float16
    wo, bo, eps_o = _ln_params(norm_out)
    lw, lb, eps = _ln_params(ln)
    ntb = 0 if w_tb is None else w_tb.shape[0]
    dyn = _dynn()
    key = ("trimul_tail",) + (("tb",) if ntb else ()) + (("dyn",) if dyn else ())
    if key not in _kernels:
        _kernels[key] = _K.metal_kernel(
            name="boltz_nax3_trimul_tail" + ("_tb" if ntb else "") + ("_dyn" if dyn else ""),
            input_names=["o", "wo", "bo", "epso", "wpo", "xn", "wgo", "z", "lnw", "lnb", "epsn", "nrows"]
            + (["wtb"] if ntb else []),
            output_names=["z_out", "ln_out"] + (["tb_out"] if ntb else []),
            source=_dsrc(_tb_variant(_TRIMUL_TAIL_SRC) if ntb else _TRIMUL_TAIL_SRC, dyn), header=HEADER)
    ntg = ((R + tm - 1) // tm + nsg - 1) // nsg
    outs = _kernels[key](
        inputs=[o, wo if wo is not None else _dummy(), bo if bo is not None else _dummy(),
                mx.array([eps_o * eps_scale], mx.float32), w_po, xn, w_go, z, lw if lw is not None else _dummy(),
                lb if lb is not None else _dummy(), mx.array([eps], mx.float32), _nrows(R, N, dyn)]
        + ([w_tb] if ntb else []),
        template=[("D", D), ("TM", tm), ("NSG", nsg)] + _nn(N, dyn) + [("TRANS_O", 0), ("HAS_WO", int(wo is not None)),
                  ("HAS_BO", int(bo is not None)), ("HAS_W", int(lw is not None)), ("HAS_B", int(lb is not None))]
        + ([("NTB", ntb)] if ntb else []),
        grid=(ntg * nsg * 32, 1, 1), threadgroup=(nsg * 32, 1, 1),
        output_shapes=[z.shape, z.shape] + ([(ntb, R)] if ntb else []),
        output_dtypes=[mx.float32, mx.float16] + ([mx.float16] if ntb else []))
    if ntb:
        return outs[0], outs[1], outs[2].reshape(1, ntb, N, M)
    return outs[0], outs[1]


# ---------------------------------------------------------------------------------------------------------------
# MLX-op references with the same rounding points (test oracles; CPU stand-ins for logic tests)
# ---------------------------------------------------------------------------------------------------------------
def _ln_ref(v, w, b, eps):
    mean = mx.mean(v, axis=-1, keepdims=True)
    var = mx.mean(mx.square(v - mean), axis=-1, keepdims=True)
    y = (v - mean) * mx.rsqrt(var + eps)
    if w is not None:
        y = y * w
    if b is not None:
        y = y + b
    return y


def _tb_ref(lo, w_tb):
    """tb = half(lo @ w_tb^T) head-major [1, H, N, N] (lo: the fp16 LN output in its own layout)."""
    f = mx.float32
    return (lo.astype(f) @ w_tb.astype(f).T).astype(mx.float16).transpose(0, 3, 1, 2)


def linear_addln_ref(x, w, z, ln, *, bias=None, trans_y=False, trans_out=False, w_tb=None, **_):
    f = mx.float32
    D = z.shape[-1]
    y = (x.astype(f).reshape(-1, D) @ w.astype(f).T).astype(mx.float16)
    if bias is not None:
        y = (y.astype(f) + bias).astype(mx.float16)
    y = y.reshape(z.shape)
    if trans_y:
        y = y.swapaxes(1, 2)
    zn = z + y.astype(f)
    lw, lb, eps = _ln_params(ln)
    lo = _ln_ref(zn, lw, lb, eps).astype(mx.float16)
    if trans_out:
        lo = lo.swapaxes(1, 2)
    return (zn, lo, _tb_ref(lo, w_tb)) if w_tb is not None else (zn, lo)


def trimul_tail_addln_ref(o, norm_out, eps_scale, w_po, xn, w_go, z, ln, w_tb=None, **_):
    f = mx.float32
    D = z.shape[-1]
    wo, bo, eps_o = _ln_params(norm_out)
    on = _ln_ref(o.astype(f).reshape(D, -1).T, wo, bo, eps_o * eps_scale).astype(mx.float16)
    p = (on.astype(f) @ w_po.astype(f).T).astype(mx.float16)
    g = (xn.astype(f).reshape(-1, D) @ w_go.astype(f).T).astype(mx.float16)
    y = (p.astype(f) * mx.sigmoid(g.astype(f))).astype(mx.float16).reshape(z.shape)
    zn = z + y.astype(f)
    lw, lb, eps = _ln_params(ln)
    lo = _ln_ref(zn, lw, lb, eps).astype(mx.float16)
    return (zn, lo, _tb_ref(lo, w_tb)) if w_tb is not None else (zn, lo)


# ---------------------------------------------------------------------------------------------------------------
# Diffusion token transformer (D = 768, fp16 diffusion matmuls)
# ---------------------------------------------------------------------------------------------------------------
class RawCond16:
    """Raw conditioning of all token-transformer layers, as the fp16 outputs of the two conditioning GEMMs
    (DiffusionTransformer.precompute_cond): h1 [Bc * N, L * 4 * D] fp16 (slots per layer: attn scale | attn shift |
    transition scale | transition shift, scales pre-sigmoid), b1 [L * 4 * D] fp32; h2 [Bc * N, L * 2 * D] fp16 (output
    gates of attention | transition, pre-sigmoid), b2 [L * 2 * D] fp32. Value = float(h) + b, exactly what the MLX f16
    path computes (mm_t: fp16 GEMM output cast to fp32, + bias). rep: samples sharing one conditioning row (shared
    noise level); ntok: tokens per sample."""

    def __init__(self, h1, b1, h2, b2, ntok: int, L: int, D: int, rep: int = 1):
        self.h1, self.b1, self.h2, self.b2 = h1, b1, h2, b2
        self.ntok, self.L, self.D, self.rep = ntok, L, D, rep

    def meta(self, M: int, off: int) -> mx.array:
        return mx.array([M, self.ntok, self.rep, off, 0, 0, 0, 0], mx.int32)

    def arrays(self, i: int):
        """Layer i's 6-tuple (attn_scale, attn_shift, out_gate, tr_scale, tr_shift, tr_gate) with sigmoids applied,
        [Bc * rep, N, D] (the MLX fast path's form; reference / fallback)."""
        L, D, f = self.L, self.D, mx.float32
        h1 = self.h1.astype(f).reshape(-1, self.ntok, L, 4, D)[:, :, i] + self.b1.reshape(L, 4, D)[i]
        h2 = self.h2.astype(f).reshape(-1, self.ntok, L, 2, D)[:, :, i] + self.b2.reshape(L, 2, D)[i]
        out = (mx.sigmoid(h1[..., 0, :]), h1[..., 1, :], mx.sigmoid(h2[..., 0, :]),
               mx.sigmoid(h1[..., 2, :]), h1[..., 3, :], mx.sigmoid(h2[..., 1, :]))
        if self.rep != 1:
            out = tuple(mx.repeat(x, self.rep, axis=0) for x in out)
        return out


_ADALN16_SRC = """
    constexpr int NV = D / 128;
    const uint row = threadgroup_position_in_grid.x * RPT + simdgroup_index_in_threadgroup;
    const uint nrows = (uint)meta[0];
    if (row >= nrows) return;
    const uint lane = thread_index_in_simdgroup;
    const uint ntok = (uint)meta[1], rep = (uint)meta[2], off = (uint)meta[3];
    const uint cr = ((row / ntok) / rep) * ntok + row % ntok;            // conditioning row (samples share it)
    const device half* hs = (const device half*)h1 + (size_t)cr * H1LD + off;   // scale (pre-sigmoid), then shift
    const device float4* xr = (const device float4*)(x + (size_t)row * D);
    float4 v[NV];
    UNR for (int i = 0; i < NV; i++) v[i] = xr[lane + 32 * i];
    float s = 0.0f;
    UNR for (int i = 0; i < NV; i++) s += v[i].x + v[i].y + v[i].z + v[i].w;
    const float mean = simd_sum(s) / D;
    float ss = 0.0f;
    UNR for (int i = 0; i < NV; i++) { const float4 d = v[i] - mean; ss += dot(d, d); }
    const float rstd = metal::precise::rsqrt(simd_sum(ss) / D + 1e-5f);
    device half4* yh = (device half4*)(out + (size_t)row * D);
    UNR for (int i = 0; i < NV; i++) {
        const uint c = (lane + 32 * i) * 4;
        const float4 sc = float4(*(const device half4*)(hs + c)) + *(const device float4*)(b1 + off + c);
        const float4 sh = float4(*(const device half4*)(hs + D + c)) + *(const device float4*)(b1 + off + D + c);
        float4 o;
        UNR for (short j = 0; j < 4; j++) o[j] = (v[i][j] - mean) * rstd * mlx_sigmoid(sc[j]) + sh[j];
        yh[lane + 32 * i] = half4(o);
    }
"""
ADALN_RPT = 8


def adaln16(a2: mx.array, rc: RawCond16, layer: int, slot: int) -> mx.array:
    """a2 [M, D] fp32 -> fp16 sigmoid(sc) * LN(a2) + sh, (sc, sh) = slots (slot, slot + 1) of layer `layer` in rc.h1."""
    M, D = a2.shape
    assert D % 128 == 0 and a2.dtype == mx.float32
    key = ("adaln16", D)
    if key not in _kernels:
        _kernels[key] = _K.metal_kernel(name=f"boltz_nax3_adaln16_{D}", input_names=["x", "h1", "b1", "meta"],
                                             output_names=["out"], source=_ADALN16_SRC, header=HEADER)
    rpt = ADALN_RPT
    (y,) = _kernels[key](inputs=[a2, rc.h1, rc.b1, rc.meta(M, (layer * 4 + slot) * D)],
                         template=[("D", D), ("H1LD", rc.L * 4 * D), ("RPT", rpt)],
                         grid=(((M + rpt - 1) // rpt) * rpt * 32, 1, 1), threadgroup=(rpt * 32, 1, 1),
                         output_shapes=[(M, D)], output_dtypes=[mx.float16])
    return y


def adaln16_ref(a2, rc: RawCond16, layer: int, slot: int):
    M, D = a2.shape
    f = mx.float32
    rows = mx.arange(M)
    cr = ((rows // rc.ntok) // rc.rep) * rc.ntok + rows % rc.ntok
    off = (layer * 4 + slot) * D
    sc = rc.h1[cr, off: off + D].astype(f) + rc.b1[off: off + D]
    sh = rc.h1[cr, off + D: off + 2 * D].astype(f) + rc.b1[off + D: off + 2 * D]
    return (_ln_ref(a2, None, None, 1e-5) * mx.sigmoid(sc) + sh).astype(mx.float16)


EPI_STORE, EPI_GRESID, EPI_SWIGLU = 0, 1, 2

_GEMM16_SRC = """
    constexpr int TMF = SM / 16, TNF = SN / 16;       // 16x16 accumulator fragments per simdgroup (TNF even)
    const int M = (int)meta[0];
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const int r0 = ((int)threadgroup_position_in_grid.y * WM + (int)(sg / WN)) * SM;
    const int c0 = ((int)threadgroup_position_in_grid.x * WN + (int)(sg % WN)) * SN;   // column within a group
    if (r0 >= M) return;
    NAX_COORD(lane, fm, fn)
    const device half* Ap[TMF][2];                    // this lane's A rows (clamped: rows >= M are never stored)
    UNR for (int t = 0; t < TMF; t++) UNR for (short r = 0; r < 2; r++)
        Ap[t][r] = (const device half*)A + (size_t)min(r0 + t * 16 + fm + 8 * r, M - 1) * K + fn;
    const device half* Wp = (const device half*)W + (size_t)(c0 + fm) * K + fn;

    metal::vec<float, 8> acc[NGRP][TMF][TNF];
    UNR for (int g = 0; g < NGRP; g++) UNR for (int t = 0; t < TMF; t++) UNR for (int f = 0; f < TNF; f++)
        acc[g][t][f] = 0.0f;
    for (int k0 = 0; k0 < K; k0 += 16 * KU) {
        metal::vec<half, 8> Af[KU][TMF];
        metal::vec<half, 8> Bf[KU][NGRP][TNF];
        UNR for (int u = 0; u < KU; u++) {
            const int k = k0 + 16 * u;
            UNR for (int t = 0; t < TMF; t++) Af[u][t] = frag2(Ap[t][0] + k, Ap[t][1] + k);
            UNR for (int g = 0; g < NGRP; g++) UNR for (int f = 0; f < TNF; f++) {
                const device half* pw = Wp + ((size_t)g * NOUT + f * 16) * K + k;
                Bf[u][g][f] = frag2(pw, pw + 8 * K);
            }
        }
        UNR for (int u = 0; u < KU; u++) UNR for (int g = 0; g < NGRP; g++) UNR for (int t = 0; t < TMF; t++)
            UNR for (int f = 0; f < TNF; f += 2)
                nax_mma<float, half, half, false, true, true>(acc[g][t][f], acc[g][t][f + 1], Af[u][t], Bf[u][g][f],
                                                              Bf[u][g][f + 1]);
    }

    // epilogue: lane element r * 4 + j of fragment (t, f) = row r0 + 16 t + fm + 8 r, column c0 + 16 f + fn + j
    const uint ntok = (uint)meta[1], rep = (uint)meta[2], goff = (uint)meta[3];
    UNR for (int t = 0; t < TMF; t++) UNR for (short r = 0; r < 2; r++) {
        const int row = r0 + t * 16 + fm + 8 * r;
        if (row >= M) continue;
        const uint cr = (((uint)row / ntok) / rep) * ntok + (uint)row % ntok;
        UNR for (int f = 0; f < TNF; f++) {
            const int col = c0 + f * 16 + fn;
            if (EPI == 0) {
                metal::vec<OT, 4> o;
                UNR for (short j = 0; j < 4; j++) o[j] = OT(acc[0][t][f][r * 4 + j] + (HAS_BIAS ? bias[col + j] : 0.0f));
                *(device metal::vec<OT, 4>*)(out + (size_t)row * NOUT + col) = o;
            } else if (EPI == 1) {
                const half4 gh = *(const device half4*)((const device half*)gate + (size_t)cr * GLD + goff + col);
                const float4 gb = *(const device float4*)(gbias + goff + col);
                const float4 rs = *(const device float4*)(resid + (size_t)row * NOUT + col);
                float4 o;
                UNR for (short j = 0; j < 4; j++)
                    o[j] = rs[j] + mlx_sigmoid(float(gh[j]) + gb[j]) * acc[0][t][f][r * 4 + j];
                *(device float4*)(out + (size_t)row * NOUT + col) = o;
            } else {
                half4 o;
                UNR for (short j = 0; j < 4; j++) {
                    const float xg = acc[0][t][f][r * 4 + j], gt = acc[1][t][f][r * 4 + j];
                    o[j] = half(gt * mlx_sigmoid(gt) * xg * acc[2][t][f][r * 4 + j]);
                }
                *(device half4*)(out + (size_t)row * NOUT + col) = o;
            }
        }
    }
"""

# (SM rows, SN cols per simdgroup, WM x WN simdgroups, KU 16-deep k steps per iteration)
GEMM_CFGS = [
    dict(sm=32, sn=32, wm=2, wn=2, ku=2),
    dict(sm=32, sn=64, wm=2, wn=2, ku=1),
    dict(sm=32, sn=64, wm=2, wn=2, ku=2),
    dict(sm=64, sn=32, wm=2, wn=2, ku=1),
    dict(sm=16, sn=64, wm=4, wn=2, ku=2),
    dict(sm=32, sn=32, wm=4, wn=1, ku=2),
    dict(sm=16, sn=32, wm=2, wn=4, ku=2),
    # 7..11: small-tile entries for few-row calls (per-chip size buckets; tools/prof/chip_sweep.py, M5 Max): more
    # threadgroups at 117-600 rows. Same per-element accumulation (k ascending in 16-steps) as every entry above.
    dict(sm=16, sn=32, wm=1, wn=1, ku=4),
    dict(sm=16, sn=32, wm=4, wn=1, ku=4),
    dict(sm=16, sn=32, wm=1, wn=2, ku=1),
    dict(sm=16, sn=32, wm=4, wn=1, ku=2),
    dict(sm=16, sn=32, wm=1, wn=1, ku=2),
]
PAIR_SITE_CFG["pproj"] = GEMM_CFGS[2]   # tri-att q|k|v|g projection (A/B 35b6a31: 1.22 vs MLX 1.52 ms at N = 400)
SWIGLU_CFGS = [
    dict(sm=16, sn=32, wm=2, wn=2, ku=1),
    dict(sm=16, sn=32, wm=2, wn=2, ku=2),
    dict(sm=32, sn=32, wm=2, wn=1, ku=1),
    dict(sm=16, sn=32, wm=4, wn=1, ku=1),
    dict(sm=16, sn=32, wm=1, wn=1, ku=2),   # 4: few-row calls (chip_sweep, M5 Max)
]
# per diffusion GEMM site (name -> cfg); "mlx" = MLX fp16 GEMM + compiled elementwise epilogue (A/B fallback)
DIFF_GEMM_CFG = {
    "qkvg": "mlx",                                   # plain fp16 GEMM: MLX unless the A/B shows a NAX win
    "o": dict(sm=32, sn=32, wm=2, wn=2, ku=2),
    "ff": dict(sm=16, sn=32, wm=2, wn=2, ku=1),
    "ba": dict(sm=32, sn=32, wm=2, wn=2, ku=2),
    "cond": "mlx",
}


def gemm16(x: mx.array, w: mx.array, *, epi: int = EPI_STORE, ngrp: int = 1, out_dtype=mx.float16,
           bias: mx.array | None = None, gate: mx.array | None = None, gate_bias: mx.array | None = None,
           gate_off: int = 0, resid: mx.array | None = None, ntok: int | None = None, rep: int = 1,
           sm: int = 32, sn: int = 32, wm: int = 2, wn: int = 2, ku: int = 2) -> mx.array:
    """y = epi(x W^T): x [M, K] fp16, w [ngrp * NOUT, K] fp16 (torch Linear layout, groups stacked by rows).
    EPI_STORE: out_dtype(acc [+ bias]); EPI_GRESID: resid + sigmoid(float(gate[cr, gate_off + col]) +
    gate_bias[gate_off + col]) * acc, fp32 (gate [Bc * ntok, GLD] fp16, cr = conditioning row of output row, see
    RawCond16); EPI_SWIGLU (ngrp 3: x | gates | ab): half(silu(acc_g) * acc_x * acc_ab)."""
    M, K = x.shape
    R, K2 = w.shape
    assert K == K2 and x.dtype == mx.float16 and w.dtype == mx.float16 and R % ngrp == 0
    NOUT = R // ngrp
    assert sm % 16 == 0 and sn % 32 == 0 and K % (16 * ku) == 0 and NOUT % (wn * sn) == 0, (sm, sn, wn, K, NOUT)
    if epi == EPI_SWIGLU:
        assert ngrp == 3
        out_dtype = mx.float16
    elif epi == EPI_GRESID:
        assert ngrp == 1 and gate is not None and gate_bias is not None and resid is not None
        assert resid.shape == (M, NOUT) and resid.dtype == mx.float32 and gate.dtype == mx.float16
        out_dtype = mx.float32
    else:
        assert ngrp == 1
    gld = gate.shape[-1] if gate is not None else 0
    key = ("gemm16",)
    if key not in _kernels:
        _kernels[key] = _K.metal_kernel(
            name="boltz_nax3_gemm16", input_names=["A", "W", "meta", "bias", "gate", "gbias", "resid"],
            output_names=["out"], source=_GEMM16_SRC, header=HEADER)
    z = _dummy()
    meta = mx.array([M, ntok if ntok else max(M, 1), rep, gate_off, 0, 0, 0, 0], mx.int32)
    (y,) = _kernels[key](
        inputs=[x, w, meta, bias if bias is not None else z, gate if gate is not None else _dummy(mx.float16),
                gate_bias if gate_bias is not None else z, resid if resid is not None else z],
        template=[("K", K), ("NOUT", NOUT), ("NGRP", ngrp), ("EPI", epi), ("HAS_BIAS", int(bias is not None)),
                  ("GLD", gld), ("SM", sm), ("SN", sn), ("WM", wm), ("WN", wn), ("KU", ku), ("OT", out_dtype)],
        grid=((NOUT // (wn * sn)) * wm * wn * 32, (M + wm * sm - 1) // (wm * sm), 1),
        threadgroup=(wm * wn * 32, 1, 1),
        output_shapes=[(M, NOUT)], output_dtypes=[out_dtype])
    return y


def gemm16_ref(x, w, *, epi=EPI_STORE, ngrp=1, out_dtype=mx.float16, bias=None, gate=None, gate_bias=None,
               gate_off=0, resid=None, ntok=None, rep=1, **_):
    f = mx.float32
    acc = x.astype(f) @ w.astype(f).T
    M = x.shape[0]
    if epi == EPI_SWIGLU:
        n = w.shape[0] // 3
        xg, gt, ab = acc[:, :n], acc[:, n: 2 * n], acc[:, 2 * n:]
        return (gt * mx.sigmoid(gt) * xg * ab).astype(mx.float16)
    if epi == EPI_GRESID:
        n = w.shape[0]
        ntok = ntok or M
        rows = mx.arange(M)
        cr = ((rows // ntok) // rep) * ntok + rows % ntok
        g = mx.sigmoid(gate[cr, gate_off: gate_off + n].astype(f) + gate_bias[gate_off: gate_off + n])
        return resid + g * acc
    if bias is not None:
        acc = acc + bias
    return acc.astype(out_dtype)


# ---------------------------------------------------------------------------------------------------------------
# Pair transition (kernels_nax2 NAX SwiGLU transition with the fused residual add) + an epilogue over the new
# residual rows (round "R3-pair"): after its rows of z_new are stored, each simdgroup reads them back (threadgroup
# barrier, device memory: the rows are L1/L2-resident) and, row by row with the LayerNorm kernel's own lane layout
# (lane + 32 i) and arithmetic (kernels._LN_SRC: two-pass fp32, precise rsqrt, (v - mean) * rstd * w + b):
#   HAS_XN: xn_out = LN_next(z_new) fp16             -> the next pairformer layer's tri_mul_out.norm_in output
#   NHB   : pb_out[h, row] = half(sum_c half(LN_s(z_new))[c] * Wz[h, c])   -> the s-track AttentionPairBias pair bias,
#           head-major [H, N*N] (= proj_z output transposed to b h i j, so the attention reads it contiguously)
# Both used to re-read z (82 MB at N = 400) and the s-track also wrote + read its fp16 LN (2 x 41 MB) before a 16-col
# GEMM. xn_out is bitwise the LN kernel's output; pb_out rounds like lin16 (fp16 LN, fp16 result) with the 128-term
# fp32 sum in another order.
# ---------------------------------------------------------------------------------------------------------------
_LNX_EPI = """
    if constexpr (LNX != 0) {
        threadgroup_barrier(mem_flags::mem_device);       // this threadgroup's z_new rows are in memory
        const uint ln_lane = thread_index_in_simdgroup;
        constexpr int NV = D / 32;
        const device float* zo = (const device float*)out;
        float wz[NHB > 0 ? NHB : 1][NV];
        if constexpr (NHB > 0) {
            UNR for (int h = 0; h < NHB; h++) UNR for (int i = 0; i < NV; i++)
                wz[h][i] = float(pbw[h * D + ln_lane + 32 * i]);
        }
        for (int rr = 0; rr < TM; rr++) {
            const int row = r0 + rr;
            float v[NV];
            float s = 0.0f;
            UNR for (int i = 0; i < NV; i++) {
                v[i] = zo[(size_t)row * D + ln_lane + 32 * i];
                s += v[i];
            }
            const float mean = simd_sum(s) / D;
            float ss = 0.0f;
            UNR for (int i = 0; i < NV; i++) {
                const float d = v[i] - mean;
                ss += d * d;
            }
            const float ssum = simd_sum(ss);
            if constexpr (HAS_XN != 0) {
                const float rstd = metal::precise::rsqrt(ssum / D + lnn_eps[0]);
                UNR for (int i = 0; i < NV; i++) {
                    const uint c = ln_lane + 32 * i;
                    float y = (v[i] - mean) * rstd;
                    if (HAS_W1) y = y * lnn_w[c];
                    if (HAS_B1) y = y + lnn_b[c];
                    xn_out[(size_t)row * D + c] = half(y);
                }
            }
            if constexpr (NHB > 0) {
                const float rstd2 = metal::precise::rsqrt(ssum / D + lns_eps[0]);
                float p[NHB];
                UNR for (int h = 0; h < NHB; h++) p[h] = 0.0f;
                UNR for (int i = 0; i < NV; i++) {
                    const uint c = ln_lane + 32 * i;
                    float y = (v[i] - mean) * rstd2;
                    if (HAS_W2) y = y * lns_w[c];
                    if (HAS_B2) y = y + lns_b[c];
                    const float yh = float(half(y));
                    UNR for (int h = 0; h < NHB; h++) p[h] += yh * wz[h][i];
                }
                UNR for (int h = 0; h < NHB; h++) {
                    const float t = simd_sum(p[h]);
                    if (ln_lane == (uint)h) pb_out[(size_t)h * N_ROWS + row] = half(t);
                }
            }
        }
    }
"""


def _trans_lnx_src() -> str:
    from boltz_mlx import kernels_nax2 as _KN2

    src = _KN2._TRANS_SRC
    old = "if (TGW == 0 && tile >= ntiles) return;"
    assert src.count(old) == 1
    # every simdgroup must reach the epilogue's barrier: clamp instead of returning (duplicate tiles rewrite identical
    # values, as the moved-back last tile already does)
    return src.replace(old, "if (TGW == 0 && LNX == 0 && tile >= ntiles) return;") + _LNX_EPI


def transition_lnx_ok(z: mx.array, cfg: dict) -> bool:
    return pair_ok(z, cfg["tm"]) and (6 if cfg.get("tgw") == 2 else 3) * cfg["hb"] * 128 * 2 <= 32768


def transition_lnx(xn: mx.array, w12: mx.array, w3: mx.array, z: mx.array, *, next_ln=None, ln_s=None,
                   w_s: mx.array | None = None, tm: int = 16, hb: int = 32, nsg: int = 8, tgw: int = 1):
    """Pair transition z + fc3(silu(fc1 xn) * fc2 xn) (kernels_nax2.transition_nax with resid, bitwise the same z_new)
    plus, from the new rows: next_ln(z_new) fp16 (if next_ln) and the s-track pair bias
    half(half(ln_s(z_new)) @ w_s^T) as [1, H, N, N] fp16 (if ln_s / w_s [H, D] fp16). Returns (z_new, xn_next, bias)
    with None for outputs not requested."""
    D = xn.shape[-1]
    H = w3.shape[1]
    R = xn.size // D
    assert xn.dtype == mx.float16 and w12.dtype == mx.float16 and w3.dtype == mx.float16
    assert w12.shape == (2 * H, D) and w3.shape == (D, H) and z.dtype == mx.float32 and z.size == xn.size
    assert D % 32 == 0 and H % hb == 0 and R >= tm
    has_xn = next_ln is not None
    nhb = 0 if w_s is None else w_s.shape[0]
    if nhb:
        assert ln_s is not None and w_s.shape == (nhb, D) and w_s.dtype == mx.float16
    key = ("trans_lnx",)
    if key not in _kernels:
        _kernels[key] = _K.metal_kernel(
            name="boltz_nax3_transition_lnx",
            input_names=["x", "w12", "w3", "scales", "resid", "nrows", "lnn_w", "lnn_b", "lnn_eps", "lns_w", "lns_b",
                         "lns_eps", "pbw"],
            output_names=["out", "xn_out", "pb_out"], source=_trans_lnx_src(), header=_trans_header())
    w1, b1, e1 = _ln_params(next_ln) if has_xn else (None, None, 1e-5)
    w2, b2, e2 = _ln_params(ln_s) if nhb else (None, None, 1e-5)
    d = _dummy()
    ntg = ((R + tm - 1) // tm + nsg - 1) // nsg
    zo, xo, pb = _kernels[key](
        inputs=[xn, w12, w3, mx.array([1.0, 1.0], mx.float32), z, mx.array([R], mx.int32),
                w1 if w1 is not None else d, b1 if b1 is not None else d, mx.array([e1], mx.float32),
                w2 if w2 is not None else d, b2 if b2 is not None else d, mx.array([e2], mx.float32),
                w_s if nhb else _dummy(mx.float16)],
        template=[("D", D), ("H", H), ("TM", tm), ("HB", hb), ("NSG", nsg), ("TGW", int(tgw)), ("RESID", 1),
                  ("OT", mx.float32), ("LNX", 1), ("HAS_XN", int(has_xn)), ("NHB", nhb),
                  ("HAS_W1", int(w1 is not None)), ("HAS_B1", int(b1 is not None)),
                  ("HAS_W2", int(w2 is not None)), ("HAS_B2", int(b2 is not None))],
        grid=(ntg * nsg * 32, 1, 1), threadgroup=(nsg * 32, 1, 1),
        output_shapes=[z.shape, xn.shape if has_xn else (8,), (nhb, R) if nhb else (8,)],
        output_dtypes=[mx.float32, mx.float16, mx.float16])
    N = z.shape[1] if z.ndim == 4 else None
    bias = pb.reshape(1, nhb, N, z.shape[2]) if (nhb and N is not None) else (pb if nhb else None)
    return zo, (xo if has_xn else None), bias


def _trans_header() -> str:
    from boltz_mlx import kernels_nax2 as _KN2

    return _KN2.HEADER


def transition_lnx_ref(xn, w12, w3, z, *, next_ln=None, ln_s=None, w_s=None, **_):
    from boltz_mlx import kernels_nax2 as _KN2

    f = mx.float32
    zn = _KN2.transition_ref(xn, w12, w3, resid=z)
    xo = None
    if next_ln is not None:
        lw, lb, e = _ln_params(next_ln)
        xo = _ln_ref(zn, lw, lb, e).astype(mx.float16)
    bias = None
    if w_s is not None:
        lw, lb, e = _ln_params(ln_s)
        y = _ln_ref(zn, lw, lb, e).astype(mx.float16)
        bias = (y.astype(f) @ w_s.astype(f).T).astype(mx.float16)  # [1, N, N, H]
        bias = bias.transpose(0, 3, 1, 2) if bias.ndim == 4 else bias.T
    return zn, xo, bias


# ---------------------------------------------------------------------------------------------------------------
# R4-pair: the tri-mult tail split in two (profile 8399ac1: the fused tail 2.81 / 2.98 ms at N = 400 vs 0.97 ms for
# linear_addln, which has one GEMM and the same epilogue -> the in-kernel strided o loads + LN cost ~1.5 ms):
#   transpose_ln16  : o [D, R] channel-major (the contraction output) -> on = norm_out(o^T) [R, D] fp16, through a
#                     threadgroup tile (coalesced reads along positions, coalesced row writes); per position one
#                     simdgroup with the LN kernel's own lane layout / arithmetic (kernels._LN_SRC) -> bitwise
#                     layer_norm(o^T, w, b, eps * scale, fp16), i.e. the unfused path's norm_out.f16 output.
#   gemm2_gate_addln: z_new = z + half(half(on Wpo^T) * sigmoid(half(xn Wgo^T))), LN_next(z_new) [+ tri bias]: the
#                     linear_addln structure (vector fragment loads from row-major operands) with two GEMMs.
# ---------------------------------------------------------------------------------------------------------------
_TLN_SRC = """
    constexpr int NV = D / 32;
    constexpr int TPS = TP + 2;                        // padded tile row (conflict-free column reads)
    const int R = (int)nrows[0];
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const int p0 = (int)threadgroup_position_in_grid.x * TP;
    threadgroup half tile[D * TPS];
    for (uint e = tid; e < (uint)(D * TP); e += NSG * 32) {
        const uint c = e / TP, p = e % TP;
        const int pos = p0 + (int)p;
        tile[c * TPS + p] = pos < R ? o[(size_t)c * R + pos] : half(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int p = (int)sg; p < TP; p += NSG) {
        const int pos = p0 + p;
        if (pos >= R) break;
        float v[NV];
        float s = 0.0f;
        for (int i = 0; i < NV; i++) {
            v[i] = float(tile[(lane + 32 * i) * TPS + p]);
            s += v[i];
        }
        float mean = simd_sum(s) / D;
        float ss = 0.0f;
        for (int i = 0; i < NV; i++) {
            float d = v[i] - mean;
            ss += d * d;
        }
        float rstd = metal::precise::rsqrt(simd_sum(ss) / D + eps[0]);
        for (int i = 0; i < NV; i++) {
            uint c = lane + 32 * i;
            float y = (v[i] - mean) * rstd;
            if (HAS_W) y = y * w[c];
            if (HAS_B) y = y + b[c];
            out[(size_t)pos * D + c] = half(y);
        }
    }
"""

TLN_CFGS = [dict(tp=32, nsg=4), dict(tp=64, nsg=8), dict(tp=16, nsg=4)]


def transpose_ln16(o: mx.array, ln, eps_scale: float = 1.0, *, tp: int = 32, nsg: int = 4) -> mx.array:
    """o [D, R] fp16 (channel-major) -> ln(o^T) [R, D] fp16 with eps * eps_scale; bitwise kernels.layer_norm of the
    transposed view (same lane layout and arithmetic)."""
    D, R = o.shape[0], o.size // o.shape[0]
    assert o.dtype == mx.float16 and D % 32 == 0 and D * (tp + 2) * 2 <= 32768
    w, b, eps = _ln_params(ln)
    key = ("tln",)
    if key not in _kernels:
        _kernels[key] = _K.metal_kernel(name="boltz_nax3_transpose_ln", input_names=["o", "w", "b", "eps", "nrows"],
                                             output_names=["out"], source=_TLN_SRC)
    (y,) = _kernels[key](
        inputs=[o, w if w is not None else _dummy(), b if b is not None else _dummy(),
                mx.array([eps * eps_scale], mx.float32), mx.array([R], mx.int32)],
        template=[("D", D), ("TP", tp), ("NSG", nsg), ("HAS_W", int(w is not None)), ("HAS_B", int(b is not None))],
        grid=(((R + tp - 1) // tp) * nsg * 32, 1, 1), threadgroup=(nsg * 32, 1, 1),
        output_shapes=[(R, D)], output_dtypes=[mx.float16])
    return y


_TAIL2_SRC = _TILE_PROLOGUE + """
    #define XROW(t, r) (r0 + (t) * 16 + fm + 8 * (r))
    #define ZROW(t, r) XROW(t, r)
    // p = half(on Wpo^T)
    metal::vec<half, 8> Ph[TMF][NF];
    {
        metal::vec<float, 8> P[TMF][NF];
        UNR for (int t = 0; t < TMF; t++) UNR for (int f = 0; f < NF; f++) P[t][f] = 0.0f;
        UNR for (int kf = 0; kf < NF; kf++) {
            metal::vec<half, 8> Af[TMF];
            UNR for (int t = 0; t < TMF; t++) {
                auto pa = on + (size_t)XROW(t, 0) * D + kf * 16 + fn;
                Af[t] = frag2(pa, pa + 8 * D);
            }
            UNR for (int f = 0; f < NF; f += 2) {
                auto pw = wpo + (size_t)(f * 16 + fm) * D + kf * 16 + fn;
                const metal::vec<half, 8> w0 = frag2(pw, pw + 8 * D);
                const metal::vec<half, 8> w1 = frag2(pw + 16 * D, pw + 24 * D);
                UNR for (int t = 0; t < TMF; t++)
                    nax_mma<float, half, half, false, true, true>(P[t][f], P[t][f + 1], Af[t], w0, w1);
            }
        }
        UNR for (int t = 0; t < TMF; t++) UNR for (int f = 0; f < NF; f++) UNR for (short i = 0; i < 8; i++)
            Ph[t][f][i] = half(P[t][f][i]);
    }
    // g = xn Wgo^T
    metal::vec<float, 8> G[TMF][NF];
    UNR for (int t = 0; t < TMF; t++) UNR for (int f = 0; f < NF; f++) G[t][f] = 0.0f;
    UNR for (int kf = 0; kf < NF; kf++) {
        metal::vec<half, 8> Af[TMF];
        UNR for (int t = 0; t < TMF; t++) {
            auto pa = xn + (size_t)XROW(t, 0) * D + kf * 16 + fn;
            Af[t] = frag2(pa, pa + 8 * D);
        }
        UNR for (int f = 0; f < NF; f += 2) {
            auto pw = wgo + (size_t)(f * 16 + fm) * D + kf * 16 + fn;
            const metal::vec<half, 8> w0 = frag2(pw, pw + 8 * D);
            const metal::vec<half, 8> w1 = frag2(pw + 16 * D, pw + 24 * D);
            UNR for (int t = 0; t < TMF; t++)
                nax_mma<float, half, half, false, true, true>(G[t][f], G[t][f + 1], Af[t], w0, w1);
        }
    }
    #define YVAL(t, f, i, c) half(float(Ph[t][f][i]) * mlx_sigmoid(float(half(G[t][f][i]))))
""" + _RESID_EPI + _LN_EPI

TAIL2_CFGS = [dict(tm=16, nsg=4), dict(tm=16, nsg=8), dict(tm=16, nsg=2), dict(tm=32, nsg=4)]


def gemm2_gate_addln(on: mx.array, w_po: mx.array, xn: mx.array, w_go: mx.array, z: mx.array, ln, *, tm: int = 16,
                     nsg: int = 4, w_tb: mx.array | None = None):
    """z + half(half(on Wpo^T) * sigmoid(half(xn Wgo^T))) and its next LN [+ tri bias], on / xn [R, D] fp16 row-major."""
    B, N, M, D = z.shape
    R = N * M
    assert pair_ok(z, tm) and on.dtype == mx.float16 and xn.dtype == mx.float16 and on.size == z.size == xn.size
    lw, lb, eps = _ln_params(ln)
    ntb = 0 if w_tb is None else w_tb.shape[0]
    dyn = _dynn()
    key = ("tail2",) + (("tb",) if ntb else ()) + (("dyn",) if dyn else ())
    if key not in _kernels:
        _kernels[key] = _K.metal_kernel(
            name="boltz_nax3_tail2" + ("_tb" if ntb else "") + ("_dyn" if dyn else ""),
            input_names=["on", "wpo", "xn", "wgo", "z", "lnw", "lnb", "epsn", "nrows"] + (["wtb"] if ntb else []),
            output_names=["z_out", "ln_out"] + (["tb_out"] if ntb else []),
            source=_dsrc(_tb_variant(_TAIL2_SRC) if ntb else _TAIL2_SRC, dyn), header=HEADER)
    ntg = ((R + tm - 1) // tm + nsg - 1) // nsg
    outs = _kernels[key](
        inputs=[on, w_po, xn, w_go, z, lw if lw is not None else _dummy(), lb if lb is not None else _dummy(),
                mx.array([eps], mx.float32), _nrows(R, N, dyn)] + ([w_tb] if ntb else []),
        template=[("D", D), ("TM", tm), ("NSG", nsg)] + _nn(N, dyn) + [("TRANS_O", 0), ("HAS_W", int(lw is not None)),
                  ("HAS_B", int(lb is not None))] + ([("NTB", ntb)] if ntb else []),
        grid=(ntg * nsg * 32, 1, 1), threadgroup=(nsg * 32, 1, 1),
        output_shapes=[z.shape, z.shape] + ([(ntb, R)] if ntb else []),
        output_dtypes=[mx.float32, mx.float16] + ([mx.float16] if ntb else []))
    if ntb:
        return outs[0], outs[1], outs[2].reshape(1, ntb, N, M)
    return outs[0], outs[1]


def trimul_tail2_addln(o, norm_out, eps_scale, w_po, xn, w_go, z, ln, *, w_tb=None, tln=None, cfg=None):
    """trimul_tail_addln as transpose_ln16 + gemm2_gate_addln (same outputs; the LN of o is bitwise the unfused
    path's norm_out.f16(o^T)). tln / cfg: configs (default PAIR_SITE_CFG['tln'] / ['tail2'])."""
    on = transpose_ln16(o, norm_out, eps_scale, **(tln or PAIR_SITE_CFG["tln"]))
    return gemm2_gate_addln(on, w_po, xn.reshape(-1, xn.shape[-1]), w_go, z, ln, w_tb=w_tb,
                            **(cfg or PAIR_SITE_CFG["tail2"]))


# Per-sub-step ablation of the fused tail (pair_profile 'tail_steps'): STAGE k stops after step k and writes a one-float
# per-lane sink (keeps the work alive, ~no traffic): 1 o loads, 2 + norm_out LN, 3 + p GEMM, 4 + g GEMM, (5 = full).
_TAIL_STAGE_ANCHORS = ["    // 2. norm_out", "    // 3. p = half", "    // 4. g = xn Wgo^T", "    // 5. _gate16"]
_TAIL_STAGE_SINK = {
    1: "Of", 2: "Of", 3: "Ph", 4: "G",
}


def _tail_stage_src(stage: int) -> str:
    src = _TRIMUL_TAIL_SRC
    anchor = _TAIL_STAGE_ANCHORS[stage - 1]
    assert src.count(anchor) == 1
    var = _TAIL_STAGE_SINK[stage]
    sink = (f"    {{ float sk = 0.0f;\n"
            f"      UNR for (int t = 0; t < TMF; t++) UNR for (int f = 0; f < NF; f++) UNR for (short i = 0; i < 8; i++)"
            f" sk += float({var}[t][f][i]);\n"
            f"      z_out[((size_t)tile * 32 + lane) % ((size_t)R * D)] = sk; return; }}\n")
    return src.replace(anchor, sink + anchor)


def trimul_tail_stage(o, norm_out, eps_scale, w_po, xn, w_go, z, ln, stage: int, *, tm=16, nsg=4):
    """The fused tail cut after sub-step `stage` (1-4) — timing only (outputs are not meaningful)."""
    B, N, M, D = z.shape
    R = N * M
    wo, bo, eps_o = _ln_params(norm_out)
    lw, lb, eps = _ln_params(ln)
    key = ("tail_stage", stage)
    if key not in _kernels:
        _kernels[key] = _K.metal_kernel(
            name=f"boltz_nax3_trimul_tail_stage{stage}",
            input_names=["o", "wo", "bo", "epso", "wpo", "xn", "wgo", "z", "lnw", "lnb", "epsn", "nrows"],
            output_names=["z_out", "ln_out"], source=_tail_stage_src(stage), header=HEADER)
    ntg = ((R + tm - 1) // tm + nsg - 1) // nsg
    zo, _ = _kernels[key](
        inputs=[o, wo if wo is not None else _dummy(), bo if bo is not None else _dummy(),
                mx.array([eps_o * eps_scale], mx.float32), w_po, xn, w_go, z, lw if lw is not None else _dummy(),
                lb if lb is not None else _dummy(), mx.array([eps], mx.float32), mx.array([R], mx.int32)],
        template=[("D", D), ("TM", tm), ("NSG", nsg), ("NN", N), ("TRANS_O", 0), ("HAS_WO", int(wo is not None)),
                  ("HAS_BO", int(bo is not None)), ("HAS_W", int(lw is not None)), ("HAS_B", int(lb is not None))],
        grid=(ntg * nsg * 32, 1, 1), threadgroup=(nsg * 32, 1, 1),
        output_shapes=[z.shape, (8,)], output_dtypes=[mx.float32, mx.float16])
    return zo


PAIR_SITE_CFG["tln"] = TLN_CFGS[0]
PAIR_SITE_CFG["tail2"] = TAIL2_CFGS[0]


# ---------------------------------------------------------------------------------------------------------------
# R5-tail: the fused tri-mult tail with lower register pressure. R4's stage cuts (prot400): o loads 0.25, + norm_out LN
# 0.27, + p GEMM 0.40, + g GEMM 1.77, full 2.74–2.86 ms; and the split tail's 2-GEMM kernel (no strided loads at all)
# was still 2.4–2.7 ms vs linear_addln (one GEMM, same epilogue) 0.97 ms -> not the o loads but the live set while
# the second GEMM runs: Ph (32 regs) + G (64 fp32) + loop temporaries (+ V in the epilogue) — consistent with spills
# (every pipeline reports 1024 max threads: the compiler spills rather than lowering occupancy).
# tail3: the same steps and arithmetic as _TRIMUL_TAIL_SRC, but
#   * p and g computed in CH column chunks (NF / CH fragments each): phase P fills Ph chunk by chunk (Of live only here),
#     phase G overwrites Ph[t][f] in place with y16 = half(p * sigmoid(half(g))) chunk by chunk -> peak live set ~ one
#     chunk's fp32 accumulators + 32 fp16 registers (was 64 + 32 + temporaries);
#   * MAXT = 1: the kernel declares max_total_threads_per_threadgroup(NSG * 32) (as MLX's own NAX GEMM does), so the
#     backend may give each thread more registers (a `kernel` macro at the end of the header rewrites both [[kernel]]
#     attributes of mx.fast.metal_kernel's generated source);
#   * UNRK = 0: the g GEMM's k loop is not unrolled (its A fragments come from device memory, so nothing is indexed by kf
#     in registers); the p loop stays unrolled (Of[t][kf] must be indexed statically).
# Every accumulator sees the same k order as before -> outputs bitwise equal to the shipped fused tail (asserted in the
# GPU tests); configs differ in speed only.
# ---------------------------------------------------------------------------------------------------------------
_TAIL3_PG = """
    // 3. p = half(LN(o^T) Wpo^T), chunk by chunk into Ph
    constexpr int CF = NF / CH;
    metal::vec<half, 8> Ph[TMF][NF];
    UNR for (int ch = 0; ch < CH; ch++) {
        metal::vec<float, 8> P[TMF][CF];
        UNR for (int t = 0; t < TMF; t++) UNR for (int f = 0; f < CF; f++) P[t][f] = 0.0f;
        UNR for (int kf = 0; kf < NF; kf++) UNR for (int f = 0; f < CF; f += 2) {
            auto pw = wpo + (size_t)((ch * CF + f) * 16 + fm) * D + kf * 16 + fn;
            const metal::vec<half, 8> w0 = frag2(pw, pw + 8 * D);
            const metal::vec<half, 8> w1 = frag2(pw + 16 * D, pw + 24 * D);
            UNR for (int t = 0; t < TMF; t++)
                nax_mma<float, half, half, false, true, true>(P[t][f], P[t][f + 1], Of[t][kf], w0, w1);
        }
        UNR for (int t = 0; t < TMF; t++) UNR for (int f = 0; f < CF; f++) UNR for (short i = 0; i < 8; i++)
            Ph[t][ch * CF + f][i] = half(P[t][f][i]);
    }
    // __STAGE3__
    // 4. g = xn Wgo^T chunk by chunk; Ph[t][f] <- y16 = half(float(p) * sigmoid(float(half(g)))) in place
    UNR for (int ch = 0; ch < CH; ch++) {
        metal::vec<float, 8> G[TMF][CF];
        UNR for (int t = 0; t < TMF; t++) UNR for (int f = 0; f < CF; f++) G[t][f] = 0.0f;
        __KLOOP__ for (int kf = 0; kf < NF; kf++) {
            metal::vec<half, 8> Af[TMF];
            UNR for (int t = 0; t < TMF; t++) {
                auto pa = xn + (size_t)XROW(t, 0) * D + kf * 16 + fn;
                Af[t] = frag2(pa, pa + 8 * D);
            }
            UNR for (int f = 0; f < CF; f += 2) {
                auto pw = wgo + (size_t)((ch * CF + f) * 16 + fm) * D + kf * 16 + fn;
                const metal::vec<half, 8> w0 = frag2(pw, pw + 8 * D);
                const metal::vec<half, 8> w1 = frag2(pw + 16 * D, pw + 24 * D);
                UNR for (int t = 0; t < TMF; t++)
                    nax_mma<float, half, half, false, true, true>(G[t][f], G[t][f + 1], Af[t], w0, w1);
            }
        }
        UNR for (int t = 0; t < TMF; t++) UNR for (int f = 0; f < CF; f++) UNR for (short i = 0; i < 8; i++)
            Ph[t][ch * CF + f][i] = half(float(Ph[t][ch * CF + f][i]) * mlx_sigmoid(float(half(G[t][f][i]))));
    }
    // __STAGE4__
    #define YVAL(t, f, i, c) Ph[t][f][i]
"""


def _tail3_src(unrk: bool, stage: int = 5) -> str:
    """_TRIMUL_TAIL_SRC with steps 3 / 4 replaced by the chunked _TAIL3_PG; stage 3 / 4: cut (one-float sink per
    lane) after phase P / G, for the profile."""
    src = _TRIMUL_TAIL_SRC
    a = src.index("    // 3. p = half(LN(o^T) Wpo^T)")
    b = src.index("    #define YVAL(t, f, i, c) half(float(Ph[t][f][i])")
    b = src.index("\n", b) + 1
    pg = _TAIL3_PG.replace("__KLOOP__", "UNR" if unrk else '_Pragma("clang loop unroll(disable)")')
    sink = ("    { float sk = 0.0f;\n"
            "      UNR for (int t = 0; t < TMF; t++) UNR for (int f = 0; f < NF; f++) UNR for (short i = 0; i < 8; i++)"
            " sk += float(Ph[t][f][i]);\n"
            "      z_out[((size_t)tile * 32 + lane) % ((size_t)R * D)] = sk; return; }\n")
    for st in (3, 4):
        pg = pg.replace(f"    // __STAGE{st}__\n", sink if stage == st else "")
    return src[:a] + pg + src[b:]


def _maxt_header(nthreads: int) -> str:
    """HEADER + a `kernel` macro turning the generated definition's [[kernel]] into
    [[kernel, max_total_threads_per_threadgroup(nthreads)]] (the header's last line). The kernel source must end with
    _MAXT_END (#undef), so mx.fast.metal_kernel's explicit instantiation after the body keeps a plain [[kernel]] (the
    attribute is not allowed there) — the same split as MLX's own NAX GEMM definition / instantiation."""
    return HEADER + f"\n#define kernel kernel, max_total_threads_per_threadgroup({nthreads})\n"


_MAXT_END = "\n#undef kernel\n"


TAIL3_CFGS = [
    dict(tm=16, nsg=4, ch=2, maxt=1, unrk=1),
    dict(tm=16, nsg=4, ch=2, maxt=0, unrk=1),
    dict(tm=16, nsg=4, ch=4, maxt=1, unrk=1),
    dict(tm=16, nsg=4, ch=4, maxt=0, unrk=1),
    dict(tm=16, nsg=4, ch=1, maxt=1, unrk=1),   # the shipped structure + the thread bound only
    dict(tm=16, nsg=4, ch=2, maxt=1, unrk=0),
    dict(tm=16, nsg=8, ch=2, maxt=1, unrk=1),
    dict(tm=16, nsg=8, ch=4, maxt=1, unrk=1),
    dict(tm=16, nsg=2, ch=2, maxt=1, unrk=1),
    dict(tm=16, nsg=16, ch=4, maxt=0, unrk=0),   # 9: 16 simdgroups (chip_sweep, M5 Max at N 384-400)
]


def trimul_tail3_addln(o: mx.array, norm_out, eps_scale: float, w_po: mx.array, xn: mx.array, w_go: mx.array,
                       z: mx.array, ln, *, w_tb: mx.array | None = None, tm: int = 16, nsg: int = 4, ch: int = 2,
                       maxt: int = 1, unrk: int = 1, stage: int = 5):
    """trimul_tail_addln with the R5 register layout (same outputs bitwise). stage 3 / 4: profile cuts."""
    B, N, M, D = z.shape
    R = N * M
    NF = D // 16
    assert pair_ok(z, tm) and o.dtype == mx.float16 and o.size == R * D and xn.dtype == mx.float16
    assert NF % ch == 0 and (NF // ch) % 2 == 0 and tm == 16
    wo, bo, eps_o = _ln_params(norm_out)
    lw, lb, eps = _ln_params(ln)
    ntb = 0 if w_tb is None else w_tb.shape[0]
    dyn = _dynn()
    key = ("tail3", ntb > 0, bool(unrk), int(maxt) * nsg, stage) + (("dyn",) if dyn else ())
    if key not in _kernels:
        src = _tail3_src(bool(unrk), stage)
        _kernels[key] = _K.metal_kernel(
            name=f"boltz_nax3_tail3{'_tb' if ntb else ''}_u{int(unrk)}_m{int(maxt) * nsg}_s{stage}" + ("_dyn" if dyn else ""),
            input_names=["o", "wo", "bo", "epso", "wpo", "xn", "wgo", "z", "lnw", "lnb", "epsn", "nrows"]
            + (["wtb"] if ntb else []),
            output_names=["z_out", "ln_out"] + (["tb_out"] if ntb else []),
            source=_dsrc(_tb_variant(src) if ntb else src, dyn) + (_MAXT_END if maxt else ""),
            header=_maxt_header(nsg * 32) if maxt else HEADER)
    ntg = ((R + tm - 1) // tm + nsg - 1) // nsg
    outs = _kernels[key](
        inputs=[o, wo if wo is not None else _dummy(), bo if bo is not None else _dummy(),
                mx.array([eps_o * eps_scale], mx.float32), w_po, xn, w_go, z, lw if lw is not None else _dummy(),
                lb if lb is not None else _dummy(), mx.array([eps], mx.float32), _nrows(R, N, dyn)]
        + ([w_tb] if ntb else []),
        template=[("D", D), ("TM", tm), ("NSG", nsg)] + _nn(N, dyn) + [("TRANS_O", 0), ("HAS_WO", int(wo is not None)),
                  ("HAS_BO", int(bo is not None)), ("HAS_W", int(lw is not None)), ("HAS_B", int(lb is not None)),
                  ("CH", ch)] + ([("NTB", ntb)] if ntb else []),
        grid=(ntg * nsg * 32, 1, 1), threadgroup=(nsg * 32, 1, 1),
        output_shapes=[z.shape, z.shape] + ([(ntb, R)] if ntb else []),
        output_dtypes=[mx.float32, mx.float16] + ([mx.float16] if ntb else []))
    if ntb:
        return outs[0], outs[1], outs[2].reshape(1, ntb, N, M)
    return outs[0], outs[1]


# Diagnostics of the shipped fused tail (profile only): stage-4 cut variants that isolate the g GEMM
#   g_only : p GEMM skipped (Ph = 0, folded away) -> g GEMM with only Of-free registers live
#   p_dead : p computed and folded into the sink before the g GEMM -> Ph not live during it
def _tail_diag_src(kind: str) -> str:
    src = _tail_stage_src(4)
    a = src.index("    // 3. p = half(LN(o^T) Wpo^T)")
    b = src.index("    // 4. g = xn Wgo^T")
    if kind == "g_only":
        mid = ("    metal::vec<half, 8> Ph[TMF][NF];\n"
               "    UNR for (int t = 0; t < TMF; t++) UNR for (int f = 0; f < NF; f++) Ph[t][f] = half(0);\n"
               "    float sk_p = float(Of[0][0][0]);\n")
    else:  # p_dead
        p_block = src[a:b]
        mid = p_block + ("    float sk_p = 0.0f;\n"
                         "    UNR for (int t = 0; t < TMF; t++) UNR for (int f = 0; f < NF; f++) UNR for (short i = 0; i < 8; i++)"
                         " sk_p += float(Ph[t][f][i]);\n")
    src = src[:a] + mid + src[b:]
    old = "sk += float(G[t][f][i]);\n"
    assert src.count(old) == 1
    return src.replace(old, "sk += float(G[t][f][i]);\n      sk += sk_p;\n")


def trimul_tail_diag(o, norm_out, eps_scale, w_po, xn, w_go, z, ln, kind: str, *, tm=16, nsg=4):
    B, N, M, D = z.shape
    R = N * M
    wo, bo, eps_o = _ln_params(norm_out)
    lw, lb, eps = _ln_params(ln)
    key = ("tail_diag", kind)
    if key not in _kernels:
        _kernels[key] = _K.metal_kernel(
            name=f"boltz_nax3_trimul_tail_diag_{kind}",
            input_names=["o", "wo", "bo", "epso", "wpo", "xn", "wgo", "z", "lnw", "lnb", "epsn", "nrows"],
            output_names=["z_out", "ln_out"], source=_tail_diag_src(kind), header=HEADER)
    ntg = ((R + tm - 1) // tm + nsg - 1) // nsg
    zo, _ = _kernels[key](
        inputs=[o, wo if wo is not None else _dummy(), bo if bo is not None else _dummy(),
                mx.array([eps_o * eps_scale], mx.float32), w_po, xn, w_go, z, lw if lw is not None else _dummy(),
                lb if lb is not None else _dummy(), mx.array([eps], mx.float32), mx.array([R], mx.int32)],
        template=[("D", D), ("TM", tm), ("NSG", nsg), ("NN", N), ("TRANS_O", 0), ("HAS_WO", int(wo is not None)),
                  ("HAS_BO", int(bo is not None)), ("HAS_W", int(lw is not None)), ("HAS_B", int(lb is not None))],
        grid=(ntg * nsg * 32, 1, 1), threadgroup=(nsg * 32, 1, 1),
        output_shapes=[z.shape, (8,)], output_dtypes=[mx.float32, mx.float16])
    return zo


def R5_PROBES(o, x16, w16, z, ln, on):
    """(name, build) pairs for tools/prof/metal_check pipeline probes of the R5 kernels."""
    out = [(f"tail3 {c}", lambda c=c: trimul_tail3_addln(o, ln, 2.0 ** -16, w16, x16, w16, z, ln, **c))
           for c in TAIL3_CFGS]
    out += [(f"tail3 stage{st}", lambda st=st: trimul_tail3_addln(o, ln, 2.0 ** -16, w16, x16, w16, z, ln, stage=st))
            for st in (3, 4)]
    out += [(f"tail3 tb", lambda: trimul_tail3_addln(o, ln, 2.0 ** -16, w16, x16, w16, z, ln,
                                                     w_tb=mx.zeros((4, 128), mx.float16)))]
    out += [(f"tail diag {k}", lambda k=k: trimul_tail_diag(o, ln, 2.0 ** -16, w16, x16, w16, z, ln, k))
            for k in ("g_only", "p_dead")]
    return out


PAIR_SITE_CFG["tail3"] = TAIL3_CFGS[0]


# Diagnostics (profile only, not integrated): the thread bound on other pair kernels, same source / arithmetic.
def linear_addln_maxt(x, w, z, ln, *, trans_y=False, trans_out=False, tm=16, nsg=4):
    B, N, M, D = z.shape
    R = N * M
    lw, lb, eps = _ln_params(ln)
    key = ("lin_addln_maxt", nsg)
    if key not in _kernels:
        _kernels[key] = _K.metal_kernel(
            name=f"boltz_nax3_lin_addln_m{nsg}", input_names=["x", "w", "bias", "z", "lnw", "lnb", "epsn", "nrows"],
            output_names=["z_out", "ln_out"], source=_LIN_ADDLN_SRC + _MAXT_END, header=_maxt_header(nsg * 32))
    ntg = ((R + tm - 1) // tm + nsg - 1) // nsg
    return _kernels[key](
        inputs=[x, w, _dummy(), z, lw if lw is not None else _dummy(), lb if lb is not None else _dummy(),
                mx.array([eps], mx.float32), mx.array([R], mx.int32)],
        template=[("D", D), ("TM", tm), ("NSG", nsg), ("NN", N), ("TRANS_Y", int(trans_y)), ("TRANS_O", int(trans_out)),
                  ("HAS_BIAS", 0), ("HAS_W", int(lw is not None)), ("HAS_B", int(lb is not None))],
        grid=(ntg * nsg * 32, 1, 1), threadgroup=(nsg * 32, 1, 1),
        output_shapes=[z.shape, z.shape], output_dtypes=[mx.float32, mx.float16])


def transition_maxt(xn, w12, w3, z, *, tm=16, hb=32, nsg=8, tgw=1):
    """kernels_nax2.transition_nax(resid=z) compiled with max_total_threads_per_threadgroup(NSG * 32)."""
    from boltz_mlx import kernels_nax2 as _KN2

    D = xn.shape[-1]
    H = w3.shape[1]
    R = xn.size // D
    key = ("trans_maxt", nsg)
    if key not in _kernels:
        _kernels[key] = _K.metal_kernel(
            name=f"boltz_nax3_transition_m{nsg}", input_names=["x", "w12", "w3", "scales", "resid", "nrows"],
            output_names=["out"], source=_KN2._TRANS_SRC + _MAXT_END,
            header=_KN2.HEADER + f"\n#define kernel kernel, max_total_threads_per_threadgroup({nsg * 32})\n")
    ntg = ((R + tm - 1) // tm + nsg - 1) // nsg
    (y,) = _kernels[key](
        inputs=[xn, w12, w3, mx.array([1.0, 1.0], mx.float32), z, mx.array([R], mx.int32)],
        template=[("D", D), ("H", H), ("TM", tm), ("HB", hb), ("NSG", nsg), ("TGW", int(tgw)), ("RESID", 1),
                  ("OT", mx.float32)],
        grid=(ntg * nsg * 32, 1, 1), threadgroup=(nsg * 32, 1, 1), output_shapes=[z.shape], output_dtypes=[mx.float32])
    return y


# ---------------------------------------------------------------------------------------------------------------
# R6-trans: pair transition candidates (prot400: shipped MPP kernel 4.34 ms, +0.42 with the R3 LNX epilogue; 63 GFLOP
# -> ~2.9 ms at ~22 TFLOP/s). Lesson of R5 (pair_profile r5): a fully unrolled GEMM k-loop whose fragments come from
# device memory lets the scheduler hoist all its loads into the preceding code -> spills; a rolled loop is a
# scheduling barrier (tail3 1.43 vs 2.48 ms, only the rolled-g-loop config won).
# rtrans (register NAX fragments, relaxed 16x32x16, TA = 0): per simdgroup 16 rows, O[16 x 128] fp32 accumulators;
#   rolled loop over hidden blocks of HB = 32 units: Sa / Sb = xn W1b^T / W2b^T (k loop over D rolled unless the A
#   fragments are kept in registers, KEEPA), Sh = half(silu(Sa) * Sb) in registers (destination layout == A layout),
#   O += Sh W3b^T; weights streamed from device memory (L1-shared by the threadgroup's simdgroups).
#   Epilogue entirely in registers: z_new = z + float(half(O)) (the shipped expression), LN statistics via xor-1 / xor-8
#   shuffles, the next layer's LN (fp16) and the s-track LN (fp16) per fragment, the pair bias as 8 NAX MMAs of the
#   LN_s fragments with Wz (16 heads; the second B^T fragment zero) — no z read-back, no barrier.
# The O accumulation order differs from the MPP kernel's (ulp-level; not bitwise), as do the LN / bias sum orders.
# ---------------------------------------------------------------------------------------------------------------
_RT_LN_EPI = """
    if constexpr (HAS_XN != 0 || NHB > 0) {
        const float eps_n = epsn[0], eps_s = lns_eps[0];
        UNR for (int t = 0; t < TMF; t++) {
            float mean[2], rn[2], rs[2];
            UNR for (short r = 0; r < 2; r++) {
                float s = 0.0f;
                UNR for (int f = 0; f < NF; f++) UNR for (short j = 0; j < 4; j++) s += V[t][f][r * 4 + j];
                s += simd_shuffle_xor(s, (ushort)1);
                s += simd_shuffle_xor(s, (ushort)8);
                mean[r] = s / D;
                float ss = 0.0f;
                UNR for (int f = 0; f < NF; f++) UNR for (short j = 0; j < 4; j++) {
                    const float d = V[t][f][r * 4 + j] - mean[r];
                    ss += d * d;
                }
                ss += simd_shuffle_xor(ss, (ushort)1);
                ss += simd_shuffle_xor(ss, (ushort)8);
                rn[r] = metal::precise::rsqrt(ss / D + eps_n);
                rs[r] = metal::precise::rsqrt(ss / D + eps_s);
            }
            metal::vec<float, 8> B0 = 0.0f, B1 = 0.0f;
            const metal::vec<half, 8> zf = half(0);
            UNR for (int f = 0; f < NF; f++) {
                const int c = f * 16 + fn;
                metal::vec<half, 8> ls;
                UNR for (short r = 0; r < 2; r++) {
                    metal::vec<half, 4> h;
                    UNR for (short j = 0; j < 4; j++) {
                        const float d = V[t][f][r * 4 + j] - mean[r];
                        if constexpr (HAS_XN != 0) {
                            float o = d * rn[r];
                            if (HAS_W) o = o * lnw[c + j];
                            if (HAS_B) o = o + lnb[c + j];
                            h[j] = half(o);
                        }
                        if constexpr (NHB > 0) {
                            float y = d * rs[r];
                            if (HAS_W2) y = y * lns_w[c + j];
                            if (HAS_B2) y = y + lns_b[c + j];
                            ls[r * 4 + j] = half(y);
                        }
                    }
                    if constexpr (HAS_XN != 0) *(device metal::vec<half, 4>*)(ln_out + (size_t)ZROW(t, r) * D + c) = h;
                }
                if constexpr (NHB > 0) {
                    auto pz = pbw + (size_t)fm * D + f * 16 + fn;
                    const metal::vec<half, 8> wz0 = frag2(pz, pz + 8 * D);
                    nax_mma<float, half, half, false, true, true>(B0, B1, ls, wz0, zf);
                }
            }
            if constexpr (NHB > 0) {
                UNR for (short r = 0; r < 2; r++) UNR for (short j = 0; j < 4; j++)
                    pb_out[(size_t)(fn + j) * R + ZROW(t, r)] = half(B0[r * 4 + j]);
            }
        }
    }
"""

_RTRANS_SRC = _TILE_PROLOGUE + """
    #define XROW(t, r) (r0 + (t) * 16 + fm + 8 * (r))
    #define ZROW(t, r) XROW(t, r)
    constexpr int HF = HB / 16;                       // hidden fragments per block (even)
    metal::vec<float, 8> O[TMF][NF];
    UNR for (int t = 0; t < TMF; t++) UNR for (int f = 0; f < NF; f++) O[t][f] = 0.0f;
    metal::vec<half, 8> Ak[TMF][KEEPA ? NF : 1];
    if constexpr (KEEPA != 0) {
        UNR for (int t = 0; t < TMF; t++) UNR for (int kf = 0; kf < NF; kf++) {
            auto pa = x + (size_t)XROW(t, 0) * D + kf * 16 + fn;
            Ak[t][kf] = frag2(pa, pa + 8 * D);
        }
    }
    _Pragma("clang loop unroll(disable)")
    for (int hb = 0; hb < H; hb += HB) {
        metal::vec<float, 8> Sa[TMF][HF], Sb[TMF][HF];
        UNR for (int t = 0; t < TMF; t++) UNR for (int f = 0; f < HF; f++) { Sa[t][f] = 0.0f; Sb[t][f] = 0.0f; }
        __KLOOP__ for (int kf = 0; kf < NF; kf++) {
            metal::vec<half, 8> Af[TMF];
            UNR for (int t = 0; t < TMF; t++) {
                if constexpr (KEEPA != 0) {
                    Af[t] = Ak[t][kf];
                } else {
                    auto pa = x + (size_t)XROW(t, 0) * D + kf * 16 + fn;
                    Af[t] = frag2(pa, pa + 8 * D);
                }
            }
            UNR for (int f = 0; f < HF; f += 2) {
                auto p1 = w12 + (size_t)(hb + f * 16 + fm) * D + kf * 16 + fn;
                auto p2 = p1 + (size_t)H * D;
                const metal::vec<half, 8> a0 = frag2(p1, p1 + 8 * D), a1 = frag2(p1 + 16 * D, p1 + 24 * D);
                const metal::vec<half, 8> b0 = frag2(p2, p2 + 8 * D), b1 = frag2(p2 + 16 * D, p2 + 24 * D);
                UNR for (int t = 0; t < TMF; t++) {
                    nax_mma<float, half, half, false, true, true>(Sa[t][f], Sa[t][f + 1], Af[t], a0, a1);
                    nax_mma<float, half, half, false, true, true>(Sb[t][f], Sb[t][f + 1], Af[t], b0, b1);
                }
            }
        }
        metal::vec<half, 8> Sh[TMF][HF];
        UNR for (int t = 0; t < TMF; t++) UNR for (int f = 0; f < HF; f++) UNR for (short i = 0; i < 8; i++) {
            const float a = Sa[t][f][i];
            Sh[t][f][i] = half(a * (1.0f / (1.0f + metal::exp(-a))) * Sb[t][f][i]);
        }
        UNR for (int kh = 0; kh < HF; kh++) UNR for (int f = 0; f < NF; f += 2) {
            auto pw = w3 + (size_t)(f * 16 + fm) * H + hb + kh * 16 + fn;
            const metal::vec<half, 8> w0 = frag2(pw, pw + 8 * H), w1 = frag2(pw + 16 * H, pw + 24 * H);
            UNR for (int t = 0; t < TMF; t++)
                nax_mma<float, half, half, false, true, true>(O[t][f], O[t][f + 1], Sh[t][kh], w0, w1);
        }
    }
    #define YVAL(t, f, i, c) half(O[t][f][i])
""" + _RESID_EPI + _RT_LN_EPI


def _rtrans_src(keepa: bool, unrk: bool) -> str:
    kloop = "UNR" if (keepa or unrk) else '_Pragma("clang loop unroll(disable)")'
    return _RTRANS_SRC.replace("__KLOOP__", kloop)


def transition_reg(xn, w12, w3, z, *, next_ln=None, ln_s=None, w_s=None, nsg: int = 4, hb: int = 32,
                   keepa: int = 0, unrk: int = 0, maxt: int = 1):
    """rtrans: pair transition (+z) on register NAX fragments with optional in-register epilogue outputs (next LN
    fp16, s-track pair bias [1, 16, N, N] fp16). Returns (z_new, xn_next | None, bias | None)."""
    D = xn.shape[-1]
    H = w3.shape[1]
    R = xn.size // D
    tm = 16
    assert xn.dtype == mx.float16 and w12.dtype == mx.float16 and w3.dtype == mx.float16 and D % 32 == 0
    assert w12.shape == (2 * H, D) and w3.shape == (D, H) and H % hb == 0 and hb % 32 == 0 and R >= tm
    assert z.dtype == mx.float32 and z.size == xn.size
    has_xn = next_ln is not None
    nhb = 0 if w_s is None else w_s.shape[0]
    assert nhb in (0, 16)
    w1, b1, e1 = _ln_params(next_ln) if has_xn else (None, None, 1e-5)
    w2, b2, e2 = _ln_params(ln_s) if nhb else (None, None, 1e-5)
    dyn = _dynn()  # R8-coldN: the source never reads NN; DYNN drops it from the template (one library for all N)
    key = ("rtrans", bool(keepa), bool(unrk), int(maxt) * nsg) + (("dyn",) if dyn else ())
    if key not in _kernels:
        _kernels[key] = _K.metal_kernel(
            name=f"boltz_nax3_rtrans_k{int(keepa)}_u{int(unrk)}_m{int(maxt) * nsg}" + ("_dyn" if dyn else ""),
            input_names=["x", "w12", "w3", "z", "lnw", "lnb", "epsn", "lns_w", "lns_b", "lns_eps", "pbw", "nrows"],
            output_names=["z_out", "ln_out", "pb_out"],
            source=_rtrans_src(bool(keepa), bool(unrk)) + (_MAXT_END if maxt else ""),
            header=_maxt_header(nsg * 32) if maxt else HEADER)
    d = _dummy()
    N = z.shape[1] if z.ndim == 4 else 1
    ntg = ((R + tm - 1) // tm + nsg - 1) // nsg
    zo, xo, pb = _kernels[key](
        inputs=[xn, w12, w3, z, w1 if w1 is not None else d, b1 if b1 is not None else d, mx.array([e1], mx.float32),
                w2 if w2 is not None else d, b2 if b2 is not None else d, mx.array([e2], mx.float32),
                w_s if nhb else _dummy(mx.float16), mx.array([R], mx.int32)],
        template=[("D", D), ("H", H), ("TM", tm), ("NSG", nsg), ("HB", hb), ("KEEPA", int(keepa))] + _nn(N, dyn) + [
                  ("HAS_XN", int(has_xn)), ("NHB", nhb), ("HAS_W", int(w1 is not None)), ("HAS_B", int(b1 is not None)),
                  ("HAS_W2", int(w2 is not None)), ("HAS_B2", int(b2 is not None))],
        grid=(ntg * nsg * 32, 1, 1), threadgroup=(nsg * 32, 1, 1),
        output_shapes=[z.shape, xn.shape if has_xn else (8,), (nhb, R) if nhb else (8,)],
        output_dtypes=[mx.float32, mx.float16, mx.float16])
    bias = pb.reshape(1, nhb, z.shape[1], z.shape[2]) if (nhb and z.ndim == 4) else (pb if nhb else None)
    return zo, (xo if has_xn else None), bias


# Standalone next-LN + pair-bias rows (for transition variants without the epilogue): one simdgroup per row, the
# LN kernel's lane layout and _LNX_EPI's arithmetic (bitwise the R3 transition_lnx epilogue outputs for the same z_new).
_LNX_ROWS_SRC = """
    constexpr int NV = D / 32;
    const int N_ROWS = (int)nrows[0];
    const int row = (int)(threadgroup_position_in_grid.x * RPT + simdgroup_index_in_threadgroup);
    if (row >= N_ROWS) return;
    const uint ln_lane = thread_index_in_simdgroup;
    float wz[NHB > 0 ? NHB : 1][NV];
    if constexpr (NHB > 0) {
        UNR for (int h = 0; h < NHB; h++) UNR for (int i = 0; i < NV; i++) wz[h][i] = float(pbw[h * D + ln_lane + 32 * i]);
    }
    float v[NV];
    float s = 0.0f;
    UNR for (int i = 0; i < NV; i++) {
        v[i] = zo[(size_t)row * D + ln_lane + 32 * i];
        s += v[i];
    }
    const float mean = simd_sum(s) / D;
    float ss = 0.0f;
    UNR for (int i = 0; i < NV; i++) {
        const float d = v[i] - mean;
        ss += d * d;
    }
    const float ssum = simd_sum(ss);
    if constexpr (HAS_XN != 0) {
        const float rstd = metal::precise::rsqrt(ssum / D + lnn_eps[0]);
        UNR for (int i = 0; i < NV; i++) {
            const uint c = ln_lane + 32 * i;
            float y = (v[i] - mean) * rstd;
            if (HAS_W1) y = y * lnn_w[c];
            if (HAS_B1) y = y + lnn_b[c];
            xn_out[(size_t)row * D + c] = half(y);
        }
    }
    if constexpr (NHB > 0) {
        const float rstd2 = metal::precise::rsqrt(ssum / D + lns_eps[0]);
        float p[NHB];
        UNR for (int h = 0; h < NHB; h++) p[h] = 0.0f;
        UNR for (int i = 0; i < NV; i++) {
            const uint c = ln_lane + 32 * i;
            float y = (v[i] - mean) * rstd2;
            if (HAS_W2) y = y * lns_w[c];
            if (HAS_B2) y = y + lns_b[c];
            const float yh = float(half(y));
            UNR for (int h = 0; h < NHB; h++) p[h] += yh * wz[h][i];
        }
        UNR for (int h = 0; h < NHB; h++) {
            const float t = simd_sum(p[h]);
            if (ln_lane == (uint)h) pb_out[(size_t)h * N_ROWS + row] = half(t);
        }
    }
"""


def lnx_rows(z_new, *, next_ln=None, ln_s=None, w_s=None, rpt: int = 8):
    D = z_new.shape[-1]
    R = z_new.size // D
    has_xn = next_ln is not None
    nhb = 0 if w_s is None else w_s.shape[0]
    if not has_xn and not nhb:
        return None, None
    w1, b1, e1 = _ln_params(next_ln) if has_xn else (None, None, 1e-5)
    w2, b2, e2 = _ln_params(ln_s) if nhb else (None, None, 1e-5)
    key = ("lnx_rows",)
    if key not in _kernels:
        _kernels[key] = _K.metal_kernel(
            name="boltz_nax3_lnx_rows",
            input_names=["zo", "lnn_w", "lnn_b", "lnn_eps", "lns_w", "lns_b", "lns_eps", "pbw", "nrows"],
            output_names=["xn_out", "pb_out"], source=_LNX_ROWS_SRC, header=HEADER)
    d = _dummy()
    xo, pb = _kernels[key](
        inputs=[z_new, w1 if w1 is not None else d, b1 if b1 is not None else d, mx.array([e1], mx.float32),
                w2 if w2 is not None else d, b2 if b2 is not None else d, mx.array([e2], mx.float32),
                w_s if nhb else _dummy(mx.float16), mx.array([R], mx.int32)],
        template=[("D", D), ("RPT", rpt), ("HAS_XN", int(has_xn)), ("NHB", nhb), ("HAS_W1", int(w1 is not None)),
                  ("HAS_B1", int(b1 is not None)), ("HAS_W2", int(w2 is not None)), ("HAS_B2", int(b2 is not None))],
        grid=(((R + rpt - 1) // rpt) * rpt * 32, 1, 1), threadgroup=(rpt * 32, 1, 1),
        output_shapes=[z_new.shape if has_xn else (8,), (nhb, R) if nhb else (8,)],
        output_dtypes=[mx.float16, mx.float16])
    bias = pb.reshape(1, nhb, z_new.shape[1], z_new.shape[2]) if (nhb and z_new.ndim == 4) else (pb if nhb else None)
    return (xo if has_xn else None), bias


def transition_mlx(xn, w12, w3, z, *, chunk: int = 0):
    """The coordinator's MLX-GEMM comparison: MLX fp16 GEMM 128 -> 2H, SwiGLU (compiled, fp32 math, fp16 out), MLX
    fp16 GEMM H -> 128, + z in fp32 (radd). chunk > 0: rows in chunks (the 2H-wide fp16 intermediate stays cache-sized).
    Rounding: the 2H-wide GEMM output is rounded to fp16 before the SwiGLU (the NAX kernels use the fp32 accumulators)."""
    from boltz_mlx import kernels as _K

    D = xn.shape[-1]
    H = w3.shape[1]
    R = xn.size // D
    xr, zr = xn.reshape(R, D), z.reshape(R, D)

    def body(xc, zc):
        return _K.radd(zc, _K.swiglu16(xc @ w12.T, H) @ w3.T)

    if not chunk or chunk >= R:
        return body(xr, zr).reshape(z.shape)
    return mx.concatenate([body(xr[i: i + chunk], zr[i: i + chunk]) for i in range(0, R, chunk)], 0).reshape(z.shape)


TRANS3_CFGS = [
    dict(impl="reg", nsg=4, keepa=0, unrk=0, maxt=1),
    dict(impl="reg", nsg=8, keepa=0, unrk=0, maxt=1),
    dict(impl="reg", nsg=4, keepa=1, unrk=1, maxt=1),
    dict(impl="reg", nsg=4, keepa=0, unrk=0, maxt=0),
    dict(impl="reg", nsg=4, keepa=0, unrk=1, maxt=1),
    dict(impl="reg", nsg=2, keepa=0, unrk=0, maxt=1),
    dict(impl="mpp", tm=16, hb=64, nsg=4, tgw=0),
    dict(impl="mpp", tm=16, hb=64, nsg=8, tgw=0),
    dict(impl="mpp", tm=16, hb=32, nsg=4, tgw=0),
    dict(impl="mlx", chunk=0),
    dict(impl="mlx", chunk=16384),
    dict(impl="reg", nsg=16, keepa=0, unrk=0, maxt=1),   # 11: 16 simdgroups (chip_sweep, M5 Max at N 384-400)
]


def transition3(xn, w12, w3, z, *, next_ln=None, ln_s=None, w_s=None, cfg=None):
    """R6 pair transition: (z_new, xn_next | None, bias | None) by the configured implementation."""
    cfg = dict(cfg or site_cfg("trans3", z.shape[1] if z.ndim == 4 else None))
    impl = cfg.pop("impl")
    if impl == "reg":
        return transition_reg(xn, w12, w3, z, next_ln=next_ln, ln_s=ln_s, w_s=w_s, **cfg)
    if impl == "mpp":
        if next_ln is None and w_s is None:
            from boltz_mlx import kernels_nax2 as _KN2

            return _KN2.transition_nax(xn, w12, w3, resid=z, **cfg), None, None
        return transition_lnx(xn, w12, w3, z, next_ln=next_ln, ln_s=ln_s, w_s=w_s, **cfg)
    zn = transition_mlx(xn, w12, w3, z, **cfg)
    xo, bias = lnx_rows(zn, next_ln=next_ln, ln_s=ln_s, w_s=w_s)
    return zn, xo, bias


def transition3_ref(xn, w12, w3, z, *, next_ln=None, ln_s=None, w_s=None, **_):
    return transition_lnx_ref(xn, w12, w3, z, next_ln=next_ln, ln_s=ln_s, w_s=w_s)


PAIR_SITE_CFG["trans3"] = TRANS3_CFGS[0]


def pair_proj16(x16: mx.array, w16: mx.array, cfg=None) -> mx.array:
    """Plain fp16 projection x [..., K] @ w^T on gemm16 (store, fp16 out; the pair-stack tri-att q|k|v|g projection).
    cfg None = PAIR_SITE_CFG['pproj']."""
    cfg = site_cfg("pproj", x16.shape[1] if x16.ndim == 4 else None) if cfg is None else cfg
    K = x16.shape[-1]
    y = gemm16(x16.reshape(-1, K), w16, **cfg)
    return y.reshape(*x16.shape[:-1], w16.shape[0])


# ---------------------------------------------------------------------------------------------------------------
# Token-transformer forward on the kernels above (DiffusionTransformer with prepared biases, --precision f16)
# ---------------------------------------------------------------------------------------------------------------
def cond_gemm(x16: mx.array, w16: mx.array) -> mx.array:
    """Conditioning GEMM (all layers' AdaLN / gate projections), fp16 out: x16 [..., D] @ w16^T."""
    D = x16.shape[-1]
    cfg = _site("cond", x16.size // D)
    if cfg == "mlx":
        return x16 @ w16.T
    return gemm16(x16.reshape(-1, D), w16, **cfg).reshape(*x16.shape[:-1], w16.shape[0])


_GEMM_LISTS = {"qkvg": GEMM_CFGS, "o": GEMM_CFGS, "ff": SWIGLU_CFGS, "ba": GEMM_CFGS, "cond": GEMM_CFGS}
_PAIR_LISTS = {"lin": PAIR_CFGS, "tail": TRIMUL_TAIL_CFGS, "pproj": GEMM_CFGS, "tln": TLN_CFGS, "tail2": TAIL2_CFGS,
               "tail3": TAIL3_CFGS, "trans3": TRANS3_CFGS, "lin_end": PAIR_CFGS}
# Split sites: a call site with its own entry when a device entry sets one, else its parent site's config (per size).
# lin_end = the ending-node tri-att linear_addln (transposed reads, no tri bias), "lin" = the starting node's.
_SITE_PARENT = {"lin_end": "lin"}
# Committed per-site choices measured by `tools/prof/nax3_ab.py suggest all --write` (the A/B's fastest candidate per
# site). Config choices within a site's NAX list are bitwise-neutral (same per-element accumulation order, k ascending
# in 16-steps; LN row statistics per lane group); 'mlx' vs a NAX config changes rounding (fp16-rounded GEMM output
# vs the fp32 accumulator). bench_mlx / run_l3 / screen strip BOLTZ_MLX_* variables, so tuned choices live here.
CFG_FILE = os.path.join(os.path.dirname(__file__), "nax3_cfg.json")


# Size buckets (per-device configs, bitwise-neutral like every choice inside a site's NAX list): "o@256=7" in a config
# string = for calls with size <= 256 use entry 7 of the site's list; the first bucket (ascending) that holds the call
# wins, else the site's plain entry. Size = GEMM rows M for the token-transformer sites (B * N; cond: tokens), N
# (tokens) for the pair sites. A plain "site=v" item clears that site's buckets (environment overrides, A/B runs), so a
# config string lists the plain entry first, then its buckets. A bucket never holds 'mlx' unless the site's plain
# entry is 'mlx' too (the tuning tools only write bitwise-equal candidates).
SITE_BUCKETS: dict = {}


def _apply_cfg(spec: str, table: dict, lists: dict, buckets: dict | None = None) -> None:
    """spec "qkvg=mlx,o=2,o@256=7,ff=1": per site an index into its config list (GEMM_CFGS; SWIGLU_CFGS for ff;
    PAIR_CFGS / TRIMUL_TAIL_CFGS for the pair sites lin / tail) or 'mlx', optionally per size bucket (SITE_BUCKETS)."""
    buckets = SITE_BUCKETS if buckets is None else buckets
    for item in filter(None, spec.split(",")):
        name, v = item.split("=")
        name, _, mx_ = name.partition("@")
        if name not in lists:  # a site this build no longer has (e.g. a dropped experiment): ignored
            continue
        cfg = "mlx" if v == "mlx" else (None if v == "off" else lists[name][int(v)])
        if mx_:
            b = [x for x in buckets.get(name, []) if x[0] != int(mx_)] + [(int(mx_), cfg)]
            buckets[name] = sorted(b, key=lambda x: x[0])
        else:
            table[name] = cfg
            buckets.pop(name, None)


def _pick(name: str, size, gemm: dict, pair: dict, buckets: dict):
    if size is not None:
        for mx_, cfg in buckets.get(name, ()):
            if size <= mx_:
                return cfg
    if name in gemm:
        return gemm[name]
    if name in pair:
        return pair[name]
    if name in _SITE_PARENT:
        return _pick(_SITE_PARENT[name], size, gemm, pair, buckets)
    return None


def site_cfg(name: str, size: int | None = None):
    """Config of GEMM / pair site `name` for a call of `size` (see SITE_BUCKETS): bucket, else the plain entry."""
    return _pick(name, size, DIFF_GEMM_CFG, PAIR_SITE_CFG, SITE_BUCKETS)


def resolve_entry(entry: dict):
    """(site, size) -> config under a per-device entry ({"gemm": ..., "pair": ...} strings) applied to the module
    defaults, without touching the live tables (tests comparing entries: tests/test_cfg_bitwise.py)."""
    gemm, pair, buckets = dict(_BASE_GEMM), dict(_BASE_PAIR), {}
    _apply_cfg(entry.get("gemm", ""), gemm, _GEMM_LISTS, buckets)
    _apply_cfg(entry.get("pair", ""), pair, _PAIR_LISTS, buckets)
    return lambda name, size=None: _pick(name, size, gemm, pair, buckets)


def _load_cfg() -> None:
    # this chip's entry of the per-device file, else "default" (one-line note when the MPP kernels are first used)
    d = _K.device_cfg(CFG_FILE, "nax", "tools/prof/nax3_ab.py all && tools/prof/nax3_ab.py suggest all --write "
                                       "(+ tools/prof/pair_profile.py r5, r6, suggest --write)")
    if d:
        _apply_cfg(d.get("gemm", ""), DIFF_GEMM_CFG, _GEMM_LISTS)
        _apply_cfg(d.get("pair", ""), PAIR_SITE_CFG, _PAIR_LISTS)
    # environment last (diagnostics / A/B runs): BOLTZ_MLX_NAX_GEMM_CFG, BOLTZ_MLX_NAX_PAIR_CFG
    _apply_cfg(os.environ.get("BOLTZ_MLX_NAX_GEMM_CFG", ""), DIFF_GEMM_CFG, _GEMM_LISTS)
    _apply_cfg(os.environ.get("BOLTZ_MLX_NAX_PAIR_CFG", ""), PAIR_SITE_CFG, _PAIR_LISTS)


_BASE_GEMM, _BASE_PAIR = dict(DIFF_GEMM_CFG), dict(PAIR_SITE_CFG)   # module defaults (before any device entry)
_load_cfg()


def _site(name: str, M: int | None = None):
    return site_cfg(name, M)


@mx.compile
def _gresid_mlx(resid, y16, g16, gb):
    return resid + mx.sigmoid(g16.astype(mx.float32) + gb) * y16.astype(mx.float32)


@mx.compile
def _swiglu_mlx(h, n):
    f = mx.float32
    xg, gt, ab = h[:, :n].astype(f), h[:, n: 2 * n].astype(f), h[:, 2 * n:].astype(f)
    return (gt * mx.sigmoid(gt) * xg * ab).astype(mx.float16)


def _gemm_site(name, x, w, cfg=None, **kw):
    """One GEMM site: the NAX kernel (cfg) or the MLX fp16 GEMM + elementwise epilogue (cfg 'mlx'); cfg None = the
    site's configured choice."""
    cfg = _site(name, x.shape[0]) if cfg is None else cfg
    if cfg != "mlx":
        return gemm16(x, w, **kw, **cfg)
    epi = kw.get("epi", EPI_STORE)
    y = x @ w.T
    if epi == EPI_SWIGLU:
        return _swiglu_mlx(y, w.shape[0] // 3)
    if epi == EPI_GRESID:
        M, n, off = x.shape[0], w.shape[0], kw["gate_off"]
        ntok, rep = kw["ntok"], kw["rep"]
        g = kw["gate"][:, off: off + n]
        if g.shape[0] != M:  # conditioning rows shared by `rep` consecutive samples
            g = mx.repeat(g.reshape(-1, ntok, n), rep, axis=0).reshape(M, n)
        return _gresid_mlx(kw["resid"], y, g, kw["gate_bias"][off: off + n])
    return y


def transformer_forward(tr, a: mx.array, rc: RawCond16, attn_bias: list, brep: int = 1, layers=None) -> mx.array:
    """Token transformer (24 x DiffusionTransformerLayer, D 768, 16 heads) in f16 with fused kernels:
    adaln16 -> qkvg GEMM (fp16) -> NAX attention (q bias, pair bias, gate) -> out GEMM + gated residual (fp32)
    -> adaln16 -> ff GEMM + SwiGLU (fp16 hidden) -> down GEMM + gated residual. attn_bias[l]: [B / brep, H, N, N] fp32
    additive (key mask folded in), shared by `brep` consecutive samples. layers: optional range of layer indices
    (module-level checks of single layers)."""
    from boltz_mlx import kernels as _K
    from boltz_mlx import kernels_nax2 as _KN2

    B, N, D = a.shape
    M = B * N
    H = tr.layers[0].pair_bias_attn.num_heads
    a2 = a.reshape(M, D)
    for l in (range(len(tr.layers)) if layers is None else layers):
        layer = tr.layers[l]
        w_qkvg, w_ff, w_o, w_ba, b_q, _ = layer._fused()
        t = adaln16(a2, rc, l, 0)
        h = _gemm_site("qkvg", t, w_qkvg)
        _K.record(f"difftx{D}.qkvg", h)
        o = _KN2.attention_nax(h, attn_bias[l], B=B, N=N, heads=H, c=D // H, qbias=b_q, brep=brep,
                               **_K.diffattn_cfg(B * N))
        a2 = _gemm_site("o", o, w_o, epi=EPI_GRESID, gate=rc.h2, gate_bias=rc.b2, gate_off=(l * 2) * D, resid=a2,
                        ntok=rc.ntok, rep=rc.rep)
        t = adaln16(a2, rc, l, 2)
        hs = _gemm_site("ff", t, w_ff, epi=EPI_SWIGLU, ngrp=3)
        a2 = _gemm_site("ba", hs, w_ba, epi=EPI_GRESID, gate=rc.h2, gate_bias=rc.b2, gate_off=(l * 2 + 1) * D,
                        resid=a2, ntok=rc.ntok, rep=rc.rep)
    return a2.reshape(B, N, D)
