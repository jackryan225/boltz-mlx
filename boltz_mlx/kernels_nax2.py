"""Experimental NAX kernels, round 2 (candidates; nothing in the default paths imports this module). GPU only.

All kernels here use the *layout-agnostic* MetalPerformancePrimitives API, unlike the register kernels in
kernels_nax.py (which move MLX steel fragments into cooperative tensors and are only valid for the relaxed
16x32x16 descriptor, see kernels_nax module docstring / tests/test_nax_layout.py):
  * operands are `metal::tensor<..., tensor_inline>` views of device / threadgroup memory, loaded with
    cooperative_tensor::load or passed straight to matmul2d::run (MPP reads them; L1-cached across simdgroups),
  * elementwise work on accumulators pairs element i of cooperative tensors that come from the same op (same layout
    by construction), and element coordinates come from get_multidimensional_index (idx[0] = column, idx[1] = row),
  * GEMM chaining in registers uses matmul2d::get_left_input_cooperative_tensor(dest) (MPP converts the layout),
  * results are written with cooperative_tensor::store.
Tiles never straddle the end of a matrix: a last tile is moved back inside (it recomputes some rows / columns of its
neighbour and writes identical values), so every kernel requires the tiled extent >= the tile size; the Python
wrappers fall back (raise) below that.

transition_nax       : fused SwiGLU transition  y = (silu(x W1^T) * (x W2^T)) W3^T  on the fp16 LayerNorm output,
                       hidden kept in registers, optional fused residual add (z + y), optional power-of-two scales.
trimul_proj_nax      : fused tri-mult input projection  ab[c, pos] = (x Wp^T)[pos, c] * sigmoid((x Wg^T)[pos, c])
                       * mask[pos] * 2^-K, written channel-major fp16 (the contraction's operand layout), plus
                       optionally g_out = x Wgo^T (position-major fp16).
trimul_contract_nax  : the per-channel contraction (outgoing a b^T / incoming a^T b), fp16 in / fp16 out.
"""

from __future__ import annotations

import mlx.core as mx

from boltz_mlx import kernels as _K  # metal_kernel(): the MPP guard (portability)

from boltz_mlx.kernels_nax import NAX_HEADER

_kernels: dict = {}

HEADER = NAX_HEADER + r"""
using ext2_t = metal::dextents<int32_t, 2>;
using str2_t = metal::array<int32_t, 2>;
template <typename T> using dtensor = metal::tensor<device T, ext2_t, metal::tensor_inline>;
template <typename T> using ttensor = metal::tensor<threadgroup T, ext2_t, metal::tensor_inline>;
"""

# ---------------------------------------------------------------------------------------------------------------
# Fused transition (SwiGLU MLP), fp16 in -> fp16 out (or fp32 z + y with RESID).
#   x [ROWS, D] fp16 (LayerNorm output), w12 [2H, D] fp16 (fc1 rows then fc2 rows), w3 [D, H] fp16
#   one simdgroup = TM rows; threadgroup = NSG simdgroups (NSG * TM rows)
#   A = x tile in a left-input cooperative tensor (TM x D, loaded once)
#   for each hidden block of HB units:
#       Sa = A W1b^T, Sb = A W2b^T          (op1: TM x HB x D, fp32 accumulate, W from device or threadgroup)
#       Sh = half(silu(Sa) * Sb * hscale)   (elementwise on same-layout cooperative tensors)
#       O += Sh W3b^T                       (op2: TM x D x HB; Sh -> left input via get_left_input_cooperative_tensor)
#   y = half(O) (or RESID: z + float(half(O)) * oscale in fp32 = exactly radd(z, Scaled(y)))
# TGW = 1: the W1/W2/W3 blocks of a hidden block are staged in threadgroup memory by the whole threadgroup (two
# barriers per block) instead of each simdgroup reading them from device memory through MPP.
# ---------------------------------------------------------------------------------------------------------------
_TRANS_SRC = """
    using namespace mpp::tensor_ops;
    const int N_ROWS = (int)nrows[0];                 // runtime (no recompile per input shape)
    const uint sg = simdgroup_index_in_threadgroup;
    const int ntiles = (N_ROWS + TM - 1) / TM;
    int tile = (int)(threadgroup_position_in_grid.x * NSG + sg);
    if (TGW == 0 && tile >= ntiles) return;          // (TGW: every simdgroup must reach the barriers)
    tile = min(tile, ntiles - 1);
    const int r0 = min(tile * TM, N_ROWS - TM);       // last tile moved back inside

    threadgroup half wbuf[TGW ? 3 * HB * D : 8];       // TGW: W1 / W2 / W3 blocks of one hidden block
    const float hscale = scales[0];
    const float oscale = scales[1];
    dtensor<half> tX((device half*)x, ext2_t(D, N_ROWS), str2_t{1, D});
    dtensor<half> tW12((device half*)w12, ext2_t(D, 2 * H), str2_t{1, D});   // [2H][D] = B^T of GEMM 1
    dtensor<half> tW3((device half*)w3, ext2_t(H, D), str2_t{1, H});          // [D][H]  = B^T of GEMM 2

    constexpr auto d1 = matmul2d_descriptor(TM, HB, D, false, true, true, matmul2d_descriptor::mode::multiply);
    constexpr auto d2 = matmul2d_descriptor(TM, D, HB, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<d1, metal::execution_simdgroup> op1;
    matmul2d<d2, metal::execution_simdgroup> op2;

    auto xs = tX.template slice<D, TM>(0, r0);
    auto Areg = op1.template get_left_input_cooperative_tensor<half, half, float>();
    Areg.load(xs);
    using AregT = metal::remove_addrspace_t<decltype(Areg)>;
    auto w1_0 = tW12.template slice<D, HB>(0, 0);
    auto w3_0 = tW3.template slice<HB, D>(0, 0);
    auto Ain0 = op2.template get_left_input_cooperative_tensor<half, half, float>();
    using AinT = metal::remove_addrspace_t<decltype(Ain0)>;
    auto O = op2.template get_destination_cooperative_tensor<AinT, decltype(w3_0), float>();
    UNR for (ushort i = 0; i < O.get_capacity(); i++) O[i] = 0.0f;

    auto block = [&](thread auto& w1b, thread auto& w2b, thread auto& w3b) {
        auto Sa = op1.template get_destination_cooperative_tensor<AregT, decltype(w1_0), float>();
        auto Sb = op1.template get_destination_cooperative_tensor<AregT, decltype(w1_0), float>();
        op1.run(Areg, w1b, Sa);
        op1.run(Areg, w2b, Sb);
        auto Sh = op1.template get_destination_cooperative_tensor<AregT, decltype(w1_0), half>();
        UNR for (ushort i = 0; i < Sh.get_capacity(); i++) {
            const float a = Sa[i];
            Sh[i] = half(a * (1.0f / (1.0f + metal::exp(-a))) * Sb[i] * hscale);
        }
        auto Ain = op2.template get_left_input_cooperative_tensor<half, half, float>(Sh);
        op2.run(Ain, w3b, O);
    };

    if constexpr (TGW == 0) {
        for (int hb = 0; hb < H; hb += HB) {
            auto w1b = tW12.template slice<D, HB>(0, hb);
            auto w2b = tW12.template slice<D, HB>(0, H + hb);
            auto w3b = tW3.template slice<HB, D>(hb, 0);
            block(w1b, w2b, w3b);
        }
    } else {
        const uint tid = thread_index_in_threadgroup;
        constexpr uint NT = NSG * 32;
        for (int hb = 0; hb < H; hb += HB) {
            threadgroup_barrier(mem_flags::mem_threadgroup);   // previous block's reads are done
            for (uint e = tid; e < (uint)(2 * HB * D / 8); e += NT) {
                const uint which = e / (HB * D / 8), off = (e % (HB * D / 8)) * 8;
                const device half* src = (const device half*)w12 + (size_t)(which * H + hb) * D + off;
                *(threadgroup vec<half, 8>*)(wbuf + which * HB * D + off) = *(const device vec<half, 8>*)src;
            }
            for (uint e = tid; e < (uint)(D * HB / 8); e += NT) {
                const uint d = e / (HB / 8), c8 = (e % (HB / 8)) * 8;
                *(threadgroup vec<half, 8>*)(wbuf + 2 * HB * D + d * HB + c8) =
                    *(const device vec<half, 8>*)((const device half*)w3 + (size_t)d * H + hb + c8);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            ttensor<half> t1(wbuf, ext2_t(D, HB), str2_t{1, D});
            ttensor<half> t2(wbuf + HB * D, ext2_t(D, HB), str2_t{1, D});
            ttensor<half> t3(wbuf + 2 * HB * D, ext2_t(HB, D), str2_t{1, HB});
            block(t1, t2, t3);
        }
    }

    auto os = dtensor<OT>(out, ext2_t(D, N_ROWS), str2_t{1, D}).template slice<D, TM>(0, r0);
    if constexpr (RESID != 0) {
        auto Z = op2.template get_destination_cooperative_tensor<AinT, decltype(w3_0), float>();
        Z.load(dtensor<float>((device float*)resid, ext2_t(D, N_ROWS), str2_t{1, D}).template slice<D, TM>(0, r0));
        UNR for (ushort i = 0; i < Z.get_capacity(); i++) Z[i] = Z[i] + float(half(O[i])) * oscale;
        Z.store(os);
    } else {
        auto Oh = op2.template get_destination_cooperative_tensor<AinT, decltype(w3_0), OT>();
        UNR for (ushort i = 0; i < Oh.get_capacity(); i++) Oh[i] = OT(O[i]);
        Oh.store(os);
    }
"""

# (TM rows / simdgroup, HB hidden block, NSG simdgroups / threadgroup, TGW weights staged in threadgroup memory)
TRANS_CFGS = [
    dict(tm=16, hb=32, nsg=4, tgw=0),
    dict(tm=16, hb=64, nsg=4, tgw=0),
    dict(tm=16, hb=32, nsg=8, tgw=0),
    dict(tm=32, hb=32, nsg=4, tgw=0),
    dict(tm=16, hb=32, nsg=4, tgw=1),
    dict(tm=16, hb=32, nsg=8, tgw=1),
    dict(tm=32, hb=32, nsg=4, tgw=1),
]
TRANS_DEFAULT = TRANS_CFGS[0]


def transition_nax(xn: mx.array, w12: mx.array, w3: mx.array, *, hidden_scale: float = 1.0,
                   out_scale: float = 1.0, resid: mx.array | None = None, tm: int = 16, hb: int = 32, nsg: int = 4,
                   tgw: int = 0) -> mx.array:
    """Fused SwiGLU transition on NAX. xn: [..., D] fp16 LayerNorm output; w12 [2H, D] fp16 (fc1; fc2);
    w3 [D, H] fp16. Returns fp16 y = fc3(silu(fc1 xn) * fc2 xn * hidden_scale) (true value y * out_scale is the
    caller's business), or with resid (fp32, xn's shape) the fp32 resid + float(y) * out_scale."""
    D = xn.shape[-1]
    H = w3.shape[1]
    n_rows = xn.size // D
    assert xn.dtype == mx.float16 and w12.dtype == mx.float16 and w3.dtype == mx.float16
    assert w12.shape == (2 * H, D) and w3.shape == (D, H)
    assert D % 16 == 0 and H % hb == 0 and hb % 16 == 0 and tm % 16 == 0
    assert not tgw or 3 * hb * D * 2 <= 32768, "TGW weight block exceeds 32 KB of threadgroup memory"
    if n_rows < tm:
        raise ValueError(f"transition_nax: {n_rows} rows < tile {tm}")
    key = ("trans",)
    if key not in _kernels:
        _kernels[key] = _K.metal_kernel(
            name="boltz_nax_transition", input_names=["x", "w12", "w3", "scales", "resid", "nrows"], output_names=["out"],
            source=_TRANS_SRC, header=HEADER)
    ntiles = (n_rows + tm - 1) // tm
    ntg = (ntiles + nsg - 1) // nsg
    has_resid = resid is not None
    if has_resid:
        assert resid.dtype == mx.float32 and resid.size == xn.size
    out_dtype = mx.float32 if has_resid else mx.float16
    (y,) = _kernels[key](
        inputs=[xn, w12, w3, mx.array([hidden_scale, out_scale], mx.float32),
                resid if has_resid else mx.zeros((8,), mx.float32), mx.array([n_rows], mx.int32)],
        template=[("D", D), ("H", H), ("TM", tm), ("HB", hb), ("NSG", nsg), ("TGW", int(tgw)),
                  ("RESID", int(has_resid)), ("OT", out_dtype)],
        grid=(ntg * nsg * 32, 1, 1), threadgroup=(nsg * 32, 1, 1),
        output_shapes=[xn.shape], output_dtypes=[out_dtype])
    return y


# ---------------------------------------------------------------------------------------------------------------
# Fused tri-mult input projection (f16 path, B = 1), channel-major output.
#   x [NM, D] fp16 (norm_in output), w [4n, D] fp16 rows [p_a | p_b | g_a | g_b] (n channels each),
#   mask [NM] fp32, scales[0] = 2^-K.  ab[c, pos] = (W_p x^T)[c, pos] * sigmoid((W_g x^T)[c, pos]) * mask[pos] * 2^-K
#   for c < 2n, written fp16 to ab [2n, NM] (the contraction reads a = ab[:n], b = ab[n:] as [n, N, M]).
#   one simdgroup = TP positions; op (TC channels x TP positions x D): left = W block (device), right = x tile^T
#   (XREG = 1: loaded once into a right-input cooperative tensor; 0: MPP reads it from device every block).
#   GOUT = 1: also g_out[pos, :] = x Wgo^T (fp16, [NM, D]) with op (TP x 32 x D) blocks.
# ---------------------------------------------------------------------------------------------------------------
_TRIMUL_PROJ_SRC = """
    using namespace mpp::tensor_ops;
    const int NM = (int)nm[0];                        // runtime (no recompile per input shape)
    const uint sg = simdgroup_index_in_threadgroup;
    const int ntiles = (NM + TP - 1) / TP;
    const int tile = (int)(threadgroup_position_in_grid.x * NSG + sg);
    if (tile >= ntiles) return;
    const int p0 = min(tile * TP, NM - TP);
    const float sc = scales[0];

    dtensor<half> tX((device half*)x, ext2_t(D, NM), str2_t{1, D});          // [NM][D]: x^T is B with transpose
    dtensor<half> tW((device half*)w, ext2_t(D, 4 * NC), str2_t{1, D});      // [4n][D]: A (row-major M x K)
    dtensor<half> tAB(ab, ext2_t(NM, 2 * NC), str2_t{1, NM});                // [2n][NM]
    auto xs = tX.template slice<D, TP>(0, p0);

    constexpr auto d1 = matmul2d_descriptor(TC, TP, D, false, true, true, matmul2d_descriptor::mode::multiply);
    matmul2d<d1, metal::execution_simdgroup> op;
    auto w0 = tW.template slice<D, TC>(0, 0);

    auto run_blocks = [&](thread auto& xop) {
        for (int c0 = 0; c0 < 2 * NC; c0 += TC) {
            auto wp = tW.template slice<D, TC>(0, c0);
            auto wg = tW.template slice<D, TC>(0, 2 * NC + c0);
            auto P = op.template get_destination_cooperative_tensor<decltype(w0), metal::remove_addrspace_t<decltype(xop)>, float>();
            auto G = op.template get_destination_cooperative_tensor<decltype(w0), metal::remove_addrspace_t<decltype(xop)>, float>();
            op.run(wp, xop, P);
            op.run(wg, xop, G);
            auto Oh = op.template get_destination_cooperative_tensor<decltype(w0), metal::remove_addrspace_t<decltype(xop)>, half>();
            UNR for (ushort i = 0; i < Oh.get_capacity(); i++) {
                const auto ix = P.get_multidimensional_index(i);            // (column = position, row = channel)
                const float m = mask[p0 + ix[0]];
                Oh[i] = half(P[i] * (1.0f / (1.0f + metal::exp(-G[i]))) * m * sc);
            }
            auto dst = tAB.template slice<TP, TC>(p0, c0);
            Oh.store(dst);
        }
    };
    if constexpr (XREG != 0) {
        auto Xr = op.template get_right_input_cooperative_tensor<half, half, float>();
        Xr.load(xs);
        run_blocks(Xr);
    } else {
        run_blocks(xs);
    }

    if constexpr (GOUT != 0) {
        constexpr auto d2 = matmul2d_descriptor(TP, 32, D, false, true, true, matmul2d_descriptor::mode::multiply);
        matmul2d<d2, metal::execution_simdgroup> op2;
        dtensor<half> tWo((device half*)wgo, ext2_t(D, D), str2_t{1, D});    // [D_out][D_in] = B^T
        dtensor<half> tG(gout, ext2_t(D, NM), str2_t{1, D});
        auto Xl = op2.template get_left_input_cooperative_tensor<half, half, float>();
        Xl.load(xs);
        auto wo0 = tWo.template slice<D, 32>(0, 0);
        for (int n0 = 0; n0 < D; n0 += 32) {
            auto wo = tWo.template slice<D, 32>(0, n0);
            auto Gf = op2.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(Xl)>, decltype(wo0), float>();
            op2.run(Xl, wo, Gf);
            auto Gh = op2.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(Xl)>, decltype(wo0), half>();
            UNR for (ushort i = 0; i < Gh.get_capacity(); i++) Gh[i] = half(Gf[i]);
            auto dst = tG.template slice<32, TP>(n0, p0);
            Gh.store(dst);
        }
    }
"""

TRIMUL_PROJ_CFGS = [
    dict(tp=32, tc=32, nsg=4, xreg=1),
    dict(tp=32, tc=16, nsg=4, xreg=1),
    dict(tp=32, tc=32, nsg=4, xreg=0),
    dict(tp=64, tc=16, nsg=4, xreg=0),
    dict(tp=32, tc=32, nsg=8, xreg=1),
]


def trimul_proj_nax(xn: mx.array, w: mx.array, mask: mx.array, scale: float, w_gout: mx.array | None = None, *,
                    tp: int = 32, tc: int = 32, nsg: int = 4, xreg: int = 1):
    """xn [NM, D] (or [1, N, M, D]) fp16; w [4n, D] fp16 rows [p_a | p_b | g_a | g_b]; mask [NM] (any float dtype);
    scale = 2^-K. Returns ab [2n, NM] fp16 (channel-major), and g_out [NM, D] fp16 if w_gout ([D, D] fp16) is given."""
    D = xn.shape[-1]
    NM = xn.size // D
    NC = w.shape[0] // 4
    assert xn.dtype == mx.float16 and w.dtype == mx.float16 and w.shape == (4 * NC, D)
    assert (2 * NC) % tc == 0 and D % 16 == 0 and tp % 16 == 0 and tc % 16 == 0
    if NM < tp:
        raise ValueError(f"trimul_proj_nax: {NM} positions < tile {tp}")
    gout = w_gout is not None
    if gout:
        assert w_gout.shape == (D, D) and w_gout.dtype == mx.float16 and D % 32 == 0
    key = ("trimul_proj",)
    if key not in _kernels:
        _kernels[key] = _K.metal_kernel(
            name="boltz_nax_trimul_proj", input_names=["x", "w", "mask", "scales", "wgo", "nm"],
            output_names=["ab", "gout"], source=_TRIMUL_PROJ_SRC, header=HEADER)
    ntiles = (NM + tp - 1) // tp
    ntg = (ntiles + nsg - 1) // nsg
    ab, g = _kernels[key](
        inputs=[xn, w, mask.reshape(-1).astype(mx.float32), mx.array([scale], mx.float32),
                w_gout if gout else mx.zeros((8,), mx.float16), mx.array([NM], mx.int32)],
        template=[("D", D), ("NC", NC), ("TP", tp), ("TC", tc), ("NSG", nsg), ("XREG", int(xreg)),
                  ("GOUT", int(gout))],
        grid=(ntg * nsg * 32, 1, 1), threadgroup=(nsg * 32, 1, 1),
        output_shapes=[(2 * NC, NM), (NM, D) if gout else (1,)], output_dtypes=[mx.float16, mx.float16])
    return (ab, g) if gout else ab


# ---------------------------------------------------------------------------------------------------------------
# Tri-mult contraction, per channel d (fp16 operands, fp32 accumulate, fp16 out):
#   outgoing: o[d, i, j] = sum_k a[d, i, k] b[d, j, k]   (A = a[d], B^T = b[d])
#   incoming: o[d, i, j] = sum_k a[d, k, i] b[d, k, j]   (A^T = a[d], B = b[d])
#   one simdgroup = TM x TN output tile; threadgroup = SM x SN simdgroups (share A / B rows through L1);
#   K in chunks of BK through MPP device tensors (dynamic tail chunk masked by MPP).
#   grid: x = tiles of the N x N output (SM*TM x SN*TN per threadgroup), y = channel.
# ---------------------------------------------------------------------------------------------------------------
_TRIMUL_CONTRACT_SRC = """
    using namespace mpp::tensor_ops;
    const uint sg = simdgroup_index_in_threadgroup;
    const int d = (int)threadgroup_position_in_grid.y;
    const int tgx = (int)threadgroup_position_in_grid.x;
    constexpr int TGM = SM * TM, TGN = SN * TN;
    const int ntm = (NN + TGM - 1) / TGM;
    const int tm_i = tgx % ntm, tn_i = tgx / ntm;
    const int i0 = min(tm_i * TGM + (int)(sg % SM) * TM, NN - TM);
    const int j0 = min(tn_i * TGN + (int)(sg / SM) * TN, NN - TN);

    const size_t plane = (size_t)NN * NN;
    device half* ad = (device half*)a + (size_t)d * plane;
    device half* bd = (device half*)b + (size_t)d * plane;
    // outgoing: A [i][k] (row-major M x K), B^T [j][k]; incoming: A^T stored [k][i], B stored [k][j]
    dtensor<half> tA(ad, ext2_t(NN, NN), str2_t{1, NN});
    dtensor<half> tB(bd, ext2_t(NN, NN), str2_t{1, NN});
    constexpr bool TL = (OUTGOING == 0), TR = (OUTGOING != 0);
    constexpr auto desc = matmul2d_descriptor(TM, TN, BK, TL, TR, true, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, metal::execution_simdgroup> op;

    auto a_chunk = [&](int k) {
        if constexpr (OUTGOING != 0) return tA.template slice<BK, TM>(k, i0); else return tA.template slice<TM, BK>(i0, k);
    };
    auto b_chunk = [&](int k) {
        if constexpr (OUTGOING != 0) return tB.template slice<BK, TN>(k, j0); else return tB.template slice<TN, BK>(j0, k);
    };
    auto a0 = a_chunk(0);
    auto b0 = b_chunk(0);
    auto C = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
    UNR for (ushort i = 0; i < C.get_capacity(); i++) C[i] = 0.0f;
    constexpr int KFULL = (NN / BK) * BK;
    for (int k = 0; k < KFULL; k += BK) {
        auto av = a_chunk(k);
        auto bv = b_chunk(k);
        op.run(av, bv, C);
    }
    if constexpr (KFULL < NN) {
        constexpr int KT = NN - KFULL;
        // tail: views that end at the matrix edge (MPP masks the rest of the BK chunk)
        if constexpr (OUTGOING != 0) {
            dtensor<half> va(ad + KFULL, ext2_t(KT, NN), str2_t{1, NN});
            dtensor<half> vb(bd + KFULL, ext2_t(KT, NN), str2_t{1, NN});
            auto av = va.template slice<metal::dynamic_extent, TM>(0, i0);
            auto bv = vb.template slice<metal::dynamic_extent, TN>(0, j0);
            op.run(av, bv, C);
        } else {
            dtensor<half> va(ad + (size_t)KFULL * NN, ext2_t(NN, KT), str2_t{1, NN});
            dtensor<half> vb(bd + (size_t)KFULL * NN, ext2_t(NN, KT), str2_t{1, NN});
            auto av = va.template slice<TM, metal::dynamic_extent>(i0, 0);
            auto bv = vb.template slice<TN, metal::dynamic_extent>(j0, 0);
            op.run(av, bv, C);
        }
    }
    auto Ch = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), half>();
    UNR for (ushort i = 0; i < Ch.get_capacity(); i++) Ch[i] = half(C[i]);
    dtensor<half> tO(o + (size_t)d * plane, ext2_t(NN, NN), str2_t{1, NN});
    auto dst = tO.template slice<TN, TM>(j0, i0);
    Ch.store(dst);
"""

TRIMUL_CFGS = [
    dict(tm=32, tn=32, bk=32, sm=2, sn=2),
    dict(tm=32, tn=32, bk=64, sm=2, sn=2),
    dict(tm=64, tn=32, bk=32, sm=2, sn=2),
    dict(tm=32, tn=64, bk=32, sm=2, sn=1),
    dict(tm=16, tn=32, bk=64, sm=4, sn=2),
]


def trimul_contract_nax(a: mx.array, b: mx.array, outgoing: bool = True, *, tm: int = 32, tn: int = 32, bk: int = 32,
                        sm: int = 2, sn: int = 2) -> mx.array:
    """a, b: [n, N, N] fp16 (channel-major; [1, n, N, N] accepted). outgoing: o = a b^T per channel, incoming:
    o = a^T b. Returns o [n, N, N] fp16."""
    if a.ndim == 4:
        a, b = a.reshape(a.shape[1:]), b.reshape(b.shape[1:])
    n, N, N2 = a.shape
    assert N == N2 and b.shape == a.shape and a.dtype == mx.float16 and b.dtype == mx.float16
    if N < max(tm, tn):
        raise ValueError(f"trimul_contract_nax: N={N} < tile {max(tm, tn)}")
    key = ("trimul_contract",)
    if key not in _kernels:
        _kernels[key] = _K.metal_kernel(
            name="boltz_nax_trimul_contract", input_names=["a", "b"], output_names=["o"],
            source=_TRIMUL_CONTRACT_SRC, header=HEADER)
    tgm, tgn = sm * tm, sn * tn
    ntg = ((N + tgm - 1) // tgm) * ((N + tgn - 1) // tgn)
    (o,) = _kernels[key](
        inputs=[a, b],
        template=[("NN", N), ("TM", tm), ("TN", tn), ("BK", bk), ("SM", sm), ("SN", sn), ("OUTGOING", int(outgoing))],
        grid=(ntg * sm * sn * 32, n, 1), threadgroup=(sm * sn * 32, 1, 1),
        output_shapes=[(n, N, N)], output_dtypes=[mx.float16])
    return o


# ---------------------------------------------------------------------------------------------------------------
# MLX-op references with the same rounding points (fp16 operands, fp32 accumulation / elementwise, fp16 storage of
# the hidden / gated operands and outputs). Test oracles and CPU stand-ins for the harness smoke run.
# ---------------------------------------------------------------------------------------------------------------
def transition_ref(xn, w12, w3, *, hidden_scale=1.0, out_scale=1.0, resid=None, **_):
    H = w3.shape[1]
    f = mx.float32
    h = xn.astype(f) @ w12.astype(f).T
    a, b = h[..., :H], h[..., H:]
    s = (a * mx.sigmoid(a) * b * hidden_scale).astype(mx.float16)
    y = (s.astype(f) @ w3.astype(f).T).astype(mx.float16)
    return y if resid is None else resid + y.astype(f) * out_scale


def trimul_proj_ref(xn, w, mask, scale, w_gout=None, **_):
    D = xn.shape[-1]
    f = mx.float32
    x = xn.reshape(-1, D).astype(f)
    NC = w.shape[0] // 4
    h = w.astype(f) @ x.T  # [4n, NM]
    ab = (h[:2 * NC] * mx.sigmoid(h[2 * NC:]) * mask.reshape(1, -1).astype(f) * scale).astype(mx.float16)
    if w_gout is None:
        return ab
    return ab, (x @ w_gout.astype(f).T).astype(mx.float16)


def trimul_contract_ref(a, b, outgoing=True, **_):
    if a.ndim == 4:
        a, b = a.reshape(a.shape[1:]), b.reshape(b.shape[1:])
    f = mx.float32
    a, b = a.astype(f), b.astype(f)
    o = a @ b.transpose(0, 2, 1) if outgoing else a.transpose(0, 2, 1) @ b
    return o.astype(mx.float16)


# ---------------------------------------------------------------------------------------------------------------
# Diffusion-transformer attention on NAX (task 4 candidate): head dim 48 (= 3 x 16), 16 heads, additive pair bias
# [B, H, Q, K] (key mask folded in), q bias added in fp32 before the fp16 rounding of Q, sigmoid(g) gate.
# Derived from the validated register flash kernel kernels_nax._TRI_NAX_SRC (relaxed 16x32x16 descriptor, MLX
# fragment layout; see tests/test_nax_layout.py) by source patches, so the shipped tri-att kernel is untouched:
#   * C % 16 == 0: an odd number of 16-wide head-dim fragments gets a zero dummy V half / dummy O accumulator,
#   * HAS_QB: Q = half(float(h_q) + qbias).
# h: [B*N, F] fp16 (q | k | v | g in the first 4HC columns), rows = batch entries (I_DIM = 1).
# ---------------------------------------------------------------------------------------------------------------
def _attn_src(unpadded: bool = False, brep: bool = False) -> str:
    """unpadded: the bias is read as [B, H, N, N] (row stride N, scalar bounds-checked loads) and the key-padding
    mask is generated in the kernel, so nothing has to be padded per call (the diffusion step runs under mx.compile;
    the bias is the step-invariant DiffusionTransformer.prepare output). brep (unpadded only): the bias has B / BREP
    batch entries, shared by BREP consecutive samples (multi-sample diffusion: identical pair bias and key mask)."""
    from boltz_mlx import kernels_nax as _KN

    src = _KN._TRI_NAX_SRC
    qb = "(HAS_QB ? qbias[hh * C + db * 16 + fn + j] : 0.0f)"
    pairs = []
    if unpadded:
        bb = "(b / BREP)" if brep else "b"
        pairs = [
            ("auto brow = bias + (size_t)(b * NH + hh) * J_DIM * KP;",
             f"auto brow = bias + (size_t)({bb} * NH + hh) * J_DIM * J_DIM;"),
            ("float4 mk = ld4(mrow + kc);",
             "float4 mk = float4(kc < J_DIM ? 0.0f : -1e30f, kc + 1 < J_DIM ? 0.0f : -1e30f, "
             "kc + 2 < J_DIM ? 0.0f : -1e30f, kc + 3 < J_DIM ? 0.0f : -1e30f);"),
            ("vec<BT, 4> ba = ld4(brow + (size_t)qra * KP + kc);",
             "vec<BT, 4> ba = vec<BT, 4>(bld4(brow + (size_t)qra * J_DIM, kc, (uint)J_DIM));"),
            ("vec<BT, 4> bb = ld4(brow + (size_t)qrb * KP + kc);",
             "vec<BT, 4> bb = vec<BT, 4>(bld4(brow + (size_t)qrb * J_DIM, kc, (uint)J_DIM));"),
        ]
    for old, new in pairs + [
        ("Qf[db][j] = (half)va[j]; Qf[db][4 + j] = (half)vb[j]; }",
         f"Qf[db][j] = (half)(float(va[j]) + {qb}); Qf[db][4 + j] = (half)(float(vb[j]) + {qb}); }}"),
        ("Qf[db][j] = (half)pa[j]; Qf[db][4 + j] = (half)pb[j]; }",
         f"Qf[db][j] = (half)(float(pa[j]) + {qb}); Qf[db][4 + j] = (half)(float(pb[j]) + {qb}); }}"),
        ("metal::vec<float, 8> O[TD];\n    UNR for (int d = 0; d < TD; d++) O[d] = 0.0f;",
         "metal::vec<float, 8> O[TD + (TD & 1)];   // + a dummy accumulator for an odd fragment count\n"
         "    UNR for (int d = 0; d < TD + (TD & 1); d++) O[d] = 0.0f;"),
        ("vec<HT, 4> a1 = ld4(pa + 16), b1 = ld4(pb + 16);",
         "vec<HT, 4> a1 = (dp + 1 < TD) ? ld4(pa + 16) : vec<HT, 4>(0), b1 = (dp + 1 < TD) ? ld4(pb + 16) : vec<HT, 4>(0);"),
        ("V1[j] = (half)pa[16 + j]; V1[4 + j] = (half)pb[16 + j];",
         "V1[j] = (dp + 1 < TD) ? (half)pa[16 + j] : (half)0; V1[4 + j] = (dp + 1 < TD) ? (half)pb[16 + j] : (half)0;"),
    ]:
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    return src


def attention_nax(h: mx.array, bias: mx.array, *, B: int, N: int, heads: int, c: int, qbias: mx.array | None = None,
                  wq: int = 4, nkf: int = 2, order: int = 1, prepared=None, out_dtype=mx.float16,
                  brep: int = 1) -> mx.array:
    """sigmoid(g) * softmax((q + qbias) k^T / sqrt(c) + bias) v on NAX. h [>= B*N, 4*heads*c] fp16 (q|k|v|g),
    bias [B, heads, N, N] additive (mask folded in; use a finite large negative, the kernel adds it in fp32),
    qbias [heads*c] fp32 or None. prepared: attention_nax_bias(bias) (padded copy; None = read the bias in place,
    the integrated mode). brep > 1 (unpadded mode): bias is [B / brep, heads, N, N], entry b // brep serves sample b
    (samples sharing the pair bias and key mask; the bias is read once per brep samples). Returns [B*N, heads*c]."""
    from boltz_mlx import kernels_nax as _KN

    assert c % 16 == 0 and nkf % 2 == 0 and h.dtype == mx.float16
    assert brep == 1 or (prepared is None and B % brep == 0 and bias.shape[0] * brep == B), (brep, B, bias.shape)
    HC = heads * c
    F = h.shape[-1]
    h = h[: B * N].reshape(B, 1, N, F)
    kpm = max(64, 16 * nkf)
    unpadded = prepared is None
    if unpadded:  # bias read in place (fp32, [B, H, N, N]); key padding masked in the kernel
        bias_p, KP = (bias if bias.dtype == mx.float32 else bias.astype(mx.float32)), ((N + kpm - 1) // kpm) * kpm
        mb = mx.zeros((8,), mx.float32)
    else:
        bias_p, KP = prepared
        mb = mx.pad(mx.zeros((1, N), mx.float32), [(0, 0), (0, KP - N)], constant_values=-1e30) if KP != N else \
            mx.zeros((1, N), mx.float32)
    shared = brep != 1  # brep == 1: the shipped kernel source, untouched
    from boltz_mlx import kernels as _K

    dyn = _K.nax("DYNN")  # R8-coldN: J / KP at run time (same arithmetic, no per-N compile)
    key = ("attn", heads, c, unpadded) + (("brep",) if shared else ()) + (("dyn",) if dyn else ())
    if key not in _kernels:
        _kernels[key] = _K.metal_kernel(
            name=f"boltz_nax_attn_h{heads}_c{c}" + ("_u" if unpadded else "") + ("_br" if shared else "")
            + ("_dyn" if dyn else ""),
            input_names=["h", "bias", "maskb", "scale_l2e", "qbias"] + (["dims"] if dyn else []),
            output_names=["out"],
            source=(_DYN_JK + _attn_src(unpadded, shared)) if dyn else _attn_src(unpadded, shared),
            header=_KN.NAX_HEADER + _BLD4)
    nqt = (N + 16 * wq - 1) // (16 * wq)
    tg = wq * 32
    grid = (nqt * tg, heads, B) if order == 0 else (B * tg, nqt, heads)
    (o,) = _kernels[key](
        inputs=[h, bias_p, mb, mx.array([c ** -0.5 * _KN.LOG2E], mx.float32),
                qbias.astype(mx.float32) if qbias is not None else mx.zeros((8,), mx.float32)]
        + ([mx.array([N, KP, 0, 0], mx.int32)] if dyn else []),
        template=[("I_DIM", 1)] + ([] if dyn else [("J_DIM", N)]) + [("NH", heads), ("C", c), ("HC", HC),
                                                                      ("ROW_STRIDE", F)] + ([] if dyn else [("KP", KP)]) + [
                  ("MASK_ROWS", 0), ("WQ", wq), ("HPT", 1), ("NKF", nkf), ("ORDER", order), ("RELAX", 1),
                  ("VEC", int(F % 4 == 0 and c % 4 == 0)), ("TRANS", 0), ("HT", mx.float16), ("BT", bias_p.dtype),
                  ("PT", mx.float32), ("OT", out_dtype), ("HAS_QB", int(qbias is not None))]
        + ([("BREP", brep)] if shared else []),
        grid=grid, threadgroup=(tg, 1, 1), output_shapes=[(B, 1, N, HC)], output_dtypes=[out_dtype])
    return o.reshape(B * N, HC)


_DYN_JK = "    const int J_DIM = (int)dims[0], KP = (int)dims[1];\n"


_BLD4 = r"""
// 4 consecutive fp32 bias values of keys kc..kc+3 (0 past the row end n), any address space / alignment
template <typename P> inline float4 bld4(P p, uint kc, uint n) {
    float4 r;
    UNR for (short j = 0; j < 4; j++) r[j] = (kc + j < n) ? float(p[kc + j]) : 0.0f;
    return r;
}
"""


def attention_nax_bias(bias: mx.array, kp_mult: int = 64, dtype=mx.float32):
    """[B, H, N, N] additive bias -> (bias padded to KP keys (pad value irrelevant: the key mask row removes the
    padding), KP). fp32 by default: the folded-in mask may be a large negative that fp16 cannot hold."""
    N = bias.shape[-1]
    KP = ((N + kp_mult - 1) // kp_mult) * kp_mult
    b = bias.astype(dtype)
    if KP != N:
        b = mx.pad(b, [(0, 0), (0, 0), (0, 0), (0, KP - N)])
    return b, KP


def attention_ref(h, bias, *, B, N, heads, c, qbias=None, **_):
    f = mx.float32
    HC = heads * c
    hh = h[: B * N].astype(f).reshape(B, N, 4, heads, c)
    q = hh[:, :, 0] + (qbias.astype(f).reshape(heads, c) if qbias is not None else 0.0)
    q = q.astype(mx.float16).astype(f)
    k, v, g = hh[:, :, 1], hh[:, :, 2], hh[:, :, 3]
    s = mx.einsum("bqhc,bkhc->bhqk", q, k) * c ** -0.5 + bias.astype(f)
    p = mx.softmax(s, axis=-1)
    o = mx.einsum("bhqk,bkhc->bqhc", p, v) * mx.sigmoid(g)
    return o.reshape(B * N, HC).astype(mx.float16)
