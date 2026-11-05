"""MSA-module kernels for --precision f16 (GPU only; mx.fast.metal_kernel, fp32 simdgroup MMA).

Numerics policy (PREREG "Numerics policy"): the OPM sum over MSA rows and the PWA weighted sum over tokens are
reductions over sequence / MSA rows, so both run here with fp32 operands and fp32 accumulation
(simdgroup_float8x8 multiply-accumulate; no fp16 / TF32 operands). fp16 appears only at rounding points the MLX f16
path already has: the proj_o GEMM input of the OPM (z / num rounded to fp16, exactly where `_div16` rounds it) and the
fp16 copy of the PWA proj_z weight (the same w16 copy the MLX f16 path multiplies with; the arithmetic here is fp32,
the logits are no longer rounded to fp16).

opm_outer    z16[(i - i_base) * N + j, c * 32 + d] = half( (sum_s a[s,i,c] * b[s,j,d]) / num[i,j] )
             with a = ab[s, i, A_OFF + c] * mask[s, i], b = ab[s, j, B_OFF + d] * mask[s, j]
             (ab = LN(m) @ [W_a; W_b]^T, one fp32 GEMM). Output is the pair-major fp16 operand of the proj_o GEMM:
             the N^2 x 1024 fp32 GEMM output, its transposed copy and the separate divide + cast pass of the MLX
             path never exist. GEMM view: rows (i, c), cols (j, d), K = S; one 32 x 32 block = one pair (i, j).
pwa_weights  w[h, i, j] = softmax_j( LN(z[i, j]) . Wz[h] + (1 - mask[i, j]) * -inf ), fp32, rows padded to NP
             (zeros) so that pwa_avg reads aligned K tiles. One threadgroup per row i.
pwa_avg      og[s, i, h*32 + d] = sigmoid(g[s, i, h*32 + d]) * sum_j w[h, i, j] * v[s, j, h*32 + d], fp32
             (v = LN(m) W_m^T, g = LN(m) W_g^T: the MLX fp32 GEMMs as before; the proj_o GEMM follows in MLX).
             GEMM view per head: rows i, cols (s, d), K = j.

Kernels follow the double-buffered pattern of the simdgroup-MMA tri-att kernel (kernels.py): the global loads of
K tile t+1 go to registers before the MMAs on tile t and to threadgroup memory after them, one barrier per tile.
Out-of-range rows / columns / K entries are zero-filled on load and skipped on store, so any N, S works.
"""

from __future__ import annotations

import mlx.core as mx

from boltz_mlx import kernels as _K  # metal_kernel(): the MPP guard (portability)

_kernels: dict = {}

HEADER = r"""
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#define UNR _Pragma("clang loop unroll(full)")
// MLX's sigmoid (unary.h), so the gate matches mx.sigmoid's formula
inline float sigmoid_mlx(float x) {
    const float y = 1.0f / (1.0f + metal::exp(metal::abs(x)));
    return (x < 0.0f) ? y : 1.0f - y;
}
"""

_TG_BYTES = 32768


# ---------------------------------------------------------------------------------------------------------------
# OPM outer sum.  Threadgroup tile: TI tokens i x TJ tokens j (BM = 32 TI rows (i, c), BN = 32 TJ cols (j, d)),
# NSG = (TI / SI) * (TJ / SJ) simdgroups, simdgroup (wm, wn) owns SI x SJ pairs = (32 SI) x (32 SJ) of the tile.
# K = S in tiles of BK.  SWZ > 0: threadgroups are swizzled in bands of 2^SWZ row tiles (L2 reuse of the B tiles).
# ---------------------------------------------------------------------------------------------------------------
_OPM_HEAD = """
    constexpr int C = 32;
    constexpr int BM = TI * C;
    constexpr int BN = TJ * C;
    constexpr int WN = TJ / SJ;
    constexpr int RM = SI * 4;
    constexpr int RN = SJ * 4;
    constexpr int LDA = AT ? (BK + PAD) : (BM + PAD);   // AT: A tile stored [(i, c)][k], else [k][(i, c)]
    constexpr int ASZ = AT ? (BM * LDA) : (BK * LDA);
    constexpr int LDB = BN + PAD;
    constexpr int NT = NSG * 32;
    constexpr int NA4 = BK * BM / 4;
    constexpr int NB4 = BK * BN / 4;
    constexpr int LA = (NA4 + NT - 1) / NT;
    constexpr int LB = (NB4 + NT - 1) / NT;

    const int S = dims[0], N = dims[1], IB = dims[2], NI = dims[3];   // rows i in [IB, IB + NI)
    const int tiles_i = (NI + TI - 1) / TI, tiles_j = (N + TJ - 1) / TJ;
    int ti = (int)threadgroup_position_in_grid.y, tj = (int)threadgroup_position_in_grid.x;
    if (SWZ > 0) {
        ti = ((int)threadgroup_position_in_grid.y << SWZ) + ((int)threadgroup_position_in_grid.x & ((1 << SWZ) - 1));
        tj = (int)threadgroup_position_in_grid.x >> SWZ;
    }
    if (ti >= tiles_i || tj >= tiles_j) return;                     // uniform per threadgroup
    const int i0 = IB + ti * TI, j0 = tj * TJ, iend = IB + NI;
    const int tid = (int)thread_index_in_threadgroup;
    const int sg = (int)simdgroup_index_in_threadgroup;
    const int lane = (int)thread_index_in_simdgroup;
    const int wm = sg / WN, wn = sg % WN;
    const size_t RS = (size_t)N * TOK;                               // row (s) stride of ab
"""

_OPM_DECL = """
    threadgroup float As[2 * ASZ];                                   // A tiles, double-buffered
    threadgroup float Bs[2 * BK * LDB];                              // [k][(j, d)]
"""

_OPM_LOOP = """
    float4 ra[LA], rb[LB];

    simdgroup_float8x8 acc[RM][RN];
    UNR for (int r = 0; r < RM; r++)
        UNR for (int c = 0; c < RN; c++) acc[r][c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const int ntk = (S + BK - 1) / BK;
    for (int it = 0; it <= ntk; it++) {
        if (it < ntk) {                                              // (A) global loads of K tile `it` (masked)
            const int s0 = it * BK;
            UNR for (int u = 0; u < LA; u++) {
                const int e = tid + u * NT;
                const int kk = e / (BM / 4), r4 = e % (BM / 4);
                const int s = s0 + kk, i = i0 + r4 / (C / 4);
                float4 v = float4(0.0f);
                if (e < NA4 && s < S && i < iend)
                    v = *(const device float4*)(ab + (size_t)s * RS + (size_t)i * TOK + A_OFF + (r4 % (C / 4)) * 4)
                        * mask[(size_t)s * N + i];
                ra[u] = v;
            }
            UNR for (int u = 0; u < LB; u++) {
                const int e = tid + u * NT;
                const int kk = e / (BN / 4), r4 = e % (BN / 4);
                const int s = s0 + kk, j = j0 + r4 / (C / 4);
                float4 v = float4(0.0f);
                if (e < NB4 && s < S && j < N)
                    v = *(const device float4*)(ab + (size_t)s * RS + (size_t)j * TOK + B_OFF + (r4 % (C / 4)) * 4)
                        * mask[(size_t)s * N + j];
                rb[u] = v;
            }
        }
        if (it > 0) {                                                // (B) MMAs on K tile it - 1
            const threadgroup float* as = As + ((it - 1) & 1) * ASZ;
            const threadgroup float* bs = Bs + ((it - 1) & 1) * (BK * LDB);
            UNR for (int kk = 0; kk < BK; kk += 8) {
                simdgroup_float8x8 af[RM], bf[RN];
                UNR for (int r = 0; r < RM; r++) {     // A[(i,c), s]
                    if (AT) simdgroup_load(af[r], as + (wm * (RM * 8) + r * 8) * LDA + kk, LDA);
                    else simdgroup_load(af[r], as + kk * LDA + wm * (RM * 8) + r * 8, LDA, ulong2(0, 0), true);
                }
                UNR for (int c = 0; c < RN; c++)
                    simdgroup_load(bf[c], bs + kk * LDB + wn * (RN * 8) + c * 8, LDB);
                UNR for (int r = 0; r < RM; r++)
                    UNR for (int c = 0; c < RN; c++)
                        simdgroup_multiply_accumulate(acc[r][c], af[r], bf[c], acc[r][c]);
            }
        }
        if (it < ntk) {                                              // (C) registers -> buffer it & 1
            threadgroup float* as = As + (it & 1) * ASZ;
            threadgroup float* bs = Bs + (it & 1) * (BK * LDB);
            UNR for (int u = 0; u < LA; u++) {
                const int e = tid + u * NT;
                if (e < NA4) {
                    const int kk = e / (BM / 4), m = (e % (BM / 4)) * 4;
                    if (AT) {
                        as[(m + 0) * LDA + kk] = ra[u].x;
                        as[(m + 1) * LDA + kk] = ra[u].y;
                        as[(m + 2) * LDA + kk] = ra[u].z;
                        as[(m + 3) * LDA + kk] = ra[u].w;
                    } else {
                        *(threadgroup float4*)(as + kk * LDA + m) = ra[u];
                    }
                }
            }
            UNR for (int u = 0; u < LB; u++) {
                const int e = tid + u * NT;
                if (e < NB4) *(threadgroup float4*)(bs + (e / (BN / 4)) * LDB + (e % (BN / 4)) * 4) = rb[u];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const int qid = lane / 4;                                        // lane owns (fm, fn..fn+1) of every fragment
    const int fm = (qid & 4) + ((lane / 2) % 4);
    const int fn = (qid & 2) * 2 + (lane % 2) * 2;
"""

_OPM_EPI_Z16 = """
    // epilogue: z / num (fp32) -> fp16, pair-major z16[(i - IB) * N + j][c * 32 + d]
    UNR for (int si = 0; si < SI; si++) {
        const int i = i0 + wm * SI + si;
        UNR for (int sj = 0; sj < SJ; sj++) {
            const int j = j0 + wn * SJ + sj;
            if (i < iend && j < N) {
                const float nm = num[(size_t)i * N + j];
                device half* zp = z16 + ((size_t)(i - IB) * N + j) * (C * C);
                UNR for (int r = 0; r < 4; r++)
                    UNR for (int c = 0; c < 4; c++) {
                        const float e0 = acc[si * 4 + r][sj * 4 + c].thread_elements()[0] / nm;
                        const float e1 = acc[si * 4 + r][sj * 4 + c].thread_elements()[1] / nm;
                        *(device half2*)(zp + (r * 8 + fm) * C + c * 8 + fn) = half2(half(e0), half(e1));
                    }
            }
        }
    }
"""

_OPM_SRC = _OPM_HEAD + _OPM_DECL + _OPM_LOOP + _OPM_EPI_Z16

# ---------------------------------------------------------------------------------------------------------------
# OPM outer sum + proj_o in one kernel (experimental candidate, flag BOLTZ_MLX_MSA_OPM_NAX): TI * TJ = 16 pairs per
# threadgroup. After the fp32 K loop (same code as above) the epilogue rounds z / num to fp16 one half of c at a time
# into threadgroup memory as rows p = (i - i0) * TJ + (j - j0), [16 pairs x 512], and simdgroups 0..3 run
#   out[16 x 32q..32q+32] += zh[16 x 512] @ W_o[32q.., 512h..512h+512]^T     (NAX matmul2d 16 x 32 x KC, fp16
# operands, fp32 accumulate: the proj_o GEMM, an ordinary fp16 GEMM by policy), then + bias in fp32, one fp16 rounding,
# rows scattered to out16[(i - IB) * N + j][o]. z16 never reaches device memory. The staging tiles of the K loop and
# the epilogue buffers alias one threadgroup allocation.
# ---------------------------------------------------------------------------------------------------------------
_OPM_DECL_FUSED = """
    constexpr int ZH = 16 * 16 * C;                                  // halves of the fp16 [16 pairs][16 c][32 d] tile
    constexpr int LOOP_F = 2 * ASZ + 2 * BK * LDB;                   // floats of the K-loop tiles
    constexpr int EPI_F = (ZH + 16 * CO) / 2;                        // floats of zh + the fp16 output staging
    constexpr int SMEM = LOOP_F > EPI_F ? LOOP_F : EPI_F;
    static_assert(TI * TJ == 16 && NSG >= 4 && CO == 128, "16 pairs per threadgroup, 4 x 32 output columns");
    threadgroup float smem[SMEM];
    threadgroup float* As = smem;
    threadgroup float* Bs = smem + 2 * ASZ;
"""

_OPM_EPI_NAX = """
    using namespace mpp::tensor_ops;
    threadgroup half* zh = (threadgroup half*)smem;                  // aliases the K-loop tiles (barrier passed)
    threadgroup half* ost = zh + ZH;
    constexpr auto dp = matmul2d_descriptor(16, 32, KC, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<dp, metal::execution_simdgroup> op;
    ttensor<half> tZ(zh, ext2_t(16 * C, 16), str2_t{1, 16 * C});    // [16 pairs][512]
    dtensor<half> tW((device half*)wo, ext2_t(C * C, CO), str2_t{1, C * C});   // W_o [CO][1024] = B^T
    auto z0 = tZ.template slice<KC, 16>(0, 0);
    auto w0 = tW.template slice<KC, 32>(0, 0);
    auto Acc = op.template get_destination_cooperative_tensor<decltype(z0), decltype(w0), float>();
    UNR for (ushort e = 0; e < Acc.get_capacity(); e++) Acc[e] = 0.0f;

    UNR for (int hh = 0; hh < 2; hh++) {
        // (1) z / num -> fp16 for c in [16 hh, 16 hh + 16): fragment rows r = 2 hh, 2 hh + 1 of each pair
        UNR for (int si = 0; si < SI; si++) {
            const int il = wm * SI + si, i = i0 + il;
            UNR for (int sj = 0; sj < SJ; sj++) {
                const int jl = wn * SJ + sj, j = j0 + jl;
                const float nm = (i < iend && j < N) ? num[(size_t)i * N + j] : 1.0f;   // invalid pairs hold zeros
                threadgroup half* zp = zh + (il * TJ + jl) * (16 * C);
                UNR for (int rr = 0; rr < 2; rr++)
                    UNR for (int c = 0; c < 4; c++) {
                        const int r = 2 * hh + rr;
                        const float e0 = acc[si * 4 + r][sj * 4 + c].thread_elements()[0] / nm;
                        const float e1 = acc[si * 4 + r][sj * 4 + c].thread_elements()[1] / nm;
                        *(threadgroup half2*)(zp + (rr * 8 + fm) * C + c * 8 + fn) = half2(half(e0), half(e1));
                    }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // (2) proj_o partial sums on NAX: simdgroup q < 4 owns output columns 32 q .. 32 q + 31
        if (sg < 4) {
            UNR for (int kc = 0; kc < 16 * C; kc += KC) {
                auto zs = tZ.template slice<KC, 16>(kc, 0);
                auto ws = tW.template slice<KC, 32>(hh * 16 * C + kc, 32 * sg);
                op.run(zs, ws, Acc);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);             // zh is rewritten by the next half
    }
    // (3) + bias (fp32), one fp16 rounding, staged [16][CO] in threadgroup memory
    if (sg < 4) {
        auto Oh = op.template get_destination_cooperative_tensor<decltype(z0), decltype(w0), half>();
        UNR for (ushort e = 0; e < Oh.get_capacity(); e++) {
            const auto ix = Acc.get_multidimensional_index(e);       // (column = o - 32 q, row = pair)
            Oh[e] = half(Acc[e] + bias[32 * sg + ix[0]]);
        }
        ttensor<half> tO(ost, ext2_t(CO, 16), str2_t{1, CO});
        Oh.store(tO.template slice<32, 16>(32 * sg, 0));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // (4) scatter the valid pair rows: out16[(i - IB) * N + j][o], 16-byte stores
    for (int e = tid; e < 16 * (CO / 8); e += NT) {
        const int p = e / (CO / 8), o8 = (e % (CO / 8)) * 8;
        const int i = i0 + p / TJ, j = j0 + p % TJ;
        if (i < iend && j < N)
            *(device vec<half, 8>*)(out16 + ((size_t)(i - IB) * N + j) * CO + o8) = *(threadgroup vec<half, 8>*)(ost + p * CO + o8);
    }
"""

_OPM_FUSED_SRC = _OPM_HEAD + _OPM_DECL_FUSED + _OPM_LOOP + _OPM_EPI_NAX

# (TI, TJ tokens per threadgroup; SI, SJ tokens per simdgroup; BK K tile; SWZ swizzle log2; AT: A tile stored
# transposed in threadgroup memory at staging time -> plain instead of transposed fragment loads)
OPM_CFGS = [
    dict(ti=2, tj=4, si=1, sj=2, bk=16, swz=0, at=0),   # 4 sg x (32 x 64), tile 64 x 128
    dict(ti=2, tj=2, si=1, sj=1, bk=16, swz=0, at=0),   # 4 sg x (32 x 32), tile 64 x 64 (MLX-like)
    dict(ti=2, tj=4, si=1, sj=1, bk=16, swz=0, at=0),   # 8 sg x (32 x 32), tile 64 x 128
    dict(ti=4, tj=4, si=1, sj=2, bk=8, swz=0, at=0),    # 8 sg x (32 x 64), tile 128 x 128
    dict(ti=4, tj=2, si=2, sj=1, bk=16, swz=0, at=0),   # 4 sg x (64 x 32), tile 128 x 64
    dict(ti=2, tj=4, si=1, sj=2, bk=16, swz=2, at=0),   # cfg 0, swizzled in bands of 4 row tiles
    dict(ti=2, tj=4, si=1, sj=2, bk=16, swz=0, at=1),   # cfg 0, A stored transposed
    dict(ti=4, tj=4, si=1, sj=2, bk=8, swz=0, at=1),    # cfg 3, A stored transposed
]
OPM_DEFAULT = OPM_CFGS[0]


def opm_tg_bytes(ti, tj, bk, pad=4, c=32, at=0) -> int:
    a = ti * c * (bk + pad) if at else bk * (ti * c + pad)
    return 4 * 2 * (a + bk * (tj * c + pad))


def opm_outer(ab: mx.array, mask: mx.array, num: mx.array, i_base: int = 0, n_i: int | None = None, *, ti: int = 2,
              tj: int = 4, si: int = 1, sj: int = 2, bk: int = 16, swz: int = 0, at: int = 0, pad: int = 4,
              c: int = 32, a_off: int = 0, b_off: int | None = None) -> mx.array:
    """ab [1, S, N, T] fp32 (a = ab[..., a_off:a_off+c], b = ab[..., b_off:b_off+c], unmasked); mask [S, N] fp32;
    num [N, N] fp32 (max(mask^T mask, 1)). Rows i in [i_base, i_base + n_i). Returns z16 [n_i * N, c * c] fp16,
    z16[(i - i_base) * N + j, c' * c + d] = half(sum_s a[s,i,c'] mask[s,i] b[s,j,d] mask[s,j] / num[i,j])."""
    _, S, N, tok = ab.shape
    b_off = c if b_off is None else b_off
    n_i = N - i_base if n_i is None else n_i
    assert c == 32 and ab.dtype == mx.float32 and mask.dtype == mx.float32 and num.dtype == mx.float32
    assert mask.shape == (S, N) and num.shape == (N, N) and tok % 4 == 0 and a_off % 4 == 0 and b_off % 4 == 0
    assert ti % si == 0 and tj % sj == 0 and bk % 8 == 0 and 0 <= i_base and i_base + n_i <= N and n_i > 0
    assert opm_tg_bytes(ti, tj, bk, pad, c, at) <= _TG_BYTES, "threadgroup tiles exceed 32 KB"
    nsg = (ti // si) * (tj // sj)
    key = ("opm",)
    if key not in _kernels:
        _kernels[key] = _K.metal_kernel(
            name="boltz_msa_opm_outer", input_names=["ab", "mask", "num", "dims"], output_names=["z16"],
            source=_OPM_SRC, header=HEADER)
    tiles_i, tiles_j = -(-n_i // ti), -(-N // tj)
    gx, gy = tiles_j << swz, -(-tiles_i // (1 << swz))
    (z16,) = _kernels[key](
        inputs=[ab, mask, num, mx.array([S, N, i_base, n_i], mx.int32)],
        template=[("TI", ti), ("TJ", tj), ("SI", si), ("SJ", sj), ("BK", bk), ("PAD", pad), ("NSG", nsg),
                  ("SWZ", swz), ("AT", int(at)), ("TOK", tok), ("A_OFF", a_off), ("B_OFF", b_off)],
        grid=(gx * nsg * 32, gy, 1), threadgroup=(nsg * 32, 1, 1),
        output_shapes=[(n_i * N, c * c)], output_dtypes=[mx.float16])
    return z16


def opm_outer_ref(ab: mx.array, mask: mx.array, num: mx.array, i_base: int = 0, n_i: int | None = None,
                  c: int = 32, a_off: int = 0, b_off: int | None = None, dtype=mx.float16) -> mx.array:
    """MLX-op reference of opm_outer (same rounding points: fp32 sum, fp32 divide, one rounding to `dtype`)."""
    _, S, N, _ = ab.shape
    b_off = c if b_off is None else b_off
    n_i = N - i_base if n_i is None else n_i
    a = ab[0, :, i_base:i_base + n_i, a_off:a_off + c] * mask[:, i_base:i_base + n_i, None]
    b = ab[0, :, :, b_off:b_off + c] * mask[:, :, None]
    z = mx.einsum("sic,sjd->ijcd", a, b) / num[i_base:i_base + n_i, :, None, None]
    return z.astype(dtype).reshape(n_i * N, c * c)


# fused outer sum + proj_o (NAX epilogue): TI * TJ must be 16
OPM_FUSED_CFGS = [
    dict(ti=4, tj=4, si=1, sj=2, bk=8, at=0, kc=128),    # 8 sg x (32 x 64)
    dict(ti=4, tj=4, si=1, sj=2, bk=8, at=1, kc=128),    # A stored transposed
    dict(ti=2, tj=8, si=1, sj=2, bk=8, at=0, kc=128),    # 8 sg, tile 64 x 256
    dict(ti=4, tj=4, si=1, sj=1, bk=8, at=0, kc=128),    # 16 sg x (32 x 32)
    dict(ti=4, tj=4, si=1, sj=2, bk=8, at=0, kc=256),    # cfg 0, K chunk 256 per NAX call
]
OPM_FUSED_DEFAULT = OPM_FUSED_CFGS[0]


def opm_fused_tg_bytes(ti, tj, bk, at=0, pad=4, c=32, co=128) -> int:
    return max(opm_tg_bytes(ti, tj, bk, pad, c, at), 2 * (16 * 16 * c + 16 * co))


def opm_outer_proj(ab: mx.array, mask: mx.array, num: mx.array, wo: mx.array, bias: mx.array, i_base: int = 0,
                   n_i: int | None = None, *, ti: int = 4, tj: int = 4, si: int = 1, sj: int = 2, bk: int = 8,
                   at: int = 0, kc: int = 128, swz: int = 0, pad: int = 4, c: int = 32, a_off: int = 0,
                   b_off: int | None = None) -> mx.array:
    """opm_outer followed by proj_o in one kernel: returns out16 [n_i * N, co] fp16 =
    half(fp32(z16 @ wo^T) + bias) with z16 = opm_outer(...) (never written to device memory). wo [co, c*c] fp16 (the
    w16 copy of proj_o.weight), bias [co] fp32. co = 128."""
    from boltz_mlx.kernels_nax2 import HEADER as NAX2_HEADER

    _, S, N, tok = ab.shape
    co = wo.shape[0]
    b_off = c if b_off is None else b_off
    n_i = N - i_base if n_i is None else n_i
    assert c == 32 and co == 128 and wo.shape == (co, c * c) and wo.dtype == mx.float16 and bias.shape == (co,)
    assert ab.dtype == mx.float32 and mask.dtype == mx.float32 and num.dtype == mx.float32
    assert mask.shape == (S, N) and num.shape == (N, N) and tok % 4 == 0 and a_off % 4 == 0 and b_off % 4 == 0
    assert ti * tj == 16 and ti % si == 0 and tj % sj == 0 and bk % 8 == 0 and (16 * c) % kc == 0
    assert 0 <= i_base and i_base + n_i <= N and n_i > 0
    nsg = (ti // si) * (tj // sj)
    assert nsg >= 4 and opm_fused_tg_bytes(ti, tj, bk, at, pad, c, co) <= _TG_BYTES
    key = ("opm_fused",)
    if key not in _kernels:
        _kernels[key] = _K.metal_kernel(
            name="boltz_msa_opm_outer_proj", input_names=["ab", "mask", "num", "wo", "bias", "dims"],
            output_names=["out16"], source=_OPM_FUSED_SRC, header=NAX2_HEADER)
    tiles_i, tiles_j = -(-n_i // ti), -(-N // tj)
    gx, gy = tiles_j << swz, -(-tiles_i // (1 << swz))
    (out,) = _kernels[key](
        inputs=[ab, mask, num, wo, bias.astype(mx.float32), mx.array([S, N, i_base, n_i], mx.int32)],
        template=[("TI", ti), ("TJ", tj), ("SI", si), ("SJ", sj), ("BK", bk), ("PAD", pad), ("NSG", nsg),
                  ("SWZ", swz), ("AT", int(at)), ("TOK", tok), ("A_OFF", a_off), ("B_OFF", b_off), ("KC", kc),
                  ("CO", co)],
        grid=(gx * nsg * 32, gy, 1), threadgroup=(nsg * 32, 1, 1),
        output_shapes=[(n_i * N, co)], output_dtypes=[mx.float16])
    return out


def opm_outer_proj_ref(ab, mask, num, wo, bias, i_base=0, n_i=None):
    """Reference of opm_outer_proj: fp16 z16, fp32-accumulated GEMM with the fp16 weight, + bias in fp32, fp16."""
    z16 = opm_outer_ref(ab, mask, num, i_base, n_i)
    y = z16.astype(mx.float32) @ wo.astype(mx.float32).T + bias
    return y.astype(mx.float16)


# ---------------------------------------------------------------------------------------------------------------
# PWA weights: one threadgroup (NSG simdgroups) per row i; simdgroups take rows j round-robin; lane holds channels
# lane + 32k. Raw logits go to `out` (device), then (after a device-memory barrier) simdgroup h normalises head h.
# ---------------------------------------------------------------------------------------------------------------
_PWAW_SRC = """
    constexpr int NT = NSG * 32;
    constexpr int KD = D / 32;
    const int N = dims[0], NP = dims[1];
    const int i = (int)threadgroup_position_in_grid.x;
    const int tid = (int)thread_index_in_threadgroup;
    const int sg = (int)simdgroup_index_in_threadgroup;
    const int lane = (int)thread_index_in_simdgroup;
    const float eps = consts[0];
    const float ninf = -consts[1];
    float lw[KD], lb[KD], wz[H][KD];
    UNR for (int k = 0; k < KD; k++) {
        const int c = lane + 32 * k;
        lw[k] = ln_w[c];
        lb[k] = ln_b[c];
        UNR for (int h = 0; h < H; h++) wz[h][k] = float(w[h * D + c]);
    }
    for (int j = sg; j < N; j += NSG) {
        const device float* zr = z + ((size_t)i * N + j) * D;
        float v[KD];
        float s = 0.0f;
        UNR for (int k = 0; k < KD; k++) { v[k] = zr[lane + 32 * k]; s += v[k]; }
        const float mean = simd_sum(s) / D;
        float ss = 0.0f;
        UNR for (int k = 0; k < KD; k++) { const float d = v[k] - mean; ss += d * d; }
        const float rstd = metal::precise::rsqrt(simd_sum(ss) / D + eps);
        UNR for (int k = 0; k < KD; k++) {
            float y = (v[k] - mean) * rstd;
            y = y * lw[k];
            v[k] = y + lb[k];
        }
        const float madd = (1.0f - pmask[(size_t)i * N + j]) * ninf;
        UNR for (int h = 0; h < H; h++) {
            float p = 0.0f;
            UNR for (int k = 0; k < KD; k++) p += v[k] * wz[h][k];
            p = simd_sum(p);
            if (lane == h) out[((size_t)h * N + i) * NP + j] = p + madd;
        }
    }
    for (int j = N + tid; j < NP; j += NT)
        UNR for (int h = 0; h < H; h++) out[((size_t)h * N + i) * NP + j] = 0.0f;
    threadgroup_barrier(mem_flags::mem_device);
    for (int h = sg; h < H; h += NSG) {
        device float* r = out + ((size_t)h * N + i) * NP;
        float mxv = -INFINITY;
        for (int j = lane; j < N; j += 32) mxv = metal::max(mxv, r[j]);
        mxv = simd_max(mxv);
        float sum = 0.0f;
        for (int j = lane; j < N; j += 32) {
            const float e = metal::exp(r[j] - mxv);
            r[j] = e;
            sum += e;
        }
        sum = simd_sum(sum);
        const float inv = 1.0f / sum;
        for (int j = lane; j < N; j += 32) r[j] = r[j] * inv;
    }
"""


def pwa_np(n: int, mult: int = 32) -> int:
    return -(-n // mult) * mult


def pwa_weights(z: mx.array, pmask: mx.array, ln_w: mx.array, ln_b: mx.array, eps: float, wz: mx.array, inf: float,
                *, nsg: int = 8) -> mx.array:
    """z [1, N, N, D] fp32, pmask [1, N, N] fp32, LN weight/bias [D] fp32, wz [H, D] (fp16 or fp32). Returns
    w [H, N, NP] fp32 (NP = N rounded up to 32, padding zeros), w[h, i, :N] = softmax_j(LN(z[i, j]) . wz[h]
    + (1 - pmask[i, j]) * -inf), LN statistics and all arithmetic fp32."""
    _, N, N2, D = z.shape
    H = wz.shape[0]
    assert N == N2 and z.dtype == mx.float32 and D % 32 == 0 and wz.shape == (H, D) and H <= 32
    assert pmask.shape == (1, N, N) and ln_w.shape == (D,) and ln_b.shape == (D,)
    NP = pwa_np(N)
    key = ("pwaw",)
    if key not in _kernels:
        _kernels[key] = _K.metal_kernel(
            name="boltz_msa_pwa_weights", input_names=["z", "pmask", "ln_w", "ln_b", "w", "consts", "dims"],
            output_names=["out"], source=_PWAW_SRC, header=HEADER)
    (w,) = _kernels[key](
        inputs=[z, pmask.astype(mx.float32), ln_w.astype(mx.float32), ln_b.astype(mx.float32), wz,
                mx.array([eps, inf], mx.float32), mx.array([N, NP], mx.int32)],
        template=[("D", D), ("H", H), ("NSG", nsg)],
        grid=(N * nsg * 32, 1, 1), threadgroup=(nsg * 32, 1, 1),
        output_shapes=[(H, N, NP)], output_dtypes=[mx.float32])
    return w


def pwa_weights_ref(z, pmask, ln_w, ln_b, eps, wz, inf):
    """MLX-op reference of pwa_weights (fp32 LN and logits)."""
    _, N, _, D = z.shape
    y = mx.fast.layer_norm(z, ln_w, ln_b, eps)
    b = (y @ wz.astype(mx.float32).T)[0].transpose(2, 0, 1)  # (H, N, N)
    b = b + (1 - pmask[0][None]) * -inf
    w = mx.softmax(b, axis=-1)
    NP = pwa_np(N)
    return mx.pad(w, [(0, 0), (0, 0), (0, NP - N)]) if NP != N else w


# ---------------------------------------------------------------------------------------------------------------
# PWA weighted average + gate. Grid: x = i tiles (BM rows; fastest, so consecutive threadgroups share the v tile),
# y = s tiles (TS sequences), z = head. Threadgroup = WM x WN simdgroups, simdgroup tile (BM / WM) x (32 TS / WN).
# A = w[h] tile [i][j] (row stride NP), B = v[s, j, h*32 + d] tiles (one [j][d] block per s), K = j.
# ---------------------------------------------------------------------------------------------------------------
_PWAA_SRC = """
    constexpr int C = 32;
    constexpr int BN = TS * C;
    constexpr int SM = BM / WM;
    constexpr int SN = BN / WN;
    constexpr int RM = SM / 8;
    constexpr int RN = SN / 8;
    constexpr int LDA = BK + PAD;
    constexpr int LDB = BN + PAD;
    constexpr int NT = WM * WN * 32;
    constexpr int NA4 = BM * BK / 4;
    constexpr int NB4 = BK * BN / 4;
    constexpr int LA = (NA4 + NT - 1) / NT;
    constexpr int LB = (NB4 + NT - 1) / NT;

    const int S = dims[0], N = dims[1], NP = dims[2];
    const int i0 = (int)threadgroup_position_in_grid.x * BM;
    const int s0 = (int)threadgroup_position_in_grid.y * TS;
    const int h = (int)threadgroup_position_in_grid.z;
    const int tid = (int)thread_index_in_threadgroup;
    const int sg = (int)simdgroup_index_in_threadgroup;
    const int lane = (int)thread_index_in_simdgroup;
    const int wm = sg / WN, wn = sg % WN;
    const device float* wh = w + (size_t)h * N * NP;

    threadgroup float As[2 * BM * LDA];                              // [i][j], double-buffered
    threadgroup float Bs[2 * BK * LDB];                              // [j][(s, d)]
    float4 ra[LA], rb[LB];

    simdgroup_float8x8 acc[RM][RN];
    UNR for (int r = 0; r < RM; r++)
        UNR for (int c = 0; c < RN; c++) acc[r][c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const int ntk = NP / BK;
    for (int it = 0; it <= ntk; it++) {
        if (it < ntk) {
            const int k0 = it * BK;
            UNR for (int u = 0; u < LA; u++) {
                const int e = tid + u * NT;
                const int r = e / (BK / 4), k4 = e % (BK / 4);
                const int i = i0 + r;
                float4 x = float4(0.0f);
                if (e < NA4 && i < N) x = *(const device float4*)(wh + (size_t)i * NP + k0 + k4 * 4);
                ra[u] = x;
            }
            UNR for (int u = 0; u < LB; u++) {
                const int e = tid + u * NT;
                const int kk = e / (BN / 4), n4 = e % (BN / 4);
                const int j = k0 + kk, s = s0 + n4 / (C / 4);
                float4 x = float4(0.0f);
                if (e < NB4 && j < N && s < S)
                    x = *(const device float4*)(v + ((size_t)s * N + j) * VS + h * C + (n4 % (C / 4)) * 4);
                rb[u] = x;
            }
        }
        if (it > 0) {
            const threadgroup float* as = As + ((it - 1) & 1) * (BM * LDA);
            const threadgroup float* bs = Bs + ((it - 1) & 1) * (BK * LDB);
            UNR for (int kk = 0; kk < BK; kk += 8) {
                simdgroup_float8x8 af[RM], bf[RN];
                UNR for (int r = 0; r < RM; r++) simdgroup_load(af[r], as + (wm * SM + r * 8) * LDA + kk, LDA);
                UNR for (int c = 0; c < RN; c++) simdgroup_load(bf[c], bs + kk * LDB + wn * SN + c * 8, LDB);
                UNR for (int r = 0; r < RM; r++)
                    UNR for (int c = 0; c < RN; c++)
                        simdgroup_multiply_accumulate(acc[r][c], af[r], bf[c], acc[r][c]);
            }
        }
        if (it < ntk) {
            threadgroup float* as = As + (it & 1) * (BM * LDA);
            threadgroup float* bs = Bs + (it & 1) * (BK * LDB);
            UNR for (int u = 0; u < LA; u++) {
                const int e = tid + u * NT;
                if (e < NA4) *(threadgroup float4*)(as + (e / (BK / 4)) * LDA + (e % (BK / 4)) * 4) = ra[u];
            }
            UNR for (int u = 0; u < LB; u++) {
                const int e = tid + u * NT;
                if (e < NB4) *(threadgroup float4*)(bs + (e / (BN / 4)) * LDB + (e % (BN / 4)) * 4) = rb[u];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // epilogue: og[s, i, h*32 + d] = sigmoid(g[s, i, h*32 + d]) * acc  (fp32)
    const int qid = lane / 4;
    const int fm = (qid & 4) + ((lane / 2) % 4);
    const int fn = (qid & 2) * 2 + (lane % 2) * 2;
    UNR for (int r = 0; r < RM; r++) {
        const int i = i0 + wm * SM + r * 8 + fm;
        UNR for (int c = 0; c < RN; c++) {
            const int n = wn * SN + c * 8 + fn;
            const int s = s0 + n / C, d = n % C;
            if (i < N && s < S) {
                const size_t row = (size_t)s * N + i;
                const float2 gp = *(const device float2*)(g + row * GS + h * C + d);
                float2 y;
                y.x = acc[r][c].thread_elements()[0] * sigmoid_mlx(gp.x);
                y.y = acc[r][c].thread_elements()[1] * sigmoid_mlx(gp.y);
                *(device float2*)(og + row * (H * C) + h * C + d) = y;
            }
        }
    }
"""

# (BM rows i per threadgroup, TS sequences per threadgroup, WM x WN simdgroups, BK K tile)
PWA_CFGS = [
    dict(bm=64, ts=4, wm=2, wn=2, bk=16),    # 4 sg x (32 x 64)
    dict(bm=32, ts=4, wm=1, wn=4, bk=16),    # 4 sg x (32 x 32)
    dict(bm=64, ts=2, wm=2, wn=2, bk=16),    # 4 sg x (32 x 32)
    dict(bm=64, ts=4, wm=2, wn=4, bk=16),    # 8 sg x (32 x 32)
    dict(bm=128, ts=2, wm=4, wn=1, bk=16),   # 4 sg x (32 x 64)
    dict(bm=32, ts=2, wm=1, wn=2, bk=32),    # 2 sg x (32 x 32), K tile 32
]
PWA_DEFAULT = PWA_CFGS[0]


def pwa_tg_bytes(bm, ts, bk, pad=4, c=32) -> int:
    return 4 * 2 * (bm * (bk + pad) + bk * (ts * c + pad))


def pwa_avg(w: mx.array, v: mx.array, g: mx.array, *, bm: int = 64, ts: int = 4, wm: int = 2, wn: int = 2,
            bk: int = 16, pad: int = 4, c: int = 32) -> mx.array:
    """w [H, N, NP] fp32 (pwa_weights), v / g [1, S, N, H*c] fp32 (proj_m / pre-sigmoid proj_g outputs). Returns
    og [1, S, N, H*c] fp32, og[s, i, h*c + d] = sigmoid(g[s, i, h*c + d]) * sum_j w[h, i, j] v[s, j, h*c + d]."""
    H, N, NP = w.shape
    _, S, N2, HC = v.shape
    assert c == 32 and N2 == N and HC == H * c and g.shape == v.shape and NP % bk == 0 and NP >= N
    assert w.dtype == mx.float32 and v.dtype == mx.float32 and g.dtype == mx.float32
    assert bm % (8 * wm) == 0 and (ts * c) % (8 * wn) == 0 and bk % 8 == 0
    assert pwa_tg_bytes(bm, ts, bk, pad, c) <= _TG_BYTES, "threadgroup tiles exceed 32 KB"
    key = ("pwaa",)
    if key not in _kernels:
        _kernels[key] = _K.metal_kernel(
            name="boltz_msa_pwa_avg", input_names=["w", "v", "g", "dims"], output_names=["og"],
            source=_PWAA_SRC, header=HEADER)
    nsg = wm * wn
    (og,) = _kernels[key](
        inputs=[w, v, g, mx.array([S, N, NP], mx.int32)],
        template=[("BM", bm), ("TS", ts), ("WM", wm), ("WN", wn), ("BK", bk), ("PAD", pad), ("H", H),
                  ("VS", HC), ("GS", HC)],
        grid=(-(-N // bm) * nsg * 32, -(-S // ts), H), threadgroup=(nsg * 32, 1, 1),
        output_shapes=[v.shape], output_dtypes=[mx.float32])
    return og


def pwa_avg_ref(w, v, g):
    """MLX-op reference of pwa_avg (fp32)."""
    H, N, NP = w.shape
    _, S, _, HC = v.shape
    c = HC // H
    vh = v.reshape(1, S, N, H, c).transpose(0, 3, 1, 2, 4)  # (1, H, S, N, c)
    o = mx.matmul(w[None, :, None, :, :N], vh)  # (1, H, S, N, c)
    o = o.transpose(0, 2, 3, 1, 4).reshape(1, S, N, HC)
    return mx.sigmoid(g) * o
