"""Experimental (rung 2N): custom Metal kernels on the M5 neural accelerators (NAX) via MetalPerformancePrimitives.

MLX 0.32.4 reaches NAX in its own kernels (steel/gemm/nax.h, steel/attn/nax.h) through
`mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup>` on register-resident cooperative tensors
(16x32x16 per call, fp32 accumulate). The header needs Metal language 4.0 (`__HAVE_TENSOR__`); whether
`mx.fast.metal_kernel` compiles user source with that language version is what `probe_lang()` checks.

The --precision f16 tri-att path uses tri_attention_nax (layers/triangular_attention/attention.py). GPU only.

Fragment layout caveat (rung 2N follow-up): `nax_mma` moves MLX steel fragments (BaseNAXFrag::get_coord layout) into
MPP cooperative tensors element by element. MPP documents the cooperative-tensor layout as *implementation defined*
(it depends on the descriptor and element types). MLX only ever uses descriptor (16, 32, 16, *, *, relaxed=true) with
these element types, and that is the only combination this file's register kernels are validated for
(`nax_layout` / tests/test_nax_layout.py check it on the device). relax=False is a different descriptor whose
layout is not MLX's; feeding it MLX-layout fragments scrambles the operands (the "nax_norelax" failure, rel 1.47).
Layout-agnostic code uses the MPP accessors instead: tensor views + cooperative_tensor load/store, op.run on device /
threadgroup tensors, and get_left_input_cooperative_tensor(dest) to chain GEMMs (see nax_mma_generic, kernels_nax2.py).
"""

from __future__ import annotations

import math

import mlx.core as mx

from boltz_mlx import kernels as _K  # metal_kernel(): the MPP guard (portability)

_kernels: dict = {}

NAX_HEADER = r"""
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#define UNR _Pragma("clang loop unroll(full)")

// One 16x16 NAX fragment per simdgroup: lane owns rows fm and fm+8, columns fn..fn+3 (element r*4+j <-> row fm+8r,
// column fn+j), same as MLX steel BaseNAXFrag::get_coord.
#define NAX_COORD(lane, fm, fn) \
    const short fm = (((lane) >> 2) & 4) | (((lane) >> 1) & 3); \
    const short fn = ((((lane) >> 2) & 2) | ((lane) & 1)) * 4;

// 4 consecutive elements (16-byte / 8-byte aligned) from a device or constant pointer. mx.fast.metal_kernel binds
// inputs with fewer than 8 elements as `const constant T*`, so input-derived pointers must not assume `device`.
template <typename T> inline metal::vec<T, 4> ld4(const device T* p) { return *(const device metal::vec<T, 4>*)p; }
template <typename T> inline metal::vec<T, 4> ld4(const constant T* p) { return *(const constant metal::vec<T, 4>*)p; }

// C[16x32] += A[16x16] * B[16x32]. B is given as two 16x16 fragments: columns 0-15 / 16-31 of B, or, with TB, rows
// 0-15 / 16-31 of B^T (i.e. fragments loaded straight from a [n][k] matrix). Same call sequence as MLX
// steel BaseNAXFrag::mma.
template <typename CT, typename AT, typename BT, bool TA, bool TB, bool RELAX>
inline void nax_mma(thread metal::vec<CT, 8>& c0, thread metal::vec<CT, 8>& c1, thread const metal::vec<AT, 8>& a,
                    thread const metal::vec<BT, 8>& b0, thread const metal::vec<BT, 8>& b1) {
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, 16, TA, TB, RELAX, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto ca = op.template get_left_input_cooperative_tensor<AT, BT, CT>();
    auto cb = op.template get_right_input_cooperative_tensor<AT, BT, CT>();
    auto cc = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(ca)>,
                                                             metal::remove_addrspace_t<decltype(cb)>, CT>();
    UNR for (short i = 0; i < 8; i++) {
        ca[i] = a[i];
        cb[i] = b0[i];
        cb[8 + i] = b1[i];
        cc[i] = c0[i];
        cc[8 + i] = c1[i];
    }
    op.run(ca, cb, cc);
    UNR for (short i = 0; i < 8; i++) {
        c0[i] = cc[i];
        c1[i] = cc[8 + i];
    }
}
"""

# ---------------------------------------------------------------------------------------------------------------
# Probe 1: language version seen by metal_kernel sources (no MPP include, always compiles).
# ---------------------------------------------------------------------------------------------------------------
_LANG_SRC = """
    if (thread_position_in_grid.x == 0) {
        out[0] = (int)__METAL_VERSION__;
#if defined(__HAVE_TENSOR__)
        out[1] = 1;
#else
        out[1] = 0;
#endif
#if defined(__HAVE_SIMDGROUP_MATRIX__)
        out[2] = 1;
#else
        out[2] = 0;
#endif
        out[3] = (int)a[0];
    }
"""


def probe_lang() -> dict:
    k = _K.metal_kernel(name="nax_probe_lang", input_names=["a"], output_names=["out"], source=_LANG_SRC)
    (o,) = k(inputs=[mx.array([7], mx.int32)], grid=(32, 1, 1), threadgroup=(32, 1, 1),
             output_shapes=[(4,)], output_dtypes=[mx.int32])
    v = o.tolist()
    return {"metal_version": v[0], "have_tensor": bool(v[1]), "have_simdgroup_matrix": bool(v[2]), "ok": v[3] == 7}


# ---------------------------------------------------------------------------------------------------------------
# Probe 2: one NAX matmul2d call through metal_kernel, fragment layout as MLX's. C = A @ B (TB=0, B [16, 32]) or
# C = A @ Bt^T (TB=1, Bt [32, 16]). A [16, 16]. One simdgroup.
# ---------------------------------------------------------------------------------------------------------------
_MMA1_SRC = """
    uint lane = thread_index_in_simdgroup;
    NAX_COORD(lane, fm, fn)
    metal::vec<AT, 8> af;
    metal::vec<half, 8> b0, b1;
    metal::vec<float, 8> c0 = 0.0f, c1 = 0.0f;
    UNR for (short r = 0; r < 2; r++) {
        UNR for (short j = 0; j < 4; j++) {
            short m = fm + 8 * r, n = fn + j;
            af[r * 4 + j] = (AT)a[m * 16 + n];
            if (TB) {   // Bt [32][16]: fragment f = rows f*16.. of Bt
                b0[r * 4 + j] = (half)b[(m) * 16 + n];
                b1[r * 4 + j] = (half)b[(16 + m) * 16 + n];
            } else {    // B [16][32]: fragment f = columns f*16..
                b0[r * 4 + j] = (half)b[m * 32 + n];
                b1[r * 4 + j] = (half)b[m * 32 + 16 + n];
            }
        }
    }
    nax_mma<float, AT, half, false, (TB != 0), (RELAX != 0)>(c0, c1, af, b0, b1);
    UNR for (short r = 0; r < 2; r++) {
        UNR for (short j = 0; j < 4; j++) {
            short m = fm + 8 * r, n = fn + j;
            c[m * 32 + n] = c0[r * 4 + j];
            c[m * 32 + 16 + n] = c1[r * 4 + j];
        }
    }
"""


def nax_mma_once(a: mx.array, b: mx.array, tb: bool, a_dtype=mx.float16, relax: bool = True) -> mx.array:
    """One 16x32x16 matmul2d through MLX-layout register fragments (the layout the flash kernel assumes)."""
    key = ("mma1",)
    if key not in _kernels:
        _kernels[key] = _K.metal_kernel(name="nax_probe_mma1", input_names=["a", "b"], output_names=["c"],
                                             source=_MMA1_SRC, header=NAX_HEADER)
    (c,) = _kernels[key](inputs=[a, b], template=[("TB", int(tb)), ("AT", a_dtype), ("RELAX", int(relax))],
                         grid=(32, 1, 1), threadgroup=(32, 1, 1), output_shapes=[(16, 32)], output_dtypes=[mx.float32])
    return c


# ---------------------------------------------------------------------------------------------------------------
# Probe 2b: the cooperative-tensor layouts MPP actually uses for a descriptor (get_capacity +
# get_multidimensional_index per lane and element), for left / right / destination operands of
# matmul2d(16, 32, 16, TA, TB, RELAX) with element types AT, BT -> CT. idx[0] = column, idx[1] = row of the operand
# as MPP sees it (as in MLX steel_gemm_thin_nax: r = idx[1], c = idx[0]).
# ---------------------------------------------------------------------------------------------------------------
_LAYOUT_SRC = """
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, TA != 0, TB != 0, RELAX != 0,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto ca = op.template get_left_input_cooperative_tensor<AT, BT, CT>();
    auto cb = op.template get_right_input_cooperative_tensor<AT, BT, CT>();
    auto cc = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(ca)>,
                                                             metal::remove_addrspace_t<decltype(cb)>, CT>();
    const uint lane = thread_index_in_simdgroup;
    caps[0 * 32 + lane] = (int)ca.get_capacity();
    caps[1 * 32 + lane] = (int)cb.get_capacity();
    caps[2 * 32 + lane] = (int)cc.get_capacity();
    for (ushort i = 0; i < MAXCAP; i++) {
        device int* pa = coords + ((0 * 32 + lane) * MAXCAP + i) * 2;
        device int* pb = coords + ((1 * 32 + lane) * MAXCAP + i) * 2;
        device int* pc = coords + ((2 * 32 + lane) * MAXCAP + i) * 2;
        pa[0] = -1; pa[1] = -1; pb[0] = -1; pb[1] = -1; pc[0] = -1; pc[1] = -1;
        if (i < ca.get_capacity()) { auto ix = ca.get_multidimensional_index(i); pa[0] = (int)ix[0]; pa[1] = (int)ix[1]; }
        if (i < cb.get_capacity()) { auto ix = cb.get_multidimensional_index(i); pb[0] = (int)ix[0]; pb[1] = (int)ix[1]; }
        if (i < cc.get_capacity()) { auto ix = cc.get_multidimensional_index(i); pc[0] = (int)ix[0]; pc[1] = (int)ix[1]; }
    }
"""
LAYOUT_MAXCAP = 32


def nax_layout(relax: bool, ta: bool, tb: bool, at=mx.float16, bt=mx.float16, ct=mx.float32):
    """-> (caps int32 [3, 32], coords int32 [3, 32, MAXCAP, 2]) for (left, right, destination); -1 = unused slot."""
    key = ("layout",)
    if key not in _kernels:
        _kernels[key] = _K.metal_kernel(name="nax_probe_layout", input_names=["dummy"],
                                             output_names=["caps", "coords"], source=_LAYOUT_SRC, header=NAX_HEADER)
    caps, coords = _kernels[key](
        inputs=[mx.zeros((8,), mx.float32)],
        template=[("TA", int(ta)), ("TB", int(tb)), ("RELAX", int(relax)), ("AT", at), ("BT", bt), ("CT", ct),
                  ("MAXCAP", LAYOUT_MAXCAP)],
        grid=(32, 1, 1), threadgroup=(32, 1, 1),
        output_shapes=[(3, 32), (3, 32, LAYOUT_MAXCAP, 2)], output_dtypes=[mx.int32, mx.int32])
    return caps, coords


def mlx_frag_coords():
    """Expected (MLX steel BaseNAXFrag) coordinates: [32 lanes, 8 elements] -> (row, col) of a 16x16 fragment."""
    import numpy as np

    lane = np.arange(32)[:, None]
    e = np.arange(8)[None, :]
    fm = ((lane >> 2) & 4) | ((lane >> 1) & 3)
    fn = (((lane >> 2) & 2) | (lane & 1)) * 4
    return np.stack([fm + 8 * (e >> 2), fn + (e & 3)], -1)  # [32, 8, 2] (row, col)


# ---------------------------------------------------------------------------------------------------------------
# Probe 2c: the same 16x32x16 product through the layout-agnostic MPP API (tensor views over device memory,
# cooperative_tensor load / store). Correct for any descriptor by construction; used to show that relax=False is a
# layout question, not an arithmetic one, and to measure what `relaxed_precision` does to float operands.
#   A [16, 16] (AT), B: TB ? Bt [32, 16] : B [16, 32] (half) -> C [16, 32] float
# ---------------------------------------------------------------------------------------------------------------
_MMA_GENERIC_SRC = """
    using ext_t = metal::dextents<int32_t, 2>;
    using str_t = metal::array<int32_t, 2>;
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, TB != 0, RELAX != 0,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    metal::tensor<device AT, ext_t, metal::tensor_inline> tA((device AT*)a, ext_t(16, 16), str_t{1, 16});
    metal::tensor<device half, ext_t, metal::tensor_inline> tB((device half*)b, TB ? ext_t(16, 32) : ext_t(32, 16),
                                                               str_t{1, TB ? 16 : 32});
    metal::tensor<device float, ext_t, metal::tensor_inline> tC(c, ext_t(32, 16), str_t{1, 32});
    auto ca = op.template get_left_input_cooperative_tensor<AT, half, float>();
    ca.load(tA);
    auto cc = op.template get_destination_cooperative_tensor<decltype(ca), decltype(tB), float>();
    op.run(ca, tB, cc);
    cc.store(tC);
"""


def nax_mma_generic(a: mx.array, b: mx.array, tb: bool, relax: bool = True) -> mx.array:
    """C = A @ (Bt^T if tb else B) through the layout-agnostic MPP API. a: [16, 16] fp16/fp32, b fp16."""
    key = ("mma_generic",)
    if key not in _kernels:
        _kernels[key] = _K.metal_kernel(name="nax_probe_mma_generic", input_names=["a", "b"],
                                             output_names=["c"], source=_MMA_GENERIC_SRC, header=NAX_HEADER)
    (c,) = _kernels[key](inputs=[a, b.astype(mx.float16)],
                         template=[("TB", int(tb)), ("RELAX", int(relax)), ("AT", a.dtype)],
                         grid=(32, 1, 1), threadgroup=(32, 1, 1), output_shapes=[(16, 32)],
                         output_dtypes=[mx.float32])
    return c


# ---------------------------------------------------------------------------------------------------------------
# Probe 3: peak MMA throughput from a custom kernel, register operands only. NAX: NACC independent 16x32x16 calls
# per iteration (16384 flop each). simdgroup_matrix 8x8x8 (1024 flop each): NACC independent accumulators.
# ---------------------------------------------------------------------------------------------------------------
_PEAK_NAX_SRC = """
    uint lane = thread_index_in_simdgroup;
    uint gid = thread_position_in_grid.x;
    metal::vec<AT, 8> af;
    metal::vec<half, 8> b0, b1;
    UNR for (short i = 0; i < 8; i++) { af[i] = (AT)seed[(lane + i) % 64]; b0[i] = (half)seed[(lane + 2 * i) % 64]; b1[i] = (half)seed[(lane + 3 * i) % 64]; }
    metal::vec<float, 8> acc[2 * NACC];
    UNR for (int k = 0; k < 2 * NACC; k++) acc[k] = 0.0f;
    for (int it = 0; it < ITERS; it++) {
        UNR for (int k = 0; k < NACC; k++) nax_mma<float, AT, half, false, (TB != 0), true>(acc[2 * k], acc[2 * k + 1], af, b0, b1);
    }
    float s = 0.0f;
    UNR for (int k = 0; k < 2 * NACC; k++) UNR for (short i = 0; i < 8; i++) s += acc[k][i];
    out[gid] = s;
"""

_PEAK_SG_SRC = """
    uint lane = thread_index_in_simdgroup;
    uint gid = thread_position_in_grid.x;
    simdgroup_matrix<DT, 8, 8> a, b;
    a.thread_elements()[0] = (DT)seed[lane % 64]; a.thread_elements()[1] = (DT)seed[(lane + 1) % 64];
    b.thread_elements()[0] = (DT)seed[(lane + 2) % 64]; b.thread_elements()[1] = (DT)seed[(lane + 3) % 64];
    simdgroup_matrix<float, 8, 8> acc[NACC];
    UNR for (int k = 0; k < NACC; k++) acc[k] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (int it = 0; it < ITERS; it++) {
        UNR for (int k = 0; k < NACC; k++) simdgroup_multiply_accumulate(acc[k], a, b, acc[k]);
    }
    float s = 0.0f;
    UNR for (int k = 0; k < NACC; k++) s += acc[k].thread_elements()[0] + acc[k].thread_elements()[1];
    out[gid] = s;
"""


def peak_kernel(kind: str, n_simdgroups: int, iters: int, nacc: int, sg_per_tg: int = 4, a_dtype=mx.float16,
                tb: bool = True):
    """Returns (fn, flops): fn() -> output array. kind: 'nax' | 'sg_half' | 'sg_float'."""
    seed = mx.random.normal((64,)) * 0.01
    tg = 32 * sg_per_tg
    grid = (n_simdgroups * 32, 1, 1)
    if kind == "nax":
        key = ("peak_nax",)
        if key not in _kernels:
            _kernels[key] = _K.metal_kernel(name="nax_probe_peak", input_names=["seed"], output_names=["out"],
                                                 source=_PEAK_NAX_SRC, header=NAX_HEADER)
        tmpl = [("ITERS", iters), ("NACC", nacc), ("AT", a_dtype), ("TB", int(tb))]
        flops = n_simdgroups * iters * nacc * 16 * 32 * 16 * 2
    else:
        key = ("peak_sg",)
        if key not in _kernels:
            _kernels[key] = _K.metal_kernel(
                name="sg_probe_peak", input_names=["seed"], output_names=["out"], source=_PEAK_SG_SRC,
                header="#include <metal_simdgroup_matrix>\n#define UNR _Pragma(\"clang loop unroll(full)\")\n")
        tmpl = [("ITERS", iters), ("NACC", nacc), ("DT", mx.float16 if kind == "sg_half" else mx.float32)]
        flops = n_simdgroups * iters * nacc * 8 * 8 * 8 * 2
    k = _kernels[key]

    def fn():
        return k(inputs=[seed], template=tmpl, grid=grid, threadgroup=(tg, 1, 1),
                 output_shapes=[(n_simdgroups * 32,)], output_dtypes=[mx.float32])[0]

    return fn, flops


# ---------------------------------------------------------------------------------------------------------------
# Flash triangle attention on NAX (fp16 h in, fp16 or fp32 out).
#   one simdgroup = 16 queries of one (row, head); a threadgroup = WQ query blocks x HPT heads of one row
#   S = Q K^T     : nax_mma(16 q x 32 keys, k = 16 head dims) x C/16, Q fragments in registers for the whole row,
#                   K/V fragments loaded straight from device memory (L1-cached, as MLX's attention_nax does)
#   logits (log2 domain, fp32): S * scale*log2e + log2e*bias + maskb'   bias = tri bias [B, H, J, KP] (BT, unscaled)
#                                                                  maskb' = log2e * inf*(m-1), -1e30 for keys >= J
#   online softmax: exp2, row max / sum across the 4 lanes of a fragment row (shuffle xor 1, 8)
#   O += P V      : nax_mma(16 q x 32 dims, k = 16 keys), P from the S registers (PT = float or half)
#   epilogue      : O / l * sigmoid(g) -> out [B, I, J, H*C]
# Keys past J are read from a clamped (valid) row and removed by the -1e30 mask padding; query rows past J are
# computed on clamped rows and not stored. No threadgroup memory, no barriers.
# ---------------------------------------------------------------------------------------------------------------
_TRI_NAX_SRC = """
    constexpr int TD = C / 16;               // 16-wide head-dim fragments (C % 32 == 0)
    const uint RS = ROW_STRIDE;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    uint qt, hg, row;
    if (ORDER == 0) { qt = threadgroup_position_in_grid.x; hg = threadgroup_position_in_grid.y; row = threadgroup_position_in_grid.z; }
    else            { row = threadgroup_position_in_grid.x; qt = threadgroup_position_in_grid.y; hg = threadgroup_position_in_grid.z; }
    const uint hh = hg * HPT + (sg % HPT);
    const uint q0 = (qt * WQ + sg / HPT) * 16;
    if (q0 >= J_DIM) return;
    const uint b = row / I_DIM;
    NAX_COORD(lane, fm, fn)
    const float scale2 = scale_l2e[0];

    // TRANS = 0: h is [B, I, J, F] (row = (b, i), keys/queries along J)
    // TRANS = 1: h is [B, J, I, F] (ending node in the original orientation: element (row r, key j) at h[b, j, r]);
    //            the output is written in the same orientation, so no transposed copies are needed around the core
    const uint r_in = row - b * I_DIM;
    const size_t JS = TRANS ? (size_t)I_DIM * RS : (size_t)RS;          // h stride between consecutive keys
    const size_t OS = TRANS ? (size_t)I_DIM * HC : (size_t)HC;          // out stride between consecutive queries
    auto hrow = h + (TRANS ? ((size_t)b * J_DIM * I_DIM + r_in) * RS : (size_t)row * J_DIM * RS);
    device OT* orow = out + (TRANS ? ((size_t)b * J_DIM * I_DIM + r_in) * HC : (size_t)row * J_DIM * HC);
    auto brow = bias + (size_t)(b * NH + hh) * J_DIM * KP;   // auto: any input may be bound in constant space (< 8 elements)
    auto mrow = maskb + (MASK_ROWS ? (size_t)row * KP : 0);
    const uint qra = min(q0 + fm, (uint)J_DIM - 1);
    const uint qrb = min(q0 + fm + 8, (uint)J_DIM - 1);

    // Q fragments (unscaled fp16; the scale is applied to S in fp32)
    metal::vec<half, 8> Qf[TD];
    UNR for (int db = 0; db < TD; db++) {
        auto pa = hrow + (size_t)qra * JS + hh * C + db * 16 + fn;
        auto pb = hrow + (size_t)qrb * JS + hh * C + db * 16 + fn;
        if (VEC) {
            vec<HT, 4> va = ld4(pa), vb = ld4(pb);
            UNR for (short j = 0; j < 4; j++) { Qf[db][j] = (half)va[j]; Qf[db][4 + j] = (half)vb[j]; }
        } else {
            UNR for (short j = 0; j < 4; j++) { Qf[db][j] = (half)pa[j]; Qf[db][4 + j] = (half)pb[j]; }
        }
    }
    metal::vec<float, 8> O[TD];
    UNR for (int d = 0; d < TD; d++) O[d] = 0.0f;
    float m_r[2] = {-3.0e38f, -3.0e38f};
    float l_r[2] = {0.0f, 0.0f};

    for (uint k0 = 0; k0 < J_DIM; k0 += 16 * NKF) {
        // key rows of this lane's K/V fragment elements (clamped)
        uint ka[NKF], kb[NKF];
        UNR for (int f = 0; f < NKF; f++) {
            ka[f] = min(k0 + f * 16 + fm, (uint)J_DIM - 1);
            kb[f] = min(k0 + f * 16 + fm + 8, (uint)J_DIM - 1);
        }
        // bias + mask for this lane's logits (independent of the MMAs)
        float bl[NKF][8];
        UNR for (int f = 0; f < NKF; f++) {
            const uint kc = k0 + f * 16 + fn;   // < KP; KP % 4 == 0, kc % 4 == 0 -> aligned vector loads
            float4 mk = ld4(mrow + kc);
            vec<BT, 4> ba = ld4(brow + (size_t)qra * KP + kc);
            vec<BT, 4> bb = ld4(brow + (size_t)qrb * KP + kc);
            UNR for (short j = 0; j < 4; j++) { bl[f][j] = fma(float(ba[j]), 1.44269504089f, mk[j]); bl[f][4 + j] = fma(float(bb[j]), 1.44269504089f, mk[j]); }
        }

        // S = Q K^T
        metal::vec<float, 8> S[NKF];
        UNR for (int f = 0; f < NKF; f++) S[f] = 0.0f;
        UNR for (int db = 0; db < TD; db++) {
            metal::vec<half, 8> Kf[NKF];
            UNR for (int f = 0; f < NKF; f++) {
                auto pa = hrow + (size_t)ka[f] * JS + HC + hh * C + db * 16 + fn;
                auto pb = hrow + (size_t)kb[f] * JS + HC + hh * C + db * 16 + fn;
                if (VEC) {
                    vec<HT, 4> va = ld4(pa), vb = ld4(pb);
                    UNR for (short j = 0; j < 4; j++) { Kf[f][j] = (half)va[j]; Kf[f][4 + j] = (half)vb[j]; }
                } else {
                    UNR for (short j = 0; j < 4; j++) { Kf[f][j] = (half)pa[j]; Kf[f][4 + j] = (half)pb[j]; }
                }
            }
            UNR for (int f = 0; f < NKF; f += 2)
                nax_mma<float, half, half, false, true, (RELAX != 0)>(S[f], S[f + 1], Qf[db], Kf[f], Kf[f + 1]);
        }

        // logits in the log2 domain, online softmax per fragment row
        UNR for (int f = 0; f < NKF; f++)
            UNR for (short i = 0; i < 8; i++) S[f][i] = fma(S[f][i], scale2, bl[f][i]);
        UNR for (short r = 0; r < 2; r++) {
            float tmax = -3.0e38f;
            UNR for (int f = 0; f < NKF; f++) UNR for (short j = 0; j < 4; j++) tmax = max(tmax, S[f][r * 4 + j]);
            tmax = max(tmax, simd_shuffle_xor(tmax, (ushort)1));
            tmax = max(tmax, simd_shuffle_xor(tmax, (ushort)8));
            const float m_new = max(m_r[r], tmax);
            const float corr = fast::exp2(m_r[r] - m_new);
            float ls = 0.0f;
            UNR for (int f = 0; f < NKF; f++) UNR for (short j = 0; j < 4; j++) {
                float p = fast::exp2(S[f][r * 4 + j] - m_new);
                S[f][r * 4 + j] = p;
                ls += p;
            }
            l_r[r] = l_r[r] * corr + ls;   // per-lane partial, reduced over the 4 lanes of the row at the end
            m_r[r] = m_new;
            UNR for (int d = 0; d < TD; d++) UNR for (short j = 0; j < 4; j++) O[d][r * 4 + j] *= corr;
        }

        // O += P V
        UNR for (int f = 0; f < NKF; f++) {
            metal::vec<PT, 8> P;
            UNR for (short i = 0; i < 8; i++) P[i] = (PT)S[f][i];
            UNR for (int dp = 0; dp < TD; dp += 2) {
                metal::vec<half, 8> V0, V1;
                auto pa = hrow + (size_t)ka[f] * JS + 2 * HC + hh * C + dp * 16 + fn;
                auto pb = hrow + (size_t)kb[f] * JS + 2 * HC + hh * C + dp * 16 + fn;
                if (VEC) {
                    vec<HT, 4> a0 = ld4(pa), b0 = ld4(pb);
                    vec<HT, 4> a1 = ld4(pa + 16), b1 = ld4(pb + 16);
                    UNR for (short j = 0; j < 4; j++) {
                        V0[j] = (half)a0[j]; V0[4 + j] = (half)b0[j];
                        V1[j] = (half)a1[j]; V1[4 + j] = (half)b1[j];
                    }
                } else {
                    UNR for (short j = 0; j < 4; j++) {
                        V0[j] = (half)pa[j]; V0[4 + j] = (half)pb[j];
                        V1[j] = (half)pa[16 + j]; V1[4 + j] = (half)pb[16 + j];
                    }
                }
                nax_mma<float, PT, half, false, false, (RELAX != 0)>(O[dp], O[dp + 1], P, V0, V1);
            }
        }
    }

    // epilogue: normalise, gate, store
    UNR for (short r = 0; r < 2; r++) {
        float l = l_r[r];
        l += simd_shuffle_xor(l, (ushort)1);
        l += simd_shuffle_xor(l, (ushort)8);
        const float inv = 1.0f / l;
        const uint qi = q0 + fm + 8 * r;
        if (qi < J_DIM) {
            UNR for (int d = 0; d < TD; d++) {
                auto gp = hrow + (size_t)qi * JS + 3 * HC + hh * C + d * 16 + fn;
                device OT* op = orow + (size_t)qi * OS + hh * C + d * 16 + fn;
                UNR for (short j = 0; j < 4; j++) {
                    float g = 1.0f / (1.0f + metal::exp(-float(gp[j])));
                    op[j] = OT(O[d][r * 4 + j] * inv * g);
                }
            }
        }
    }
"""

LOG2E = 1.4426950408889634
# R8-coldN: the size constants as run-time locals with the template parameters' names and type (int), read from a small
# int buffer `dims` (bound in constant space): identical expressions and arithmetic, one compiled library for every N.
DYN_IJK = "    const int I_DIM = (int)dims[0], J_DIM = (int)dims[1], KP = (int)dims[2];\n"


def nax_tri_inputs(h: mx.array, n_heads: int, c: int, mask: mx.array | None, inf: float, bias: mx.array | None = None,
                   kp_mult: int = 64, bias_dtype=mx.float16, trans: bool = False):
    """Prepare the tri bias [B, H, J, KP] (bias_dtype, unscaled) and log2-domain key-mask bias [B*I or 1, KP] (fp32).
    h: [B, I, J, F] ([B, J, I, F] with trans). mask: [B, I, J] in the attention orientation (row i, key j).
    bias: [B, H, Q, K] tri bias (bias(b,h,q,k)); None -> from h[..., 4HC:] (I == J only)."""
    B = h.shape[0]
    I, J = (h.shape[2], h.shape[1]) if trans else (h.shape[1], h.shape[2])
    F = h.shape[3]
    HC = n_heads * c
    KP = ((J + kp_mult - 1) // kp_mult) * kp_mult
    if bias is None:
        assert F == 4 * HC + n_heads and I == J
        tb = h[..., 4 * HC:]
        # attention-orientation tensor t[b, q, k, h]; with trans t[b, q, k] = h[b, k, q]
        bias = tb.transpose(0, 3, 2, 1) if trans else tb.transpose(0, 3, 1, 2)  # [b, h, q, k]
    bias = bias.astype(bias_dtype)  # unscaled (exact fp16 h values); log2e is applied in fp32 in the kernel
    if KP != J:
        bias = mx.pad(bias, [(0, 0), (0, 0), (0, 0), (0, KP - J)])
    if mask is None:
        mb = mx.zeros((1, J), mx.float32)
        rows = 0
    else:
        mb = (inf * LOG2E) * (mask.reshape(B * I, J).astype(mx.float32) - 1.0)
        rows = 1
    if KP != J:
        mb = mx.pad(mb, [(0, 0), (0, KP - J)], constant_values=-1e30)
    return bias, mb, KP, rows


def tri_attention_nax(h: mx.array, n_heads: int, c: int, mask: mx.array | None, inf: float,
                      bias: mx.array | None = None, out_dtype=mx.float16, *, wq: int = 4, hpt: int = 1, nkf: int = 2,
                      order: int = 0, relax: bool = True, p_dtype=mx.float32, bias_dtype=mx.float16,
                      trans: bool = False, prepared=None, allow_unverified_layout: bool = False) -> mx.array:
    """sigmoid(g) * softmax(q k^T / sqrt(c) + mask_bias + tri_bias) v, NAX MMAs.
    h: [B, I, J, F] (q|k|v|g in the first 4HC columns, any F) -> out [B, I, J, H*C].
    trans: h is [B, J, I, F] (rows along axis 2, e.g. the ending node without swapaxes) -> out [B, J, I, H*C];
    mask stays [B, I, J] in the attention orientation. prepared: output of nax_tri_inputs (reuse).
    relax=False selects a matmul2d descriptor whose cooperative-tensor layout is not the MLX fragment layout this
    kernel is written for (see the module docstring); it is refused unless allow_unverified_layout (probes only)."""
    if not relax and not allow_unverified_layout:
        raise ValueError("tri_attention_nax: relax=False uses a cooperative-tensor layout the register kernel does not "
                         "handle (wrong results); see kernels_nax module docstring / nax_layout()")
    B, A1, A2, F = h.shape
    I, J = (A2, A1) if trans else (A1, A2)
    HC = n_heads * c
    assert c % 32 == 0 and nkf % 2 == 0 and n_heads % hpt == 0
    if prepared is None:
        prepared = nax_tri_inputs(h, n_heads, c, mask, inf, bias, kp_mult=max(64, 16 * nkf), bias_dtype=bias_dtype,
                                  trans=trans)
    bias_l2, mb, KP, rows = prepared
    assert KP % (16 * nkf) == 0
    vec = int(F % 4 == 0)
    from boltz_mlx import kernels as _K

    dyn = _K.nax("DYNN")  # R8-coldN: I / J / KP at run time (same arithmetic, no per-N compile)
    key = ("tri_nax", n_heads, c) + (("dyn",) if dyn else ())
    if key not in _kernels:
        _kernels[key] = _K.metal_kernel(
            name=f"boltz_tri_att_nax_h{n_heads}_c{c}" + ("_dyn" if dyn else ""),
            input_names=["h", "bias", "maskb", "scale_l2e"] + (["dims"] if dyn else []),
            output_names=["out"],
            source=(DYN_IJK + _TRI_NAX_SRC) if dyn else _TRI_NAX_SRC,
            header=NAX_HEADER,
        )
    nqt = (J + 16 * wq - 1) // (16 * wq)
    nhg = n_heads // hpt
    tg = wq * hpt * 32
    grid = (nqt * tg, nhg, B * I) if order == 0 else (B * I * tg, nqt, nhg)
    sizes = [] if dyn else [("I_DIM", I), ("J_DIM", J)]
    (o,) = _kernels[key](
        inputs=[h, bias_l2, mb, mx.array([c**-0.5 * LOG2E], mx.float32)]
        + ([mx.array([I, J, KP, 0], mx.int32)] if dyn else []),
        template=sizes + [("NH", n_heads), ("C", c), ("HC", HC), ("ROW_STRIDE", F)] + ([] if dyn else [("KP", KP)])
        + [("MASK_ROWS", rows), ("WQ", wq), ("HPT", hpt), ("NKF", nkf), ("ORDER", order), ("RELAX", int(relax)),
           ("VEC", vec), ("TRANS", int(trans)), ("HT", h.dtype), ("BT", bias_l2.dtype), ("PT", p_dtype),
           ("OT", out_dtype)],
        grid=grid,
        threadgroup=(tg, 1, 1),
        output_shapes=[(B, A1, A2, HC)],
        output_dtypes=[out_dtype],
    )
    return o


assert math  # (kept for callers that compute scales)
