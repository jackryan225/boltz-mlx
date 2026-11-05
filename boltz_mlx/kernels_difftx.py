"""Fused Metal kernels for the Boltz-2 diffusion token transformer (BOLTZ_MLX_DIFFTX_IMPL=fused).

Per layer (dim 768, 16 heads of 48, hidden 1536, fp32) the fused path runs 7 kernels instead of ~20 MLX ops:

  attention half : AdaLN -> QKVG GEMM (+q bias) -> flash attention with additive pair bias, sigmoid(g) gate
                   -> output GEMM + sigmoid(out_gate) * . + residual
  transition half: AdaLN -> GEMM1 (x | gates | ab) + silu(gates) * x * ab -> GEMM2 + sigmoid(gate) * . + residual

`_GEMM_SRC` is a streaming-weight simdgroup-MMA GEMM for skinny A (M = 117 .. ~1000 token rows):
  * threadgroup = BM rows x (NSG simdgroups x NB 8-wide column blocks); grid = (row tiles, column tiles), so the
    many weight column tiles fill the GPU even for 117 rows (M5 Pro: 16 cores).
  * A fragments are loaded straight from device memory (the [M, K] activations are a few hundred KB and stay in
    cache): no threadgroup staging, no barriers, tiny register-only footprint -> high occupancy. Row tiles that
    overrun M read padded rows (activation buffers are allocated with 48 slack rows) and never store them.
  * B (weights) is packed once into the simdgroup fragment layout [block][k pair][lane][4], so every lane's
    B-fragment data for two consecutive k steps is one coalesced 16 B load; loads are software pipelined one k pair
    ahead. NGRP weight groups share the A fragments (GEMM1 of the transition: x | gates | ab).
  * Epilogues (MODE): 0 store (+bias), 1 silu(g) * x * ab, 2 resid + sigmoid(gate_raw) * acc.

The conditioning (AdaLN scale/shift, output gates) is consumed in its raw, pre-sigmoid, un-sliced form
(`RawCond`: the two cond GEMM outputs of `DiffusionTransformer.precompute_cond(raw=True)`), so no per-layer slice
copies or sigmoid kernels exist; sigmoids are evaluated inside the AdaLN / epilogue kernels.

The attention kernel is the tri-attention simdgroup-MMA kernel (kernels._TRI_ATT_MMA_SRC, flash style, online softmax,
S and P in registers) with I = 1 and the bias read as [Q][K].

BOLTZ_MLX_DIFF_DTYPE=fp16 uses a hybrid (MLX fp16 GEMMs run on the neural accelerators and cannot be beaten with
simdgroup MMA): fused AdaLN -> fp16 operand and fused attention reading the fp16 projection; GEMMs and elementwise
epilogues stay MLX ops. bf16, non-768 dims (atom transformers), keys-windowed attention and the CPU device fall back to
the MLX path. Select with BOLTZ_MLX_DIFFTX_IMPL=mlx|fused (default mlx).
"""

from __future__ import annotations

import os

import mlx.core as mx
import numpy as np

from boltz_mlx import kernels as _K

DIFFTX_IMPL = os.environ.get("BOLTZ_MLX_DIFFTX_IMPL", "fused")   # mlx | fused (read at call time)

_HEADER = "#include <metal_simdgroup_matrix>\n#define UNR _Pragma(\"clang loop unroll(full)\")\n"

_GEMM_SRC = """
    constexpr int RB = BM / 8;
    constexpr int KS = K / 8;
    constexpr int NBLK = NOUT / 8;                           // 8-wide column blocks per weight group
    constexpr int GSTRIDE = NBLK * (KS / 2) * 32 * 4;        // floats per packed weight group

    uint sg = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    const uint nrows = meta[0];
    uint r0 = threadgroup_position_in_grid.x * BM;
    uint cb0 = (threadgroup_position_in_grid.y * NSG + sg) * NB;     // first column block of this simdgroup

    simdgroup_float8x8 acc[NGRP][RB][NB];
    UNR for (int g = 0; g < NGRP; g++)
        UNR for (int rb = 0; rb < RB; rb++)
            UNR for (int nb = 0; nb < NB; nb++) acc[g][rb][nb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const device float* Ar = A + (size_t)r0 * K;
    float4 bcur[NGRP][NB];
    UNR for (int g = 0; g < NGRP; g++)
        UNR for (int nb = 0; nb < NB; nb++)
            bcur[g][nb] = *(const device float4*)(Wp + (size_t)g * GSTRIDE + (((size_t)(cb0 + nb) * (KS / 2)) * 32 + lane) * 4);

    for (int kp = 0; kp < KS / 2; kp++) {
        float4 bnx[NGRP][NB];
        UNR for (int g = 0; g < NGRP; g++)
            UNR for (int nb = 0; nb < NB; nb++) {
                bnx[g][nb] = bcur[g][nb];
                if (kp + 1 < KS / 2)
                    bnx[g][nb] = *(const device float4*)(
                        Wp + (size_t)g * GSTRIDE + (((size_t)(cb0 + nb) * (KS / 2) + kp + 1) * 32 + lane) * 4);
            }
        UNR for (int sub = 0; sub < 2; sub++) {
            simdgroup_float8x8 Bf[NGRP][NB];
            UNR for (int g = 0; g < NGRP; g++)
                UNR for (int nb = 0; nb < NB; nb++) {
                    Bf[g][nb].thread_elements()[0] = sub ? bcur[g][nb].z : bcur[g][nb].x;
                    Bf[g][nb].thread_elements()[1] = sub ? bcur[g][nb].w : bcur[g][nb].y;
                }
            UNR for (int rb = 0; rb < RB; rb++) {
                simdgroup_float8x8 Af;
                simdgroup_load(Af, Ar + (size_t)rb * 8 * K + (kp * 2 + sub) * 8, K);
                UNR for (int g = 0; g < NGRP; g++)
                    UNR for (int nb = 0; nb < NB; nb++)
                        simdgroup_multiply_accumulate(acc[g][rb][nb], Af, Bf[g][nb], acc[g][rb][nb]);
            }
        }
        UNR for (int g = 0; g < NGRP; g++)
            UNR for (int nb = 0; nb < NB; nb++) bcur[g][nb] = bnx[g][nb];
    }

    // epilogue
    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;
    UNR for (int rb = 0; rb < RB; rb++) {
        uint row = r0 + rb * 8 + fm;
        if (row < nrows) {
            UNR for (int nb = 0; nb < NB; nb++) {
                uint col = (cb0 + nb) * 8 + fn;
                float2 o;
                if (MODE == 1) {
                    UNR for (int e = 0; e < 2; e++) {
                        float x = acc[0][rb][nb].thread_elements()[e];
                        float gt = acc[1][rb][nb].thread_elements()[e];
                        float ab = acc[2][rb][nb].thread_elements()[e];
                        o[e] = gt / (1.0f + metal::exp(-gt)) * x * ab;
                    }
                } else {
                    o = float2(acc[0][rb][nb].thread_elements()[0], acc[0][rb][nb].thread_elements()[1]);
                    if (HAS_BIAS) o += *(const device float2*)(bias + col);
                    if (MODE == 2) {
                        // gate: pre-sigmoid slot of the raw cond tensor h2 (B,N,L,2,D); samples broadcast by rep
                        const uint ntok = meta[1], rep = meta[2];
                        const uint cr = ((row / ntok) / rep) * ntok + row % ntok;
                        float2 gt = *(const device float2*)(gate + (size_t)cr * GLD + meta[3] + col);
                        gt = 1.0f / (1.0f + metal::exp(-gt));
                        float2 rs = *(const device float2*)(resid + (size_t)row * NOUT + col);
                        o = rs + gt * o;
                    }
                }
                *(device float2*)(out + (size_t)row * NOUT + col) = o;
            }
        }
    }
"""

_kernels: dict = {}
_Z = []


def _dummy():
    if not _Z:
        _Z.append(mx.zeros((8,), mx.float32))
    return _Z[0]


def _pad_rows(M: int) -> int:
    """Row count of activation buffers: M plus 48 slack rows, so a row tile (BM <= 48) overrunning M stays in bounds."""
    return M + 48


def pack_weights(w: mx.array, ngrp: int = 1) -> mx.array:
    """w [ngrp * NOUT, K] (rows = output features, torch Linear layout) -> flat fp32 fragment-packed
    [ngrp * NOUT/8][K/16][32 lanes][4]: the 4 floats of a lane are the B-fragment elements (cols fn, fn+1) of the
    two consecutive k steps (2*kp, 2*kp+1) of 8, B[k][n] = W[n][k]."""
    R, K = w.shape
    assert R % ngrp == 0 and (R // ngrp) % 8 == 0 and K % 16 == 0
    lane = np.arange(32)
    qid = lane // 4
    fm = (qid & 4) + ((lane // 2) % 4)
    fn = (qid & 2) * 2 + (lane % 2) * 2
    # lane l, element j of a fragment holds W[8 nb + fn(l) + j][8 k8 + fm(l)]: gather within each 8x8 block
    idx = ((fn[:, None] + np.arange(2)[None, :]) * 8 + fm[:, None]).astype(np.uint32)       # (32, 2) into n*8 + k8
    w6 = w.reshape(R // 8, 8, K // 16, 2, 8).transpose(0, 2, 3, 1, 4).reshape(R // 8, K // 16, 2, 64)
    out = mx.take(w6, mx.array(idx), axis=-1)                                                # (nb, kp, sub, 32, 2)
    out = out.transpose(0, 1, 3, 2, 4).reshape(-1)                                           # (nb, kp, lane, sub, j)
    mx.eval(out)
    return out


def gemm(A, Wp, *, M, K, NOUT, NGRP=1, mode=0, bias=None, gate=None, gate_ld=0, resid=None, meta=None,
         cfg=(40, 2, 2)):
    """C[M, NOUT] = A @ W^T with packed weights Wp (pack_weights; NGRP groups of NOUT rows). A must have at least
    ceil(M / BM) * BM rows (padded rows are read, never stored). Epilogue `mode`: see the module docstring;
    mode 2 needs gate (raw cond tensor, row stride gate_ld floats), resid [M, NOUT] and
    meta = uint32 [n_rows, tokens per sample, sample repeat, gate offset]. The output has _pad_rows(M) rows."""
    BM, NSG, NB = cfg
    NBLK = NOUT // 8
    assert NBLK % (NSG * NB) == 0, (NBLK, NSG, NB)
    assert A.shape[0] >= ((M + BM - 1) // BM) * BM
    key = ("gemm", K, NOUT, NGRP, mode, bias is not None, gate_ld, BM, NSG, NB)
    if key not in _kernels:
        _kernels[key] = _K.metal_kernel(
            name=f"boltz_dtx_gemm_k{K}_n{NOUT}_g{NGRP}_m{mode}_b{BM}_{NSG}_{NB}_{gate_ld}",
            input_names=["A", "Wp", "meta", "bias", "gate", "resid"], output_names=["out"],
            source=_GEMM_SRC, header=_HEADER)
    z = _dummy()
    if meta is None:
        meta = mx.array([M, M, 1, 0], mx.uint32)
    (y,) = _kernels[key](
        inputs=[A, Wp, meta, z if bias is None else bias, z if gate is None else gate, z if resid is None else resid],
        template=[("K", K), ("NOUT", NOUT), ("NGRP", NGRP), ("MODE", mode), ("HAS_BIAS", int(bias is not None)),
                  ("GLD", gate_ld), ("BM", BM), ("NSG", NSG), ("NB", NB)],
        grid=(((M + BM - 1) // BM) * NSG * 32, NBLK // (NSG * NB), 1),
        threadgroup=(NSG * 32, 1, 1),
        output_shapes=[(_pad_rows(M), NOUT)], output_dtypes=[mx.float32])
    return y


# ----------------------------------------------------------------------------------------------
# AdaLN: t = sigmoid(h1[.., slot]) * LayerNorm(x) + h1[.., slot + 1]   (one simdgroup per row)
# ----------------------------------------------------------------------------------------------
_ADALN_SRC = """
    constexpr int NV = D / 128;
    uint row = threadgroup_position_in_grid.x * RPT + simdgroup_index_in_threadgroup;
    const uint nrows = meta[0];
    if (row >= nrows) return;
    uint lane = thread_index_in_simdgroup;
    const uint ntok = meta[1], rep = meta[2];
    const uint cr = ((row / ntok) / rep) * ntok + row % ntok;
    const device float4* sc = (const device float4*)(h1 + (size_t)cr * H1LD + meta[3]);
    const device float4* sh = sc + D / 4;
    const device float4* xr = (const device float4*)(x + (size_t)row * D);
    float4 v[NV], g[NV], b[NV];
    UNR for (int i = 0; i < NV; i++) { v[i] = xr[lane + 32 * i]; g[i] = sc[lane + 32 * i]; b[i] = sh[lane + 32 * i]; }
    float s = 0.0f;
    UNR for (int i = 0; i < NV; i++) s += v[i].x + v[i].y + v[i].z + v[i].w;
    float mean = simd_sum(s) / D;
    float ss = 0.0f;
    UNR for (int i = 0; i < NV; i++) { float4 d = v[i] - mean; ss += dot(d, d); }
    float rstd = metal::precise::rsqrt(simd_sum(ss) / D + 1e-5f);
    if (OUT_HALF) {
        device half4* yh = (device half4*)(out + (size_t)row * D);
        UNR for (int i = 0; i < NV; i++)
            yh[lane + 32 * i] = half4((v[i] - mean) * rstd * (1.0f / (1.0f + metal::exp(-g[i]))) + b[i]);
    } else {
        device float4* yr = (device float4*)(out + (size_t)row * D);
        UNR for (int i = 0; i < NV; i++)
            yr[lane + 32 * i] = (v[i] - mean) * rstd * (1.0f / (1.0f + metal::exp(-g[i]))) + b[i];
    }
"""


ADALN_RPT = 8


def adaln(x, h1, *, M, D, h1_ld, meta, out_half=False):
    """x [M, D] -> sigmoid(scale) * LN(x) + shift, scale/shift from the raw cond tensor h1 (see RawCond).
    The output has _pad_rows(M) rows (padding rows are not written)."""
    key = ("adaln", D, h1_ld, out_half)
    if key not in _kernels:
        _kernels[key] = _K.metal_kernel(name=f"boltz_dtx_adaln_{D}_{h1_ld}_{int(out_half)}", input_names=["x", "h1", "meta"],
                                             output_names=["out"], source=_ADALN_SRC, header=_HEADER)
    rpt = ADALN_RPT
    (y,) = _kernels[key](inputs=[x, h1, meta], template=[("D", D), ("H1LD", h1_ld), ("RPT", rpt), ("OUT_HALF", int(out_half))],
                         grid=(((M + rpt - 1) // rpt) * rpt * 32, 1, 1), threadgroup=(rpt * 32, 1, 1),
                         output_shapes=[(_pad_rows(M), D)], output_dtypes=[mx.float16 if out_half else mx.float32])
    return y


# ----------------------------------------------------------------------------------------------
# Attention with additive pair bias [B,H,Q,K] (key mask folded in), head dim 48, flash style.
# Reads q|k|v|g straight from the fused projection output; writes sigmoid(g) * attn(q, k, v) as [B*N, H*C].
# ----------------------------------------------------------------------------------------------
def _patched_att_src() -> str:
    """Tri-attention MMA kernel source with the bias read as [Q][K] instead of the transposed [K][Q]."""
    src = _K._TRI_ATT_MMA_SRC
    for old, new in (("brow[(size_t)col * J_DIM + qi]", "brow[(size_t)qi * J_DIM + col]"),
                     ("brow[(size_t)(col + 1) * J_DIM + qi]", "brow[(size_t)qi * J_DIM + col + 1]"),
                     ("brow[(size_t)col * J_DIM + qc]", "brow[(size_t)qc * J_DIM + col]"),
                     ("brow[(size_t)(col + 1) * J_DIM + qc]", "brow[(size_t)qc * J_DIM + col + 1]"),
                     # h may be fp16 (fp16 diffusion mode): untyped pointers, q bias in the Q load, fp32 sigmoid
                     ("const device HT* hrow", "auto hrow"),
                     ("const device HT* qp", "auto qp"),
                     ("const device HT* p = hrow", "auto p = hrow"),
                     ("const device HT* gp", "auto gp"),
                     ("device OT* op", "auto op"),
                     ("a0 = float(qp[0]) * scale; a1 = float(qp[1]) * scale;",
                      "a0 = float(qp[0]); a1 = float(qp[1]);"
                      " if (HAS_QB) { a0 += qbias[hh * C + db * 8 + fn]; a1 += qbias[hh * C + db * 8 + fn + 1]; }"
                      " a0 *= scale; a1 *= scale;")):
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    return src


# (NSG simdgroups / threadgroup, RQ query rows / simdgroup, BK keys / tile, PAD); tuned at N = 117, B = 1
ATT_CFG = (2, 16, 16, 4)


def attention(h, bias, *, B, N, heads, c, cfg=None, half=False, qbias=None, out_half=False):
    """h [>=B*N, 4*heads*c] (q|k|v|g; fp32, or fp16 with qbias = the q bias still to be added), bias [B, heads, N, N]
    additive (mask included) -> sigmoid(g) * softmax(q k^T / sqrt(c) + bias) v as [_pad_rows(B*N), heads*c]."""
    nsg, rq, bk, pad = ATT_CFG if cfg is None else cfg
    HC = heads * c
    F = 4 * HC
    h_half = h.dtype == mx.float16
    vec4 = int(F % 4 == 0 and c % 4 == 0 and not h_half)
    key = ("att", heads, c, nsg, rq, bk, pad, vec4, half, h_half, qbias is not None, out_half)
    if key not in _kernels:
        _kernels[key] = _K.metal_kernel(
            name=f"boltz_dtx_att_h{heads}_c{c}_{nsg}_{rq}_{bk}{'h' if half else ''}{int(h_half)}{int(out_half)}",
            input_names=["h", "bias", "mask", "scale", "inf_val", "qbias"], output_names=["out"],
            source=_patched_att_src(), header=_K._TRI_ATT_MMA_HEADER)
    qpt = nsg * rq
    (o,) = _kernels[key](
        inputs=[h, bias, _dummy(), mx.array(c**-0.5, mx.float32), mx.array(0.0, mx.float32),
                _dummy() if qbias is None else qbias],
        template=[("I_DIM", 1), ("J_DIM", N), ("NH", heads), ("C", c), ("HC", HC), ("ROW_STRIDE", F),
                  ("NSG", nsg), ("RQ", rq), ("BK", bk), ("PAD", pad), ("HAS_MASK", 0),
                  ("HAS_QB", int(qbias is not None)), ("VEC4", vec4), ("DT", mx.float16 if half else mx.float32),
                  ("HT", h.dtype), ("OT", mx.float16 if out_half else mx.float32)],
        grid=(((N + qpt - 1) // qpt) * nsg * 32, heads, B),
        threadgroup=(nsg * 32, 1, 1),
        output_shapes=[(_pad_rows(B * N), HC)], output_dtypes=[mx.float16 if out_half else mx.float32])
    return o


# ----------------------------------------------------------------------------------------------
# Raw (pre-sigmoid) conditioning of all layers, consumed in place by the kernels (no per-layer slices/copies)
# ----------------------------------------------------------------------------------------------
class RawCond:
    """h1 (B,N,L,4,D): attn scale|shift, transition scale|shift (scales pre-sigmoid); h2 (B,N,L,2,D): output gates
    (pre-sigmoid) of attention and transition. rep: samples sharing one conditioning row (shared noise level)."""

    def __init__(self, h1, h2, rep: int = 1):
        self.h1, self.h2, self.rep = h1, h2, rep
        self.L, self.D = h1.shape[2], h1.shape[4]

    def layer(self, i):
        return RawCondLayer(self, i)


class RawCondLayer:
    def __init__(self, rc: RawCond, i: int):
        self.rc, self.i = rc, i

    def gate(self, slot):
        """sigmoid output gate (slot 0: attention, 1: transition), (B, N, D), broadcast over shared samples."""
        rc = self.rc
        g = mx.sigmoid(rc.h2[:, :, self.i, slot, :])
        if rc.rep != 1:
            g = mx.broadcast_to(g, (rc.rep * g.shape[0],) + g.shape[1:]) if g.shape[0] == 1 else mx.repeat(g, rc.rep, axis=0)
        return g

    def arrays(self):
        """The 6-tuple (attn_scale, attn_shift, out_gate, tr_scale, tr_shift, tr_gate) with sigmoids applied."""
        rc, i = self.rc, self.i
        h1, h2 = rc.h1[:, :, i], rc.h2[:, :, i]
        out = (mx.sigmoid(h1[..., 0, :]), h1[..., 1, :], mx.sigmoid(h2[..., 0, :]),
               mx.sigmoid(h1[..., 2, :]), h1[..., 3, :], mx.sigmoid(h2[..., 1, :]))
        if rc.rep != 1:
            out = tuple(mx.repeat(x, rc.rep, axis=0) for x in out)
        return out


# Diffusion matmul dtypes that use the fused path. fp32 by default (1.17x per step at parity); the fp16
# hybrid is opt-in (BOLTZ_MLX_DIFFTX_FP16=1): only ~1.08x and larger fixed-noise deviation than MLX fp16.
_DTYPES = (mx.float32, mx.float16) if os.environ.get("BOLTZ_MLX_DIFFTX_FP16") == "1" else (mx.float32,)


def transformer_supported(tr, to_keys) -> bool:
    """Whole-transformer gate: the cond tensors may be produced in raw form for the fused kernels."""
    if DIFFTX_IMPL != "fused" or not _K.USE_KERNELS or _K.DIFF_DTYPE not in _DTYPES or to_keys is not None:
        return False
    L0 = tr.layers[0]
    at = L0.pair_bias_attn
    return (at.c_s == 768 and at.num_heads == 16 and L0.transition.b_to_a.weight.shape == (768, 1536)
            and not L0.post_layer_norm)


def _common_ok(layer, a, cond) -> bool:
    return (isinstance(cond, RawCondLayer) and a.dtype == mx.float32 and a.ndim == 3 and a.shape[-1] == 768
            and DIFFTX_IMPL == "fused" and _K.USE_KERNELS and _K.DIFF_DTYPE in _DTYPES)


def supported_attn(layer, a, to_keys, attn_bias, cond) -> bool:
    return (_common_ok(layer, a, cond) and to_keys is None and attn_bias is not None
            and layer.pair_bias_attn.num_heads == 16)


def supported_transition(layer, a, cond) -> bool:
    return _common_ok(layer, a, cond) and layer.transition.b_to_a.weight.shape == (768, 1536)


def _packed(layer):
    """Fragment-packed fp32 weights of one layer, cached on the module (re-packed if the weights are replaced)."""
    at, tr = layer.pair_bias_attn, layer.transition
    srcs = (at.proj_q.weight, at.proj_k.weight, at.proj_v.weight, at.proj_g.weight, at.proj_o.weight,
            tr.swish_gate[0].weight, tr.a_to_b.weight, tr.b_to_a.weight)
    st = getattr(layer, "_dtx", None)
    if st is None or any(x is not y for x, y in zip(st["srcs"], srcs)):
        qkvg = mx.concatenate(srcs[:4], 0)
        ff = mx.concatenate([srcs[5], srcs[6]], 0)       # x | gates | ab rows (3 groups of 1536)
        bias = mx.concatenate([at.proj_q.bias, mx.zeros((3 * 768,), mx.float32)])
        st = {"srcs": srcs, "qkvg": pack_weights(qkvg), "bias": bias, "o": pack_weights(srcs[4]),
              "ff": pack_weights(ff, 3), "ba": pack_weights(srcs[7])}
        mx.eval(bias)
        layer._dtx = st
    return st


# (BM rows, NSG simdgroups, NB column blocks per simdgroup); tuned on M5 Pro at M = 117
GEMM_CFG = {"qkvg": (40, 2, 2), "o": (24, 2, 1), "ff": (24, 2, 1), "ba": (24, 2, 1)}
# many rows (>= 256, e.g. 5 samples x 117 tokens): bigger row tiles / more columns per simdgroup
GEMM_CFG_LARGE = {"qkvg": (40, 2, 2), "o": (40, 2, 1), "ff": (40, 2, 1), "ba": (24, 2, 2)}


def _cfg(name: str, M: int):
    return (GEMM_CFG_LARGE if M >= 256 else GEMM_CFG)[name]


def attn_fused(layer, a, cond: RawCondLayer, attn_bias):
    """a + out_gate * proj_o(sigmoid(g) * attn(AdaLN(a))): 4 kernels, no intermediate elementwise passes."""
    if _K.DIFF_DTYPE == mx.float16:
        return _attn_fused_h(layer, a, cond, attn_bias)
    B, N, D = a.shape
    M = B * N
    rc, l = cond.rc, cond.i
    st = _packed(layer)
    a2 = a.reshape(M, D)
    t = adaln(a2, rc.h1, M=M, D=D, h1_ld=rc.L * 4 * D, meta=mx.array([M, N, rc.rep, l * 4 * D], mx.uint32))
    h = gemm(t, st["qkvg"], M=M, K=D, NOUT=4 * D, mode=0, bias=st["bias"], cfg=_cfg("qkvg", M))
    o = attention(h, attn_bias, B=B, N=N, heads=16, c=D // 16)
    y = gemm(o, st["o"], M=M, K=D, NOUT=D, mode=2, gate=rc.h2, gate_ld=rc.L * 2 * D, resid=a2,
             meta=mx.array([M, N, rc.rep, (l * 2) * D], mx.uint32), cfg=_cfg("o", M))
    return y[:M].reshape(B, N, D)


def transition_fused(layer, a, cond: RawCondLayer):
    """a + tr_gate * ConditionedTransition(a): AdaLN, GEMM1 + gating (hidden stays [M,1536]), GEMM2 + gate + residual."""
    if _K.DIFF_DTYPE == mx.float16:
        return _transition_fused_h(layer, a, cond)
    B, N, D = a.shape
    M = B * N
    rc, l = cond.rc, cond.i
    st = _packed(layer)
    a2 = a.reshape(M, D)
    t = adaln(a2, rc.h1, M=M, D=D, h1_ld=rc.L * 4 * D, meta=mx.array([M, N, rc.rep, (l * 4 + 2) * D], mx.uint32))
    hs = gemm(t, st["ff"], M=M, K=D, NOUT=2 * D, NGRP=3, mode=1, cfg=_cfg("ff", M))
    y = gemm(hs, st["ba"], M=M, K=2 * D, NOUT=D, mode=2, gate=rc.h2, gate_ld=rc.L * 2 * D, resid=a2,
             meta=mx.array([M, N, rc.rep, (l * 2 + 1) * D], mx.uint32), cfg=_cfg("ba", M))
    return y[:M].reshape(B, N, D)


# ----------------------------------------------------------------------------------------------
# fp16 diffusion matmuls (BOLTZ_MLX_DIFF_DTYPE=fp16): MLX's fp16 GEMMs run on the M5 neural accelerators (~3x the
# simdgroup-MMA peak), so only the surrounding passes are fused: AdaLN writes the fp16 GEMM operand directly, the
# attention kernel reads the fp16 projection (adds the q bias in fp32, writes the gated fp16 operand of the output
# projection), and the residual / gating epilogues stay MLX elementwise ops (fused by mx.compile).
# ----------------------------------------------------------------------------------------------
def _attn_fused_h(layer, a, cond: RawCondLayer, attn_bias):
    B, N, D = a.shape
    M = B * N
    rc, l = cond.rc, cond.i
    w_qkvg, _, w_o, _, b_q, _ = layer._fused()
    t = adaln(a.reshape(M, D), rc.h1, M=M, D=D, h1_ld=rc.L * 4 * D,
              meta=mx.array([M, N, rc.rep, l * 4 * D], mx.uint32), out_half=True)
    h = t[:M] @ w_qkvg.T                                        # fp16 GEMM, fp16 result
    o = attention(h, attn_bias, B=B, N=N, heads=16, c=D // 16, qbias=b_q, out_half=True)
    return a + cond.gate(0) * (o[:M] @ w_o.T).astype(mx.float32).reshape(B, N, D)


def _transition_fused_h(layer, a, cond: RawCondLayer):
    B, N, D = a.shape
    M = B * N
    rc, l = cond.rc, cond.i
    _, w_ff, _, w_ba, _, _ = layer._fused()
    inner = layer.transition.b_to_a.weight.shape[1]
    t = adaln(a.reshape(M, D), rc.h1, M=M, D=D, h1_ld=rc.L * 4 * D,
              meta=mx.array([M, N, rc.rep, (l * 4 + 2) * D], mx.uint32), out_half=True)
    hf = (t[:M] @ w_ff.T).astype(mx.float32)
    x, gates, ab = hf[:, :inner], hf[:, inner : 2 * inner], hf[:, 2 * inner :]
    hs = (gates * mx.sigmoid(gates) * x * ab).astype(mx.float16)
    return a + cond.gate(1) * (hs @ w_ba.T).astype(mx.float32).reshape(B, N, D)


# ----------------------------------------------------------------------------------------------
# s-conditioning of all layers (the two big GEMMs of DiffusionTransformer.precompute_cond, ~20 GFLOP per step in
# fp32): the streaming GEMM with the bias in its epilogue; outputs stay raw (RawCond).
# ----------------------------------------------------------------------------------------------
COND_CFG = (40, 2, 2)


def _cond_srcs(tr):
    out = []
    for L in tr.layers:
        for ad in (L.adaln, L.transition.adaln):
            out += [ad.s_norm.weight, ad.s_scale.weight, ad.s_bias.weight, ad.s_scale.bias]
        op = L.transition.output_projection[0]
        out += [L.output_projection_linear.weight, op.weight, L.output_projection_linear.bias, op.bias]
    return out


def cond_packed(tr):
    """Fragment-packed cond weights cached on the DiffusionTransformer (re-packed if any source weight is replaced).
    The unpacked concatenation of the MLX path (tr._cw) is dropped to keep the memory footprint equal."""
    srcs = _cond_srcs(tr)
    st = getattr(tr, "_dtx_cond", None)
    if st is None or any(x is not y for x, y in zip(st["srcs"], srcs)):
        w_ln, b_ln, w_raw, b_raw, _ = tr._cond_weights()
        st = {"srcs": srcs, "ln": pack_weights(w_ln), "raw": pack_weights(w_raw), "b_ln": b_ln, "b_raw": b_raw}
        mx.eval(b_ln, b_raw)
        tr._dtx_cond = st
        tr._cw = None
    return st


def cond_raw(tr, s):
    """RawCond of all layers from s (B, N, D), fp32 diffusion matmuls."""
    st = cond_packed(tr)
    B, N, D = s.shape
    M, L = B * N, len(tr.layers)
    s2 = s.reshape(M, D)
    pad = mx.zeros((_pad_rows(M) - M, D), mx.float32)
    t = mx.concatenate([_K.layer_norm(s2, None, None, 1e-5), pad], axis=0)
    h1 = gemm(t, st["ln"], M=M, K=D, NOUT=st["b_ln"].shape[0], mode=0, bias=st["b_ln"], cfg=COND_CFG)
    h2 = gemm(mx.concatenate([s2, pad], axis=0), st["raw"], M=M, K=D, NOUT=st["b_raw"].shape[0], mode=0,
              bias=st["b_raw"], cfg=COND_CFG)
    return RawCond(h1[:M].reshape(B, N, L, 4, D), h2[:M].reshape(B, N, L, 2, D))
