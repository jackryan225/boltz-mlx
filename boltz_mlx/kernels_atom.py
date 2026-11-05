"""Atom attention encoder / decoder on NAX (speed-first round, --precision f16 only; flag BOLTZ_MLX_NAX_ATOM). GPU only.

The two AtomTransformers of the diffusion module (3 layers each: atom dim 128, 4 heads of 32, windows of 32 queries x
128 keys, AdaLN from the step-invariant atom conditioning c, conditioned SwiGLU transition) run per layer as

    adaln16 (kernels_nax3)        t = sigmoid(sc) * LN(a) + sh          fp16      (sc / sh: raw fp16 cond + fp32 bias)
    gemm16 qkvg                   h = t Wqkvg^T                         fp16 [M, 512]
    atom_attention (this file)    o = sigmoid(g) * softmax(q k^T / sqrt(c) + bias) v   on NAX, keys read in place
    gemm16 o   + EPI_GRESID       a = a + sigmoid(gate) * (o Wo^T)      fp32
    adaln16                       t = ...
    gemm16 ff  + EPI_SWIGLU       hs = silu(t Wg^T) * (t Wx^T) * (t Wab^T)  fp16 [M, 256]
    gemm16 ba  + EPI_GRESID       a = a + sigmoid(gate) * (hs Wba^T)    fp32

instead of ~30 MLX ops per layer (fp32 AdaLN passes, fp16 GEMMs with fp32 casts, the k|v key-window gather
(ToKeys: 4x the k|v bytes, 0.53 ms per layer at lig_aff x5), transposed copies, the unfused fp32 SDPA fallback for
head dim 32 and elementwise gating). Rows of `a` are (sample, atom), M = samples * NA; windows are contiguous row
blocks of W atoms (global window w = sample * NWS + window).

atom_attention: one simdgroup = 16 queries of one (window, head); a threadgroup = the window's W / 16 query blocks x HPT
heads. Key slot j of window k is atom k * W - KOFF + j (KOFF = (H - W) / 2; encodersv2.get_indexing_matrix: key window
k holds half-windows 2k - 3 .. 2k + 4), read straight from the qkvg GEMM output (rows clamped into the sample; slots
outside the sequence carry the -1e6 key mask in the bias, exactly as in the MLX path). The NAX code is the validated
register flash kernel of kernels_nax._TRI_NAX_SRC (relaxed 16x32x16 descriptor, MLX fragment layout pinned by
tests/test_nax_layout.py): S = Q K^T in fp32, logits and online softmax in fp32 (log2 domain), P (PT) x V fp16,
fp32 accumulate. Q = half(float(h_q) + q bias). The additive bias [NWb, heads, W, H] fp32 (pair bias + key mask, the
step-invariant DiffusionTransformer.full_attn_bias) is shared by BREP consecutive samples (multiplicity): window w
reads bias window (w / NWS / BREP) * NWS + w % NWS. Output half(O / l * sigmoid(g)) [M, 128] = the MLX path's fp16
operand of the output projection (same rounding point).

Glue (per step, outside the transformers):
  segment_mean_relu   encoder atom -> token aggregation: a[s, t] = sum_{atoms i of t} relu(y[s, i]) / (n_t + 1e-6)
                      (y = q Wa2t^T, fp32), replaces the dense (B, T, NA) x (B, NA, 768) fp32 matmul with the
                      normalised one-hot (and its per-step repeat / normalise / transposed copy).
  atom_tokens         decoder token -> atom broadcast as an index gather (exact: the one-hot matmul adds zeros).
Index arrays are built once per prediction (prepare(), host side); nothing here syncs inside the compiled step.
"""

from __future__ import annotations

import os

import mlx.core as mx
import numpy as np

from boltz_mlx import kernels as _K  # metal_kernel(): the MPP guard (portability)
from boltz_mlx import kernels_nax3 as _N3

_kernels: dict = {}
LOG2E = 1.4426950408889634

# ---------------------------------------------------------------------------------------------------------------
# Windowed atom attention on NAX
# ---------------------------------------------------------------------------------------------------------------
_ATT_SRC = """
    constexpr int TD = C / 16;                  // 16-wide head-dim fragments (C % 32 == 0 -> TD even)
    constexpr int KOFF = (H - W) / 2;           // key slot j of window k <-> atom k * W - KOFF + j
    const int NA = meta[0], NWS = meta[1], BREP = meta[2], NS = meta[3];
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    // global window = sample * NWS + window; ORDER 1: threadgroups of the samples of one window are adjacent (the
    // shared bias block is read by consecutive threadgroups), ORDER 0: windows of one sample adjacent (key overlap)
    const uint tgx = threadgroup_position_in_grid.x;
    const uint win = ORDER == 0 ? tgx : (tgx % (uint)NS) * (uint)NWS + tgx / (uint)NS;
    const uint hh = threadgroup_position_in_grid.y * HPT + sg % HPT;
    const uint q0 = (sg / HPT) * 16;                               // query block within the window
    const uint smp = win / (uint)NWS, kw = win % (uint)NWS;
    NAX_COORD(lane, fm, fn)
    const float scale2 = scale_l2e[0];
    auto hq = h + (size_t)win * W * F;                             // the window's W query rows
    auto hk = h + (size_t)smp * (size_t)NA * F;                    // atom 0 of this sample (key rows)
    auto brow = bias + (size_t)(((smp / (uint)BREP) * (uint)NWS + kw) * NH + hh) * (W * H);   // [W][H] block
    const int kbase = (int)(kw * W) - KOFF;

    // Q fragments: rows q0 + fm / q0 + fm + 8 (always inside the window); q bias added in fp32, then fp16
    metal::vec<half, 8> Qf[TD];
    UNR for (int db = 0; db < TD; db++) {
        const int c = hh * C + db * 16 + fn;
        const vec<half, 4> va = ld4(hq + (size_t)(q0 + fm) * F + c);
        const vec<half, 4> vb = ld4(hq + (size_t)(q0 + fm + 8) * F + c);
        UNR for (short j = 0; j < 4; j++) {
            const float qb = HAS_QB ? float(qbias[c + j]) : 0.0f;
            Qf[db][j] = (half)(float(va[j]) + qb);
            Qf[db][4 + j] = (half)(float(vb[j]) + qb);
        }
    }
    metal::vec<float, 8> O[TD];
    UNR for (int d = 0; d < TD; d++) O[d] = 0.0f;
    float m_r[2] = {-3.0e38f, -3.0e38f};
    float l_r[2] = {0.0f, 0.0f};

    for (int k0 = 0; k0 < H; k0 += 16 * NKF) {
        // key rows (element offsets) of this lane's K / V fragment elements, clamped into the sample
        size_t ka[NKF], kb[NKF];
        UNR for (int f = 0; f < NKF; f++) {
            const int a0 = kbase + k0 + f * 16 + fm;
            ka[f] = (size_t)metal::clamp(a0, 0, NA - 1) * F;
            kb[f] = (size_t)metal::clamp(a0 + 8, 0, NA - 1) * F;
        }
        // additive bias (pair bias + key mask) for this lane's logits, log2 domain (independent of the MMAs)
        float bl[NKF][8];
        UNR for (int f = 0; f < NKF; f++) {
            const int kc = k0 + f * 16 + fn;
            const float4 ba = ld4(brow + (size_t)(q0 + fm) * H + kc);
            const float4 bb = ld4(brow + (size_t)(q0 + fm + 8) * H + kc);
            UNR for (short j = 0; j < 4; j++) { bl[f][j] = ba[j] * 1.44269504089f; bl[f][4 + j] = bb[j] * 1.44269504089f; }
        }

        // S = Q K^T (keys as B^T rows)
        metal::vec<float, 8> S[NKF];
        UNR for (int f = 0; f < NKF; f++) S[f] = 0.0f;
        UNR for (int db = 0; db < TD; db++) {
            metal::vec<half, 8> Kf[NKF];
            UNR for (int f = 0; f < NKF; f++) {
                const int c = HC + hh * C + db * 16 + fn;
                const vec<half, 4> va = ld4(hk + ka[f] + c), vb = ld4(hk + kb[f] + c);
                UNR for (short j = 0; j < 4; j++) { Kf[f][j] = va[j]; Kf[f][4 + j] = vb[j]; }
            }
            UNR for (int f = 0; f < NKF; f += 2)
                nax_mma<float, half, half, false, true, true>(S[f], S[f + 1], Qf[db], Kf[f], Kf[f + 1]);
        }

        // logits in the log2 domain, online softmax per fragment row (rows shared by lanes l, l^1, l^8, l^9)
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
                const float p = fast::exp2(S[f][r * 4 + j] - m_new);
                S[f][r * 4 + j] = p;
                ls += p;
            }
            l_r[r] = l_r[r] * corr + ls;   // per-lane partial, reduced over the 4 lanes of the row at the end
            m_r[r] = m_new;
            UNR for (int d = 0; d < TD; d++) UNR for (short j = 0; j < 4; j++) O[d][r * 4 + j] *= corr;
        }

        // O += P V (P = this lane's S elements as the A operand, V rows = keys)
        UNR for (int f = 0; f < NKF; f++) {
            metal::vec<PT, 8> P;
            UNR for (short i = 0; i < 8; i++) P[i] = (PT)S[f][i];
            UNR for (int dp = 0; dp < TD; dp += 2) {
                const int c = 2 * HC + hh * C + dp * 16 + fn;
                const vec<half, 4> a0 = ld4(hk + ka[f] + c), b0 = ld4(hk + kb[f] + c);
                const vec<half, 4> a1 = ld4(hk + ka[f] + c + 16), b1 = ld4(hk + kb[f] + c + 16);
                metal::vec<half, 8> V0, V1;
                UNR for (short j = 0; j < 4; j++) {
                    V0[j] = a0[j]; V0[4 + j] = b0[j];
                    V1[j] = a1[j]; V1[4 + j] = b1[j];
                }
                nax_mma<float, PT, half, false, false, true>(O[dp], O[dp + 1], P, V0, V1);
            }
        }
    }

    // epilogue: normalise, gate with sigmoid(g) (MLX's sigmoid), fp16 store [M, HC]
    UNR for (short r = 0; r < 2; r++) {
        float l = l_r[r];
        l += simd_shuffle_xor(l, (ushort)1);
        l += simd_shuffle_xor(l, (ushort)8);
        const float inv = 1.0f / l;
        const uint qi = q0 + fm + 8 * r;
        UNR for (int d = 0; d < TD; d++) {
            const int c = hh * C + d * 16 + fn;
            const vec<half, 4> g = ld4(hq + (size_t)qi * F + 3 * HC + c);
            vec<half, 4> o;
            UNR for (short j = 0; j < 4; j++) o[j] = half(O[d][r * 4 + j] * inv * mlx_sigmoid(float(g[j])));
            *(device vec<half, 4>*)(out + ((size_t)win * W + qi) * HC + c) = o;
        }
    }
"""

# (HPT heads per threadgroup, NKF 16-key fragments per online-softmax step (even), PT dtype of P, ORDER of windows)
ATT_CFGS = [
    dict(hpt=1, nkf=2),
    dict(hpt=1, nkf=4),
    dict(hpt=2, nkf=4),
    dict(hpt=4, nkf=4),
    dict(hpt=1, nkf=8),
    dict(hpt=4, nkf=8),
    dict(hpt=4, nkf=2),
    dict(hpt=4, nkf=4, pt="half"),
    dict(hpt=4, nkf=4, order=1),
    dict(hpt=1, nkf=4, order=1),
]


def atom_attention(h: mx.array, bias: mx.array, *, NA: int, W: int = 32, H: int = 128, heads: int = 4, c: int = 32,
                   qbias: mx.array | None = None, brep: int = 1, hpt: int = 4, nkf: int = 4,
                   pt: str = "float", order: int = 0) -> mx.array:
    """sigmoid(g) * softmax((q + qbias) k^T / sqrt(c) + bias) v over key windows, NAX. h [M, >= 4 * heads * c] fp16
    (q | k | v | g per atom, rows (sample, atom), M = samples * NA, NA % W == 0); bias [M / W / brep, heads, W, H] fp32
    additive (key mask folded in, -1e6 for keys outside the sequence / padded atoms), shared by `brep` consecutive
    samples; qbias [heads * c] fp32 or None. Returns [M, heads * c] fp16."""
    M, F = h.shape
    HC = heads * c
    assert h.dtype == mx.float16 and c % 32 == 0 and nkf % 2 == 0 and H % (16 * nkf) == 0 and W % 16 == 0
    assert (H - W) % 2 == 0 and NA % W == 0 and M % NA == 0 and heads % hpt == 0 and F >= 4 * HC
    NWS = NA // W
    nwin = M // W
    assert bias.shape == (nwin // brep, heads, W, H) and (M // NA) % brep == 0, (bias.shape, nwin, brep)
    bias = bias if bias.dtype == mx.float32 else bias.astype(mx.float32)
    key = ("att", W, H, heads, c, F, hpt, nkf, pt, order, qbias is not None)
    if key not in _kernels:
        _kernels[key] = _K.metal_kernel(
            name=f"boltz_atom_att_w{W}_h{H}_n{heads}_c{c}", input_names=["h", "bias", "qbias", "scale_l2e", "meta"],
            output_names=["out"], source=_ATT_SRC, header=_N3.HEADER)
    tg = (W // 16) * hpt * 32
    (o,) = _kernels[key](
        inputs=[h, bias, qbias.astype(mx.float32) if qbias is not None else mx.zeros((8,), mx.float32),
                mx.array([c ** -0.5 * LOG2E], mx.float32), mx.array([NA, NWS, brep, M // NA, 0, 0, 0, 0], mx.int32)],
        template=[("W", W), ("H", H), ("NH", heads), ("C", c), ("HC", HC), ("F", F), ("HPT", hpt), ("NKF", nkf),
                  ("PT", mx.float16 if pt == "half" else mx.float32), ("HAS_QB", int(qbias is not None)),
                  ("ORDER", int(order))],
        grid=(nwin * tg, heads // hpt, 1), threadgroup=(tg, 1, 1),
        output_shapes=[(M, HC)], output_dtypes=[mx.float16])
    return o


def window_keys(x: mx.array, NA: int, W: int = 32, H: int = 128) -> mx.array:
    """x [S * NA, D] (rows (sample, atom)) -> key windows [S * NA / W, H, D]: slot j of window k = atom k * W - (H - W)
    / 2 + j of the same sample, zero outside [0, NA) (= encodersv2.ToKeys on the same rows; reference / tests)."""
    D = x.shape[-1]
    S = x.shape[0] // NA
    NWS = NA // W
    a = (np.arange(NWS)[:, None] * W - (H - W) // 2 + np.arange(H)[None, :])          # [NWS, H] atom per slot
    valid = (a >= 0) & (a < NA)
    g = mx.take(x.reshape(S, NA, D), mx.array(np.clip(a, 0, NA - 1).reshape(-1).astype(np.uint32)), axis=1)
    g = mx.where(mx.array(valid.reshape(1, -1, 1)), g, mx.zeros((), x.dtype))
    return g.reshape(S * NWS, H, D)


def atom_attention_ref(h, bias, *, NA, W=32, H=128, heads=4, c=32, qbias=None, brep=1, **_):
    """MLX-op reference with the kernel's rounding points (fp16 Q after the q bias, fp32 logits / softmax / P V,
    fp16 output after the gate)."""
    f = mx.float32
    M = h.shape[0]
    HC = heads * c
    hh = h[:, : 4 * HC].astype(f)
    q = hh[:, :HC] + (qbias.astype(f) if qbias is not None else 0.0)
    q = q.astype(mx.float16).astype(f).reshape(M // W, W, heads, c)
    kv = window_keys(hh[:, HC: 3 * HC], NA, W, H)                                       # [nwin, H, 2 HC]
    k = kv[..., :HC].reshape(-1, H, heads, c)
    v = kv[..., HC:].reshape(-1, H, heads, c)
    g = hh[:, 3 * HC:].reshape(M // W, W, heads, c)
    b = bias if brep == 1 else mx.repeat(bias.reshape(-1, NA // W, heads, W, H), brep, axis=0)  # per input -> sample
    b = b.reshape(M // W, heads, W, H).astype(f)
    s = mx.einsum("wqhc,wkhc->whqk", q, k) * c ** -0.5 + b
    p = mx.softmax(s, axis=-1)
    o = mx.einsum("whqk,wkhc->wqhc", p, v) * mx.sigmoid(g)
    return o.reshape(M, HC).astype(mx.float16)


# ---------------------------------------------------------------------------------------------------------------
# Encoder aggregation atoms -> tokens: relu, segment sum in a fixed (ascending atom) order, weight 1 / (n_t + 1e-6)
# ---------------------------------------------------------------------------------------------------------------
_SEG_SRC = """
    const uint c4 = thread_position_in_grid.x;
    const uint t = thread_position_in_grid.y;
    const uint s = thread_position_in_grid.z;
    if (c4 >= D / 4) return;
    const int T = meta[0], NA = meta[1];
    const int p0 = off[t], p1 = off[t + 1];
    const float w = wt[t];
    const device float4* xs = (const device float4*)x + (size_t)s * NA * (D / 4) + c4;
    float4 acc = 0.0f;
    for (int p = p0; p < p1; p++) acc += metal::max(xs[(size_t)idx[p] * (D / 4)], 0.0f) * w;
    ((device float4*)out)[((size_t)s * T + t) * (D / 4) + c4] = acc;
"""


def token_segments(att: np.ndarray):
    """att [NA, T] one-hot atom -> token (rows of padded atoms are zero) -> (off [T + 1] int32, idx [P] int32 atoms of
    each token in ascending order, wt [T] fp32 = 1 / (n_t + 1e-6) as the MLX path computes it)."""
    att = np.asarray(att, np.float32)
    NA, T = att.shape
    valid = att.sum(1) > 0
    tok = np.where(valid, att.argmax(1), T)
    order = np.argsort(tok, kind="stable")
    order = order[tok[order] < T]
    counts = np.bincount(tok[valid], minlength=T)
    off = np.zeros(T + 1, np.int64)
    off[1:] = np.cumsum(counts)
    wt = (np.float32(1.0) / (att.sum(0).astype(np.float32) + np.float32(1e-6))).astype(np.float32)
    return off.astype(np.int32), order.astype(np.int32), wt


def segment_mean_relu(y: mx.array, seg) -> mx.array:
    """y [S, NA, D] fp32 -> [S, T, D]: a[s, t] = sum_{i in token t} relu(y[s, i]) * wt[t]. seg: mx arrays (off, idx, wt)."""
    off, idx, wt = seg
    S, NA, D = y.shape
    T = wt.shape[0]
    assert y.dtype == mx.float32 and D % 4 == 0
    key = ("seg", D)
    if key not in _kernels:
        _kernels[key] = _K.metal_kernel(name=f"boltz_atom_seg_{D}", input_names=["x", "off", "idx", "wt", "meta"],
                                             output_names=["out"], source=_SEG_SRC)
    tgx = min(D // 4, 256)
    (o,) = _kernels[key](inputs=[y, off, idx, wt, mx.array([T, NA, 0, 0, 0, 0, 0, 0], mx.int32)],
                         template=[("D", D)], grid=(D // 4, T, S), threadgroup=(tgx, 1, 1),
                         output_shapes=[(S, T, D)], output_dtypes=[mx.float32])
    return o


def segment_mean_relu_ref(y, seg):
    off, idx, wt = (np.asarray(x) for x in seg)
    yy = np.maximum(np.asarray(y, np.float32), 0)
    S, NA, D = yy.shape
    T = wt.shape[0]
    out = np.zeros((S, T, D), np.float32)
    for t in range(T):
        for p in range(off[t], off[t + 1]):
            out[:, t] += yy[:, idx[p]] * wt[t]
    return out


def atom_tokens(att: np.ndarray):
    """att [NA, T] -> (token index per atom [NA] uint32 (0 for padded atoms), valid [NA] bool) for the decoder's
    token -> atom broadcast."""
    att = np.asarray(att, np.float32)
    valid = att.sum(1) > 0
    return np.where(valid, att.argmax(1), 0).astype(np.uint32), valid


# ---------------------------------------------------------------------------------------------------------------
# Fused conditioned transition (A/B candidate, site "trans"): one kernel for
#     a + sigmoid(gate) * ( half(silu(t Wg^T) * (t Wx^T) * (t Wab^T)) Wba^T ),   t = half(sigmoid(sc) * LN(a) + sh)
# = adaln16 -> gemm16 ff (EPI_SWIGLU) -> gemm16 ba (EPI_GRESID) with the same rounding points (t and the hidden in
# fp16, fp32 accumulators, fp32 gated residual); only the LN statistics are summed in another order. One simdgroup
# owns TM rows: AdaLN of the whole 128-wide rows in registers (row statistics: lane partials -> xor 1 -> xor 8), the
# hidden in blocks of 32 units (up GEMM: 3 weight groups x 16 x 32 x 128; SwiGLU in the accumulator layout, which is
# the A-fragment layout of the down GEMM), output accumulators O [TM x 128] in registers. Weights read from device
# memory (L1-cached across the simdgroups of a core). Removes the t and hidden round trips and 2 launches per layer.
# ---------------------------------------------------------------------------------------------------------------
_TRANS_SRC = """
    constexpr int KF = D / 16;                        // k fragments of the up projection (the whole row)
    constexpr int TMF = TM / 16;                      // 16-row blocks per simdgroup
    constexpr int OF = D / 16;                        // output fragments
    const int M = meta[0];
    const uint ntok = (uint)meta[1], rep = (uint)meta[2], off1 = (uint)meta[3], off2 = (uint)meta[4];
    const uint lane = thread_index_in_simdgroup;
    const int tile = (int)(threadgroup_position_in_grid.x * NSG + simdgroup_index_in_threadgroup);
    const int ntiles = (M + TM - 1) / TM;
    if (tile >= ntiles) return;
    const int r0 = min(tile * TM, M - TM);            // last tile moved back inside (rewrites identical values)
    NAX_COORD(lane, fm, fn)

    // 1. AdaLN of this simdgroup's rows -> fp16 A fragments over the whole row (K = D); = adaln16 arithmetic
    metal::vec<half, 8> Af[TMF][KF];
    UNR for (int t = 0; t < TMF; t++) UNR for (short r = 0; r < 2; r++) {
        const uint row = (uint)(r0 + t * 16 + fm + 8 * r);
        const uint cr = ((row / ntok) / rep) * ntok + row % ntok;
        float4 v[KF];
        float s = 0.0f;
        UNR for (int kf = 0; kf < KF; kf++) {
            v[kf] = *(const device float4*)(a + (size_t)row * D + kf * 16 + fn);
            s += v[kf].x + v[kf].y + v[kf].z + v[kf].w;
        }
        s += simd_shuffle_xor(s, (ushort)1);
        s += simd_shuffle_xor(s, (ushort)8);
        const float mean = s / D;
        float ss = 0.0f;
        UNR for (int kf = 0; kf < KF; kf++) { const float4 d = v[kf] - mean; ss += dot(d, d); }
        ss += simd_shuffle_xor(ss, (ushort)1);
        ss += simd_shuffle_xor(ss, (ushort)8);
        const float rstd = metal::precise::rsqrt(ss / D + 1e-5f);
        const device half* hc = (const device half*)h1 + (size_t)cr * H1LD + off1;
        UNR for (int kf = 0; kf < KF; kf++) {
            const int c = kf * 16 + fn;
            const float4 sc = float4(*(const device half4*)(hc + c)) + *(const device float4*)(b1 + off1 + c);
            const float4 sh = float4(*(const device half4*)(hc + D + c)) + *(const device float4*)(b1 + off1 + D + c);
            UNR for (short j = 0; j < 4; j++)
                Af[t][kf][r * 4 + j] = half((v[kf][j] - mean) * rstd * mlx_sigmoid(sc[j]) + sh[j]);
        }
    }

    // 2. hidden in blocks of 32 units: up (x | gates | ab) -> half(silu(g) * x * ab) -> down, accumulated in O
    metal::vec<float, 8> O[TMF][OF];
    UNR for (int t = 0; t < TMF; t++) UNR for (int f = 0; f < OF; f++) O[t][f] = 0.0f;
    for (int hb = 0; hb < HID; hb += 32) {
        metal::vec<half, 8> Hs[TMF][2];
        {
            metal::vec<float, 8> X[TMF][2], G[TMF][2], AB[TMF][2];
            UNR for (int t = 0; t < TMF; t++) UNR for (int f = 0; f < 2; f++) { X[t][f] = 0.0f; G[t][f] = 0.0f; AB[t][f] = 0.0f; }
            UNR for (int kf = 0; kf < KF; kf++) {
                const device half* px = (const device half*)wff + (size_t)(hb + fm) * D + kf * 16 + fn;
                const device half* pg = px + (size_t)HID * D;
                const device half* pb = pg + (size_t)HID * D;
                const metal::vec<half, 8> x0 = frag2(px, px + 8 * D), x1 = frag2(px + 16 * D, px + 24 * D);
                const metal::vec<half, 8> g0 = frag2(pg, pg + 8 * D), g1 = frag2(pg + 16 * D, pg + 24 * D);
                const metal::vec<half, 8> e0 = frag2(pb, pb + 8 * D), e1 = frag2(pb + 16 * D, pb + 24 * D);
                UNR for (int t = 0; t < TMF; t++) {
                    nax_mma<float, half, half, false, true, true>(X[t][0], X[t][1], Af[t][kf], x0, x1);
                    nax_mma<float, half, half, false, true, true>(G[t][0], G[t][1], Af[t][kf], g0, g1);
                    nax_mma<float, half, half, false, true, true>(AB[t][0], AB[t][1], Af[t][kf], e0, e1);
                }
            }
            UNR for (int t = 0; t < TMF; t++) UNR for (int f = 0; f < 2; f++) UNR for (short i = 0; i < 8; i++) {
                const float gt = G[t][f][i];
                Hs[t][f][i] = half(gt * mlx_sigmoid(gt) * X[t][f][i] * AB[t][f][i]);
            }
        }
        UNR for (int kf2 = 0; kf2 < 2; kf2++) UNR for (int of = 0; of < OF; of += 2) {
            const device half* pw = (const device half*)wba + (size_t)(of * 16 + fm) * HID + hb + kf2 * 16 + fn;
            const metal::vec<half, 8> w0 = frag2(pw, pw + 8 * HID), w1 = frag2(pw + 16 * HID, pw + 24 * HID);
            UNR for (int t = 0; t < TMF; t++)
                nax_mma<float, half, half, false, true, true>(O[t][of], O[t][of + 1], Hs[t][kf2], w0, w1);
        }
    }

    // 3. gated residual (= EPI_GRESID): a + sigmoid(float(gate16) + gate bias) * O, fp32
    UNR for (int t = 0; t < TMF; t++) UNR for (short r = 0; r < 2; r++) {
        const uint row = (uint)(r0 + t * 16 + fm + 8 * r);
        const uint cr = ((row / ntok) / rep) * ntok + row % ntok;
        UNR for (int f = 0; f < OF; f++) {
            const int col = f * 16 + fn;
            const half4 gh = *(const device half4*)((const device half*)h2 + (size_t)cr * GLD + off2 + col);
            const float4 gb = *(const device float4*)(b2 + off2 + col);
            const float4 rs = *(const device float4*)(a + (size_t)row * D + col);
            float4 o;
            UNR for (short j = 0; j < 4; j++) o[j] = rs[j] + mlx_sigmoid(float(gh[j]) + gb[j]) * O[t][f][r * 4 + j];
            *(device float4*)(out + (size_t)row * D + col) = o;
        }
    }
"""

# (TM rows per simdgroup, NSG simdgroups per threadgroup)
TRANS_CFGS = [dict(tm=16, nsg=4), dict(tm=16, nsg=8), dict(tm=16, nsg=2), dict(tm=32, nsg=4)]


def atom_transition(a2: mx.array, rc, layer: int, w_ff: mx.array, w_ba: mx.array, *, tm: int = 16,
                    nsg: int = 4) -> mx.array:
    """Fused conditioned transition of layer `layer` (see _TRANS_SRC): a2 [M, D] fp32, rc kernels_nax3.RawCond16
    (AdaLN slots 2 / 3, gate slot 1 of the layer), w_ff [3 * HID, D] fp16 (x | gates | ab), w_ba [D, HID] fp16.
    Returns the new residual stream [M, D] fp32."""
    M, D = a2.shape
    HID = w_ba.shape[1]
    assert a2.dtype == mx.float32 and w_ff.dtype == mx.float16 and w_ba.dtype == mx.float16
    assert w_ff.shape == (3 * HID, D) and w_ba.shape == (D, HID) and D % 32 == 0 and HID % 32 == 0
    assert tm % 16 == 0 and M >= tm and rc.D == D
    key = ("trans", D, HID, tm, nsg)
    if key not in _kernels:
        _kernels[key] = _K.metal_kernel(
            name=f"boltz_atom_trans_d{D}_h{HID}", input_names=["a", "h1", "b1", "h2", "b2", "wff", "wba", "meta"],
            output_names=["out"], source=_TRANS_SRC, header=_N3.HEADER)
    ntg = ((M + tm - 1) // tm + nsg - 1) // nsg
    meta = mx.array([M, rc.ntok, rc.rep, (layer * 4 + 2) * D, (layer * 2 + 1) * D, 0, 0, 0], mx.int32)
    (y,) = _kernels[key](
        inputs=[a2, rc.h1, rc.b1, rc.h2, rc.b2, w_ff, w_ba, meta],
        template=[("D", D), ("HID", HID), ("TM", tm), ("NSG", nsg), ("H1LD", rc.L * 4 * D), ("GLD", rc.L * 2 * D)],
        grid=(ntg * nsg * 32, 1, 1), threadgroup=(nsg * 32, 1, 1),
        output_shapes=[(M, D)], output_dtypes=[mx.float32])
    return y


# AdaLN prologue + plain GEMM (A/B candidate, site "aqkvg"): t = AdaLN slot SLOT of the layer (the _TRANS_SRC
# prologue, verbatim) kept in registers, y = half(t W^T) [M, NOUT] in blocks of 32 output columns (= adaln16 -> gemm16
# EPI_STORE rounding). Replaces adaln16 + the qkvg GEMM of the attention half (t round trip, one launch).
_ADALN_GEMM_SRC = _TRANS_SRC.split("    // 2. hidden")[0] + """
    // 2. y = half(t W^T), 32 output columns at a time
    for (int nb = 0; nb < NOUT; nb += 32) {
        metal::vec<float, 8> C[TMF][2];
        UNR for (int t = 0; t < TMF; t++) { C[t][0] = 0.0f; C[t][1] = 0.0f; }
        UNR for (int kf = 0; kf < KF; kf++) {
            const device half* pw = (const device half*)w + (size_t)(nb + fm) * D + kf * 16 + fn;
            const metal::vec<half, 8> w0 = frag2(pw, pw + 8 * D), w1 = frag2(pw + 16 * D, pw + 24 * D);
            UNR for (int t = 0; t < TMF; t++)
                nax_mma<float, half, half, false, true, true>(C[t][0], C[t][1], Af[t][kf], w0, w1);
        }
        UNR for (int t = 0; t < TMF; t++) UNR for (short r = 0; r < 2; r++) {
            const int row = r0 + t * 16 + fm + 8 * r;
            UNR for (int f = 0; f < 2; f++) {
                half4 o;
                UNR for (short j = 0; j < 4; j++) o[j] = half(C[t][f][r * 4 + j]);
                *(device half4*)(out + (size_t)row * NOUT + nb + f * 16 + fn) = o;
            }
        }
    }
"""
ADALN_GEMM_CFGS = [dict(tm=16, nsg=4), dict(tm=16, nsg=8), dict(tm=32, nsg=4)]


def adaln_gemm(a2: mx.array, rc, layer: int, slot: int, w: mx.array, *, tm: int = 16, nsg: int = 4) -> mx.array:
    """half(AdaLN(a2) W^T): AdaLN from slots (slot, slot + 1) of layer `layer` in rc (as adaln16), w [NOUT, D] fp16."""
    M, D = a2.shape
    NOUT = w.shape[0]
    assert a2.dtype == mx.float32 and w.dtype == mx.float16 and w.shape[1] == D and NOUT % 32 == 0 and D % 16 == 0
    assert tm % 16 == 0 and M >= tm and rc.D == D
    key = ("adaln_gemm", D, NOUT, tm, nsg)
    if key not in _kernels:
        _kernels[key] = _K.metal_kernel(
            name=f"boltz_atom_adaln_gemm_d{D}_n{NOUT}", input_names=["a", "h1", "b1", "w", "meta"],
            output_names=["out"], source=_ADALN_GEMM_SRC, header=_N3.HEADER)
    ntg = ((M + tm - 1) // tm + nsg - 1) // nsg
    meta = mx.array([M, rc.ntok, rc.rep, (layer * 4 + slot) * D, 0, 0, 0, 0], mx.int32)
    (y,) = _kernels[key](
        inputs=[a2, rc.h1, rc.b1, w, meta],
        template=[("D", D), ("NOUT", NOUT), ("TM", tm), ("NSG", nsg), ("H1LD", rc.L * 4 * D)],
        grid=(ntg * nsg * 32, 1, 1), threadgroup=(nsg * 32, 1, 1),
        output_shapes=[(M, NOUT)], output_dtypes=[mx.float16])
    return y


def adaln_gemm_ref(a2, rc, layer, slot, w, **_):
    return _N3.gemm16_ref(_N3.adaln16_ref(a2, rc, layer, slot), w)


def atom_transition_ref(a2, rc, layer, w_ff, w_ba, **_):
    """The split composition the fused kernel replaces (MLX-op references of adaln16 / gemm16)."""
    D = a2.shape[1]
    t = _N3.adaln16_ref(a2, rc, layer, 2)
    hs = _N3.gemm16_ref(t, w_ff, epi=_N3.EPI_SWIGLU, ngrp=3)
    return _N3.gemm16_ref(hs, w_ba, epi=_N3.EPI_GRESID, gate=rc.h2, gate_bias=rc.b2, gate_off=(layer * 2 + 1) * D,
                          resid=a2, ntok=rc.ntok, rep=rc.rep)


# ---------------------------------------------------------------------------------------------------------------
# AtomTransformer forward (DiffusionTransformer layers of dim 128 / 4 heads, keys in windows)
# ---------------------------------------------------------------------------------------------------------------
# per GEMM site: a kernels_nax3 GEMM_CFGS / SWIGLU_CFGS entry or "mlx" (MLX fp16 GEMM + compiled elementwise epilogue);
# "trans": "split" (adaln16 -> ff -> ba as three kernels, sites ff / ba) or a TRANS_CFGS entry (fused kernel)
GEMM_CFG = {
    "qkvg": dict(sm=32, sn=32, wm=2, wn=2, ku=2),
    "o": dict(sm=32, sn=32, wm=2, wn=2, ku=2),
    "ff": dict(sm=32, sn=32, wm=2, wn=1, ku=1),
    "ba": dict(sm=32, sn=32, wm=2, wn=2, ku=2),
    "trans": "split",
    "aqkvg": "split",
}
ATT_CFG = dict(hpt=4, nkf=4)
_LISTS = {"qkvg": _N3.GEMM_CFGS, "o": _N3.GEMM_CFGS, "ff": _N3.SWIGLU_CFGS, "ba": _N3.GEMM_CFGS, "trans": TRANS_CFGS,
          "aqkvg": ADALN_GEMM_CFGS}
# Committed choices (tools/prof/atom_ab.py suggest --write); env BOLTZ_MLX_ATOM_CFG="qkvg=mlx,o=0,ff=2,ba=3,att=3,trans=0"
CFG_FILE = os.path.join(os.path.dirname(__file__), "atom_cfg.json")


# Size buckets (per-device configs; bitwise-neutral within a site's list, as kernels_nax3.SITE_BUCKETS): "o@1024=7" =
# for M = samples * atoms <= 1024 use entry 7 of the site's list; the first bucket (ascending) that holds M wins, else
# the plain entry; a plain "site=v" item clears the site's buckets. att buckets index ATT_CFGS.
BUCKETS: dict = {}


def apply_cfg(spec: str, gemm: dict | None = None, att: dict | None = None, buckets: dict | None = None) -> None:
    gemm = GEMM_CFG if gemm is None else gemm
    att = ATT_CFG if att is None else att
    buckets = BUCKETS if buckets is None else buckets
    for item in filter(None, spec.split(",")):
        name, v = item.split("=")
        name, _, mx_ = name.partition("@")
        if name == "att":
            cfg = dict(ATT_CFGS[int(v)])
        elif name in _LISTS:
            cfg = v if v in ("mlx", "split") else _LISTS[name][int(v)]
        else:
            continue
        if mx_:
            b = [x for x in buckets.get(name, []) if x[0] != int(mx_)] + [(int(mx_), cfg)]
            buckets[name] = sorted(b, key=lambda x: x[0])
            continue
        buckets.pop(name, None)
        if name == "att":
            att.clear()
            att.update(cfg)
        else:
            gemm[name] = cfg


def _pick(name: str, M: int, gemm: dict, att: dict, buckets: dict):
    for mx_, cfg in buckets.get(name, ()):
        if M <= mx_:
            return cfg
    return att if name == "att" else gemm.get(name, "split")


def site_cfg(name: str, M: int):
    """Config of atom site `name` (a GEMM_CFG key or 'att') for M rows: bucket, else the plain entry."""
    return _pick(name, M, GEMM_CFG, ATT_CFG, BUCKETS)


def resolve_entry(spec: str):
    """(site, M) -> config under an atom_cfg.json "atom" string applied to the module defaults, without touching the
    live tables (tests/test_cfg_bitwise.py)."""
    gemm, att, buckets = dict(_BASE_GEMM), dict(_BASE_ATT), {}
    apply_cfg(spec, gemm, att, buckets)
    return lambda name, M: _pick(name, M, gemm, att, buckets)


def _load_cfg() -> None:
    # this chip's entry of the per-device file, else "default" (one-line note when the MPP kernels are first used)
    apply_cfg(_K.device_cfg(CFG_FILE, "nax", "tools/prof/atom_ab.py && tools/prof/atom_ab.py suggest --write").get("atom", ""))
    apply_cfg(os.environ.get("BOLTZ_MLX_ATOM_CFG", ""))


_BASE_GEMM, _BASE_ATT = dict(GEMM_CFG), dict(ATT_CFG)   # module defaults (before any device entry)
_load_cfg()


def supported(tr, W: int, H: int) -> bool:
    """Layer shapes this path handles (the Boltz-2 atom transformers): dim 128, 4 heads of 32, hidden 256, no post-LN,
    windows W % 16 == 0, H % 64 == 0, H > W."""
    L0 = tr.layers[0]
    at = L0.pair_bias_attn
    return (at.c_s == 128 and at.num_heads == 4 and L0.transition.b_to_a.weight.shape == (128, 256)
            and not L0.post_layer_norm and tr.pair_bias_attn and W % 16 == 0 and H % 64 == 0 and H > W
            and (H - W) % 2 == 0)


def transformer_forward(tr, a2: mx.array, P: dict, layers=None) -> mx.array:
    """a2 [M, 128] fp32 (rows (sample, atom)) -> the AtomTransformer output [M, 128] fp32. P: AtomTransformer.prepare()
    output of the NAX path: cond (kernels_nax3.RawCond16 of c, rep = multiplicity), attn_bias [L] x [NWb, 4, W, H]
    fp32, attn_brep, NA / W / H."""
    from boltz_mlx import kernels as _K

    M, D = a2.shape
    rc, NA, W, H, brep = P["cond"], P["NA"], P["W"], P["H"], P["attn_brep"]
    heads = tr.layers[0].pair_bias_attn.num_heads
    c = {k: site_cfg(k, M) for k in ("aqkvg", "qkvg", "o", "trans", "ff", "ba", "att")}  # per-size choices (M fixed)
    for l in (range(len(tr.layers)) if layers is None else layers):
        layer = tr.layers[l]
        w_qkvg, w_ff, w_o, w_ba, b_q, _ = layer._fused()
        if c["aqkvg"] != "split" and M >= c["aqkvg"]["tm"]:
            h = adaln_gemm(a2, rc, l, 0, w_qkvg, **c["aqkvg"])
        else:
            t = _N3.adaln16(a2, rc, l, 0)
            h = _N3._gemm_site("qkvg", t, w_qkvg, cfg=c["qkvg"])
        _K.record(f"atom{D}.qkvg", h)
        o = atom_attention(h, P["attn_bias"][l], NA=NA, W=W, H=H, heads=heads, c=D // heads, qbias=b_q, brep=brep,
                           **c["att"])
        a2 = _N3._gemm_site("o", o, w_o, cfg=c["o"], epi=_N3.EPI_GRESID, gate=rc.h2, gate_bias=rc.b2,
                            gate_off=(l * 2) * D, resid=a2, ntok=rc.ntok, rep=rc.rep)
        if c["trans"] != "split" and M >= c["trans"]["tm"]:
            a2 = atom_transition(a2, rc, l, w_ff, w_ba, **c["trans"])
            continue
        t = _N3.adaln16(a2, rc, l, 2)
        hs = _N3._gemm_site("ff", t, w_ff, cfg=c["ff"], epi=_N3.EPI_SWIGLU, ngrp=3)
        a2 = _N3._gemm_site("ba", hs, w_ba, cfg=c["ba"], epi=_N3.EPI_GRESID, gate=rc.h2, gate_bias=rc.b2,
                            gate_off=(l * 2 + 1) * D, resid=a2, ntok=rc.ntok, rep=rc.rep)
    return a2
