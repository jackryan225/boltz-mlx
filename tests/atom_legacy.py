"""The shipped (BOLTZ_MLX_NAX_ATOM=0) f16 atom-attention core as an independent reference for kernels_atom, on the
production layouts (tests/test_atom_cpu.py with the MLX references, tests/test_atom_kernels.py with the Metal kernels).

Bias layouts (DiffusionTransformer.full_attn_bias output, [windows, heads, W, H] fp32, pair bias + key mask):
  * MLX path: per sample, SAMPLE-MAJOR — AtomTransformer._shape repeats the pair bias over the multiplicity on its
    sample axis before the windows are flattened, so row = sample * NW + window.
  * kernels_atom path: one block per input ([NW, heads, W, H] for one input), read with attn_brep = multiplicity:
    window w of sample s uses row (s // brep) * NW + w.
A per-sample copy for brep = 1 must therefore be sample-major (mx.repeat over the input axis of [Bin, NW, ...]);
mx.repeat(bias, S, axis=0) on the flattened [Bin * NW, ...] interleaves windows (w0 w0 w0 .. w1 ..) and is NOT the
production layout.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np


def per_sample(bias: mx.array, NA: int, S: int, W: int = 32) -> mx.array:
    """One input's bias block [NW, ...] -> the MLX path's per-sample layout [S * NW, ...] (sample-major)."""
    NW = NA // W
    return mx.repeat(bias.reshape(-1, NW, *bias.shape[1:]), S, axis=0).reshape(-1, *bias.shape[1:])


def legacy_core(h16: mx.array, attn_bias: mx.array, tkn, b_q: mx.array, *, W: int = 32, heads: int = 4) -> mx.array:
    """DiffusionTransformerLayer.fast_attn's attention core on the shipped f16 path: h16 [M, 4D] fp16 (the q|k|v|g GEMM
    output, cast to fp32 as mm_t returns it), keys through the windowed to_keys map, MLX SDPA in fp32 with the
    per-sample additive bias [M / W, heads, W, H], sigmoid(g) gate. Returns the fp32 operand of the output projection
    [M, D] (the shipped path rounds it to fp16 inside mm_t)."""
    h = h16.astype(mx.float32)
    M = h.shape[0]
    D = h.shape[1] // 4
    hd = D // heads
    h = h.reshape(M // W, W, 4 * D)
    q, k, v, g = h[..., :D] + b_q, h[..., D: 2 * D], h[..., 2 * D: 3 * D], h[..., 3 * D:]
    kv = tkn(mx.concatenate([k, v], axis=-1))
    k, v = kv[..., :D], kv[..., D:]
    B = q.shape[0]
    q = q.reshape(B, -1, heads, hd).transpose(0, 2, 1, 3)
    k = k.reshape(B, -1, heads, hd).transpose(0, 2, 1, 3)
    v = v.reshape(B, -1, heads, hd).transpose(0, 2, 1, 3)
    o = mx.fast.scaled_dot_product_attention(q, k, v, scale=hd ** -0.5, mask=attn_bias)
    o = o.transpose(0, 2, 1, 3).reshape(B, -1, D)
    return (mx.sigmoid(g) * o).reshape(M, D)


def production_inputs(NA: int, S: int, seed: int = 0, n_pad: int = 3):
    """A random-init Boltz-2 atom transformer (dim 128, 4 heads, 3 layers, 32 x 128 windows) and real-shaped inputs:
    returns dict(at, P (kernels_atom prepare_nax: one bias block per input, brep = S), P0 (the MLX path's prepare:
    per-sample biases / cond), tkn (the MLX path's key map), h (fp16 q|k|v|g [S * NA, 512]), b_q, q (atom activations
    [S, NA, 128]))."""
    from boltz_mlx.modules.encodersv2 import make_to_keys
    from boltz_mlx.modules.transformersv2 import AtomTransformer

    mx.random.seed(seed)
    at = AtomTransformer(attn_window_queries=32, attn_window_keys=128, dim=128, dim_single_cond=128, depth=3, heads=4)
    rng = np.random.default_rng(seed)
    for L in at.diffusion_transformer.layers:  # non-trivial conditioning and q bias
        for lin in (L.adaln.s_scale, L.transition.adaln.s_scale, L.output_projection_linear,
                    L.transition.output_projection[0]):
            lin.weight = mx.array(rng.normal(size=lin.weight.shape).astype(np.float32) * 0.05)
        L.pair_bias_attn.proj_q.bias = mx.array(rng.normal(size=128).astype(np.float32) * 0.3)
    c = mx.array(rng.normal(size=(1, NA, 128)).astype(np.float32))
    bias = mx.array(rng.normal(size=(1, NA // 32, 32, 128, 12)).astype(np.float32))
    mask = mx.array(np.r_[np.ones(NA - n_pad), np.zeros(n_pad)].astype(np.float32)[None])
    tk = make_to_keys(NA, 32, 128)
    c_rep, m_rep = mx.repeat(c, S, 0), mx.repeat(mask, S, 0)
    P0 = at.prepare((S, NA, 1), c_rep, bias, tk, m_rep, S)
    P = at.prepare_nax(c, bias, tk, mask, S)
    q = mx.array(rng.normal(size=(S, NA, 128)).astype(np.float32))
    tkn = at._shape(q, c_rep, bias, tk, m_rep, S)[3]
    h = mx.array((rng.normal(size=(S * NA, 512)) * 0.7).astype(np.float16))
    b_q = at.diffusion_transformer.layers[0].pair_bias_attn.proj_q.bias
    mx.eval(P0, h, q)
    return dict(at=at, P=P, P0=P0, tkn=tkn, h=h, b_q=b_q, q=q, c_rep=c_rep, m_rep=m_rep, bias=bias, tk=tk, S=S, NA=NA)
