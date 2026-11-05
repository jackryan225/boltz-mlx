"""Per-kernel A/B inside the diffusion token transformer (fp32): MLX ops vs the fused kernels of kernels_difftx.

Each part is chained over 24 distinct weight sets (weights stream from DRAM like in the real model) and compiled; the
chain glue (`x + 1e-3 * f(x)`, identical for both sides) is included in both timings. Alternating A/B, min over rounds.

  parts: adaln   LN * sigmoid(scale) + shift
         qkvg    t @ Wqkvg^T (+ q bias)
         attn    attention with additive pair bias + key mask, sigmoid(g) gate (from the projection output h)
         oproj   a + sigmoid(gate) * (o @ Wo^T)
         ff      t @ Wff^T -> silu(gates) * x * ab (GEMM1 + gating)
         ba      a + sigmoid(gate) * (hs @ Wba^T)

  MULT=1,5 ROUNDS=3 python tools/prof/ab_difftx_parts.py
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # this checkout, not an editable install elsewhere

import mlx.core as mx
import mlx.nn as nn
import numpy as np

import boltz_mlx  # noqa: F401
from boltz_mlx import kernels as K
from boltz_mlx import kernels_difftx as KD
from boltz_mlx.kernels import layer_norm

mx.set_default_device(mx.gpu)
N = int(os.environ.get("N", "117"))
MULTS = [int(m) for m in os.environ.get("MULT", "1,5").split(",")]
ROUNDS = int(os.environ.get("ROUNDS", "3"))
PARTS = os.environ.get("PARTS", "adaln,qkvg,attn,oproj,ff,ba").split(",")
NL, D, HEADS, HID = 24, 768, 16, 1536
rng = np.random.default_rng(0)


def rnd(shape, scale=1.0):
    return mx.array((rng.normal(size=shape) * scale).astype(np.float32))


W = {
    "qkvg": [rnd((4 * D, D), D**-0.5) for _ in range(NL)],
    "o": [rnd((D, D), D**-0.5) for _ in range(NL)],
    "ff": [rnd((3 * HID, D), D**-0.5) for _ in range(NL)],
    "ba": [rnd((D, HID), HID**-0.5) for _ in range(NL)],
}
bq = [mx.concatenate([rnd((D,), 0.1), mx.zeros((3 * D,))]) for _ in range(NL)]
PK = {k: [KD.pack_weights(w, 3 if k == "ff" else 1) for w in ws] for k, ws in W.items()}
mx.eval(W, bq, PK)


def build(B):
    M = B * N
    h1 = rnd((B, N, NL, 4, D))
    h2 = rnd((B, N, NL, 2, D))
    rc = KD.RawCond(h1, h2)
    six = [mx.sigmoid(h1[:, :, :, 0]), h1[:, :, :, 1], mx.sigmoid(h2[:, :, :, 0]),
           mx.sigmoid(h1[:, :, :, 2]), h1[:, :, :, 3], mx.sigmoid(h2[:, :, :, 1])]
    x = rnd((B, N, D))
    hqkvg = [rnd((B * N + 48, 4 * D)) for _ in range(4)]
    ab = [rnd((B, HEADS, N, N)) for _ in range(NL)]
    hs = [rnd((M + 48, HID)) for _ in range(4)]
    og = [rnd((M + 48, D)) for _ in range(4)]
    mx.eval(h1, h2, six, x, hqkvg, ab, hs, og)
    return M, rc, six, x, hqkvg, ab, hs, og


def make(part, impl, B, ctx):
    M, rc, six, x0, hqkvg, ab, hs, og = ctx
    sl = lambda i: [t[..., i, :] for t in six]

    def f_mlx(i, x):
        c = sl(i)
        if part == "adaln":
            return c[0] * layer_norm(x, None, None, 1e-5) + c[1]
        if part == "qkvg":
            return (x @ W["qkvg"][i].T + bq[i])[..., :D]
        if part == "attn":
            h = hqkvg[i % 4][:M].reshape(B, N, 4 * D)
            q, k, v, g = h[..., :D], h[..., D:2 * D], h[..., 2 * D:3 * D], h[..., 3 * D:]
            sh = lambda t: t.reshape(B, N, HEADS, D // HEADS).transpose(0, 2, 1, 3)
            o = mx.fast.scaled_dot_product_attention(sh(q), sh(k), sh(v), scale=(D // HEADS) ** -0.5, mask=ab[i])
            return mx.sigmoid(g) * o.transpose(0, 2, 1, 3).reshape(B, N, D)
        if part == "oproj":
            return x + c[2] * (og[i % 4][:M].reshape(B, N, D) @ W["o"][i].T)
        if part == "ff":
            hf = x @ W["ff"][i].T
            return (nn.silu(hf[..., HID:2 * HID]) * hf[..., :HID] * hf[..., 2 * HID:])[..., :D]
        if part == "ba":
            return x + c[5] * (hs[i % 4][:M].reshape(B, N, HID) @ W["ba"][i].T)

    def f_fused(i, x):
        x2 = x.reshape(M, D)
        if part == "adaln":
            return KD.adaln(x2, rc.h1, M=M, D=D, h1_ld=NL * 4 * D, meta=mx.array([M, N, 1, i * 4 * D], mx.uint32))[:M].reshape(B, N, D)
        if part == "qkvg":
            t = mx.concatenate([x2, mx.zeros((48, D))])
            return KD.gemm(t, PK["qkvg"][i], M=M, K=D, NOUT=4 * D, mode=0, bias=bq[i], cfg=KD._cfg("qkvg", M))[:M, :D].reshape(B, N, D)
        if part == "attn":
            return KD.attention(hqkvg[i % 4], ab[i], B=B, N=N, heads=HEADS, c=D // HEADS)[:M].reshape(B, N, D)
        if part == "oproj":
            return KD.gemm(og[i % 4], PK["o"][i], M=M, K=D, NOUT=D, mode=2, gate=rc.h2, gate_ld=NL * 2 * D, resid=x2,
                           meta=mx.array([M, N, 1, i * 2 * D], mx.uint32), cfg=KD._cfg("o", M))[:M].reshape(B, N, D)
        if part == "ff":
            t = mx.concatenate([x2, mx.zeros((48, D))])
            return KD.gemm(t, PK["ff"][i], M=M, K=D, NOUT=HID, NGRP=3, mode=1, cfg=KD._cfg("ff", M))[:M, :D].reshape(B, N, D)
        if part == "ba":
            return KD.gemm(hs[i % 4], PK["ba"][i], M=M, K=HID, NOUT=D, mode=2, gate=rc.h2, gate_ld=NL * 2 * D, resid=x2,
                           meta=mx.array([M, N, 1, (i * 2 + 1) * D], mx.uint32), cfg=KD._cfg("ba", M))[:M].reshape(B, N, D)

    fn = f_mlx if impl == "mlx" else f_fused

    def chain(x):
        for i in range(NL):
            x = x + 1e-3 * fn(i, x)
        return x
    return mx.compile(chain)


def bench(f, x, reps=10):
    mx.eval(f(x))
    t = time.perf_counter()
    for _ in range(reps):
        mx.eval(f(x))
    return (time.perf_counter() - t) / reps * 1e3


if __name__ == "__main__":
    print("boltz_mlx:", boltz_mlx.__file__)
    KD.DIFFTX_IMPL = "fused"
    for B in MULTS:
        ctx = build(B)
        x0 = ctx[3]
        for part in PARTS:
            fns = {i: make(part, i, B, ctx) for i in ("mlx", "fused")}
            ys = {i: fns[i](x0) for i in fns}
            mx.eval(ys)
            err = float(mx.abs(ys["fused"] - ys["mlx"]).max() / mx.abs(ys["mlx"]).max())
            res = {i: [] for i in fns}
            for _ in range(ROUNDS):
                for i in fns:
                    res[i].append(bench(fns[i], x0))
            tm, tf = min(res["mlx"]) / NL * 1e3, min(res["fused"]) / NL * 1e3
            print(f"mult={B} {part:6s} mlx {tm:7.1f} us/layer   fused {tf:7.1f} us/layer   {tm / tf:.2f}x   rel diff {err:.1e}", flush=True)
