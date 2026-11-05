"""A/B of the diffusion token-transformer implementations (BOLTZ_MLX_DIFFTX_IMPL mlx | fused).

Builds 24 random-weight DiffusionTransformerLayers (dim 768, 16 heads; distinct weights per layer so weights stream
from DRAM like the real model) and times chains of the whole layer / attention half / transition half, compiled
like the production denoise graph. Alternating A/B, min over rounds. GPU only; one process at a time.

  python tools/prof/ab_difftx.py                      # N=117 tokens, mult 1, fp32
  MULT=1,5 DT=fp32,fp16 PARTS=layer,attn,trans python tools/prof/ab_difftx.py
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # this checkout, not an editable install elsewhere

import mlx.core as mx
import numpy as np

import boltz_mlx  # noqa: F401
from boltz_mlx import kernels as K
from boltz_mlx import kernels_difftx as KD
from boltz_mlx.modules.transformersv2 import DiffusionTransformerLayer

mx.set_default_device(mx.gpu)
N = int(os.environ.get("N", "117"))
MULTS = [int(m) for m in os.environ.get("MULT", "1").split(",")]
DTS = os.environ.get("DT", "fp32").split(",")
PARTS = os.environ.get("PARTS", "layer,attn,trans").split(",")
IMPLS = os.environ.get("IMPLS", "mlx,fused").split(",")
REPS = int(os.environ.get("REPS", "10"))
ROUNDS = int(os.environ.get("ROUNDS", "3"))
NL, D, HEADS = int(os.environ.get("LAYERS", "24")), 768, 16
COMPILE = os.environ.get("COMPILE", "1") == "1"


def rnd(shape, scale, rng):
    return mx.array((rng.normal(size=shape) * scale).astype(np.float32))


def sigm(shape, rng):
    return mx.array((1 / (1 + np.exp(-rng.normal(size=shape)))).astype(np.float32))


def build_layers(rng):
    layers = []
    for _ in range(NL):
        L = DiffusionTransformerLayer(HEADS, D, D)
        at, tr = L.pair_bias_attn, L.transition
        for lin in (at.proj_q, at.proj_k, at.proj_v, at.proj_g, at.proj_o, tr.swish_gate[0], tr.a_to_b, tr.b_to_a):
            lin.weight = rnd(lin.weight.shape, lin.weight.shape[1] ** -0.5, rng)
        at.proj_q.bias = rnd(at.proj_q.bias.shape, 0.1, rng)
        mx.eval(L.parameters())
        layers.append(L)
    return layers


def build_inputs(B, rng, keep_frac=0.9):
    a = rnd((B, N, D), 1.0, rng)
    # raw (pre-sigmoid) conditioning of all layers, as precompute_cond(raw=True) produces it
    h1 = rnd((B, N, NL, 4, D), 1.0, rng)
    h2 = rnd((B, N, NL, 2, D), 1.0, rng)
    rc = KD.RawCond(h1, h2)
    # mlx impl: the 6 sigmoid-applied (B, N, L, D) arrays of precompute_cond, sliced per layer (as in production)
    six = [mx.sigmoid(h1[:, :, :, 0]), h1[:, :, :, 1], mx.sigmoid(h2[:, :, :, 0]),
           mx.sigmoid(h1[:, :, :, 2]), h1[:, :, :, 3], mx.sigmoid(h2[:, :, :, 1])]
    mx.eval(six)
    conds = {"mlx": [[x[..., i, :] for x in six] for i in range(NL)], "fused": [rc.layer(i) for i in range(NL)]}
    mask = (rng.uniform(size=(B, N)) < keep_frac).astype(np.float32)
    mask[:, 0] = 1.0
    abias = [rnd((B, HEADS, N, N), 1.0, rng) + (1 - mx.array(mask)[:, None, None, :]) * -1e6 for _ in range(NL)]
    mx.eval(a, rc.h1, rc.h2, abias)
    return a, conds, abias


def make_fn(layers, conds, abias, part):
    def f(a):
        for L, c, ab in zip(layers, conds, abias):
            if part in ("layer", "attn"):
                a = L.fast_attn(a, c, None, ab)
            if part in ("layer", "trans"):
                a = L.fast_transition(a, c)
        return a
    return mx.compile(f) if COMPILE else f


def bench(f, a, reps):
    mx.eval(f(a))
    t = time.perf_counter()
    for _ in range(reps):
        mx.eval(f(a))
    return (time.perf_counter() - t) / reps * 1e3


if __name__ == "__main__":
    print("boltz_mlx:", boltz_mlx.__file__)
    rng = np.random.default_rng(0)
    layers = build_layers(rng)
    for dt in DTS:
        K.set_diff_dtype(dt)
        for L in layers:
            L._fw = None
        for mult in MULTS:
            a, conds, abias = build_inputs(mult, rng)
            for part in PARTS:
                fns = {}
                for impl in IMPLS:
                    KD.DIFFTX_IMPL = impl
                    fns[impl] = make_fn(layers, conds[impl if impl in conds else 'mlx'], abias, part)
                    mx.eval(fns[impl](a))   # trace + compile under this impl
                res = {i: [] for i in IMPLS}
                for _ in range(ROUNDS):
                    for impl in IMPLS:
                        KD.DIFFTX_IMPL = impl
                        res[impl].append(bench(fns[impl], a, REPS))
                ref = None
                for impl in IMPLS:
                    KD.DIFFTX_IMPL = impl
                    y = fns[impl](a)
                    mx.eval(y)
                    ref = y if ref is None else ref
                    err = float(mx.abs(y - ref).max() / mx.abs(ref).max())
                    t = min(res[impl])
                    print(f"{dt} mult={mult} {part:5s} {impl:6s} {t:8.3f} ms (x{NL} layers, {t/NL*1e3:7.1f} us/layer)  "
                          f"{min(res[IMPLS[0]]) / t:.2f}x vs {IMPLS[0]}  rel diff {err:.1e}", flush=True)
    KD.DIFFTX_IMPL = "mlx"
