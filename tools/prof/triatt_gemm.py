"""Rung 2N, candidate (c): triangle attention as two batched MLX fp16 GEMMs (NAX) with an fp32 softmax in between,
chunked over rows i to bound the [rows, H, J, K] score memory.

  per chunk of r rows:  S = Q K^T             fp16 GEMM (fp32 accumulate, fp16 scores)   [r, H, J, K]
                        P = softmax(S*scale + tri_bias + mask_bias) in fp32, P stored fp16 (one compiled kernel)
                        O = P V                fp16 GEMM                                  [r, H, J, C]
                        core = O * sigmoid(g)  (compiled, fp16 out)
Variants: rows r in {25, 50, 100, 200, 400}; `prescale` folds C^-0.5 into Q before the GEMM (fp16 rounding of q);
`p32` keeps P in fp32 and runs PV as an fp32 GEMM (TF32 off -> not NAX; accuracy reference for the P rounding).
Extra diagnostics: max |S| (fp16 range check of the scores; ACT_RANGES has no row for raw q.k).

  python tools/prof/triatt_gemm.py               # GPU, ~3 min
  python tools/prof/triatt_gemm.py --cpu-smoke   # logic / shapes on CPU at N=48
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from triatt_common import C, CPU_SMOKE, H, HC, K, Ctx, baseline_candidates, captures, emit, mx, run  # noqa: E402

EXP = "gemm2"
SCALE = C**-0.5


@mx.compile
def _softmax16(s, tri, mb):
    """s fp16 [r, H, J, K]; tri fp16 [1, H, J, K]; mb fp32 [r, 1, 1, K] -> fp16 P (softmax in fp32)."""
    a = s.astype(mx.float32) * SCALE + mb + tri.astype(mx.float32)
    return mx.softmax(a, axis=-1).astype(mx.float16)


@mx.compile
def _softmax16_pre(s, tri, mb):
    a = s.astype(mx.float32) + mb + tri.astype(mx.float32)
    return mx.softmax(a, axis=-1).astype(mx.float16)


@mx.compile
def _softmax32(s, tri, mb):
    a = s.astype(mx.float32) * SCALE + mb + tri.astype(mx.float32)
    return mx.softmax(a, axis=-1)


def gemm_core(h, mask, r, prescale=False, p32=False):
    B, I, J, F = h.shape
    assert B == 1
    tri = h[..., 4 * HC:4 * HC + H].transpose(0, 3, 1, 2)  # [1, H, Q, K]
    outs = []
    for i0 in range(0, I, r):
        rr = min(r, I - i0)
        hc = h[0, i0:i0 + rr]
        q, k, v = (hc[..., o:o + HC].reshape(rr, J, H, C).transpose(0, 2, 1, 3) for o in (0, HC, 2 * HC))
        if prescale:
            q = (q.astype(mx.float32) * SCALE).astype(mx.float16)
        s = q @ k.swapaxes(-1, -2)  # fp16 out
        mb = (1e9 * (mask[0, i0:i0 + rr] - 1.0))[:, None, None, :]
        if p32:
            o = (_softmax32(s, tri, mb) @ v.astype(mx.float32)).astype(mx.float16)
        else:
            p = (_softmax16_pre if prescale else _softmax16)(s, tri, mb)
            o = p @ v
        o = o.transpose(0, 2, 1, 3).reshape(1, rr, J, HC)
        outs.append(K._gate16(o, hc[None, ..., 3 * HC:4 * HC]))
    return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=1)


def cand(r, prescale=False, p32=False):
    def core_h(ctx, h, m):
        return gemm_core(h, m, min(r, ctx.I), prescale, p32)

    name = f"gemm2_R{r}" + ("_prescale" if prescale else "") + ("_p32" if p32 else "")
    return {"name": name, "cfg": {"rows": r, "prescale": prescale, "p32": p32},
            "core": lambda ctx, s: core_h(ctx, ctx.h16, ctx.mask), "core_h": core_h}


cands = baseline_candidates() + [cand(r) for r in (25, 50, 100, 200, 400)] + [cand(100, prescale=True),
                                                                             cand(100, p32=True)]
run(EXP, cands, case="prot400", calls=(0, 3, 7), time_calls=(0,), reps=5, extra_cases=("prot117",))

# score range diagnostic (fp16 scores: overflow above 65504; precision |s| * 2^-11)
fs = [None] if CPU_SMOKE else captures("prot400", (0, 3, 7))
for f in fs:
    ctx = Ctx(f)
    h = ctx.h16
    mx_s = 0.0
    for i0 in range(0, ctx.I, 100):
        hc = h[0, i0:i0 + 100]
        q, k = (hc[..., o:o + HC].reshape(-1, ctx.J, H, C).transpose(0, 2, 1, 3).astype(mx.float32) for o in (0, HC))
        mx_s = max(mx_s, float(mx.abs(q @ k.swapaxes(-1, -2)).max().item()))
    tri = h[..., 4 * HC:4 * HC + H]
    emit({"exp": EXP, "probe": "score_range", "input": ctx.name, "max_abs_qk_unscaled": mx_s,
          "max_abs_logit_scaled": mx_s * SCALE, "max_abs_tri_bias": float(mx.abs(tri).max().item())})
