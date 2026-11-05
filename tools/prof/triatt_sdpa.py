"""Rung 2N, candidate (b): mx.fast.scaled_dot_product_attention in fp16 as the triangle-attention core.

MLX 0.32.4 facts (from libmlx strings / mlx.metallib, not measured): the fused full-attention kernel supports head
dims {64, 72, 80, 96, 128, 192, 256} (+512 on NAX); Boltz tri-att has head dim 32, so a plain call takes the unfused
fallback (fp16 QK^T GEMM -> softmax -> fp16 PV GEMM, [rows, H, J, K] scores materialised). Additive masks must have
the query dtype (fp16) and broadcast to [B, H, Tq, Tk].

Candidates (core = sigmoid(g) * attention, fp16 [B, I, J, HC]):
  sdpa_d32_R<r>        : head dim 32 (fallback path), rows chunked by r; additive mask = tri bias [1, H, J, K] fp16
                         broadcast over the B*I batch (mask all ones), or per-chunk [r, H, J, K] = bias + key mask
  sdpa_d32_forced      : force_fused=True at head dim 32 (expected: error, recorded)
  sdpa_pad64_R<r>      : q/k/v zero-padded to head dim 64 *inside the projection* (padded weight rows, no copies) so
                         the fused (NAX on M5) attention kernel runs; 2x the attention FLOPs, force_fused=True
  sdpa_pad64_auto      : same without force_fused (MLX's own heuristic)
The module timing for pad64 includes the larger projection (900 instead of 516 output columns).

  python tools/prof/triatt_sdpa.py                       # GPU, ~3 min
  python tools/prof/triatt_sdpa.py --pad-env 1           # MLX_SDPA_PAD_HEAD_DIM=1 set before import: d32 forced
  python tools/prof/triatt_sdpa.py --cpu-smoke           # logic/shape check on CPU at N=48 (no timings that matter)
"""

import os
import sys

if "--pad-env" in sys.argv:
    os.environ["MLX_SDPA_PAD_HEAD_DIM"] = sys.argv[sys.argv.index("--pad-env") + 1]

from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from triatt_common import C, CPU_SMOKE, H, HC, K, baseline_candidates, mx, run  # noqa: E402

EXP = "sdpa" + (f"_padenv{os.environ['MLX_SDPA_PAD_HEAD_DIM']}" if "--pad-env" in sys.argv else "")
NEG = -1e4  # masked keys, added in half


def _addmask(tri, mask, i0, r, all_ones):
    """tri: [B, H, J, K] fp16 (B == 1); mask [B, I, K]. Returns an fp16 additive mask broadcastable to [r, H, J, K]."""
    if all_ones:
        return tri  # [1, H, J, K]: batch stride 0, never materialised per row
    mb = (NEG * (1.0 - mask[0, i0:i0 + r])).astype(mx.float16)  # [r, K]
    return tri + mb[:, None, None, :]


def sdpa_core(h, mask, r, all_ones, dh=C, force=False, layout="plain"):
    """h: [1, I, J, F]. layout plain: q|k|v|g|bias with head dim C; pad64: q|k|v with head dim 64 (zero upper half)."""
    B, I, J, F = h.shape
    assert B == 1
    if layout == "plain":
        offs, gofs, bofs = (0, HC, 2 * HC), 3 * HC, 4 * HC
    else:
        offs, gofs = (0, H * dh, 2 * H * dh), 3 * H * dh
        bofs = gofs + HC
    tri = h[..., bofs:bofs + H].transpose(0, 3, 1, 2)  # [1, H, Q, K] = h[0, q, k, h]
    outs = []
    for i0 in range(0, I, r):
        rr = min(r, I - i0)
        hc = h[0, i0:i0 + rr]
        q, k, v = (hc[..., o:o + H * dh].reshape(rr, J, H, dh).transpose(0, 2, 1, 3) for o in offs)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=C**-0.5, mask=_addmask(tri, mask, i0, rr, all_ones),
                                                 force_fused=force)
        if dh != C:
            o = o[..., :C]
        o = o.transpose(0, 2, 1, 3).reshape(1, rr, J, HC)
        outs.append(K._gate16(o, hc[None, ..., gofs:gofs + HC]))
    return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=1)


def pad64_weights(ctx):
    """[3*H*64 + HC + H, 128] fp16: per head [W_h (32 rows); zeros (32 rows)] for q, k, v, then g, tri bias."""
    w = ctx.w32
    blocks = []
    for t in range(3):
        wt = w[t * HC:(t + 1) * HC].reshape(H, C, -1)
        blocks.append(mx.concatenate([wt, mx.zeros_like(wt)], axis=1).reshape(H * 2 * C, -1))
    blocks += [w[3 * HC:]]
    return mx.concatenate(blocks, 0).astype(mx.float16)


def d32(r, force=False):
    def core_h(ctx, h, m):
        return sdpa_core(h, m, min(r, ctx.I), ctx.mask_all_ones, force=force)

    return {"name": f"sdpa_d32_R{r}" + ("_forced" if force else ""), "cfg": {"rows": r, "dh": 32, "force": force},
            "core": lambda ctx, s: core_h(ctx, ctx.h16, ctx.mask), "core_h": core_h}


def pad64(r, force=True):
    def prep(ctx):
        wp = getattr(ctx, "_wp", None)
        if wp is None:
            wp = ctx._wp = pad64_weights(ctx)
        hp = ctx.x16 @ wp.T
        mx.eval(hp)
        return hp

    def module(ctx):
        x16 = ctx.mod.layer_norm.f16(ctx.x)
        hp = x16 @ ctx._wp.T
        return ctx.finish(sdpa_core(hp, ctx.mask, min(r, ctx.I), ctx.mask_all_ones, dh=64, force=force, layout="pad64"))

    return {"name": f"sdpa_pad64_R{r}" + ("" if force else "_auto"), "cfg": {"rows": r, "dh": 64, "force": force},
            "prep": prep,
            "core": lambda ctx, hp: sdpa_core(hp, ctx.mask, min(r, ctx.I), ctx.mask_all_ones, dh=64, force=force,
                                              layout="pad64"),
            "module": module}


if "--pad-env" in sys.argv:
    cands = [d32(400, force=True), d32(100, force=True), d32(400)]
else:
    cands = baseline_candidates() + [d32(400), d32(100), d32(50), d32(400, force=True),
                                     pad64(400), pad64(200), pad64(100), pad64(400, force=False)]
if CPU_SMOKE:
    cands = [c for c in cands if not c["cfg"].get("force")]  # no fused kernels on CPU
run(EXP, cands, case="prot400", calls=(0, 3, 7), time_calls=(0,), reps=5, extra_cases=("prot117",))
