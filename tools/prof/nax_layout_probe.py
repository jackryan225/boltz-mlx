"""NAX follow-up, task 1: why does the NAX tri-att kernel with relax=False give wrong output?

  python tools/prof/nax_layout_probe.py        # GPU, < 1 min; rows -> logs/nax2.jsonl

Hypothesis: the register kernels move MLX steel fragments (BaseNAXFrag layout) into MPP cooperative tensors element
by element. MPP's cooperative-tensor layout is implementation defined per descriptor and element types; MLX only uses
matmul2d(16, 32, 16, *, *, relaxed_precision=true). With relaxed_precision=false the layout differs, so the operands
are scrambled (not an arithmetic error).
Rows:
  layout   : capacity and per-lane coordinates (get_multidimensional_index) of left / right / destination for
             relax in {1, 0} x the operand combos of tri_attention_nax; match_mlx_* = equals the MLX fragment layout
  mma_mlx  : one 16x32x16 product through MLX-layout fragments (what the kernels do), relax 1/0 -> max_rel
  mma_api  : the same product through the layout-agnostic MPP API (tensor views, coop load/store) -> max_rel
  float_op : what relaxed_precision does to *float* left operands (tri_attention_nax feeds P as float):
             error of the product vs exact / A rounded to half (RNE) / A truncated to half / A in bf16 / A in tf32
  triatt   : tri_attention_nax relax 1 / 0 vs the fp32 MMA kernel on the same fp16 h (N = 64, 117)
Expected if the hypothesis holds: layout relax=1 match_mlx true, relax=0 false; mma_mlx relax=0 fails; mma_api
passes for both; triatt relax=0 fails like nax_norelax (rel ~1).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import triatt_common as tc  # noqa: E402
from triatt_common import ROOT, emit, mx, np  # noqa: E402

from boltz_mlx import kernels as K  # noqa: E402
from boltz_mlx import kernels_nax as KN  # noqa: E402

tc.OUT = ROOT / "logs" / "nax2.jsonl"
EXP = "nax_layout"
rng = np.random.default_rng(0)
exp_frag = KN.mlx_frag_coords()  # [32, 8, (row, col)]

# 1. layouts
for relax in (True, False):
    for tb, at in ((True, mx.float16), (False, mx.float16), (False, mx.float32)):
        row = {"exp": EXP, "probe": "layout", "relax": relax, "tb": tb, "a_dtype": str(at)}
        try:
            caps, coords = KN.nax_layout(relax, False, tb, at, mx.float16, mx.float32)
            caps, coords = np.array(caps), np.array(coords)
            row["caps"] = {k: sorted(set(caps[i].tolist())) for i, k in enumerate(("left", "right", "dest"))}
            left, dest = coords[0, :, :8], coords[2, :, :16]
            row["match_mlx_left"] = bool((left[..., ::-1] == exp_frag).all() or (left == exp_frag).all())
            row["match_mlx_dest"] = bool((dest[:, :8, ::-1] == exp_frag).all()
                                         and (dest[:, 8:, ::-1] == exp_frag + np.array([0, 16])).all())
            row["lanes0_3"] = {k: coords[i, :4, :int(caps[i].max())].tolist()
                               for i, k in enumerate(("left", "right", "dest"))}
            row["coords"] = coords.tolist()  # full dump (idx[0], idx[1]); -1 = unused
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {str(e)[:3000]}"
        emit(row)


def _rel(c, ref):
    return float(np.abs(np.asarray(c, np.float64) - ref).max() / np.abs(ref).max())


# 2. one product, MLX-layout fragments vs MPP API
for relax in (True, False):
    for tb in (True, False):
        for at in (mx.float16, mx.float32):
            a = rng.standard_normal((16, 16)).astype(np.float16)
            b = rng.standard_normal((32, 16) if tb else (16, 32)).astype(np.float16)
            ref = a.astype(np.float64) @ (b.T if tb else b).astype(np.float64)
            for probe, fn in (("mma_mlx", lambda: KN.nax_mma_once(mx.array(a.astype(np.float32)),
                                                                   mx.array(b.astype(np.float32)), tb, at, relax)),
                              ("mma_api", lambda: KN.nax_mma_generic(mx.array(a).astype(at), mx.array(b), tb, relax))):
                row = {"exp": EXP, "probe": probe, "relax": relax, "tb": tb, "a_dtype": str(at)}
                try:
                    row["max_rel"] = _rel(np.array(fn()), ref)
                    row["pass"] = row["max_rel"] < 1e-3
                except Exception as e:
                    row["error"] = f"{type(e).__name__}: {str(e)[:3000]}"
                emit(row)


# 3. float left operands under relaxed precision
def _round_mantissa(x, bits):
    """Round fp32 x to `bits` explicit mantissa bits (RNE), fp32 exponent range."""
    xi = x.astype(np.float32).view(np.uint32).astype(np.uint64)
    drop = 23 - bits
    half = np.uint64(1) << np.uint64(drop - 1)
    lsb = (xi >> np.uint64(drop)) & np.uint64(1)
    xi = ((xi + half - np.uint64(1) + lsb) >> np.uint64(drop)) << np.uint64(drop)
    return xi.astype(np.uint32).view(np.float32)


def _trunc_half(x):
    h = x.astype(np.float16)
    over = np.abs(h.astype(np.float32)) > np.abs(x)
    h2 = np.nextafter(h, np.float16(0)).astype(np.float16)
    return np.where(over, h2, h).astype(np.float32)


a32 = (rng.standard_normal((16, 16)) * (1 + rng.random((16, 16)) * 2.0 ** -11)).astype(np.float32)
a32 += (rng.standard_normal((16, 16)) * 2.0 ** -13).astype(np.float32)   # bits below fp16 precision
b16 = rng.standard_normal((16, 32)).astype(np.float16)
bb = b16.astype(np.float64)
cands = {"exact": a32.astype(np.float64), "a_half_rne": a32.astype(np.float16).astype(np.float64),
         "a_half_trunc": _trunc_half(a32).astype(np.float64), "a_bf16_rne": _round_mantissa(a32, 7).astype(np.float64),
         "a_tf32_rne": _round_mantissa(a32, 10).astype(np.float64)}
for relax in (True, False):
    for probe, fn in (("float_op_mlx", lambda: KN.nax_mma_once(mx.array(a32), mx.array(b16.astype(np.float32)), False,
                                                               mx.float32, relax)),
                      ("float_op_api", lambda: KN.nax_mma_generic(mx.array(a32), mx.array(b16), False, relax))):
        if probe == "float_op_mlx" and not relax:
            continue  # layout-scrambled, see mma_mlx
        row = {"exp": EXP, "probe": probe, "relax": relax}
        try:
            c = np.array(fn()).astype(np.float64)
            row["max_abs_vs"] = {k: float(np.abs(c - v @ bb).max()) for k, v in cands.items()}
            row["best_match"] = min(row["max_abs_vs"], key=row["max_abs_vs"].get)
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {str(e)[:3000]}"
        emit(row)

# 4. the flash kernel itself
H, C = 4, 32
for N in (64, 117):
    h = mx.array(rng.normal(size=(1, N, N, 4 * H * C + H)).astype(np.float16))
    ref = K.tri_attention_fused(h.astype(mx.float32), H, C, None, 1e9, impl="mma")
    for relax in (True, False):
        row = {"exp": EXP, "probe": "triatt", "N": N, "relax": relax}
        try:
            out = KN.tri_attention_nax(h, H, C, None, 1e9, relax=relax, allow_unverified_layout=True)
            row.update(tc.metrics(out, ref))
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {str(e)[:3000]}"
        emit(row)
