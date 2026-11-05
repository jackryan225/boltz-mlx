"""NAX follow-up, task 4: diffusion-transformer attention (token transformer: D=768, 16 heads, head dim 48, pair
bias) on NAX (kernels_nax2.attention_nax, derived from the validated NAX tri-att flash kernel) vs what --precision f16
runs today.

  python tools/prof/diffattn_nax.py          # GPU, ~2 min; rows -> logs/nax2.jsonl

What f16 runs today (DiffusionTransformerLayer.fast_attn with DIFF_DTYPE fp16; the fused difftx kernels only run
with BOLTZ_MLX_DIFFTX_FP16=1): AdaLN -> fp16 GEMM (mm_t, fp32 result) -> + q bias -> MLX SDPA in fp32 (head dim 48:
no fused kernel) -> sigmoid(g) * o -> fp16 GEMM -> a + gate * o.
Candidate: AdaLN -> fp16 GEMM with fp16 output -> NAX flash kernel (q bias in fp32, scores fp32, gate fused,
fp16 out) -> fp16 GEMM -> a + gate * o.
There are no L1 captures for the diffusion transformer, so inputs are recorded here: the model runs prot400 / prot117
in --precision f16 (1 trunk pass, 2 sampling steps, replayed seed) and the first token-transformer fast_attn calls'
(layer, a, cond, attn_bias) are captured.
Rows per captured call: update (= out_gate * attention, the layer's residual increment) of each candidate vs the same
call in strict fp32 (DIFF_DTYPE fp32: fp32 GEMMs + fp32 SDPA) -> rel_fro / cosine; timing on prot400 call 0:
block_ms (whole fast_attn), core_ms (SDPA + gating vs NAX kernel), peak MB; B = 5 (affinity samples) timing.
Integration gate (diffusion is covered by L2/L3, not L1): update rel_fro vs fp32 <= the shipped f16 path's x 1.5
and <= 3e-3, no non-finite; then L2/L3.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import triatt_common as tc  # noqa: E402
from triatt_common import DATA, ROOT, bench, emit, metrics, mx, np, peak_mb  # noqa: E402

from boltz_mlx import kernels as K  # noqa: E402
from boltz_mlx import kernels_nax2 as KN2  # noqa: E402
from boltz_mlx.modules import transformersv2 as TV  # noqa: E402

tc.OUT = ROOT / "logs" / "nax2.jsonl"
EXP = "nax_diffattn"
CFGS = [dict(wq=4, nkf=2, order=1), dict(wq=2, nkf=2, order=1), dict(wq=5, nkf=2, order=1),
        dict(wq=4, nkf=4, order=1), dict(wq=4, nkf=2, order=0)]
LayerCls = next(c for c in vars(TV).values() if isinstance(c, type) and "fast_attn" in c.__dict__)


def capture(case: str, n_calls: int = 3):
    from boltz_mlx.model import PredictArgs, load_model
    from boltz_mlx.sampler import record_noise

    K.PRECISION = "f16"
    K.set_diff_dtype("fp16")
    model = load_model("conf", compile_diffusion=False)
    z = np.load(DATA / case / "feats.npz")
    feats = {k: mx.array(z[k]) for k in z.files}
    rec = []
    orig = LayerCls.fast_attn

    def wrap(self, a, cond, to_keys, attn_bias):
        if len(rec) < n_calls and to_keys is None and a.shape[-1] == 768 and self.pair_bias_attn.num_heads == 16:
            c = cond.arrays() if hasattr(cond, "arrays") else cond
            mx.eval(a, attn_bias, *c[:3])
            rec.append({"layer": self, "a": a, "cond": tuple(c[:3]), "attn_bias": attn_bias})
        return orig(self, a, cond, to_keys, attn_bias)

    LayerCls.fast_attn = wrap
    try:
        model(feats, PredictArgs(0, 2, 1), record_noise(0, case, "structure", 1))
    finally:
        LayerCls.fast_attn = orig
    return rec


def shipped(layer, a, cond, attn_bias):
    return LayerCls.fast_attn(layer, a, cond, None, attn_bias)


def nax_block(layer, a, cond, attn_bias, prep, cfg):
    attn_scale, attn_shift, out_gate = cond
    B, N, D = a.shape
    w_qkvg, _, w_o, _, b_q, _ = layer._fused()  # fp16 weights (DIFF_DTYPE fp16)
    b = attn_scale * TV._ln_kernel(a, None, None, 1e-5) + attn_shift
    h16 = b.astype(mx.float16) @ w_qkvg.T
    o16 = KN2.attention_nax(h16.reshape(B * N, 4 * D), attn_bias, B=B, N=N, heads=16, c=D // 16, qbias=b_q,
                            prepared=prep, **cfg)
    o = (o16 @ w_o.T).astype(mx.float32).reshape(B, N, D)
    return a + out_gate * o


def sdpa_core(layer, a, cond, attn_bias):
    """The part of the shipped block the NAX kernel replaces (q bias, head split, SDPA, merge, gate)."""
    attn_scale, attn_shift, _ = cond
    B, N, D = a.shape
    w_qkvg, _, _, _, b_q, _ = layer._fused()
    h = K.mm_t(attn_scale * TV._ln_kernel(a, None, None, 1e-5) + attn_shift, w_qkvg)
    mx.eval(h)

    def core():
        q, k, v, g = h[..., :D] + b_q, h[..., D:2 * D], h[..., 2 * D:3 * D], h[..., 3 * D:]
        q, k, v = (t.reshape(B, -1, 16, D // 16).transpose(0, 2, 1, 3) for t in (q, k, v))
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=(D // 16) ** -0.5, mask=attn_bias)
        return mx.sigmoid(g) * o.transpose(0, 2, 1, 3).reshape(B, -1, D)

    return core


for case in ("prot400", "prot117"):
    try:
        calls = capture(case)
    except Exception as e:
        emit({"exp": EXP, "case": case, "error": f"capture: {type(e).__name__}: {str(e)[:2000]}"})
        continue
    for ci, r in enumerate(calls):
        layer, a, cond, ab = r["layer"], r["a"], r["cond"], r["attn_bias"]
        B, N, D = a.shape
        base = {"exp": EXP, "input": f"{case}/fast_attn_call{ci}", "B": B, "N": N,
                "bias_min": float(mx.min(ab).item()), "bias_max": float(mx.max(ab).item())}
        K.set_diff_dtype("fp32")
        upd32 = shipped(layer, a, cond, ab) - a
        mx.eval(upd32)
        K.set_diff_dtype("fp16")
        timed = case == "prot400" and ci == 0
        a16 = shipped(layer, a, cond, ab)
        mx.eval(a16)
        row = {**base, "cand": "f16_shipped_sdpa", "update_vs_fp32": metrics(a16 - a, upd32),
               "max_abs_update": float(mx.max(mx.abs(upd32)).item())}
        if timed:
            fn = lambda: shipped(layer, a, cond, ab)  # noqa: E731
            row["block_ms"] = round(bench(fn, 20), 4)
            row["block_peak_mb"] = round(peak_mb(fn), 1)
            core = sdpa_core(layer, a, cond, ab)
            row["core_ms"] = round(bench(core, 20), 4)
        emit(row)
        for cfg in CFGS:
            row = {**base, "cand": "nax_wq{wq}_nkf{nkf}_o{order}".format(**cfg), "cfg": cfg}
            try:
                prep = KN2.attention_nax_bias(ab, kp_mult=max(64, 16 * cfg["nkf"]))
                mx.eval(prep[0])
                an = nax_block(layer, a, cond, ab, prep, cfg)
                mx.eval(an)
                row["update_vs_fp32"] = metrics(an - a, upd32)
                row["update_vs_f16_shipped"] = metrics(an - a, a16 - a)
                if timed:
                    fn = lambda c_=cfg, p=prep: nax_block(layer, a, cond, ab, p, c_)  # noqa: E731
                    row["block_ms"] = round(bench(fn, 20), 4)
                    row["block_peak_mb"] = round(peak_mb(fn), 1)
                    w_qkvg, _, _, _, b_q, _ = layer._fused()
                    h16 = (cond[0] * TV._ln_kernel(a, None, None, 1e-5) + cond[1]).astype(mx.float16) @ w_qkvg.T
                    h16 = h16.reshape(B * N, 4 * D)
                    mx.eval(h16)
                    row["core_ms"] = round(bench(lambda c_=cfg, p=prep: KN2.attention_nax(
                        h16, ab, B=B, N=N, heads=16, c=D // 16, qbias=b_q, prepared=p, **c_), 20), 4)
                    row["core_tflops"] = round(4 * B * 16 * N * N * (D // 16) / row["core_ms"] / 1e9, 2)
                    row["bias_pad_ms"] = round(bench(lambda c_=cfg: KN2.attention_nax_bias(
                        ab, kp_mult=max(64, 16 * c_["nkf"]))[0], 10), 4)
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {str(e)[:2000]}"
            emit(row)
        if timed:  # 5 samples (affinity pass / multi-sample runs): tiled a and bias
            a5 = mx.concatenate([a] * 5, 0)
            ab5 = mx.concatenate([ab] * 5, 0)
            c5 = tuple(mx.concatenate([x] * 5, 0) if x.shape[0] == B else x for x in cond)
            mx.eval(a5, ab5, *c5)
            fn = lambda: shipped(layer, a5, c5, ab5)  # noqa: E731
            emit({**base, "cand": "f16_shipped_sdpa_B5", "block_ms": round(bench(fn, 10), 4)})
            prep5 = KN2.attention_nax_bias(ab5)
            fn = lambda: nax_block(layer, a5, c5, ab5, prep5, CFGS[0])  # noqa: E731
            o5 = fn()
            emit({**base, "cand": "nax_B5", "block_ms": round(bench(fn, 10), 4),
                  "sample0_vs_B1": metrics(o5[:B], nax_block(layer, a, cond, ab, None, CFGS[0]))})
            del a5, ab5, c5, prep5
        mx.clear_cache()
