"""Shared harness for the rung-2N triangle-attention experiments (tools/prof/triatt_*.py).

Inputs: the recorded strict-fp32 L1 captures data/cache/l1/<case>/trunk.pairformer__TriangleAttention[EndingNode]__<call>.pkl
(x [1, N, N, 128] fp32 + pair mask), weights of the captured module (pairformer_module.layers.<l>.tri_att_*).

For one capture we build
  x16      = LN(x) in fp16 (the f16 path's fused-cast LayerNorm)
  h16      = x16 @ W16^T          [B, I, J, 4HC + H]   (f16 path projection, fp16 out)
  h32      = LN(x) @ W^T in fp32  (strict path projection)
and references
  ref_same = fp32 MMA kernel on h16 upcast to fp32  -> the core error of a candidate on identical inputs
  ref32    = fp32 MMA kernel on h32 (strict core)
  out_ref  = the captured strict module output (L1 gate: rel_fro <= 3e-3, cosine >= 0.99999)

A candidate is a dict with
  name, prep(ctx) -> state (anything precomputed outside the timed core, e.g. a padded bias),
  core(ctx, state) -> fp16 core output [B, I, J, HC] (sigmoid(g) * attention, before linear_o),
  module(ctx) -> fp16 module output (LN -> projection -> core -> linear_o), optional (defaults to the
  f16 module with the candidate core).
Rows are printed as JSON (one per candidate x capture) and appended to logs/triatt_2n.jsonl.
GPU only for the kernel candidates; `--cpu-smoke` runs the MLX-op candidates at N=48 on CPU (logic/shape check only).
"""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # this checkout, not an editable install elsewhere

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

import boltz_mlx  # noqa: E402,F401  (TF32 off)
from boltz_mlx import kernels as K  # noqa: E402
from boltz_mlx.layers.triangular_attention.attention import TriangleAttention  # noqa: E402
from boltz_mlx.weights import load_module  # noqa: E402

DATA = Path(os.environ.get("BOLTZ_MLX_DATA", Path(__file__).resolve().parents[2] / "data")).expanduser()
L1 = DATA / "cache" / "l1"
OUT = ROOT / "logs" / "triatt_2n.jsonl"
H, C = 4, 32
HC = H * C
GATE = {"rel_fro": 3e-3, "cosine": 0.99999}
FLOPS = lambda B, I, J: 4.0 * B * I * J * J * C * H  # noqa: E731  QK^T + PV, 2 flop/FMA

CPU_SMOKE = "--cpu-smoke" in sys.argv
if CPU_SMOKE:
    mx.set_default_device(mx.cpu)
else:
    mx.set_default_device(mx.gpu)


def git_rev():
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "?"


def metrics(out, ref):
    out = np.asarray(out.astype(mx.float32) if isinstance(out, mx.array) else out, np.float64).ravel()
    ref = np.asarray(ref.astype(mx.float32) if isinstance(ref, mx.array) else ref, np.float64).ravel()
    nonfinite = int((~np.isfinite(out)).sum())
    if nonfinite:
        return {"rel_fro": float("nan"), "cosine": float("nan"), "max_abs": float("nan"), "nonfinite": nonfinite}
    d = np.linalg.norm(ref)
    return {"rel_fro": float(np.linalg.norm(out - ref) / d), "cosine": float(np.dot(out, ref) / (np.linalg.norm(out) * d)),
            "max_abs": float(np.abs(out - ref).max()), "nonfinite": 0}


def captures(case: str, calls=(0, 3, 7), classes=("TriangleAttention", "TriangleAttentionEndingNode")):
    res = []
    for cls in classes:
        for c in calls:
            f = L1 / case / f"trunk.pairformer__{cls}__{c}.pkl"
            if f.exists():
                res.append(f)
    return res


_MODS: dict = {}


def module_for(path: str, starting: bool):
    if path not in _MODS:
        _MODS[path] = load_module(TriangleAttention(128, C, H, starting=starting, inf=1e9), path)
    return _MODS[path]


class Ctx:
    """Everything derived from one capture (or a synthetic CPU-smoke input)."""

    def __init__(self, f: Path | None = None, n_smoke: int = 48):
        if f is None:  # synthetic: random x, all-ones mask, layer-0 weights
            mx.random.seed(0)
            self.name = f"smoke_N{n_smoke}"
            self.mod = module_for("pairformer_module.layers.0.tri_att_start", True)
            x = mx.random.normal((1, n_smoke, n_smoke, 128))
            mask = mx.ones((1, n_smoke, n_smoke))
            self.out_ref = None
        else:
            rec = pickle.load(open(f, "rb"))
            self.name = f"{f.parent.name}/{f.stem}"
            self.mod = module_for(rec["path"], rec["cls"] == "TriangleAttention")
            x = mx.array(rec["args"][1][0][1])
            mask = mx.array(rec["kwargs"][1]["mask"][1])
            self.out_ref = rec["out"][1]
        m = self.mod
        self.x_in, self.mask_in = x, mask  # module-call inputs (before the ending-node transpose)
        if not m.starting:
            x = x.swapaxes(-2, -3)
            mask = mask.swapaxes(-1, -2)
        self.x, self.mask = x, mask
        self.mask_all_ones = bool(mx.all(mask == 1).item())
        w = mx.concatenate([m.mha.linear_q.weight, m.mha.linear_k.weight, m.mha.linear_v.weight,
                            m.mha.linear_g.weight, m.linear.weight], 0)
        self.w32 = w
        self.w16 = w.astype(mx.float16)
        self.x16 = m.layer_norm.f16(x)
        self.h16 = self.x16 @ self.w16.T
        self.h32 = m.layer_norm(x) @ w.T
        mx.eval(self.x16, self.h16, self.h32)
        self.B, self.I, self.J = x.shape[:3]
        self.ref_same = ref_core(self.h16.astype(mx.float32), mask)
        self.ref32 = ref_core(self.h32, mask)
        mx.eval(self.ref_same, self.ref32)

    def finish(self, core16):
        """linear_o (f16 path) + ending-node transpose: module output from a core output."""
        o = K.lin16(core16, self.mod.mha.linear_o)
        return o if self.mod.starting else o.swapaxes(-2, -3)

    def f16_module(self, core_fn):
        """The f16 module call (LN fp16 -> fused fp16 projection -> core_fn(h16, mask) -> linear_o fp16)."""
        m = self.mod
        x16 = m.layer_norm.f16(self.x)
        h = x16 @ self.w16.T
        return self.finish(core_fn(h, self.mask))


def ref_core(h32, mask):
    """fp32 reference core: the strict fp32 MMA kernel on GPU; MLX ops (fp32, TF32 off) on CPU."""
    if mx.default_device() == mx.gpu:
        return K.tri_attention_fused(h32, H, C, mask, 1e9, impl="mma", out_dtype=mx.float32)
    B, I, J, _ = h32.shape
    q = h32[..., :HC].reshape(B * I, J, H, C).transpose(0, 2, 1, 3)
    k = h32[..., HC:2 * HC].reshape(B * I, J, H, C).transpose(0, 2, 1, 3)
    v = h32[..., 2 * HC:3 * HC].reshape(B * I, J, H, C).transpose(0, 2, 1, 3)
    g = h32[..., 3 * HC:4 * HC]
    tri = h32[..., 4 * HC:].transpose(0, 3, 1, 2)[:, None]
    mb = (1e9 * (mask - 1))[:, :, None, None, :]
    s = (q * C**-0.5) @ k.swapaxes(-1, -2) + (mb + tri).reshape(B * I, H, J, J)
    o = mx.softmax(s, axis=-1) @ v
    return o.transpose(0, 2, 1, 3).reshape(B, I, J, HC) * mx.sigmoid(g)


def bench(f, reps: int, rounds: int = 3) -> float:
    """ms per call: min over rounds of (reps pipelined calls, one eval)."""
    mx.eval(f())
    best = 1e9
    for _ in range(rounds):
        t = time.perf_counter()
        mx.eval([f() for _ in range(reps)])
        best = min(best, (time.perf_counter() - t) / reps * 1e3)
    return best


def peak_mb(f) -> float:
    mx.eval(f())  # warm (compile) outside the measurement
    mx.clear_cache()
    mx.reset_peak_memory()
    base = mx.get_active_memory()
    mx.eval(f())
    return (mx.get_peak_memory() - base) / 1e6


def emit(row: dict, log: bool = True):
    row = {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "git": git_rev(), "chip": K.chip_name(), **row}
    print(json.dumps(row), flush=True)
    if log and not CPU_SMOKE:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT, "a") as fo:
            fo.write(json.dumps(row) + "\n")


def run(experiment: str, candidates: list, case: str = "prot400", calls=(0, 3, 7), time_calls=(0,),
        reps: int = 10, extra_cases=("prot117",)):
    """For every capture: numerics of every candidate; timing (core + module) on calls in time_calls, at `case`.
    extra_cases: numerics only (tail / odd-J coverage). A candidate that raises is reported with its error."""
    if CPU_SMOKE:
        ctxs = [(Ctx(None), True)]
    else:
        ctxs = [(f, int(f.stem.rsplit("__", 1)[1]) in time_calls) for f in captures(case, calls)]
        for ec in extra_cases:
            ctxs += [(f, False) for f in captures(ec, (0,))]
    for item, timed in ctxs:
        ctx = item if isinstance(item, Ctx) else Ctx(item)
        base = {"exp": experiment, "input": ctx.name, "N": ctx.J, "mask_all_ones": ctx.mask_all_ones}
        flops = FLOPS(ctx.B, ctx.I, ctx.J)
        for cand in candidates:
            row = {**base, "cand": cand["name"], **cand.get("cfg", {})}
            if CPU_SMOKE and cand.get("gpu_only"):
                continue
            try:
                state = cand["prep"](ctx) if "prep" in cand else None
                core = cand["core"](ctx, state)
                if "module" in cand:
                    mod_fn = lambda c=cand, x=ctx: c["module"](x)  # noqa: E731
                elif "core_h" in cand:
                    mod_fn = lambda c=cand, x=ctx: x.f16_module(lambda h, m: c["core_h"](x, h, m))  # noqa: E731
                else:
                    mod_fn = None
                out = mod_fn() if mod_fn else ctx.finish(core)
                mx.eval(core, out)
                row["core_vs_same"] = metrics(core, ctx.ref_same)
                row["core_vs_ref32"] = metrics(core, ctx.ref32)
                if ctx.out_ref is not None:
                    mo = metrics(out, ctx.out_ref)
                    row["module_vs_capture"] = mo
                    row["l1_pass"] = bool(mo["nonfinite"] == 0 and mo["rel_fro"] <= GATE["rel_fro"]
                                          and mo["cosine"] >= GATE["cosine"])
                if timed:
                    tc = bench(lambda: cand["core"](ctx, state), reps)
                    row["core_ms"] = round(tc, 3)
                    row["core_tflops"] = round(flops / tc / 1e9, 2)
                    if mod_fn is not None:
                        row["module_ms"] = round(bench(mod_fn, reps), 3)
                        row["module_peak_mb"] = round(peak_mb(mod_fn), 1)
                    row["core_peak_mb"] = round(peak_mb(lambda: cand["core"](ctx, state)), 1)
            except Exception as e:  # report and continue (e.g. kernel compile errors)
                row["error"] = f"{type(e).__name__}: {str(e)[:2000]}"
            emit(row)
        del ctx
        mx.clear_cache()


def baseline_candidates():
    """The shipped f16 core (fp32-operand simdgroup MMA on fp16 h, fp16 out) and the half-operand variant."""
    def mk(impl):
        return {"name": f"f16_{impl}", "cfg": {"impl": impl}, "gpu_only": True,
                "core": lambda ctx, s: K.tri_attention_fused(ctx.h16, H, C, ctx.mask, 1e9, impl=impl, out_dtype=mx.float16),
                "core_h": lambda ctx, h, m: K.tri_attention_fused(h, H, C, m, 1e9, impl=impl, out_dtype=mx.float16)}
    return [mk("mma"), mk("mma_half")]
