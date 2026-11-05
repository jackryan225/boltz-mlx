"""Level-1 module checks: fp16 path vs strict fp32 on identical recorded fp32 inputs.

  python tools/validation/l1_ops.py capture <case> [--affinity]      # strict process: record module inputs + outputs
  python tools/validation/l1_ops.py compare <case> [--affinity] --mode f16|tf32   (BOLTZ_MLX_KERNEL_PATH / _NAX pass
                                                     through: rows are labelled f16-portable / f16-tensorops off NAX)
  python tools/validation/l1_ops.py compare <case> [--affinity] --mode f16 --fused   # layer-level, production path
  python tools/validation/l1_ops.py all [--mode f16]                  # capture (if missing) + compare f16 + tf32, L1 cases
  python tools/validation/l1_ops.py report                            # logs/l1_ops.jsonl -> table

Capture: for each recorded module class, calls number 0, 3 and 7 of each (phase, class) are saved
(inputs, kwargs, fp32 output) to data/cache/l1/<case>[_aff]/. Compare re-runs that module in a fresh process
with the candidate precision (f16: BOLTZ_MLX_PRECISION=f16; tf32: BOLTZ_MLX_TF32=1 + MLX paths) and reports
relative Frobenius error, cosine, non-finite count and the module's guard count. Gate (PREREG-A section L1):
rel_fro <= 3e-3, cosine >= 0.99999, zero non-finite, zero guard counts; per pairformer layer z and s rel_fro <= 3e-3.
TF32 rows are the reference error (diagnostic). GPU only.

--fused (f16 only): the layer captures (PairformerLayer, PairformerNoSeqLayer, MSALayer) replayed on the production
f16 path instead of the per-op guard mode: compare's f16 env sets BOLTZ_MLX_GUARD_MODE=op (per-op guards need each
module's fp16 output), which turns off the fused pair-block epilogues (kernels.pair_epi_ok: PAIREPI tri-mult tails,
linear_addln, the transition with the next layer's LN = PAIRLNX), so the rows above never compare those kernels. Here
the guard mode is the default (residual), the captured next_ln module kwarg is restored
(restore_module_kwargs: captures hold it as a parameter dict) and the same GATE applies. Rows -> logs/l1_layers.jsonl
(field replay = "fused-layer"; kept apart from the PREREG-A L1 rows).
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
sys.path.insert(0, str(ROOT))  # this checkout's boltz_mlx first: the editable install maps boltz_mlx to the main checkout
DATA = Path(os.environ.get("BOLTZ_MLX_DATA", Path(__file__).resolve().parents[2] / "data")).expanduser()
CACHE = DATA / "cache" / "l1"
OUT = ROOT / "logs" / "l1_ops.jsonl"
OUT_LAYERS = ROOT / "logs" / "l1_layers.jsonl"   # --fused rows
LAYER_CLASSES = ("PairformerLayer", "PairformerNoSeqLayer", "MSALayer")
CASES = [("prot117", False), ("prot400", False), ("msa384", False), ("lig_aff", False), ("lig_aff", True),
         ("ex_ligand", False), ("cdk2_17", False), ("tyk2_ejm_31", False)]  # PREREG-A section C
CALLS = (0, 3, 7)  # per (phase, class): e.g. pairformer layers 0, 3, 7 of recycle 0
CLASSES = {
    "boltz_mlx.layers.triangular_mult": ["TriangleMultiplicationOutgoing", "TriangleMultiplicationIncoming"],
    "boltz_mlx.layers.triangular_attention.attention": ["TriangleAttention", "TriangleAttentionEndingNode"],
    "boltz_mlx.layers.transition": ["Transition"],
    "boltz_mlx.layers.attentionv2": ["AttentionPairBias"],
    "boltz_mlx.layers.outer_product_mean": ["OuterProductMean"],
    "boltz_mlx.layers.pair_averaging": ["PairWeightedAveraging"],
    "boltz_mlx.layers.pairformer": ["PairformerLayer", "PairformerNoSeqLayer"],
    "boltz_mlx.modules.trunkv2": ["MSALayer"],
}  # diffusion transformer layers: f16 there is the round-1 fast-mode path (fp16 matmuls via mm_t), covered by L2/L3
GATE = {"rel_fro": 3e-3, "cosine": 0.99999}
PATH_ENV = ("BOLTZ_MLX_KERNEL_PATH", "BOLTZ_MLX_NAX")  # kernel-path selection (kernels.py "Hardware capability")


def _tag(case, aff):
    return case + ("_aff" if aff else "")


def _to_np(v):
    import mlx.core as mx
    import numpy as np

    if isinstance(v, mx.array):
        return ("arr", np.array(v.astype(mx.float32)) if v.dtype in (mx.float16, mx.bfloat16) else np.array(v), str(v.dtype))
    if isinstance(v, (tuple, list)):
        return ("seq", [_to_np(x) for x in v])
    if isinstance(v, dict):
        return ("dict", {k: _to_np(x) for k, x in v.items()})
    return ("py", v)


def _from_np(t):
    import mlx.core as mx

    kind = t[0]
    if kind == "arr":
        return mx.array(t[1])
    if kind == "seq":
        return [_from_np(x) for x in t[1]]
    if kind == "dict":
        return {k: _from_np(x) for k, x in t[1].items()}
    return t[1]


def restore_module_kwargs(rec: dict, kwargs: dict, mods: dict) -> dict:
    """Captured kwargs that were modules come back from _from_np as their parameter dicts (an MLX nn.Module is a dict,
    so _to_np stores it as one): PairformerModule / PairformerNoSeqModule pass next_ln = the next layer's
    tri_mul_out.norm_in, which arrives as {weight, bias} without eps (kernels_nax3._ln_params then fails on the f16
    PAIRLNX path). For standalone-layer harnesses: put the model's module back (mods = dict(model.named_modules());
    its parameters are checked against the captured ones, to fp16 rounding). Returns kwargs (changed in place)."""
    import mlx.core as mx

    nl = kwargs.get("next_ln")
    if not isinstance(nl, dict) or hasattr(nl, "eps"):
        return kwargs
    pre, _, i = rec["path"].rpartition(".")
    mod = mods[f"{pre}.{int(i) + 1}.tri_mul_out.norm_in"]
    for k, v in nl.items():
        assert bool(mx.allclose(mod[k].astype(mx.float32), v.astype(mx.float32), rtol=1e-3, atol=1e-3).item()), \
            (rec["path"], k)
    kwargs["next_ln"] = mod
    return kwargs


def _model(affinity: bool):
    from boltz_mlx.model import load_model

    return load_model("aff" if affinity else "conf", compile_diffusion=False)


def _feats(case: str, affinity: bool):
    import mlx.core as mx
    import numpy as np

    z = np.load(DATA / case / ("feats_affinity.npz" if affinity else "feats.npz"))
    return {k: mx.array(z[k]) for k in z.files}


def _wrap_classes(fn):
    """Wrap __call__ of every recorded class with fn(path, orig, self, args, kwargs)."""
    import importlib

    owners = set()
    for mod, names in CLASSES.items():
        m = importlib.import_module(mod)
        for n in names:  # wrap the class in the MRO that defines __call__, once
            owners.add(next(c for c in getattr(m, n).__mro__ if "__call__" in c.__dict__))
    for cls in owners:
        if True:
            orig = cls.__call__

            def wrapped(self, *args, _orig=orig, **kwargs):
                return fn(_orig, self, args, kwargs)

            cls.__call__ = wrapped


def capture(case: str, affinity: bool):
    os.environ.setdefault("BOLTZ_MLX_TRANSITION_IMPL", "mlx")  # module outputs comparable to the f16 MLX path
    import boltz_mlx  # noqa: F401
    import mlx.core as mx

    from boltz_mlx import kernels as K
    from boltz_mlx.model import PredictArgs
    from boltz_mlx.sampler import record_noise

    assert K.PRECISION == "fp32" and os.environ.get("BOLTZ_MLX_TF32", "0") != "1"
    model = _model(affinity)
    paths = {id(m): p for p, m in model.named_modules()}
    counts: dict = {}
    d = CACHE / _tag(case, affinity)
    d.mkdir(parents=True, exist_ok=True)

    def fn(orig, self, args, kwargs):
        p = paths.get(id(self))
        key = (K.ACT_PHASE, type(self).__name__)
        i = counts.get(key, 0)
        counts[key] = i + 1
        out = orig(self, *args, **kwargs)
        if p is None or i not in CALLS:
            return out
        mx.eval(out)
        rec = {"phase": K.ACT_PHASE, "path": p, "cls": type(self).__name__, "call": i,
               "args": _to_np(list(args)), "kwargs": _to_np(kwargs), "out": _to_np(out)}
        with open(d / f"{K.ACT_PHASE}__{type(self).__name__}__{i}.pkl", "wb") as f:
            pickle.dump(rec, f, protocol=5)
        return out

    _wrap_classes(fn)
    feats = _feats(case, affinity)
    if affinity:
        args, noise = PredictArgs(5, 200, 5), record_noise(0, case, "affinity", 5)
    else:
        args, noise = PredictArgs(3, 200, 1), record_noise(0, case, "structure", 1)
    model(feats, args, noise)
    print(f"captured {len(list(d.glob('*.pkl')))} module calls for {_tag(case, affinity)}")


def _metrics(ref, out):
    import numpy as np

    ref = np.asarray(ref, np.float64).ravel()
    out = np.asarray(out, np.float64).ravel()
    nonfinite = int((~np.isfinite(out)).sum())
    if nonfinite:
        return {"rel_fro": float("nan"), "cosine": float("nan"), "nonfinite": nonfinite, "max_abs_err": float("nan")}
    num = np.linalg.norm(out - ref)
    den = np.linalg.norm(ref)
    cos = float(np.dot(out, ref) / (np.linalg.norm(out) * den)) if den > 0 else 1.0
    return {"rel_fro": float(num / den) if den > 0 else float(num), "cosine": cos, "nonfinite": 0,
            "max_abs_err": float(np.abs(out - ref).max())}


def _flat_outputs(t):
    """List of (name, ndarray) for a captured output tree."""
    kind = t[0]
    if kind == "arr":
        return [("out", t[1])]
    if kind == "seq":
        names = ["s", "z"] if len(t[1]) == 2 else [f"o{i}" for i in range(len(t[1]))]
        res = []
        for n, x in zip(names, t[1]):
            res += [(n if m == "out" else f"{n}.{m}", a) for m, a in _flat_outputs(x)]
        return res
    if kind == "dict":
        res = []
        for k, x in t[1].items():
            res += [(k if m == "out" else f"{k}.{m}", a) for m, a in _flat_outputs(x)]
        return res
    return []


def compare(case: str, affinity: bool, mode: str, fused: bool = False):
    import boltz_mlx  # noqa: F401
    import mlx.core as mx
    import numpy as np

    from boltz_mlx import kernels as K

    model = _model(affinity)
    mods = dict(model.named_modules())
    d = CACHE / _tag(case, affinity)
    files = sorted(d.glob("*.pkl"))
    assert files, f"no capture for {_tag(case, affinity)}: run capture first"
    if fused:
        assert mode == "f16" and K.GUARD_MODE != "op" and K.f16(), "--fused: the production f16 path (guard mode not op)"
        files = [f for f in files if f.name.split("__")[1] in LAYER_CLASSES]
    git = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"]).decode().strip()
    pkg = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD:boltz_mlx"]).decode().strip()
    dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "boltz_mlx"],
                                capture_output=True, text=True).stdout.strip())
    msa_env = {k: v for k, v in os.environ.items() if k.startswith("BOLTZ_MLX_MSA_") or k in PATH_ENV}
    dev = K.device_tag()  # chip / architecture / kernel path: f16 off the NAX path is f16-portable / f16-tensorops
    rows = []
    for f in files:
        rec = pickle.load(open(f, "rb"))
        m = mods[rec["path"]]
        K.set_phase(rec["phase"])
        K.guard_take()
        kwargs = _from_np(rec["kwargs"])
        if fused:
            restore_module_kwargs(rec, kwargs, mods)
        out = type(m).__call__(m, *_from_np(rec["args"]), **kwargs)
        m.__dict__.pop("_pb_carry", None)   # PAIRLNX carry for the module loop (none here)
        out = tuple(K.unscale(o) for o in out) if isinstance(out, tuple) else K.unscale(out)
        mx.eval(out)
        guards = K.guard_take()
        got = dict(_flat_outputs(_to_np(out)))
        for name, ref in _flat_outputs(rec["out"]):
            r = {"case": _tag(case, affinity), "mode": K.mode_label(mode), "device": dev, "phase": rec["phase"], "path": rec["path"], "cls": rec["cls"],
                 "call": rec["call"], "output": name, "guards": guards, **_metrics(ref, got[name]),
                 "ref_maxabs": float(np.abs(ref).max()), "git": git, "pkg_tree": pkg + ("-dirty" if dirty else ""),
                 "date": time.strftime("%Y-%m-%d %H:%M:%S")}
            if msa_env:
                r["env"] = msa_env
            if fused:
                r["replay"] = "fused-layer"
                r["next_ln"] = "module" if kwargs.get("next_ln") is not None else None
            r["pass"] = bool(r["nonfinite"] == 0 and r["guards"] == 0 and r["rel_fro"] <= GATE["rel_fro"]
                             and r["cosine"] >= GATE["cosine"])
            rows.append(r)
    with open(OUT_LAYERS if fused else OUT, "a") as fo:
        for r in rows:
            fo.write(json.dumps(r) + "\n")
    n_fail = sum(not r["pass"] for r in rows)
    worst = max(rows, key=lambda r: r["rel_fro"] if r["rel_fro"] == r["rel_fro"] else 1e9)
    print(f"{_tag(case, affinity)} {K.mode_label(mode)}{' fused-layer' if fused else ''} ({dev['chip']}, path {dev['path']}): {len(rows)} outputs, {n_fail} fail; worst rel_fro {worst['rel_fro']:.2e} "
          f"({worst['phase']} {worst['cls']} {worst['output']})")


MODE_ENV = {"f16": {"BOLTZ_MLX_PRECISION": "f16", "BOLTZ_MLX_GUARD_MODE": "op"},
            "tf32": {"BOLTZ_MLX_TF32": "1", "BOLTZ_MLX_TRANSITION_IMPL": "mlx"}}


def _child(args, extra_env):
    env = {k: v for k, v in os.environ.items()  # BOLTZ_MLX_MSA_* (MSA-kernel flags) + the kernel path pass through
           if (not k.startswith("BOLTZ_MLX_") or k.startswith("BOLTZ_MLX_MSA_") or k in PATH_ENV)
           and k != "MLX_ENABLE_TF32"}
    env.update(extra_env)
    env["PYTHONPATH"] = str(ROOT)
    return subprocess.run([sys.executable, __file__, *args], env=env).returncode


def report():
    rows = [json.loads(l) for l in open(OUT) if l.strip()]
    latest = {}
    for r in rows:  # last row per key wins
        latest[(r["case"], r["mode"], r["phase"], r["path"], r["call"], r["output"])] = r
    rows = list(latest.values())
    from collections import defaultdict

    agg = defaultdict(lambda: {"n": 0, "fail": 0, "worst": 0.0, "mincos": 1.0, "guards": 0, "nonfinite": 0, "where": ""})
    for r in rows:
        a = agg[(r["mode"], r["phase"], r["cls"], r["output"])]
        a["n"] += 1
        a["fail"] += not r["pass"]
        rf = r["rel_fro"] if r["rel_fro"] == r["rel_fro"] else float("inf")
        if rf >= a["worst"]:
            a["worst"], a["where"] = rf, r["case"]
        a["mincos"] = min(a["mincos"], r["cosine"] if r["cosine"] == r["cosine"] else -1)
        a["guards"] += r["guards"]
        a["nonfinite"] += r["nonfinite"]
    print("| mode | phase | module | output | n | fail | worst rel_fro (case) | min cosine | guards | non-finite |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for k in sorted(agg):
        a = agg[k]
        print(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {a['n']} | {a['fail']} | {a['worst']:.2e} ({a['where']}) | "
              f"{a['mincos']:.7f} | {a['guards']} | {a['nonfinite']} |")


def main():
    cmd = sys.argv[1]
    aff = "--affinity" in sys.argv
    if cmd == "capture":
        capture(sys.argv[2], aff)
    elif cmd == "compare":
        mode = sys.argv[sys.argv.index("--mode") + 1]
        fused = "--fused" in sys.argv
        if os.environ.get("_L1_CHILD") != "1":
            env = dict(MODE_ENV[mode])
            if fused:   # the production f16 path: default guard mode (residual), fused pair-block epilogues on
                assert mode == "f16", "--fused is an f16 check"
                env.pop("BOLTZ_MLX_GUARD_MODE")
            sys.exit(_child(sys.argv[1:], {**env, "_L1_CHILD": "1"}))
        compare(sys.argv[2], aff, mode, fused)
    elif cmd == "all":
        modes = [sys.argv[sys.argv.index("--mode") + 1]] if "--mode" in sys.argv else ["f16", "tf32"]
        for case, a in CASES:
            if not (DATA / case / ("feats_affinity.npz" if a else "feats.npz")).exists():
                print(f"{_tag(case, a)}: no features — skipped")
                continue
            if not any((CACHE / _tag(case, a)).glob("*.pkl")):
                _child(["capture", case] + (["--affinity"] if a else []), {})
            for mode in modes:
                _child(["compare", case, "--mode", mode] + (["--affinity"] if a else []), {**MODE_ENV[mode], "_L1_CHILD": "1"})
    elif cmd == "report":
        report()


if __name__ == "__main__":
    main()
