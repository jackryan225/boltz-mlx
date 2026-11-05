"""Mixed fp16/fp32 checkpoint for --precision f16 (boltz_mlx/mixed.py explains the format).

  python tools/make_mixed_checkpoint.py [--names conf aff] [--src DIR] [--out DIR] [--report docs/weights_fp16.md]
      Writes <out>/boltz2_{name}_mixed.safetensors from <src>/boltz2_{name}.safetensors (src default: the weights dir
      of boltz_mlx.paths.weights_dir(); out default: src, where load_model looks), using the committed
      boltz_mlx/fp16_manifest.json. Fails on any fp16 overflow. Writes the fp16 range report + sha256s + a memory
      estimate to --report (skipped with --report '').
  python tools/make_mixed_checkpoint.py --derive [--device cpu|gpu] [--write-manifest]
      Re-derives the manifest from the code: loads each fp32 checkpoint (one at a time), runs one f16 forward with
      kernels.W16_TRACE set (prot117 structure pass for conf, lig_aff affinity pass for aff; 0 recycles, 2 steps) and
      records every fp16 weight built and its source parameters. Compares with the committed manifest (exit 1 on any
      difference) or overwrites it (--write-manifest). On the CPU the f16 paths run through mixed.cpu_f16() (Metal-only
      pieces replaced); --device gpu runs the shipped GPU f16 paths (NAX, fused add+LN) and must give the same manifest.

Memory: one checkpoint at a time; tensors are converted one by one (peak ~ mixed size + one tensor).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = Path(os.environ.get("BOLTZ_MLX_DATA", Path(__file__).resolve().parents[1] / "data")).expanduser()
TRACE_INPUTS = {"conf": ("prot117", "feats.npz"), "aff": ("lig_aff", "feats_affinity.npz")}


def sha256(path: Path) -> str:
    from boltz_mlx.assets import sha256_file

    return sha256_file(Path(path)).hexdigest()


# ------------------------------------------------------------------------------------------------ derive
def derive(name: str, device: str) -> dict:
    import mlx.core as mx
    import numpy as np

    from boltz_mlx import kernels as K
    from boltz_mlx import mixed as MX
    from boltz_mlx.model import PredictArgs, load_model
    from boltz_mlx.sampler import MLXNoise

    case, fname = TRACE_INPUTS[name]
    z = np.load(DATA / case / fname)
    feats = {k: mx.array(z[k]) for k in z.files}
    model = load_model(name, weights="fp32", compile_diffusion=False)
    ctx = MX.cpu_f16() if device == "cpu" else _null()
    with ctx:
        assert K.f16() and K.DIFF_DTYPE == mx.float16, "f16 mode is not active"
        with MX.trace_fp16_weights(model, convert=True) as groups:
            out = model(feats, PredictArgs(0, 2, 1), MLXNoise(0))
            mx.eval({k: v for k, v in out.items() if isinstance(v, mx.array)})
    entry = MX.manifest_entry(model, groups)
    del model, out
    mx.clear_cache()
    return entry


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def manifest_diff(old: dict, new: dict) -> list[str]:
    msgs = []
    og = {(g["owner"], g["cache"]): g for g in old.get("groups", [])}
    ng = {(g["owner"], g["cache"]): g for g in new["groups"]}
    for k in sorted(set(og) | set(ng)):
        if og.get(k) != ng.get(k):
            msgs.append(f"group {k[0]}.{k[1]}: committed {og.get(k)} != derived {ng.get(k)}")
    if set(old.get("params", [])) != set(new["params"]):
        a, b = set(old.get("params", [])), set(new["params"])
        msgs.append(f"params: only committed {sorted(a - b)[:5]}, only derived {sorted(b - a)[:5]}")
    return msgs


# ------------------------------------------------------------------------------------------------ convert
def convert(name: str, src_dir: Path, out_dir: Path, entry: dict) -> dict:
    """Mixed checkpoint of one model (the code lives in boltz_mlx/assets.py, shared with `boltz-mlx setup`)."""
    from boltz_mlx.assets import convert_mixed

    return convert_mixed(name, src_dir, out_dir, entry)


# ------------------------------------------------------------------------------------------------ memory estimate
def memory_estimate(name: str, entry: dict) -> dict:
    """Resident weight bytes in f16 mode, from tensor sizes (an estimate: allocator rounding, activations and the MLX
    buffer cache are not included). Unique arrays of the model (aliased modules counted once)."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    from boltz_mlx.model import Boltz2
    from boltz_mlx.weights import WEIGHTS

    hp = json.loads((WEIGHTS / f"boltz2_{name}.json").read_text())
    with mx.stream(mx.cpu):
        model = Boltz2(hp, affinity=(name == "aff"))  # lazy init: shapes only, nothing is evaluated
    seen, sizes = {}, {}
    for k, v in tree_flatten(model.parameters()):
        if id(v) not in seen:
            seen[id(v)] = k
            sizes[k] = v.size
    params16 = {k for k in entry["params"] if k in sizes}
    total = sum(sizes.values())
    n16 = sum(sizes[k] for k in params16)
    groups = [g for g in entry["groups"]]
    g_store = sum(_gsize(g) for g in groups if g["storable"])
    g_store_multi = sum(_gsize(g) for g in groups if g["storable"] and len(g["srcs"]) > 1)
    g_derived = sum(_gsize(g) for g in groups if not g["storable"])
    # fp32 concatenations the f16 paths built before step 4.2 (Transition._w12, TriMult._w_t, TriAtt._w)
    old_cat32 = sum(_gsize(g) for g in groups if g["cache"] in ("_w12h", "_w_th", "_wh"))
    gb = 1e9
    before = (4 * total + 2 * (g_store + g_derived) + 4 * old_cat32) / gb
    runtime = (4 * total + 2 * (g_store + g_derived)) / gb
    mixed_ = (4 * (total - n16) + 2 * n16 + 2 * g_derived) / gb
    mixed_noprefuse = mixed_ + 2 * g_store_multi / gb
    del model
    return {"name": name, "params_total": total, "params_fp16": n16, "group_elems_storable": g_store,
            "group_elems_derived": g_derived, "fp32_concat_elems_removed": old_cat32,
            "gb_before_step42": round(before, 3), "gb_runtime_cast": round(runtime, 3), "gb_mixed": round(mixed_, 3),
            "gb_mixed_without_prefuse": round(mixed_noprefuse, 3), "gb_strict_fp32": round(4 * total / gb, 3)}


def _gsize(g) -> int:
    n = 1
    for d in g["shape"]:
        n *= d
    return n


# ------------------------------------------------------------------------------------------------ report
def _role(k: str) -> str:
    return re.sub(r"\.\d+\.", ".*.", re.sub(r"\.\d+\.", ".*.", k))


def write_report(path: Path, results: list, mems: list, manifest_path: Path) -> None:
    L = ["# fp16 weights of the mixed checkpoint", "",
         "Generated by `tools/make_mixed_checkpoint.py` (do not edit by hand). The mixed checkpoints",
         "`boltz2_{conf,aff}_mixed.safetensors` hold fp16 copies of exactly the parameters that `--precision f16` consumes",
         "only through fp16 GEMMs (list: `boltz_mlx/fp16_manifest.json`, derived by tracing `kernels.w16` in an f16",
         "forward; see `boltz_mlx/mixed.py`) and the fp32 values of everything else. The `.safetensors` files are not in",
         "git (gitignored); their sha256 are recorded here.", "",
         "Counts per converted tensor: **overflow** = finite fp32 value that becomes inf in fp16 (|w| >= 65520; must be",
         "0, the converter fails otherwise); **subnormal** = the fp16 value is nonzero and below 2^-14 = 6.1e-5 (stored",
         "with fewer mantissa bits); **flushed** = nonzero fp32 value that rounds to 0 in fp16 (|w| <= 2^-25 = 3.0e-8).",
         "Both builds (runtime cast and mixed) see exactly these fp16 values, so the counts describe f16 mode itself.",
         "Tensor counts are per checkpoint key: the diffusion transformers' `output_projection_linear` is also stored as",
         "`output_projection.0` (one torch module registered twice), so those tensors appear twice.", ""]
    L += ["## Files", "", "| checkpoint | file | size | sha256 | source (fp32) | source sha256 | manifest sha |",
          "|---|---|---|---|---|---|---|"]
    for r in results:
        L.append(f"| {r['name']} | `{Path(r['path']).name}` | {r['bytes'] / 1e9:.3f} GB | `{r['sha256']}` | "
                 f"`{Path(r['source']).name}` {r['source_bytes'] / 1e9:.3f} GB | `{r['source_sha256']}` | "
                 f"`{r['manifest_sha']}` |")
    L += ["", "## Totals", "", "| checkpoint | fp16 tensors | fp16 params | fp32 tensors | fp32 params | overflow | "
          "subnormal | flushed to 0 | tensors with subnormals | max abs (fp16 set) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        st = r["stats"].values()
        L.append(f"| {r['name']} | {r['n_tensors_fp16']} | {r['params_fp16'] / 1e6:.2f} M | {r['n_tensors_fp32']} | "
                 f"{r['params_fp32'] / 1e6:.2f} M | {sum(s['overflow'] for s in st)} | "
                 f"{sum(s['subnormal'] for s in st)} ({100 * sum(s['subnormal'] for s in st) / max(1, r['params_fp16']):.4f}%) | "
                 f"{sum(s['flushed'] for s in st)} | {sum(1 for s in st if s['subnormal'])} | "
                 f"{max((s['max_abs'] for s in st), default=0):.4g} |")
    if mems:
        L += ["", "## Resident weight memory in f16 mode (estimate from tensor sizes)", "",
              "Bytes of weight arrays held by a loaded model, from parameter counts x dtype size; allocator rounding,",
              "activations and the MLX buffer cache are not included, so this is an **estimate**, not a measured peak",
              "(the GPU queue has the measured peak). *before 4.2*: fp32 checkpoint + fp16 caches + the fp32",
              "concatenations the f16 trunk paths also built; *runtime cast*: fp32 checkpoint + fp16 caches (this",
              "branch with BOLTZ_MLX_MIXED=0 or no mixed file); *mixed*: fp32 rest + fp16 set + the runtime-cast",
              "derived groups (AdaLN weights with the LN weight folded in), with prefuse (concatenated fp16 weights",
              "stored once, parameters are views).", "",
              "| checkpoint | strict fp32 | f16 before 4.2 | f16 runtime cast | f16 mixed (no prefuse) | **f16 mixed** | "
              "saving vs before | saving vs runtime cast |", "|---|---|---|---|---|---|---|---|"]
        for m in mems:
            L.append(f"| {m['name']} | {m['gb_strict_fp32']:.2f} GB | {m['gb_before_step42']:.2f} GB | "
                     f"{m['gb_runtime_cast']:.2f} GB | {m['gb_mixed_without_prefuse']:.2f} GB | **{m['gb_mixed']:.2f} GB** | "
                     f"{m['gb_before_step42'] - m['gb_mixed']:.2f} GB | {m['gb_runtime_cast'] - m['gb_mixed']:.2f} GB |")
        L += ["", "Also on this branch: `load_model` no longer keeps the loaded checkpoint dict in an lru_cache. Before,",
              "the structure-pass checkpoint stayed resident through the affinity pass after `del model` (CLI / screen",
              f"phases): about {mems[0]['gb_strict_fp32']:.2f} GB fp32 (estimate) for every mode, not only f16."]
    for r in results:
        st = r["stats"]
        roles: dict = {}
        for k, s in st.items():
            a = roles.setdefault(_role(k), {"tensors": 0, "n": 0, "subnormal": 0, "flushed": 0, "overflow": 0,
                                            "max_abs": 0.0})
            a["tensors"] += 1
            for f in ("n", "subnormal", "flushed", "overflow"):
                a[f] += s[f]
            a["max_abs"] = max(a["max_abs"], s["max_abs"])
        L += ["", f"## {r['name']}: by tensor role (layer indices replaced by *)", "",
              "| role | tensors | params | overflow | subnormal | flushed | max abs |", "|---|---|---|---|---|---|---|"]
        for k in sorted(roles):
            a = roles[k]
            L.append(f"| `{k}` | {a['tensors']} | {a['n']} | {a['overflow']} | {a['subnormal']} | {a['flushed']} | "
                     f"{a['max_abs']:.4g} |")
        L += ["", f"## {r['name']}: per tensor", "", "| tensor | params | overflow | subnormal | flushed |",
              "|---|---|---|---|---|"]
        for k in sorted(st):
            s = st[k]
            L.append(f"| {k} | {s['n']} | {s['overflow']} | {s['subnormal']} | {s['flushed']} |")
    path.write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", default=["conf", "aff"])
    ap.add_argument("--src", default=None, help="dir with boltz2_{name}.safetensors (default: boltz_mlx weights dir)")
    ap.add_argument("--out", default=None, help="output dir (default: --src)")
    ap.add_argument("--report", default=str(ROOT / "docs" / "weights_fp16.md"))
    ap.add_argument("--derive", action="store_true", help="re-derive the manifest by tracing an f16 forward")
    ap.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--write-manifest", action="store_true")
    a = ap.parse_args()
    if a.derive and a.device == "gpu":
        os.environ.setdefault("BOLTZ_MLX_PRECISION", "f16")  # before boltz_mlx / MLX initialize (as --precision f16)
    import boltz_mlx  # noqa: F401
    import mlx.core as mx

    mx.set_default_device(mx.gpu if a.device == "gpu" else mx.cpu)
    mx.set_cache_limit(0)  # freed fp32 buffers go back to the OS at once (keeps the process well under 4 GB)
    from boltz_mlx import mixed as MX
    from boltz_mlx.weights import WEIGHTS

    if a.derive:
        committed = MX.load_manifest() if MX.MANIFEST.exists() else {}
        new, ok = dict(committed), True
        for n in a.names:
            t0 = time.time()
            e = derive(n, a.device)
            ns = sum(1 for g in e["groups"] if g["storable"])
            print(f"{n}: {len(e['groups'])} fp16 weight groups ({ns} storable), {len(e['params'])} fp16 parameters "
                  f"[{time.time() - t0:.0f} s, {a.device}]")
            d = manifest_diff(committed.get(n, {}), e)
            for m in d[:20]:
                print("  DIFF", m)
            ok &= not d
            new[n] = e
        if a.write_manifest:
            MX.MANIFEST.write_text(MX.dump_manifest(new))
            print(f"wrote {MX.MANIFEST}")
        elif not ok:
            sys.exit("derived manifest differs from boltz_mlx/fp16_manifest.json (re-run with --write-manifest, then "
                     "rebuild the mixed checkpoints)")
        else:
            print("derived manifest == committed manifest")
        return
    manifest = MX.load_manifest()
    src = Path(a.src).expanduser() if a.src else WEIGHTS
    out = Path(a.out).expanduser() if a.out else src
    results, mems = [], []
    for n in a.names:
        r = convert(n, src, out, manifest[n])
        st = r["stats"].values()
        print(f"{n}: {r['path']} {r['bytes'] / 1e9:.3f} GB (fp32 {r['source_bytes'] / 1e9:.3f} GB) sha256 {r['sha256'][:16]} "
              f"fp16 tensors {r['n_tensors_fp16']} ({r['params_fp16'] / 1e6:.1f} M) overflow "
              f"{sum(s['overflow'] for s in st)} subnormal {sum(s['subnormal'] for s in st)} flushed "
              f"{sum(s['flushed'] for s in st)} [{r['seconds']} s]")
        results.append(r)
        mems.append(memory_estimate(n, manifest[n]))
        print(json.dumps(mems[-1]))
    if a.report:
        write_report(Path(a.report), results, mems, MX.MANIFEST)
        print(f"wrote {a.report}")


if __name__ == "__main__":
    main()
