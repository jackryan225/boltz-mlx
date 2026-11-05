"""Mixed fp16/fp32 checkpoint for --precision f16.

`weights/boltz2_{conf,aff}_mixed.safetensors` holds, under the PyTorch names of the fp32 checkpoint, fp16 copies of
exactly the parameters that f16 mode consumes only through fp16 GEMMs, and the fp32 values of everything else. f16 mode
loads it directly: the fp16 GEMM weights are never held in fp32, and no fp16 cache is built at runtime (this removes
the fp32 + fp16 double caching of the runtime-cast path).

Which parameters are fp16 is derived from the code, not listed by hand:
  every fp16 GEMM weight is built by `kernels.w16(owner, name, *srcs)` from the parameters it is made of. With
  `kernels.W16_TRACE` set, an f16 forward reports each (owner module, cache name, source arrays); sources that are
  parameters (by identity) are mapped to their qualified names. A group whose sources are all parameters is
  "storable": its parameters go fp16 into the mixed checkpoint. A group built from a derived array (e.g. the diffusion
  AdaLN weights with the LayerNorm weight folded in, a product computed in fp32) is not storable: its parameters stay
  fp32 and the fp16 copy is still cast at runtime (small: 3 transformers' cond weights).
The result is committed as boltz_mlx/fp16_manifest.json (tools/make_mixed_checkpoint.py --derive writes it and the
converter's --check re-derives it and fails on any difference); tests/test_mixed_checkpoint.py checks that the code paths
of a small model trace to the manifest's groups and that mixed == runtime cast bit for bit.

Bit identity: cast(concatenate(srcs)) == concatenate(cast(srcs)) element by element, and the fp32 -> fp16 cast is
deterministic (round to nearest even), so the fp16 GEMM operands are the same bits either way. A parameter that some
other f16-mode op read in fp32 would break this; the bit-identity tests (CPU synthetic, GPU prot117 + lig_aff) catch it.

Guard fallback (a trunk recycle recomputed with fp32 GEMMs after an fp16 overflow): with the mixed checkpoint the fp32
recompute reads the fp16-rounded values of the converted trunk weights (fp32 activations and accumulation; MLX promotes
fp16 weights in fp32 GEMMs). That recompute exists for activation range, which weight rounding (|rel| <= 2^-11) does
not affect; it is not bit-identical to the runtime-cast build's fallback.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from boltz_mlx import kernels as _K

FORMAT = "boltz-mlx-mixed-v1"
MANIFEST = Path(__file__).with_name("fp16_manifest.json")
FP16_MAX = 65504.0
FP16_MIN_NORMAL = 2.0 ** -14  # 6.10e-5
FP16_MIN_SUBNORMAL = 2.0 ** -24  # 5.96e-8; |w| < 2^-25 rounds to 0


def mixed_path(name: str, wdir: str | os.PathLike | None = None) -> Path:
    from boltz_mlx.weights import WEIGHTS

    return Path(wdir if wdir is not None else WEIGHTS) / f"boltz2_{name}_mixed.safetensors"


def load_manifest(path: str | os.PathLike | None = None) -> dict:
    return json.loads(Path(path or MANIFEST).read_text())


def dump_manifest(manifest: dict) -> str:
    """JSON with one group / parameter per line (readable diffs when the f16 code paths change)."""
    out = ["{"]
    for i, name in enumerate(sorted(manifest)):
        e = manifest[name]
        out += [f' "{name}": {{', '  "groups": [']
        gs = [json.dumps(g, sort_keys=True) for g in e["groups"]]
        out += [f"   {g}," for g in gs[:-1]] + [f"   {g}" for g in gs[-1:]]
        out += ["  ],", '  "params": [']
        ps = [json.dumps(p) for p in e["params"]]
        out += [f"   {p}," for p in ps[:-1]] + [f"   {p}" for p in ps[-1:]]
        out += ["  ]", " }" + ("," if i < len(manifest) - 1 else "")]
    out.append("}")
    return "\n".join(out) + "\n"


def entry_sha(entry: dict) -> str:
    """Hash of one checkpoint's manifest entry (stored in the mixed file; a mismatch at load = stale mixed file)."""
    return hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()[:16]


# ------------------------------------------------------------------------------------------------ derivation (trace)
def _module_paths(model: nn.Module) -> dict:
    out: dict = {}
    for p, m in model.named_modules():
        out.setdefault(id(m), p)
    return out


def _param_names(model: nn.Module) -> dict:
    """id(array) -> [qualified names] (a module registered twice, e.g. output_projection_linear / output_projection.0,
    gives two names for one array)."""
    out: dict = {}
    for k, v in tree_flatten(model.parameters()):
        out.setdefault(id(v), []).append(k)
    return out


@contextmanager
def trace_fp16_weights(model: nn.Module, convert: bool = False):
    """Collect the fp16 weight groups built while the block runs (yields the list, filled in place). Caches already
    filled before entering are not seen: trace a freshly loaded model. convert=True replaces each storable group's
    source parameters by their fp16 casts as soon as the group is built (the model turns into the mixed one while the
    trace runs: lower peak memory for full-size traces; the fp16 values are the same)."""
    mods, names = _module_paths(model), _param_names(model)
    groups: dict = {}

    def hook(owner, name, srcs):
        op = mods.get(id(owner))
        if op is None:
            raise RuntimeError(f"fp16 weight {name!r}: owner {type(owner).__name__} is not a submodule of the model")
        pre = op + "." if op else ""
        src = []
        for a in srcs:
            n = [x for x in names.get(id(a), []) if x.startswith(pre)]
            src.append(n[0][len(pre):] if n else None)
        key = (op, name)
        g = {"owner": op, "cache": name, "srcs": src,
             "storable": all(s is not None for s in src), "shape": [int(sum(a.shape[0] for a in srcs)),
                                                                    *map(int, srcs[0].shape[1:])]}
        if key in groups and groups[key] != g:
            raise RuntimeError(f"fp16 weight {op}.{name} built twice from different sources")
        groups[key] = g
        if convert and g["storable"]:
            for s, a in zip(src, srcs):
                *pp, leaf = s.split(".")
                if a.dtype != mx.float16:
                    h = a.astype(mx.float16)
                    mx.eval(h)
                    setattr(_resolve(owner, ".".join(pp)), leaf, h)
                    names[id(h)] = names.pop(id(a))  # (the old id may be reused once the fp32 array is freed)

    old = _K.W16_TRACE
    _K.W16_TRACE = hook
    out: list = []
    try:
        yield out
    finally:
        _K.W16_TRACE = old
        out.extend(sorted(groups.values(), key=lambda g: (g["owner"], g["cache"])))


def manifest_entry(model: nn.Module, groups: list) -> dict:
    """{"groups": [...], "params": [all qualified names of storable sources, aliases included]}."""
    names = _param_names(model)
    by_name = {n: ns for ns in names.values() for n in ns}
    params = set()
    for g in groups:
        if g["storable"]:
            for s in g["srcs"]:
                full = f"{g['owner']}.{s}" if g["owner"] else s
                params.update(by_name[full])
    return {"groups": groups, "params": sorted(params)}


def _tri_attention_ref(h, n_heads, c, mask, inf, bias=None, out_dtype=mx.float32, **_):
    """MLX-op stand-in for the Metal triangle-attention core (CPU traces / tests only; weights never pass through it)."""
    B, I, J, F = h.shape
    HC = n_heads * c
    f = mx.float32
    hf = h.astype(f)
    if bias is None:
        bias = hf[..., 4 * HC:].transpose(0, 3, 2, 1)  # [B, H, K, Q]
    tri = bias.astype(f).transpose(0, 1, 3, 2)  # [B, H, Q, K]
    q, k, v = (hf[..., i * HC:(i + 1) * HC].reshape(B * I, J, n_heads, c).transpose(0, 2, 1, 3) for i in range(3))
    s = (q @ k.transpose(0, 1, 3, 2)) * c ** -0.5
    s = s.reshape(B, I, n_heads, J, J) + tri[:, None]
    if mask is not None:
        s = s + (inf * (mask.astype(f) - 1))[:, :, None, None, :]
    p = mx.softmax(s, axis=-1).reshape(B * I, n_heads, J, J)
    o = (p @ v).transpose(0, 2, 1, 3).reshape(B, I, J, HC)
    return (mx.sigmoid(hf[..., 3 * HC:4 * HC]) * o).astype(out_dtype)


@contextmanager
def cpu_f16():
    """Run the --precision f16 code paths on the MLX CPU device (traces and CPU tests; numerics are not GPU f16):
    f16() on, Metal-only pieces off (NAX groups, fused add+LN pair block) or replaced by MLX ops (tri-attention core).
    The fp16 weights are built by the same w16 calls as on the GPU (the GPU-only paths read the same caches)."""
    saved = (_K.f16, _K.PRECISION, _K.DIFF_DTYPE, dict(_K.NAX_FLAGS), _K.FUSE_ADD_LN, _K.TRIATT_F16,
             _K.tri_attention_fused, _K.GUARDS)
    _K.f16 = lambda: _K.PRECISION == "f16"
    _K.PRECISION, _K.DIFF_DTYPE = "f16", mx.float16
    _K.NAX_FLAGS.update({k: "0" for k in _K.NAX_FLAGS})
    _K.FUSE_ADD_LN, _K.TRIATT_F16 = False, "mma"
    _K.tri_attention_fused = _tri_attention_ref
    try:
        yield
    finally:
        (_K.f16, _K.PRECISION, _K.DIFF_DTYPE, nax, _K.FUSE_ADD_LN, _K.TRIATT_F16, _K.tri_attention_fused,
         _K.GUARDS) = saved
        _K.NAX_FLAGS.clear()
        _K.NAX_FLAGS.update(nax)


# ------------------------------------------------------------------------------------------------ conversion
def fp16_range(w) -> dict:
    """Counts for an fp32 tensor converted to fp16: overflow (finite |w| that becomes inf: |w| >= 65520), subnormal
    (fp16 value nonzero and below 2^-14 = 6.1e-5: stored with reduced precision), flushed (nonzero, rounds to 0:
    |w| <= 2^-25)."""
    import numpy as np

    a = np.abs(np.asarray(w, dtype=np.float32))
    nz = a > 0
    h = a.astype(np.float16)
    return {
        "n": int(a.size),
        "overflow": int(np.count_nonzero(~np.isfinite(h) & np.isfinite(a))),
        "subnormal": int(np.count_nonzero((h > 0) & (h < np.float16(FP16_MIN_NORMAL)))),
        "flushed": int(np.count_nonzero(nz & (h == 0))),
        "max_abs": float(a.max()) if a.size else 0.0,
        "min_abs_nonzero": float(a[nz].min()) if nz.any() else 0.0,
    }


def to_mixed(weights: dict, entry: dict) -> tuple[dict, dict]:
    """fp32 checkpoint dict -> (mixed dict, per-tensor fp16 range stats). Raises on any fp16 overflow."""
    params = set(entry["params"])
    missing = params - set(weights)
    if missing:
        raise KeyError(f"manifest parameters missing from the checkpoint: {sorted(missing)[:5]}")
    out, stats = {}, {}
    for k, v in weights.items():
        if k in params:
            if v.dtype != mx.float32:
                raise TypeError(f"{k}: expected fp32 source, got {v.dtype}")
            stats[k] = fp16_range(v)
            out[k] = v.astype(mx.float16)
        else:
            out[k] = v
    bad = {k: s["overflow"] for k, s in stats.items() if s["overflow"]}
    if bad:
        raise OverflowError(f"fp16 overflow (|w| > {FP16_MAX}) in {len(bad)} tensors: {bad}")
    return out, stats


# ------------------------------------------------------------------------------------------------ loading
def _resolve(model: nn.Module, path: str):
    m = model
    for part in path.split(".") if path else []:
        m = m[int(part)] if isinstance(m, list) else getattr(m, part, None)
        if m is None:
            return None
    return m


def prefuse(model: nn.Module, entry: dict) -> int:
    """After loading a mixed checkpoint: build each multi-parameter fp16 weight once, put it in its w16 cache, and make
    the source parameters views into it (their own buffers are then freed), so a concatenated fp16 weight exists once.
    Returns the number of groups fused. Owners missing from this model (e.g. confidence=False) are skipped."""
    groups = [g for g in entry["groups"] if g["storable"] and len(g["srcs"]) > 1]
    seen: dict = {}  # a parameter in two fused groups keeps its own buffer (views of one group only would be wrong)
    for g in groups:
        for s in g["srcs"]:
            seen[f"{g['owner']}.{s}"] = seen.get(f"{g['owner']}.{s}", 0) + 1
    n = 0
    for g in groups:
        owner = _resolve(model, g["owner"])
        if owner is None:
            continue
        parents, arrays = [], []
        for s in g["srcs"]:
            *pp, leaf = s.split(".")
            parent = _resolve(owner, ".".join(pp))
            arrays.append(getattr(parent, leaf))
            parents.append((parent, leaf))
        if any(a.dtype != mx.float16 for a in arrays):
            continue  # not a mixed parameter set (should not happen); w16 casts at first use
        h = mx.concatenate(arrays, 0)
        mx.eval(h)
        owner.__dict__[g["cache"]] = h
        if all(seen[f"{g['owner']}.{s}"] == 1 for s in g["srcs"]):
            r = 0
            for (parent, leaf), a in zip(parents, arrays):
                setattr(parent, leaf, h[r:r + a.shape[0]])
                r += a.shape[0]
        n += 1
    return n


def load_mixed(name: str, path: Path, entry: dict, tops: set) -> list:
    """(key, array) list from the mixed checkpoint after checking its format and manifest hash."""
    w, meta = mx.load(str(path), return_metadata=True)
    if meta.get("format") != FORMAT or meta.get("manifest_sha") != entry_sha(entry):
        raise ValueError(f"{path}: format {meta.get('format')!r} / manifest {meta.get('manifest_sha')!r} do not match "
                         f"this build ({FORMAT} / {entry_sha(entry)}); re-run tools/make_mixed_checkpoint.py")
    return [(k, v) for k, v in w.items() if k.split(".")[0] in tops]


def want_mixed() -> bool:
    """f16 mode as shipped (fp16 trunk GEMMs on the GPU and fp16 diffusion GEMMs), unless BOLTZ_MLX_MIXED=0."""
    return _K.f16() and _K.DIFF_DTYPE == mx.float16 and os.environ.get("BOLTZ_MLX_MIXED", "1") != "0"


def warn(msg: str) -> None:
    print(f"boltz_mlx: {msg}", file=sys.stderr)
