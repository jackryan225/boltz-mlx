"""Mixed fp16/fp32 checkpoint (boltz_mlx/mixed.py, tools/make_mixed_checkpoint.py).

CPU (default):
  - the fp16 weight groups a small synthetic model builds in an f16 forward (kernels.w16 trace) are, role by role,
    the committed manifest's groups (the manifest is derived from the code and stays in sync with it);
  - f16 mode with the mixed weights (+ prefuse) is bit-identical to f16 mode with fp32 weights + runtime w16 casts,
    on the small conf model (prot117 features) and the small aff model (lig_aff affinity features);
  - mixed loading keeps no fp32 copy of a converted weight and stores each concatenated fp16 weight once;
  - the f16 guard fallback (fp32 recompute of a recycle) runs with fp16 parameters;
  - fp16 range counting and the overflow failure;
  - the real mixed files (if built): header keys / dtypes match the manifest, load_model picks them only in f16 mode.
GPU (BOLTZ_MLX_DEVICE=gpu): the shipped GPU f16 paths on prot117 (structure pass) and lig_aff
(affinity pass): load_model(weights="mixed") == load_model(weights="fp32") bit for bit.
"""

from __future__ import annotations

import json
import re
import struct
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

sys.path.insert(0, str(Path(__file__).parent))
from conftest import DATA  # noqa: E402
from mixed_small import checkpoint, load_into, small_model  # noqa: E402

from boltz_mlx import kernels as K  # noqa: E402
from boltz_mlx import mixed as MX  # noqa: E402
from boltz_mlx.model import PredictArgs  # noqa: E402
from boltz_mlx.sampler import MLXNoise  # noqa: E402

CPU = mx.default_device() == mx.cpu
CASES = {"conf": ("prot117", "feats.npz", PredictArgs(1, 2, 2)), "aff": ("lig_aff", "feats_affinity.npz", PredictArgs(0, 2, 1))}


@pytest.fixture(autouse=True, scope="module")
def _small_buffer_cache():
    """Freed MLX buffers go back to the OS (memory rule: one test process stays well under 4 GB)."""
    old = mx.set_cache_limit(0)
    yield
    mx.clear_cache()
    mx.set_cache_limit(old)


def _feats(name):
    case, fname, _ = CASES[name]
    p = DATA / case / fname
    if not p.exists():
        pytest.skip(f"missing {p}")
    z = np.load(p)
    return {k: mx.array(z[k]) for k in z.files}


def _run(model, feats, args):
    out = model(feats, args, MLXNoise(0))
    arrs = {k: v for k, v in out.items() if isinstance(v, mx.array)}
    mx.eval(arrs)
    return {k: np.array(v) for k, v in arrs.items()}, out


def _role(owner: str) -> str:
    return re.sub(r"\.\d+(?=\.|$)", ".*", owner)


def _roles(groups) -> set:
    # (a group over all layers of a transformer, e.g. _cwh_raw, has one source role per layer: deduplicated)
    return {(_role(g["owner"]), g["cache"], tuple(dict.fromkeys(_role(s) for s in g["srcs"])) if g["storable"] else None)
            for g in groups}


def _assert_bitwise(a: dict, b: dict):
    assert set(a) == set(b)
    bad = {k: float(np.nanmax(np.abs(a[k].astype(np.float64) - b[k]))) for k in a
           if not np.array_equal(a[k], b[k], equal_nan=True) or a[k].dtype != b[k].dtype}
    assert not bad, f"mixed != runtime cast: {bad}"


# ------------------------------------------------------------------------------------------------ CPU, small models
@pytest.fixture(scope="module", params=["conf", "aff"])
def small_run(request):
    """(name, fp32 checkpoint dict, traced groups, runtime-cast outputs) of a small model in CPU f16 mode."""
    if not CPU:
        pytest.skip("CPU emulation of the f16 paths (GPU runs the shipped paths below)")
    name = request.param
    feats = _feats(name)
    model = small_model(name)
    sd = checkpoint(model)
    with MX.cpu_f16():
        with MX.trace_fp16_weights(model) as groups:
            ref, _ = _run(model, feats, CASES[name][2])
    entry = MX.manifest_entry(model, groups)
    del model
    return name, feats, sd, groups, entry, ref


def test_trace_matches_committed_manifest(small_run):
    """Every fp16 weight site of the f16 code paths (by module role) is in the committed manifest and vice versa: a
    code change that adds, removes or rewires an fp16 GEMM weight fails here until the manifest is re-derived."""
    name, _, _, groups, _, _ = small_run
    committed = MX.load_manifest()[name]["groups"]
    assert _roles(groups) == _roles(committed)
    derived = [g for g in groups if not g["storable"]]
    assert {g["cache"] for g in derived} == {"_cwh_ln"}, derived  # only the LN-folded AdaLN weights are cast at runtime


def test_manifest_params_are_storable_sources():
    m = MX.load_manifest()
    for name, e in m.items():
        assert e["params"] == sorted(set(e["params"]))
        srcs = {f"{g['owner']}.{s}" for g in e["groups"] if g["storable"] for s in g["srcs"]}
        assert srcs <= set(e["params"])
        extra = set(e["params"]) - srcs  # aliases only (output_projection.0 == output_projection_linear)
        assert all(".output_projection.0." in k for k in extra), sorted(extra)[:5]
        assert len({(g["owner"], g["cache"]) for g in e["groups"]}) == len(e["groups"])


def test_mixed_bit_identical_cpu(small_run):
    """f16 mode: mixed weights (+ prefuse) == fp32 weights + runtime w16 casts, bit for bit, all outputs."""
    name, feats, sd, _, entry, ref = small_run
    mixed, stats = MX.to_mixed(sd, entry)
    assert all(s["overflow"] == 0 for s in stats.values())
    model = small_model(name, seed=1)  # different init, overwritten by the mixed weights
    load_into(model, mixed)
    del mixed
    n = MX.prefuse(model, entry)
    assert n == sum(1 for g in entry["groups"] if g["storable"] and len(g["srcs"]) > 1)
    params = dict(tree_flatten(model.parameters()))
    conv = set(entry["params"])
    assert all(params[k].dtype == mx.float16 for k in conv)  # no fp32 copy of a converted weight is held
    assert all(v.dtype == mx.float32 for k, v in params.items() if k not in conv)
    with MX.cpu_f16():
        out, _ = _run(model, feats, CASES[name][2])
    _assert_bitwise(ref, out)


def test_prefuse_shares_buffers(small_run):
    """After prefuse a concatenated fp16 weight exists once: the w16 cache, with the parameters as views into it."""
    name, _, sd, _, entry, _ = small_run
    mixed, _ = MX.to_mixed(sd, entry)
    model = small_model(name, seed=1)
    load_into(model, mixed)
    del mixed, sd
    g = next(g for g in entry["groups"] if g["cache"] == "_w12h" and g["owner"].startswith("pairformer_module"))
    owner = MX._resolve(model, g["owner"])
    mx.eval(model.parameters())
    mx.synchronize()
    a0 = mx.get_active_memory()
    MX.prefuse(model, entry)
    mx.eval(model.parameters())
    h = owner.__dict__["_w12h"]
    np.testing.assert_array_equal(np.array(h[: owner.fc1.weight.shape[0]]), np.array(owner.fc1.weight))
    a1 = mx.get_active_memory()
    # the source buffers are released (views share the fused buffer): active memory does not grow by the fused size
    fused = sum(2 * int(np.prod(gg["shape"])) for gg in entry["groups"] if gg["storable"] and len(gg["srcs"]) > 1)
    assert a1 - a0 < 0.5 * fused, (a0, a1, fused)


def test_guard_fallback_runs_with_fp16_params(monkeypatch):
    """A trunk guard hit recomputes the recycle with fp32 GEMMs; with the mixed weights the fp32 paths read fp16
    parameters (promoted). It must run and stay finite (not bit-identical to the runtime-cast fallback, mixed.py)."""
    if not CPU:
        pytest.skip("CPU")
    name = "conf"
    feats = _feats(name)
    model = small_model(name)
    entry = MX.manifest_entry(model, _trace_groups(model, feats))
    model = small_model(name)
    load_into(model, MX.to_mixed(checkpoint(model), entry)[0])
    MX.prefuse(model, entry)
    calls = {"n": 0}
    real = K.guard_take

    def fake():
        real()
        calls["n"] += 1
        return 1 if calls["n"] == 1 else 0

    with MX.cpu_f16():
        monkeypatch.setattr(K, "guard_take", fake)
        out, raw = _run(model, feats, PredictArgs(1, 2, 1))
    assert raw["fp16_fallbacks"] == 1
    assert np.isfinite(out["sample_atom_coords"]).all() and np.isfinite(out["z"]).all()


def _trace_groups(model, feats):
    with MX.cpu_f16():
        with MX.trace_fp16_weights(model) as groups:
            _run(model, feats, PredictArgs(0, 2, 1))
    return groups


def test_fp16_range_and_overflow():
    w = np.array([0.0, 1.0, -70000.0, 1e-5, -3e-6, 1e-9, 65504.0, 2.0 ** -14], np.float32)
    s = MX.fp16_range(mx.array(w))
    assert s == {"n": 8, "overflow": 1, "subnormal": 2, "flushed": 1, "max_abs": 70000.0,
                 "min_abs_nonzero": pytest.approx(1e-9)}
    e = {"groups": [], "params": ["a"]}
    with pytest.raises(OverflowError):
        MX.to_mixed({"a": mx.array(w), "b": mx.array(w)}, e)
    out, st = MX.to_mixed({"a": mx.array(w[:2]), "b": mx.array(w)}, e)
    assert out["a"].dtype == mx.float16 and out["b"].dtype == mx.float32 and set(st) == {"a"}


# ------------------------------------------------------------------------------------------------ real files (CPU ok)
def _header(path: Path) -> tuple[dict, dict]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    return h.pop("__metadata__", {}), h


@pytest.mark.parametrize("name", ["conf", "aff"])
def test_mixed_file_matches_manifest(name):
    p = MX.mixed_path(name)
    if not p.exists():
        pytest.skip(f"{p} not built (tools/make_mixed_checkpoint.py)")
    meta, h = _header(p)
    _, h32 = _header(p.with_name(f"boltz2_{name}.safetensors"))
    e = MX.load_manifest()[name]
    assert meta["format"] == MX.FORMAT and meta["manifest_sha"] == MX.entry_sha(e)
    assert set(h) == set(h32)
    conv = set(e["params"])
    assert {k for k, v in h.items() if v["dtype"] == "F16"} == conv
    assert all(h[k]["shape"] == h32[k]["shape"] for k in h)
    assert all(h32[k]["dtype"] == "F32" for k in conv)


def test_load_model_selects_checkpoint(monkeypatch, capsys):
    """auto: fp32 outside f16 mode (strict / fast / turbo); mixed in f16 mode; fp32 + one warning line if the mixed
    file is missing. Model construction is patched out (no 2 GB load in a CPU test)."""
    from boltz_mlx import model as M

    loaded = []

    class Stop(Exception):
        pass

    def fake_load(path, *a, **k):
        loaded.append(Path(path).name)
        raise Stop

    monkeypatch.setattr(M, "Boltz2", lambda hp, **kw: _Tiny())
    monkeypatch.setattr(M.mx, "load", fake_load)
    monkeypatch.setattr(MX, "load_mixed", lambda name, path, entry, tops: fake_load(path))
    monkeypatch.setattr(K, "PRECISION", "fp32")
    with pytest.raises(Stop):
        M.load_model("conf")
    assert loaded[-1] == "boltz2_conf.safetensors"
    monkeypatch.setattr(K, "f16", lambda: True)
    monkeypatch.setattr(K, "DIFF_DTYPE", mx.float16)
    monkeypatch.setattr(MX, "mixed_path", lambda name: Path("/nonexistent") / f"boltz2_{name}_mixed.safetensors")
    with pytest.raises(Stop):
        M.load_model("conf")
    assert loaded[-1] == "boltz2_conf.safetensors"
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "mixed checkpoint" in err
    monkeypatch.setattr(MX, "mixed_path", lambda name: Path(__file__))  # exists
    with pytest.raises(Stop):
        M.load_model("conf")
    assert loaded[-1] == Path(__file__).name
    monkeypatch.setenv("BOLTZ_MLX_MIXED", "0")
    with pytest.raises(Stop):
        M.load_model("conf")
    assert loaded[-1] == "boltz2_conf.safetensors"


class _Tiny:
    def children(self):
        return {}


# ------------------------------------------------------------------------------------------------ GPU
@pytest.mark.skipif(CPU, reason="GPU f16 paths (NAX, fused add+LN); BOLTZ_MLX_DEVICE=gpu")
@pytest.mark.parametrize("name", ["conf", "aff"])
def test_mixed_bit_identical_gpu(name, monkeypatch):
    """Shipped GPU f16 mode: load_model(weights="mixed") == load_model(weights="fp32") (runtime w16 casts), bit for
    bit: prot117 structure pass (3 recycles, 50 steps, confidence) / lig_aff affinity pass (5 recycles, 50 steps,
    5 samples). One model resident at a time."""
    from boltz_mlx.model import load_model

    if not MX.mixed_path(name).exists():
        pytest.skip("mixed checkpoint not built")
    monkeypatch.setattr(K, "PRECISION", "f16")
    monkeypatch.setattr(K, "DIFF_DTYPE", mx.float16)
    assert K.f16()
    case = {"conf": ("prot117", "feats.npz", PredictArgs(3, 50, 1)),
            "aff": ("lig_aff", "feats_affinity.npz", PredictArgs(5, 50, 5))}[name]
    z = np.load(DATA / case[0] / case[1])
    feats = {k: mx.array(z[k]) for k in z.files}
    outs = []
    for kind in ("fp32", "mixed"):
        model = load_model(name, weights=kind)
        assert model.weights_kind == kind
        o, raw = _run(model, feats, case[2])
        assert raw["fp16_fallbacks"] == 0
        outs.append(o)
        del model, raw
        mx.clear_cache()
    _assert_bitwise(*outs)


@pytest.mark.skipif(CPU, reason="GPU fp32 fallback paths (custom kernels); BOLTZ_MLX_DEVICE=gpu")
def test_guard_fallback_gpu_mixed(monkeypatch):
    """Mixed weights + a forced trunk guard hit on the GPU: the fp32 recompute (Metal fp32 kernels, fp16 parameters
    promoted in MLX GEMMs; transition_supported() keeps fp16 weights away from the fp32-only fused transition) runs
    and stays finite; prot117, 1 recycle, 10 steps."""
    from boltz_mlx.model import load_model

    if not MX.mixed_path("conf").exists():
        pytest.skip("mixed checkpoint not built")
    monkeypatch.setattr(K, "PRECISION", "f16")
    monkeypatch.setattr(K, "DIFF_DTYPE", mx.float16)
    z = np.load(DATA / "prot117" / "feats.npz")
    feats = {k: mx.array(z[k]) for k in z.files}
    model = load_model("conf", weights="mixed")
    calls = {"n": 0}
    real = K.guard_take

    def fake():
        real()
        calls["n"] += 1
        return 1 if calls["n"] == 1 else 0

    monkeypatch.setattr(K, "guard_take", fake)
    out, raw = _run(model, feats, PredictArgs(1, 10, 1))
    assert raw["fp16_fallbacks"] == 1
    assert np.isfinite(out["sample_atom_coords"]).all() and np.isfinite(out["plddt"]).all()
