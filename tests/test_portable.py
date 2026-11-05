"""Portability (M1-M5): kernel-path detection, the MPP guard, device-aware labels and per-device configs.

Kernel paths (boltz_mlx/kernels.py "Hardware capability"): nax (M5), tensorops (MPP kernels on M1-M4 shader cores,
opt-in, only after the on-device tensor-op check), portable (no MetalPerformancePrimitives source is ever built).

CPU (default; nothing is compiled or dispatched - mx.fast.metal_kernel is replaced by a recorder that returns zeros):
  - detection logic for every (architecture, env, check result) combination, the per-device + OS disk cache;
  - on the portable path every MPP kernel group is off and kernels.metal_kernel refuses MPP sources;
  - a full f16 predict (small synthetic conf + aff models on the prot117 / lig_aff features, every module class of the
    model, the GPU code paths taken by pretending the device is a GPU) builds NO MPP source on the portable path, and
    does build MPP sources on the nax path (positive control: the recorder sees them when they exist);
  - labels (f16 -> f16-portable / f16-tensorops), screen presets, run_l3 arms, per-device tuned configs.
GPU (BOLTZ_MLX_DEVICE=gpu): the same predict with the real kernels on the portable path builds no
MPP source and is finite; the tensor-op check passes on NAX hardware; portable vs nax outputs of the small models agree
to fp16 noise.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from conftest import DATA  # noqa: E402

from boltz_mlx import kernels as K  # noqa: E402

CPU = mx.default_device() == mx.cpu
KERNEL_MODULES = ("kernels", "kernels_nax", "kernels_nax2", "kernels_nax3", "kernels_msa", "kernels_atom",
                  "kernels_difftx")


def _pin(monkeypatch, path):
    monkeypatch.setattr(K, "_CAP", K.pinned_status(path, f"test pin {path}"))


# ------------------------------------------------------------------------------------------------ detection (CPU)
def test_arch_generation():
    assert K.arch_generation("applegpu_g17s") == 17
    assert K.arch_generation("applegpu_g13g") == 13
    assert K.arch_generation("applegpu_g16p") == 16
    assert K.arch_generation("") is None and K.arch_generation("amd_gfx") is None


def _detect(monkeypatch, arch, env=None, check=(True, "ok"), macos="26.2"):
    calls = []

    def fake_check(rec):
        calls.append(rec["architecture"])
        return check[0], check[1], False

    monkeypatch.setattr(K, "gpu_info", lambda: {"chip": "Apple Mx", "architecture": arch,
                                                "gpu_gen": K.arch_generation(arch), "memory_gb": 16.0})
    monkeypatch.setattr(K, "os_info", lambda: {"macos": macos, "os_build": "25A1"})
    monkeypatch.setattr(K, "tensorops_check", fake_check)
    monkeypatch.setattr(K.mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(K, "_WARNED", [])
    for k in ("BOLTZ_MLX_KERNEL_PATH", "BOLTZ_MLX_NAX"):
        monkeypatch.delenv(k, raising=False)
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    return K._detect(), calls


@pytest.mark.parametrize("arch", ["applegpu_g13g", "applegpu_g14s", "applegpu_g15d", "applegpu_g16s"])
def test_m1_to_m4_tensorops_after_the_check(monkeypatch, capsys, arch):
    """Owner decision 2026-10-28 22:30: M1-M4 run the MPP kernels on the shader cores (tensorops), gated by the
    on-device check; a failed check falls back to the portable path with a loud warning."""
    s, calls = _detect(monkeypatch, arch)
    assert s["path"] == "tensorops" and s["mpp"] and not s["nax"] and not s["nax_hw"] and calls == [arch]
    assert capsys.readouterr().err == ""
    s, _ = _detect(monkeypatch, arch, check=(False, "tensor-op check WRONG on this device (layout_tb1_float16)"))
    assert s["path"] == "portable" and not s["mpp"] and s["reason"].startswith("FALLBACK")
    err = capsys.readouterr().err
    assert "WARNING" in err and "FALLBACK" in err and "layout_tb1_float16" in err


def test_m5_nax_needs_the_check(monkeypatch):
    s, calls = _detect(monkeypatch, "applegpu_g17s")
    assert s["path"] == "nax" and s["nax"] and s["mpp"] and calls == ["applegpu_g17s"]
    s, _ = _detect(monkeypatch, "applegpu_g17s", check=(False, "tensor-op check failed: RuntimeError: x"))
    assert s["path"] == "portable" and not s["mpp"]


def test_mlx_nax_rules(monkeypatch):
    """Same NAX hardware rule as MLX (is_nax_available): gen >= 17, >= 18 for 'p' architectures; on macOS < 26.2 an M5
    still takes the nax path (our MPP kernels need Metal 4 only) with a note (MLX's own GEMMs skip NAX there)."""
    assert K.nax_hardware("applegpu_g17s") and K.nax_hardware("applegpu_g18p")
    assert not K.nax_hardware("applegpu_g17p") and not K.nax_hardware("applegpu_g16s") and not K.nax_hardware("")
    s, calls = _detect(monkeypatch, "applegpu_g17p")
    assert s["path"] == "tensorops" and calls and not s["nax_hw"]
    s, calls = _detect(monkeypatch, "applegpu_g17s", macos="26.1")
    assert s["path"] == "nax" and calls and "26.2" in s["reason"]


def test_requests(monkeypatch, capsys):
    for env in ({"BOLTZ_MLX_KERNEL_PATH": "portable"}, {"BOLTZ_MLX_NAX": "0"}):
        s, calls = _detect(monkeypatch, "applegpu_g17s", env)
        assert s["path"] == "portable" and calls == []
    assert capsys.readouterr().err == ""  # a requested debug path is not a fallback: no warning
    s, _ = _detect(monkeypatch, "applegpu_g14s", {"BOLTZ_MLX_KERNEL_PATH": "nax"})  # the hardware decides the label
    assert s["path"] == "tensorops"
    s, _ = _detect(monkeypatch, "applegpu_g17s", {"BOLTZ_MLX_KERNEL_PATH": "tensorops"})
    assert s["path"] == "nax"
    s, _ = _detect(monkeypatch, "applegpu_g13g", {"BOLTZ_MLX_NAX": "1"})
    assert s["path"] == "tensorops"
    s, calls = _detect(monkeypatch, "weird_gpu")
    assert s["path"] == "tensorops" and calls  # unknown architecture: the check decides
    s, calls = _detect(monkeypatch, "applegpu_g13g", macos="15.7")  # only with BOLTZ_MLX_ALLOW_OLD_MACOS=1
    assert s["path"] == "portable" and calls == [] and "Metal 4" in s["reason"]
    monkeypatch.setenv("BOLTZ_MLX_KERNEL_PATH", "fast")
    with pytest.raises(ValueError):
        K._requested()


def test_macos_26_required(monkeypatch):
    import boltz_mlx

    monkeypatch.delenv("BOLTZ_MLX_ALLOW_OLD_MACOS", raising=False)
    for v in ("26.0", "26.6.2", "27.1", ""):
        monkeypatch.setattr(boltz_mlx, "macos_version", lambda v=v: v)
        boltz_mlx.require_macos()
    for v in ("15.7.1", "14.0"):
        monkeypatch.setattr(boltz_mlx, "macos_version", lambda v=v: v)
        with pytest.raises(RuntimeError, match="requires macOS 26"):
            boltz_mlx.require_macos()
    monkeypatch.setenv("BOLTZ_MLX_ALLOW_OLD_MACOS", "1")
    boltz_mlx.require_macos()
    assert boltz_mlx.macos_version().split(".")[0] not in ("16",)  # compatibility numbering mapped


def test_tensorops_check_cached_per_device_and_os(monkeypatch, tmp_path):
    runs = []

    def fake(*_):
        runs.append(1)
        return True, "tensor-op check passed", {"mma_tb1_float16": 0.0}

    monkeypatch.setattr(K, "_tensorops_check", fake)
    cache = tmp_path / "cap.json"
    monkeypatch.setenv("BOLTZ_MLX_CAP_CACHE", str(cache))
    rec = {"chip": "Apple M1", "architecture": "applegpu_g13g", "macos": "26.0", "os_build": "25A1", "mlx": "0.32.4"}
    assert K.tensorops_check(rec) == (True, "tensor-op check passed", False)
    assert K.tensorops_check(rec) == (True, "tensor-op check passed", True) and len(runs) == 1
    assert K.tensorops_check({**rec, "os_build": "25B2"})[2] is False and len(runs) == 2  # OS update -> re-check
    assert K.tensorops_check({**rec, "chip": "Apple M2"})[2] is False and len(runs) == 3
    assert len(json.loads(cache.read_text())) == 3
    monkeypatch.setenv("BOLTZ_MLX_CAP_CACHE", "0")
    K.tensorops_check(rec)
    assert len(runs) == 4  # cache off


# ------------------------------------------------------------------------------------------------ gating + guard
def _pretend_gpu_f16(monkeypatch):
    monkeypatch.setattr(K, "_use_kernels", lambda: True)
    monkeypatch.setattr(K, "PRECISION", "f16")
    monkeypatch.setattr(K, "DIFF_DTYPE", mx.float16)


def test_every_mpp_group_off_on_the_portable_path(monkeypatch):
    _pretend_gpu_f16(monkeypatch)
    _pin(monkeypatch, "portable")
    for g in K.NAX_FLAGS:
        assert K.nax(g) == (g in K.NAX_INDEPENDENT), g
        monkeypatch.setitem(K.NAX_FLAGS, g, "1")  # forcing a group by env does not override the path
        assert K.nax(g) == (g in K.NAX_INDEPENDENT), g
    assert not K.triatt_nax() and not K.msa_opm_nax()
    assert K.msa_kernel("OPM") and K.msa_kernel("PWA")  # the non-MPP MSA kernels stay on
    assert K.mode_label("f16") == "f16-portable" and K.mode_label("strict") == "strict"
    for path, label in (("nax", "f16"), ("tensorops", "f16-tensorops")):
        _pin(monkeypatch, path)
        assert K.triatt_nax() and K.mode_label("f16") == label
        assert all(K.nax(g) for g in ("TRANSITION", "PAIREPI", "ATOM", "GEMM"))


def _recorder(rec):
    def make(**kw):
        rec.append((kw["name"], K.is_mpp_source(kw.get("source", ""), kw.get("header", ""))))

        def call(inputs=None, template=None, grid=None, threadgroup=None, output_shapes=(), output_dtypes=(), **_):
            return [mx.zeros(tuple(s), d) for s, d in zip(output_shapes, output_dtypes)]

        return call

    return make


def test_guard_refuses_mpp_sources_on_the_portable_path(monkeypatch):
    from boltz_mlx import kernels_nax as KN

    rec = []
    monkeypatch.setattr(mx.fast, "metal_kernel", _recorder(rec))
    _pin(monkeypatch, "portable")
    with pytest.raises(K.NaxUnavailableError):
        K.metal_kernel(name="x", input_names=["a"], output_names=["o"], source="o[0] = a[0];", header=KN.NAX_HEADER)
    with pytest.raises(K.NaxUnavailableError):
        K.metal_kernel(name="y", input_names=["a"], output_names=["o"], source="mpp::tensor_ops::matmul2d<d> op;")
    assert rec == []
    K.metal_kernel(name="z", input_names=["a"], output_names=["o"], source="o[0] = simd_sum(a[0]);",
                   header="#include <metal_simdgroup_matrix>\n")
    with K.probing():  # the capability check's own kernels
        K.metal_kernel(name="p", input_names=["a"], output_names=["o"], source="o[0] = a[0];", header=KN.NAX_HEADER)
    assert rec == [("z", False), ("p", True)]
    _pin(monkeypatch, "tensorops")
    K.metal_kernel(name="t", input_names=["a"], output_names=["o"], source="o[0] = a[0];", header=KN.NAX_HEADER)
    assert rec[-1] == ("t", True)


# ------------------------------------------------------------------------------------------------ full f16 predict
CASES = {"conf": ("prot117", "feats.npz", (1, 2, 2)), "aff": ("lig_aff", "feats_affinity.npz", (0, 2, 1))}


def _feats(name):
    case, fname, _ = CASES[name]
    p = DATA / case / fname
    if not p.exists():
        pytest.skip(f"missing {p}")
    z = np.load(p)
    return {k: mx.array(z[k]) for k in z.files}


def _predict(name):
    from mixed_small import small_model

    from boltz_mlx.model import PredictArgs
    from boltz_mlx.sampler import MLXNoise

    model = small_model(name)
    out = model(_feats(name), PredictArgs(*CASES[name][2]), MLXNoise(0))
    arrs = {k: v for k, v in out.items() if isinstance(v, mx.array)}
    mx.eval(arrs)
    res = {k: np.array(v.astype(mx.float32)) for k, v in arrs.items()}
    del model, out, arrs
    mx.clear_cache()
    return res


def _clear_kernel_caches(monkeypatch):
    import importlib

    for m in KERNEL_MODULES:
        monkeypatch.setattr(importlib.import_module(f"boltz_mlx.{m}"), "_kernels", {})


@pytest.fixture
def small_buffers():
    old = mx.set_cache_limit(0)
    yield
    mx.clear_cache()
    mx.set_cache_limit(old)


@pytest.mark.skipif(not CPU, reason="CPU logic check (the GPU variant below runs the real kernels)")
@pytest.mark.parametrize("name", ["conf", "aff"])
def test_portable_f16_predict_builds_no_mpp_source(monkeypatch, name, small_buffers):
    """Every Metal kernel a full f16 predict constructs (GPU code paths, kernels replaced by a zero-returning
    recorder): none includes MetalPerformancePrimitives on the portable path; the nax path does build MPP kernels
    (so the recorder would see one)."""
    _pretend_gpu_f16(monkeypatch)
    found = {}
    for path in ("portable", "nax"):
        rec = []
        _clear_kernel_caches(monkeypatch)
        monkeypatch.setattr(mx.fast, "metal_kernel", _recorder(rec))
        _pin(monkeypatch, path)
        _predict(name)
        found[path] = rec
    names = {n for n, _ in found["portable"]}
    assert len(names) >= 5, names  # LN, add_ln, tri-attention MMA, MSA / difftx ... kernels were built
    assert not [n for n, m in found["portable"] if m], "MPP kernel built on the portable path"
    assert [n for n, m in found["nax"] if m], "positive control: the nax path builds MPP kernels"
    print(sorted(names), sorted({n for n, m in found["nax"]}))
    print(f"{name}: portable path built {len(names)} kernels, 0 MPP; nax path "
          f"{len({n for n, m in found['nax'] if m})} MPP kernels")


@pytest.mark.skipif(not CPU, reason="CPU logic check")
def test_portable_msa_opm_uses_the_simdgroup_kernel(monkeypatch):
    """Many MSA rows (S * 32 > OPM_REASSOC_MAX_K: not reached by the prot117 / lig_aff features above): the OPM runs
    kernels_msa.opm_outer (fp32 simdgroup MMA) + the MLX fp16 proj_o on the portable path, opm_outer_proj (MPP) on nax."""
    from boltz_mlx.layers.outer_product_mean import OuterProductMean

    _pretend_gpu_f16(monkeypatch)
    mx.random.seed(0)
    mod = OuterProductMean(64, 32, 128)
    m, mask = mx.random.normal((1, 40, 24, 64)), mx.ones((1, 40, 24))
    got = {}
    for path in ("portable", "nax"):
        rec = []
        _clear_kernel_caches(monkeypatch)
        monkeypatch.setattr(mx.fast, "metal_kernel", _recorder(rec))
        _pin(monkeypatch, path)
        mx.eval(mod(m, mask))
        got[path] = rec
    assert "boltz_msa_opm_outer" in [n for n, _ in got["portable"]] and not any(mpp for _, mpp in got["portable"])
    assert ("boltz_msa_opm_outer_proj", True) in got["nax"], got["nax"]


@pytest.mark.skipif(CPU, reason="GPU: real kernels (BOLTZ_MLX_DEVICE=gpu)")
@pytest.mark.parametrize("name", ["conf", "aff"])
def test_gpu_portable_f16_predict_no_mpp_and_close_to_nax(monkeypatch, name, small_buffers):
    """GPU: the small models' f16 predict on the portable path builds no MPP kernel (real kernels, wrapped recorder),
    is finite, and (on NAX hardware) agrees with the nax path to fp16 noise."""
    monkeypatch.setattr(K, "PRECISION", "f16")
    monkeypatch.setattr(K, "DIFF_DTYPE", mx.float16)
    orig = mx.fast.metal_kernel
    rec = []

    def wrap(**kw):
        rec.append((kw["name"], K.is_mpp_source(kw.get("source", ""), kw.get("header", ""))))
        return orig(**kw)

    monkeypatch.setattr(mx.fast, "metal_kernel", wrap)
    _clear_kernel_caches(monkeypatch)
    real = K.nax_status()
    _pin(monkeypatch, "portable")
    port = _predict(name)
    assert rec and not [n for n, m in rec if m], [n for n, m in rec if m]
    for k, v in port.items():
        assert np.isfinite(v).all(), k
    if not real["nax"]:
        pytest.skip(f"no NAX path on this GPU ({real['reason']}): portable-only check done")
    monkeypatch.setattr(K, "_CAP", real)
    _clear_kernel_caches(monkeypatch)
    rec.clear()
    ref = _predict(name)
    assert [n for n, m in rec if m]
    worst = {}
    for k in ref:
        a, b = port[k].astype(np.float64), ref[k].astype(np.float64)
        d = np.linalg.norm(b)
        worst[k] = float(np.linalg.norm(a - b) / d) if d > 0 else float(np.abs(a - b).max())
    print(f"{name}: portable vs nax rel_fro {worst}")
    assert max(worst.values()) < 0.1, worst  # random-weight small models; real-weight accuracy = L1 / portable_check


@pytest.mark.skipif(CPU, reason="GPU device check")
def test_gpu_tensorops_check_on_this_device():
    """The tensor-op check itself (matmul2d exact through MLX fragments, cooperative-tensor layout == MLX's, generic
    API exact). Must pass on NAX hardware; elsewhere it reports (the result decides whether tensorops is allowed)."""
    import platform

    if int((platform.mac_ver()[0] or "0").split(".")[0]) < 26:
        pytest.skip("Metal 4 (macOS 26) absent: tensor ops unavailable, portable path only")
    ok, detail, res = K._tensorops_check()
    print(f"{K.gpu_info()}: {detail} {res}")
    if K.nax_status()["nax_hw"]:
        assert ok, (detail, res)


# ------------------------------------------------------------------------------------------------ labels / configs
def test_screen_presets_device_aware():
    from boltz_mlx.screen.presets import PRESETS, estimate_measured_here, for_device

    assert for_device(PRESETS["f16"], "nax") is PRESETS["f16"]
    assert for_device(PRESETS["strict"], "portable") is PRESETS["strict"]
    p = for_device(PRESETS["f16"], "portable")
    assert p.name == "f16-portable" and p.env == PRESETS["f16"].env and not p.validated
    assert for_device(PRESETS["screen"], "tensorops").name == "screen-tensorops"
    assert for_device(p, "portable") is p
    assert estimate_measured_here(PRESETS["f16"], "Apple M5 Pro")
    assert not estimate_measured_here(p, "Apple M5 Pro") and not estimate_measured_here(PRESETS["strict"], "Apple M1")


def test_screen_dry_run_says_not_measured(tmp_path, monkeypatch):
    from boltz_mlx.screen.inputs import Ligand, Target
    from boltz_mlx.screen.presets import PRESETS, for_device
    from boltz_mlx.screen.runner import ScreenConfig, plan_lines

    p = for_device(PRESETS["f16"], "portable")
    t = Target(doc={"sequences": []}, ligand_chain="L", n_residues=10, msa_source="empty", msa_paths=[], source="t.yaml")
    lig = Ligand(ligand_id="x", smiles="CCO", index=0, record_id="x", smiles_used="CCO", n_atoms=3)
    hw = {"chip": "Apple M1", "architecture": "applegpu_g13g", "path": "portable", "nax": False}
    cfg = ScreenConfig(out=tmp_path, preset=p, steps=p.steps, recycling=3, affinity_steps=200, hw=hw)
    text = "\n".join(plan_lines(cfg, t, [lig], set()))
    assert "f16-portable" in text and "ESTIMATE (not a measurement; not measured on this chip (Apple M1" in text
    assert "kernel path portable" in text


def test_run_l3_arm_label():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "validation"))
    import run_l3

    assert run_l3.hw_mode("f16", {"path": "nax"}) == "f16"
    assert run_l3.hw_mode("f16", {"path": "portable"}) == "f16-portable"
    assert run_l3.hw_mode("f16", {"path": "tensorops"}) == "f16-tensorops"
    assert run_l3.hw_mode("strict", {"path": "portable"}) == "strict" and run_l3.hw_mode("f16", None) == "f16"


def test_device_cfg(monkeypatch, tmp_path, capsys):
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps({"default": {"x": "1", "tuned_on": "Apple M5 Pro"}, "Apple M5 Pro": {"x": "1"},
                             "Apple M5 Max": {"x": "2"}}))
    monkeypatch.setattr(K, "chip_name", lambda: "Apple M5 Max")
    monkeypatch.setattr(K, "_CFG_NOTES", {"nax": [], "msa": []})
    assert K.device_cfg(str(p), "nax", "tool")["x"] == "2" and K._CFG_NOTES["nax"] == []
    monkeypatch.setattr(K, "chip_name", lambda: "Apple M5 Ultra")
    assert K.device_cfg(str(p), "nax", "tools/prof/nax3_ab.py")["x"] == "1"
    assert len(K._CFG_NOTES["nax"]) == 1 and "'Apple M5 Ultra'" in K._CFG_NOTES["nax"][0]
    _pretend_gpu_f16(monkeypatch)
    _pin(monkeypatch, "nax")
    K.nax("TRANSITION")  # first use of the MPP configs prints the note once
    K.nax("TRANSITION")
    err = capsys.readouterr().err
    assert err.count("no entry for 'Apple M5 Ultra'") == 1 and err.count("\n") == 1
    # write: this chip's entry, others kept; a legacy flat file becomes the default
    d = K.write_device_cfg(str(p), {"x": "3"})
    assert d["Apple M5 Ultra"] == {"x": "3", "tuned_on": "Apple M5 Ultra"} and d["Apple M5 Max"] == {"x": "2"}
    q = tmp_path / "legacy.json"
    q.write_text(json.dumps({"x": "0"}))
    assert K.device_cfg(str(q), "nax", "tool") == {"x": "0"}
    d = K.write_device_cfg(str(q), {"x": "4"})
    assert d["default"] == {"x": "0"} and d["Apple M5 Ultra"]["x"] == "4"
    assert K.rows_for_chip([{"a": 1}, {"chip": "Apple M5 Ultra"}, {"chip": "Apple M1"}], "Apple M5 Pro") == [{"a": 1}]


def test_shipped_cfg_files_are_per_device():
    root = Path(K.__file__).parent
    for f in ("nax3_cfg.json", "atom_cfg.json", "msa_cfg.json"):
        d = json.loads((root / f).read_text())
        assert "default" in d and "Apple M5 Pro" in d, f
        e = {k: v for k, v in d["Apple M5 Pro"].items()}
        assert e == d["default"], f  # default = the tuned M5 Pro values (the only measured chip so far)
    text = (root.parent / "pyproject.toml").read_text()
    line = next(x for x in text.splitlines() if x.startswith("boltz_mlx = ["))
    for f in ("nax3_cfg.json", "atom_cfg.json", "msa_cfg.json", "fp16_manifest.json"):
        assert f'"{f}"' in line, f  # package data: shipped in non-editable installs


def test_portable_check_kit_logic(tmp_path):
    """tools/portable_check.py on synthetic outputs (no model, no GPU): superposition RMSD, deltas, the decision rule
    and the markdown summary."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    import portable_check as PC

    rng = np.random.default_rng(0)
    x = rng.normal(size=(1, 40, 3)) * 10
    th = 0.7
    rot = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    mask = np.ones((1, 40), np.float32)
    mask[0, 35:] = 0
    for lab, coords, val in (("strict", x, 1.0), ("f16-portable", x @ rot.T + 5.0, 1.25)):
        d = tmp_path / lab
        d.mkdir()
        np.savez(d / "prot117.npz", sample_atom_coords=coords.astype(np.float32), plddt=np.full((1, 10), 0.8),
                 atom_pad_mask=mask, token_pad_mask=np.ones((1, 10)))
        np.savez(d / "lig_aff_affinity.npz", sample_atom_coords=np.repeat(coords, 5, 0).astype(np.float32),
                 affinity_pred_value=np.array([val]), affinity_probability_binary=np.array([0.5]), atom_pad_mask=mask,
                 token_pad_mask=np.ones((1, 10)))
    c = PC.compare(tmp_path / "f16-portable", tmp_path / "strict")
    assert c["prot117"]["coords_rmsd_A"] < 1e-3 and c["prot117"]["finite"]  # rigid motion only
    assert abs(c["lig_aff_affinity"]["affinity_pred_value_delta"] - 0.25) < 1e-9
    info = {"chip": "Apple M1", "architecture": "applegpu_g13g", "nax_hw": False, "tensorops": "ok", "path": "tensorops",
            "reason": "r", "memory_gb": 16.0, "macos": "26.1", "os_build": "x", "mlx": "0.32.4"}
    res = {"info": info, "date": "d", "git": "g", "main_path": "tensorops", "paths": ["tensorops", "portable"],
           "tests": {"tensorops": {"ok": True, "kernel_path": "tensorops"}},
           "predict": {"f16-tensorops": {"deterministic": True}},
           "bench": {"prot117_f16@50_portable": {"median_runs2N_s": 2.0},
                     "prot117_f16@50_tensorops": {"median_runs2N_s": 1.5}},
           "accuracy": {"f16-portable_vs_strict": c, "f16-tensorops_vs_strict": c}}
    d = PC.decide(res)
    assert d["recommendation"] == "tensorops confirmed as this chip's path"
    assert d["speedup_tensorops_vs_portable_ref"]["prot117_f16@50"] > 1
    res["bench"]["prot117_f16@50_tensorops"]["median_runs2N_s"] = 2.5
    assert "NOTE slower than the portable reference" in PC.decide(res)["recommendation"]
    res["tests"]["tensorops"]["ok"] = False
    assert PC.decide(res)["recommendation"].startswith("FALLBACK needed (MPP per-kernel tests FAILED")
    res["decision"] = PC.decide(res)
    assert PC.decide({**res, "main_path": "portable"})["recommendation"].startswith("FALLBACK: portable path")
    md = PC.summary_md(res)
    assert "Apple M1" in md and "## Decision" in md and "| f16-portable_vs_strict | prot117 |" in md
