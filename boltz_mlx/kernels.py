"""Custom Metal kernels (mx.fast.metal_kernel) for hot ops.

layer_norm: MLX's fast.layer_norm is slow for narrow rows (D=128 pair rep, ~5x a plain
elementwise pass on M5). Here one simdgroup handles one row; each lane holds D/32 values in
registers, mean/variance via simd_sum (two-pass, same as the reference formula).
"""

from __future__ import annotations

import os

import mlx.core as mx

USE_KERNELS_ENV = os.environ.get("BOLTZ_MLX_KERNELS", "1") == "1"


def _use_kernels() -> bool:
    """Custom Metal kernels only on the GPU device (CPU-device tests use the MLX reference ops)."""
    return USE_KERNELS_ENV and mx.default_device() == mx.gpu


class _UseKernels:
    def __bool__(self):
        return _use_kernels()


USE_KERNELS = _UseKernels()


# ----------------------------------------------------------------------------------------------
# Hardware capability: one codebase for every Apple-silicon Mac (M1-M5) on macOS 26+ (Metal 4; boltz_mlx/__init__.py
# refuses older macOS). Two shipped kernel paths for --precision f16 (owner decision 2026-10-28 22:30):
#   nax       M5-class GPU (MLX's rule: architecture generation >= 17, >= 18 for 'p' class; M1..M4 = g13..g16): the
#             MetalPerformancePrimitives (Metal 4 tensor_ops matmul2d) kernels of kernels_nax*.py / kernels_atom.py /
#             kernels_msa.opm_outer_proj run on the per-core neural accelerators ("NAX").
#   tensorops M1-M4: the same MPP kernels; Metal 4 runs tensor ops on the shader cores there.
# Both are gated by the on-device tensor-op check (_tensorops_check): the register kernels copy MLX steel-layout
# fragments into MPP cooperative tensors, whose layout is implementation defined (tests/test_nax_layout.py), so before
# any MPP kernel is built the device must show (a) matmul2d(16, 32, 16, relaxed) exact on integer operands through those
# fragments for every (B orientation, A type) the kernels use, (b) the left / destination cooperative-tensor
# coordinates == the MLX layout, (c) the layout-agnostic MPP API (kernels_nax2, opm_outer_proj) exact, (d) relative
# error <= 1e-3 on non-integer fp16 operands. One simdgroup per check; result cached in memory and on disk per device +
# OS build + MLX version + probe source (~/.cache/boltz_mlx/tensorops_check.json; BOLTZ_MLX_CAP_CACHE=<file> or 0 = off).
#   portable  internal fallback / debug path only (no tuning, no validation track): taken - with a loud warning - if
#             the check fails, or on request (BOLTZ_MLX_KERNEL_PATH=portable, alias BOLTZ_MLX_NAX=0). No MPP source is
#             ever built: MLX fp16 GEMMs + the simdgroup-MMA / simd-reduction kernels of this file, kernels_difftx.py and
#             kernels_msa.py (opm_outer, pwa_*).
# BOLTZ_MLX_KERNEL_PATH = auto (default) | nax | tensorops | portable: nax and tensorops both mean "the MPP kernels" (the
# hardware decides the label) and still need the check. Never run detection inside an mx.compile trace: load_model()
# and Boltz2.__call__ call nax_status() eagerly; CPU test sessions pin the status (tests/conftest.py) and never probe.
# ----------------------------------------------------------------------------------------------
NAX_MIN_GEN = 17
NAX_MIN_MACOS = (26, 2)  # MLX's own NAX kernels need macOS >= 26.2 (mlx device.cpp is_nax_available): noted
KERNEL_PATHS = ("nax", "tensorops", "portable")
MPP_MARKERS = ("MetalPerformancePrimitives", "mpp::")
_CAP: dict = {}           # cached capability record (nax_status())
_PROBING = False          # metal_kernel() lets the capability check's own MPP kernels through


class NaxUnavailableError(RuntimeError):
    """A kernel source using MetalPerformancePrimitives was about to be built while the kernel path is portable."""


def arch_generation(arch: str):
    """'applegpu_g17s' -> 17 (None if the string is not an Apple GPU architecture)."""
    import re

    m = re.match(r"applegpu_g(\d+)", arch or "")
    return int(m.group(1)) if m else None


def nax_hardware(arch: str) -> bool:
    """Neural accelerators by MLX's rule (mlx/backend/metal/device.cpp is_nax_available, v0.32.4): generation >= 17,
    >= 18 for architecture strings ending in 'p' (phone-class GPUs, e.g. A18 Pro Macs). docs/apple_gpu_generations.md."""
    gen = arch_generation(arch)
    return gen is not None and gen >= (NAX_MIN_GEN + 1 if arch.endswith("p") else NAX_MIN_GEN)


def gpu_info() -> dict:
    """Chip name, GPU architecture string, generation and memory (a device query; nothing is dispatched)."""
    try:
        d = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
    except Exception:  # no Metal device (CPU-only MLX build)
        d = {}
    arch = str(d.get("architecture", ""))
    return {"chip": str(d.get("device_name", "unknown")), "architecture": arch, "gpu_gen": arch_generation(arch),
            "memory_gb": round(float(d.get("memory_size", 0)) / 2 ** 30, 1)}


def os_info() -> dict:
    import subprocess

    from boltz_mlx import macos_version

    try:
        build = subprocess.run(["sysctl", "-n", "kern.osversion"], capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        build = ""
    return {"macos": macos_version(), "os_build": build or os.uname().version[:60]}


class probing:
    """Context manager: let MPP diagnostic kernels through the metal_kernel() guard (the capability check itself and
    the layout diagnostics of tests/test_nax_layout.py)."""

    def __enter__(self):
        global _PROBING
        self._old, _PROBING = _PROBING, True
        return self

    def __exit__(self, *exc):
        global _PROBING
        _PROBING = self._old
        return False


def _int_operands(rows: int, cols: int, mul_r: int, mul_c: int, mod: int):
    import numpy as np

    r, c = np.arange(rows)[:, None], np.arange(cols)[None, :]
    return ((r * mul_r + c * mul_c) % mod - mod // 2).astype(np.float32)


def _tensorops_check() -> tuple[bool, str, dict]:
    """(ok, detail, per-check results) of the tensor-op check described above. Exceptions (no Metal 4 header on
    macOS < 26, compile errors) -> not ok."""
    import numpy as np

    res: dict = {}
    try:
        from boltz_mlx import kernels_nax as _KN

        a = _int_operands(16, 16, 7, 3, 7)          # [16, 16], entries -3..3, position dependent (layout errors show)
        b = _int_operands(16, 32, 5, 2, 5)          # [16, 32], entries -2..2
        ref = a @ b                                 # |ref| <= 96: exact in fp16 operands / fp32 accumulation
        with probing(), mx.stream(mx.gpu):
            for tb, at in ((True, mx.float16), (False, mx.float16), (False, mx.float32)):
                bb = np.ascontiguousarray(b.T) if tb else b
                c = np.array(_KN.nax_mma_once(mx.array(a), mx.array(bb), tb, at, relax=True))
                res[f"mma_tb{int(tb)}_{str(at).split('.')[-1]}"] = float(np.abs(c - ref).max())
                caps, coords = _KN.nax_layout(True, False, tb, at, mx.float16, mx.float32)
                caps, coords = np.array(caps), np.array(coords)
                exp = _KN.mlx_frag_coords()
                left, dest = coords[0, :, :8], coords[2, :, :16][..., ::-1]
                res[f"layout_tb{int(tb)}_{str(at).split('.')[-1]}"] = bool(
                    (caps[0] == 8).all() and (caps[2] == 16).all() and ((left[..., ::-1] == exp).all() or (left == exp).all())
                    and (dest[:, :8] == exp).all() and (dest[:, 8:] == exp + np.array([0, 16])).all())
            for tb in (True, False):
                bb = np.ascontiguousarray(b.T) if tb else b
                c = np.array(_KN.nax_mma_generic(mx.array(a.astype(np.float16)), mx.array(bb), tb, relax=True))
                res[f"generic_tb{int(tb)}"] = float(np.abs(c - ref).max())
            # precision on non-integer fp16 operands (the integer checks above stay exact even with a coarser
            # shader fallback): relative max error vs fp64, the bound tests/test_nax_layout.py validates on M5
            rng = np.random.default_rng(0)
            af = rng.standard_normal((16, 16)).astype(np.float16).astype(np.float32)
            bf = rng.standard_normal((32, 16)).astype(np.float16).astype(np.float32)
            reff = af.astype(np.float64) @ bf.T.astype(np.float64)
            c = np.array(_KN.nax_mma_once(mx.array(af), mx.array(bf), True, mx.float16, relax=True))
            res["precision_rel"] = float(np.abs(c - reff).max() / np.abs(reff).max())
    except Exception as e:
        if "transformation" in str(e):  # mx.compile / vmap trace: the check needs a real eval
            raise RuntimeError("kernel-path detection ran inside an mx.compile trace; call "
                               "boltz_mlx.kernels.nax_status() before compiling (load_model does)") from e
        first = str(e).strip().splitlines()[0][:200] if str(e).strip() else ""
        return False, f"tensor-op check failed: {type(e).__name__}: {first}", res
    bad = [k for k, v in res.items() if (v is False) or (k == "precision_rel" and v > 1e-3)
           or (not isinstance(v, bool) and k != "precision_rel" and v != 0.0)]
    if bad:
        return False, f"tensor-op check WRONG on this device ({', '.join(bad)}): MPP fragment layout differs", res
    return True, "tensor-op check passed (matmul2d exact, MLX fragment layout, layout-agnostic API)", res


def _probe_source_hash() -> str:
    import hashlib

    from boltz_mlx import kernels_nax as _KN

    src = _KN.NAX_HEADER + _KN._MMA1_SRC + _KN._LAYOUT_SRC + _KN._MMA_GENERIC_SRC + "v2"
    return hashlib.sha256(src.encode()).hexdigest()[:12]


def _cache_path():
    from pathlib import Path

    v = os.environ.get("BOLTZ_MLX_CAP_CACHE", "")
    if v == "0":
        return None
    return Path(v).expanduser() if v else Path("~/.cache/boltz_mlx/tensorops_check.json").expanduser()


def tensorops_check(rec: dict) -> tuple[bool, str, bool]:
    """The tensor-op check for this device + OS (+ MLX + probe source), from the disk cache when present.
    Returns (ok, detail, from_cache)."""
    import json

    key = json.dumps([rec["chip"], rec["architecture"], rec["macos"], rec["os_build"], rec["mlx"], _probe_source_hash()])
    p = _cache_path()
    try:
        cache = json.loads(p.read_text()) if p is not None and p.exists() else {}
    except Exception:
        cache = {}
    if key in cache:
        return bool(cache[key]["ok"]), cache[key]["detail"], True
    ok, detail, res = _tensorops_check()
    if p is not None:
        try:
            import time

            cache[key] = {"ok": ok, "detail": detail, "checks": res, "date": time.strftime("%Y-%m-%d %H:%M:%S")}
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_name(p.name + f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(cache, indent=1))
            os.replace(tmp, p)
        except Exception:
            pass
    return ok, detail, False


def _requested() -> str | None:
    v = os.environ.get("BOLTZ_MLX_KERNEL_PATH", "").strip().lower()
    if v in ("", "auto"):
        return "portable" if os.environ.get("BOLTZ_MLX_NAX") == "0" else None
    if v not in KERNEL_PATHS:
        raise ValueError(f"BOLTZ_MLX_KERNEL_PATH must be one of {KERNEL_PATHS} or auto, not {v!r}")
    return v


_WARNED = []


def _fallback_warning(reason: str) -> None:
    import sys

    if _WARNED:
        return
    _WARNED.append(reason)
    bar = "!" * 100
    print(f"{bar}\nboltz-mlx WARNING: Metal tensor-op kernels DISABLED on this Mac - {reason}\n"
          "Running the internal FALLBACK kernel path ('portable'): unvalidated for speed and accuracy, results are "
          "labelled f16-portable. Please report this (python -m boltz_mlx.capability).\n" + bar, file=sys.stderr)


def _detect() -> dict:
    req = _requested()
    rec = {**gpu_info(), **os_info(), "mlx": mx.__version__, "requested": req or "auto", "tensorops": "not checked",
           "tensorops_cached": False}
    gen = rec["gpu_gen"]
    rec["nax_hw"] = nax_hardware(rec["architecture"])
    try:
        mac = tuple(int(x) for x in (rec["macos"] or "0").split(".")[:2])
    except ValueError:
        mac = (0,)

    def done(path, reason):
        return {**rec, "path": path, "nax": path == "nax", "mpp": path != "portable", "reason": reason}

    if req == "portable":
        return done("portable", "requested (BOLTZ_MLX_KERNEL_PATH=portable / BOLTZ_MLX_NAX=0: debug fallback path)")
    if not mx.metal.is_available():
        return done("portable", "no Metal GPU")
    if mac < (26, 0):  # only reachable with BOLTZ_MLX_ALLOW_OLD_MACOS=1 (boltz_mlx/__init__.py refuses otherwise)
        why = f"macOS {rec['macos']} < 26: no Metal 4 tensor ops (unsupported OS)"
        _fallback_warning(why)
        return done("portable", why)
    ok, detail, cached = tensorops_check(rec)
    rec.update(tensorops="ok" if ok else detail, tensorops_cached=cached)
    if not ok:
        _fallback_warning(detail)
        return done("portable", f"FALLBACK: {detail}")
    if rec["nax_hw"]:
        note = "" if mac >= NAX_MIN_MACOS else (f"; note: macOS {rec['macos']} < "
                                                f"{'.'.join(map(str, NAX_MIN_MACOS))}, MLX's own GEMMs do not use NAX")
        return done("nax", f"GPU {rec['architecture']} has neural accelerators; {detail}{note}")
    return done("tensorops", f"GPU {rec['architecture']} (generation {gen}): no neural accelerators, Metal 4 tensor ops "
                             f"on the shader cores; {detail}")


def nax_status() -> dict:
    """Capability record (cached per process): chip, architecture, gpu_gen, memory_gb, macos, os_build, mlx,
    requested, nax_hw, tensorops, path (nax | tensorops | portable), nax, mpp, reason."""
    if not _CAP:
        _CAP.update(_detect())
    return dict(_CAP)


def kernel_path() -> str:
    return nax_status()["path"]


def mpp_enabled() -> bool:
    """MetalPerformancePrimitives kernels may be built (path nax or tensorops)."""
    return bool(nax_status()["mpp"])


def nax_available() -> bool:
    """NAX hardware present and the tensor-op check passed (the M5 path)."""
    return bool(nax_status()["nax"])


def pinned_status(path: str = "nax", reason: str = "pinned") -> dict:
    """A capability record without detection (CPU test sessions; tests monkeypatch kernels._CAP with it)."""
    assert path in KERNEL_PATHS
    return {"chip": "pinned", "architecture": "", "gpu_gen": None, "memory_gb": 0.0, "macos": "", "os_build": "",
            "mlx": mx.__version__, "requested": "pinned", "nax_hw": path == "nax", "tensorops": "pinned",
            "tensorops_cached": False, "path": path, "nax": path == "nax", "mpp": path != "portable", "reason": reason}


def is_mpp_source(*texts) -> bool:
    return any(m in t for t in texts if t for m in MPP_MARKERS)


KERNEL_LOG = None  # list -> metal_kernel() appends (name, uses_mpp) for every kernel it builds (audits / tests)


def metal_kernel(**kw):
    """mx.fast.metal_kernel with the portability guard: a source / header using MetalPerformancePrimitives is never
    built on the portable path (mx.fast.metal_kernel is looked up at call time: tests that replace it still work)."""
    mpp_src = is_mpp_source(kw.get("source", ""), kw.get("header", ""))
    if mpp_src and not _PROBING and not mpp_enabled():
        raise NaxUnavailableError(f"refusing to build MPP kernel {kw.get('name')!r} on the portable kernel path: "
                                  f"{_CAP.get('reason')}")
    if KERNEL_LOG is not None and not _PROBING:  # (the capability check's own kernels are not part of a run)
        KERNEL_LOG.append((kw.get("name"), mpp_src))
    return mx.fast.metal_kernel(**kw)


def device_record() -> dict:
    """Fields every bench / fingerprint row carries: chip, architecture, kernel path (+ why), macOS, MLX."""
    s = nax_status()
    return {k: s[k] for k in ("chip", "architecture", "gpu_gen", "memory_gb", "macos", "os_build", "mlx", "path", "nax",
                              "tensorops", "reason")}


def device_tag() -> dict:
    """Compact form of device_record() for per-output / per-ligand rows."""
    s = nax_status()
    return {"chip": s["chip"], "architecture": s["architecture"], "path": s["path"], "nax": s["nax"]}


def mode_label(mode: str) -> str:
    """Label of a precision mode as it runs here: f16 = the NAX build; 'f16-tensorops' / 'f16-portable' otherwise."""
    if mode == "f16":
        p = kernel_path()
        return "f16" if p == "nax" else f"f16-{p}"
    return mode


# ----------------------------------------------------------------------------------------------
# Per-device tuned kernel configs (boltz_mlx/nax3_cfg.json, atom_cfg.json, msa_cfg.json):
#   {"default": {...}, "Apple M5 Pro": {...}, "<chip name as in mx.device_info()['device_name']>": {...}}
# The entry of this chip is used, else "default" with a one-line note (printed when the configs are first used: nax()
# for the MPP sites, msa_kernel() for the MSA kernels). A flat file without "default" is a legacy single config for
# every device. The suggest tools (tools/prof/nax3_ab.py suggest all --write, pair_profile.py suggest --write,
# atom_ab.py suggest --write, msa_kernels.py suggest --write) write this chip's entry (write_device_cfg) from rows of
# this chip only (rows_for_chip); docs/researcher.md "Tuning on another Mac".
# ----------------------------------------------------------------------------------------------
LEGACY_ROW_CHIP = "Apple M5 Pro"  # tuning rows written before rows carried "chip" (2026-10-28) were measured here
_CFG_NOTES: dict = {"nax": [], "msa": []}


def _cfg_note(kind: str) -> None:
    import sys

    while _CFG_NOTES[kind]:
        print(_CFG_NOTES[kind].pop(0), file=sys.stderr)


def chip_name() -> str:
    return gpu_info()["chip"]


def cfg_chip() -> str:
    """Name of the per-device config entry to use: this chip's, or BOLTZ_MLX_CFG_CHIP (an entry name such as "default"
    or "Apple M5 Pro"; A/B runs of tuned configs, tools/bench_mlx.py --cfg-chip). Config choices inside a kernel family
    are bitwise-neutral, so this selects speed only."""
    return os.environ.get("BOLTZ_MLX_CFG_CHIP") or chip_name()


def device_cfg(path: str, kind: str, tool: str) -> dict:
    """This chip's entry of a per-device config file (see above), else its "default" entry (note queued under
    `kind`: 'nax' | 'msa'). {} if the file does not exist. BOLTZ_MLX_CFG_CHIP selects another entry (cfg_chip)."""
    import json

    if not os.path.exists(path):
        return {}
    with open(path) as f:
        d = json.load(f)
    if "default" not in d:
        return d
    chip = cfg_chip()
    if chip in d:
        return d[chip]
    _CFG_NOTES[kind].append(f"boltz_mlx: {os.path.basename(path)} has no entry for {chip!r}; using 'default' (tuned on "
                            f"{d['default'].get('tuned_on', '?')}); to tune for this Mac: {tool}")
    return d["default"]


def device_cfg_entry(path: str) -> dict:
    """Copy of the entry device_cfg would use (no note); for tools that merge into this chip's entry."""
    import json

    if not os.path.exists(path):
        return {}
    with open(path) as f:
        d = json.load(f)
    if "default" not in d:
        return dict(d)
    return dict(d.get(chip_name(), d["default"]))


def write_device_cfg(path: str, entry: dict, chip: str | None = None) -> dict:
    """Store `entry` as this chip's entry (other chips' entries and "default" kept; a legacy flat file becomes the
    "default"). Returns the whole file."""
    import json

    d: dict = {}
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        if d and "default" not in d:
            d = {"default": d}
    chip = chip or chip_name()
    d[chip] = {**{k: v for k, v in entry.items() if k != "tuned_on"}, "tuned_on": chip}
    d.setdefault("default", d[chip])
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        f.write(json.dumps(d, indent=1) + "\n")
    os.replace(tmp, path)
    return d


def rows_for_chip(rows: list, chip: str | None = None) -> list:
    """Tuning rows measured on this chip (rows without a chip field: LEGACY_ROW_CHIP)."""
    chip = chip or chip_name()
    return [r for r in rows if r.get("chip", LEGACY_ROW_CHIP) == chip]


_LN_SRC = """
    uint row = threadgroup_position_in_grid.x * ROWS_PER_TG + simdgroup_index_in_threadgroup;
    if (row >= n_rows) return;
    uint lane = thread_index_in_simdgroup;
    const device IT* xr = x + row * D;
    device OT* yr = out + row * D;
    float v[D / 32];
    float s = 0.0f;
    for (int i = 0; i < D / 32; i++) {
        v[i] = float(xr[lane + 32 * i]);
        s += v[i];
    }
    float mean = simd_sum(s) / D;
    float ss = 0.0f;
    for (int i = 0; i < D / 32; i++) {
        float d = v[i] - mean;
        ss += d * d;
    }
    float rstd = metal::precise::rsqrt(simd_sum(ss) / D + eps[0]);
    for (int i = 0; i < D / 32; i++) {
        uint c = lane + 32 * i;
        float y = (v[i] - mean) * rstd;
        if (HAS_W) y = y * w[c];
        if (HAS_B) y = y + b[c];
        yr[c] = OT(y);
    }
"""

_kernels: dict = {}
ROWS_PER_TG = 8


def _ln_kernel(has_w: bool, has_b: bool):
    key = (has_w, has_b)
    if key not in _kernels:
        _kernels[key] = metal_kernel(
            name=f"boltz_ln_w{int(has_w)}_b{int(has_b)}",
            input_names=["x", "w", "b", "eps", "n_rows"],
            output_names=["out"],
            source=_LN_SRC,
        )
    return _kernels[key]


def layer_norm(x: mx.array, weight: mx.array | None, bias: mx.array | None, eps: float = 1e-5,
               out_dtype=mx.float32) -> mx.array:
    """fp32 statistics always; out_dtype=float16 writes the normalized output in fp16 (the cast is fused)."""
    D = x.shape[-1]
    if not USE_KERNELS or x.dtype not in (mx.float32, mx.float16) or D % 32 != 0 or D > 1024:
        y = mx.fast.layer_norm(x.astype(mx.float32), weight, bias, eps)
        return y if out_dtype == mx.float32 else y.astype(out_dtype)
    n_rows = x.size // D
    k = _ln_kernel(weight is not None, bias is not None)
    dummy = mx.zeros((1,), mx.float32)
    (y,) = k(
        inputs=[
            x,  # metal_kernel copies to row-contiguous if needed (mx.contiguous is not exportable)
            weight if weight is not None else dummy,
            bias if bias is not None else dummy,
            mx.array([eps], mx.float32),
            mx.array(n_rows, mx.uint32),
        ],
        template=[("D", D), ("ROWS_PER_TG", ROWS_PER_TG), ("HAS_W", int(weight is not None)), ("HAS_B", int(bias is not None)),
                  ("OT", out_dtype), ("IT", x.dtype)],
        grid=(((n_rows + ROWS_PER_TG - 1) // ROWS_PER_TG) * 32 * ROWS_PER_TG, 1, 1),
        threadgroup=(32 * ROWS_PER_TG, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[out_dtype],
    )
    return y


# Reduced-precision matmuls for the diffusion transformers only (opt-in; trunk stays fp32).
# BOLTZ_MLX_DIFF_DTYPE = fp32 (default, parity build) | fp16 | bf16. Accuracy of the shipped presets: docs/report.md sections 5 and 7.
_DT = {"fp32": mx.float32, "fp16": mx.float16, "bf16": mx.bfloat16}
# --precision f16 implies fp16 diffusion matmuls unless BOLTZ_MLX_DIFF_DTYPE says otherwise.
DIFF_DTYPE = _DT[os.environ.get("BOLTZ_MLX_DIFF_DTYPE",
                                "fp16" if os.environ.get("BOLTZ_MLX_PRECISION", "fp32") == "f16" else "fp32")]


def set_diff_dtype(name: str) -> None:
    global DIFF_DTYPE
    DIFF_DTYPE = _DT[name]


def mm_t(x: mx.array, w: mx.array, dtype=None) -> mx.array:
    """x @ w.T in `dtype` (fp32 accumulate inside the GEMM), result fp32."""
    dt = DIFF_DTYPE if dtype is None else dtype
    if dt == mx.float32:
        return x @ w.T
    return (x.astype(dt) @ w.astype(dt).T).astype(mx.float32)


# ----------------------------------------------------------------------------------------------
# Fused triangle attention (flash-style, fp32): one thread per query row, online softmax in
# registers, K/V tiles staged in threadgroup memory. Reads q/k/v/g straight out of the fused
# projection output h[B*I, J, 4*H*C + H] and writes sigmoid(g) * attn(q,k,v) as [B*I, J, H*C].
# bias: tri bias laid out [B, H, K, J] (coalesced over queries), optional key mask [B*I, K].
# Never materializes the [B*I, H, J, K] score tensor.
# ----------------------------------------------------------------------------------------------
_TRI_ATT_SRC = """
    // grid: x = query blocks of TQ, y = head, z = row (b*I + i)
    uint tid = thread_position_in_threadgroup.x;
    uint qb = threadgroup_position_in_grid.x;
    uint hh = threadgroup_position_in_grid.y;
    uint row = threadgroup_position_in_grid.z;
    uint b = row / I_DIM;
    uint j = qb * TQ + tid;
    bool valid = j < J_DIM;
    const uint RS = ROW_STRIDE;                     // row stride of h (q|k|v|g[|bias])
    const device float* hrow = h + (size_t)row * J_DIM * RS;

    float q[C];
    float acc[C];
    for (int d = 0; d < C; d++) { acc[d] = 0.0f; q[d] = valid ? hrow[(size_t)j * RS + hh * C + d] * scale : 0.0f; }
    float m = -INFINITY;
    float l = 0.0f;

    threadgroup float Ks[TK][C];
    threadgroup float Vs[TK][C];
    auto brow = bias + ((size_t)(b * NH + hh) * J_DIM) * J_DIM;   // [K][J] for this (b,h); auto: bias may be bound in constant space (< 8 elements, N=1)

    for (uint k0 = 0; k0 < J_DIM; k0 += TK) {
        // cooperative tile load: TK keys x C dims for K and V
        for (uint e = tid; e < TK * C; e += TQ) {
            uint kk = e / C, d = e % C;
            uint key = k0 + kk;
            float kv = 0.0f, vv = 0.0f;
            if (key < J_DIM) {
                kv = hrow[(size_t)key * RS + HC + hh * C + d];
                vv = hrow[(size_t)key * RS + 2 * HC + hh * C + d];
            }
            Ks[kk][d] = kv;
            Vs[kk][d] = vv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint kend = min((uint)TK, J_DIM - k0);
        for (uint kk = 0; kk < kend; kk++) {
            uint key = k0 + kk;
            float s = 0.0f;
            for (int d = 0; d < C; d++) s += q[d] * Ks[kk][d];
            // same summation order as torch: (qk + mask_bias) + tri_bias
            if (HAS_MASK) s += inf_val * (mask[(size_t)row * J_DIM + key] - 1.0f);
            s += valid ? brow[(size_t)key * J_DIM + j] : 0.0f;
            float m_new = max(m, s);
            float corr = metal::exp(m - m_new);
            float p = metal::exp(s - m_new);
            l = l * corr + p;
            for (int d = 0; d < C; d++) acc[d] = acc[d] * corr + p * Vs[kk][d];
            m = m_new;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (valid) {
        float inv = 1.0f / l;
        device float* orow = out + ((size_t)row * J_DIM + j) * HC + hh * C;
        const device float* grow = hrow + (size_t)j * RS + 3 * HC + hh * C;
        for (int d = 0; d < C; d++) {
            float g = 1.0f / (1.0f + metal::exp(-grow[d]));
            orow[d] = acc[d] * inv * g;
        }
    }
"""


# ----------------------------------------------------------------------------------------------
# MMA variant (BOLTZ_MLX_TRIATT_IMPL=mma): flash attention with simdgroup 8x8 matrix ops.
# A threadgroup = NSG simdgroups handles one (row, head, block of NSG*RQ queries); each simdgroup owns RQ
# query rows (Q fragments live in registers), K/V tiles (BK keys) are staged in threadgroup memory and shared.
#   S = Q K^T      : 8x8x8 MMAs over C (C/8 k-steps), accumulators = fragments in registers
#   + mask + bias  : applied elementwise on fragment elements (same order as scalar: (qk + mask) + tri_bias)
#   online softmax : row max / sum via shuffle-xor over the 4 lanes that share a fragment row
#                    (Apple fragment layout: lane -> (row fm, cols fn, fn+1); see MLX steel BaseMMAFrag)
#   O += P V       : P fragments (the S fragments after exp) are used directly as the A operand
# No threadgroup round trip for S/P is needed. DT is the Q/K/V/P operand type (float, or half for the
# "mma_half" variant; accumulation is always fp32).
# ----------------------------------------------------------------------------------------------
_TRI_ATT_MMA_HEADER = "#include <metal_simdgroup_matrix>\n#define UNR _Pragma(\"clang loop unroll(full)\")\n"
_TRI_ATT_MMA_SRC = """
    constexpr int RB = RQ / 8;          // 8-row query blocks per simdgroup
    constexpr int NKB = BK / 8;         // 8-key blocks per key tile
    constexpr int NDB = C / 8;          // 8-wide head-dim blocks
    constexpr int LD = C + PAD;         // threadgroup row stride of the K/V tiles
    constexpr int NT = NSG * 32;
    constexpr int NLD = (BK * (C / 4) + NT - 1) / NT;   // float4 global loads per thread, per matrix and tile
    constexpr int NLS = (BK * C + NT - 1) / NT;         // scalar loads (row stride not 16B aligned)
    constexpr int NLM = (BK + NT - 1) / NT;
    const uint RS = ROW_STRIDE;         // row stride of h (q|k|v|g[|bias])

    uint tid = thread_position_in_threadgroup.x;
    uint sg = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint qb = threadgroup_position_in_grid.x;
    uint hh = threadgroup_position_in_grid.y;
    uint row = threadgroup_position_in_grid.z;
    uint b = row / I_DIM;
    uint q0 = qb * (NSG * RQ) + sg * RQ;

    // fragment coordinates of this lane (row fm, columns fn and fn+1 of every 8x8 fragment)
    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;

    const device HT* hrow = h + (size_t)row * J_DIM * RS;
    auto brow = bias + ((size_t)(b * NH + hh) * J_DIM) * J_DIM;   // [K][J] for this (b,h); auto: bias may be bound in constant space (< 8 elements, N=1)

    // Q fragments (pre-scaled like the scalar kernel), zero for out-of-range queries
    simdgroup_matrix<DT, 8, 8> Qf[RB][NDB];
    UNR for (int rb = 0; rb < RB; rb++) {
        uint qi = q0 + rb * 8 + fm;
        UNR for (int db = 0; db < NDB; db++) {
            float a0 = 0.0f, a1 = 0.0f;
            if (qi < J_DIM) {
                const device HT* qp = hrow + (size_t)qi * RS + hh * C + db * 8 + fn;
                a0 = float(qp[0]) * scale; a1 = float(qp[1]) * scale;
            }
            Qf[rb][db].thread_elements()[0] = (DT)a0;
            Qf[rb][db].thread_elements()[1] = (DT)a1;
        }
    }
    simdgroup_float8x8 Of[RB][NDB];
    UNR for (int rb = 0; rb < RB; rb++)
        UNR for (int db = 0; db < NDB; db++) Of[rb][db] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    float m_i[RB], l_i[RB];
    UNR for (int rb = 0; rb < RB; rb++) { m_i[rb] = -INFINITY; l_i[rb] = 0.0f; }

    // double-buffered K/V tiles (+ key-mask bias); the global loads of tile t+1 are issued before, and
    // stored to threadgroup memory after, the compute on tile t -> one barrier per tile
    threadgroup DT Ks[2 * BK * LD];
    threadgroup DT Vs[2 * BK * LD];
    threadgroup float Ms[2 * BK];
    float4 kreg[NLD], vreg[NLD];
    float ksr[NLS], vsr[NLS];
    float mreg[NLM];

    const uint ntiles = (J_DIM + BK - 1) / BK;
    for (uint it = 0; it <= ntiles; it++) {
        // (A) issue global loads of tile `it`
        if (it < ntiles) {
            const uint k0 = it * BK;
            if (VEC4) {
                UNR for (int u = 0; u < NLD; u++) {
                    uint e = tid + u * NT;
                    uint kk = e / (C / 4), d4 = e % (C / 4);
                    uint key = k0 + kk;
                    kreg[u] = float4(0.0f); vreg[u] = float4(0.0f);
                    if (e < BK * (C / 4) && key < J_DIM) {
                        const device HT* p = hrow + (size_t)key * RS + hh * C + d4 * 4;
                        kreg[u] = float4(*(const device vec<HT, 4>*)(p + HC));
                        vreg[u] = float4(*(const device vec<HT, 4>*)(p + 2 * HC));
                    }
                }
            } else {
                UNR for (int u = 0; u < NLS; u++) {
                    uint e = tid + u * NT;
                    uint kk = e / C, d = e % C;
                    uint key = k0 + kk;
                    ksr[u] = 0.0f; vsr[u] = 0.0f;
                    if (e < BK * C && key < J_DIM) {
                        ksr[u] = float(hrow[(size_t)key * RS + HC + hh * C + d]);
                        vsr[u] = float(hrow[(size_t)key * RS + 2 * HC + hh * C + d]);
                    }
                }
            }
            if (HAS_MASK) {
                UNR for (int u = 0; u < NLM; u++) {
                    uint e = tid + u * NT;
                    uint key = k0 + e;
                    mreg[u] = (e < BK && key < J_DIM) ? inf_val * (mask[(size_t)row * J_DIM + key] - 1.0f) : 0.0f;
                }
            }
        }

        // (B) compute on tile it-1 (already in threadgroup buffer (it-1)%2)
        if (it > 0) {
            const uint k0 = (it - 1) * BK;
            const uint buf = (it - 1) & 1;
            threadgroup DT* Kt = Ks + buf * (BK * LD);
            threadgroup DT* Vt = Vs + buf * (BK * LD);
            const bool tail = (k0 + BK > J_DIM);

            // tri bias for this thread's fragment elements (independent of the MMAs -> latency overlaps them)
            float bv[RB][NKB][2];
            UNR for (int kb = 0; kb < NKB; kb++) {
                uint col = k0 + kb * 8 + fn;
                UNR for (int rb = 0; rb < RB; rb++) {
                    uint qi = q0 + rb * 8 + fm;
                    if (tail) {
                        bv[rb][kb][0] = (qi < J_DIM && col < J_DIM)     ? brow[(size_t)col * J_DIM + qi] : 0.0f;
                        bv[rb][kb][1] = (qi < J_DIM && col + 1 < J_DIM) ? brow[(size_t)(col + 1) * J_DIM + qi] : 0.0f;
                    } else {
                        uint qc = min(qi, (uint)J_DIM - 1);   // padded queries (outputs discarded) read a valid address
                        bv[rb][kb][0] = brow[(size_t)col * J_DIM + qc];
                        bv[rb][kb][1] = brow[(size_t)(col + 1) * J_DIM + qc];
                    }
                }
            }

            // S = Q K^T
            simdgroup_float8x8 Sf[RB][NKB];
            UNR for (int rb = 0; rb < RB; rb++)
                UNR for (int kb = 0; kb < NKB; kb++) Sf[rb][kb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            UNR for (int ds = 0; ds < NDB; ds++) {
                UNR for (int kb = 0; kb < NKB; kb++) {
                    simdgroup_matrix<DT, 8, 8> Kf;
                    simdgroup_load(Kf, Kt + (kb * 8) * LD + ds * 8, LD, ulong2(0, 0), true);
                    UNR for (int rb = 0; rb < RB; rb++)
                        simdgroup_multiply_accumulate(Sf[rb][kb], Qf[rb][ds], Kf, Sf[rb][kb]);
                }
            }

            // + key mask, + tri bias (same order as the scalar kernel), -inf for out-of-range keys
            UNR for (int kb = 0; kb < NKB; kb++) {
                uint col = k0 + kb * 8 + fn;
                float mb0 = 0.0f, mb1 = 0.0f;
                if (HAS_MASK) { mb0 = Ms[buf * BK + kb * 8 + fn]; mb1 = Ms[buf * BK + kb * 8 + fn + 1]; }
                UNR for (int rb = 0; rb < RB; rb++) {
                    float s0 = Sf[rb][kb].thread_elements()[0];
                    float s1 = Sf[rb][kb].thread_elements()[1];
                    if (HAS_MASK) { s0 += mb0; s1 += mb1; }
                    s0 += bv[rb][kb][0];
                    s1 += bv[rb][kb][1];
                    if (tail) {
                        if (col >= J_DIM)     s0 = -INFINITY;
                        if (col + 1 >= J_DIM) s1 = -INFINITY;
                    }
                    Sf[rb][kb].thread_elements()[0] = s0;
                    Sf[rb][kb].thread_elements()[1] = s1;
                }
            }

            // online softmax (per fragment row) and rescale of the O accumulators
            UNR for (int rb = 0; rb < RB; rb++) {
                float tmax = -INFINITY;
                UNR for (int kb = 0; kb < NKB; kb++) {
                    tmax = max(tmax, Sf[rb][kb].thread_elements()[0]);
                    tmax = max(tmax, Sf[rb][kb].thread_elements()[1]);
                }
                tmax = max(tmax, simd_shuffle_xor(tmax, (ushort)1));
                tmax = max(tmax, simd_shuffle_xor(tmax, (ushort)8));
                float m_new = max(m_i[rb], tmax);
                float corr = metal::exp(m_i[rb] - m_new);
                float lsum = 0.0f;
                UNR for (int kb = 0; kb < NKB; kb++) {
                    float p0 = metal::exp(Sf[rb][kb].thread_elements()[0] - m_new);
                    float p1 = metal::exp(Sf[rb][kb].thread_elements()[1] - m_new);
                    Sf[rb][kb].thread_elements()[0] = p0;
                    Sf[rb][kb].thread_elements()[1] = p1;
                    lsum += p0 + p1;
                }
                l_i[rb] = l_i[rb] * corr + lsum;   // per-lane partial; reduced across the 4 lanes at the end
                m_i[rb] = m_new;
                UNR for (int db = 0; db < NDB; db++) {
                    Of[rb][db].thread_elements()[0] *= corr;
                    Of[rb][db].thread_elements()[1] *= corr;
                }
            }

            // O += P V
            UNR for (int kb = 0; kb < NKB; kb++) {
                simdgroup_matrix<DT, 8, 8> Pf[RB];
                UNR for (int rb = 0; rb < RB; rb++) {
                    Pf[rb].thread_elements()[0] = (DT)Sf[rb][kb].thread_elements()[0];
                    Pf[rb].thread_elements()[1] = (DT)Sf[rb][kb].thread_elements()[1];
                }
                UNR for (int db = 0; db < NDB; db++) {
                    simdgroup_matrix<DT, 8, 8> Vf;
                    simdgroup_load(Vf, Vt + (kb * 8) * LD + db * 8, LD, ulong2(0, 0), false);
                    UNR for (int rb = 0; rb < RB; rb++)
                        simdgroup_multiply_accumulate(Of[rb][db], Pf[rb], Vf, Of[rb][db]);
                }
            }
        }

        // (C) store the prefetched tile `it` into threadgroup buffer it%2
        if (it < ntiles) {
            const uint buf = it & 1;
            if (VEC4) {
                UNR for (int u = 0; u < NLD; u++) {
                    uint e = tid + u * NT;
                    if (e < BK * (C / 4)) {
                        uint kk = e / (C / 4), d4 = e % (C / 4);
                        threadgroup DT* kd = Ks + buf * (BK * LD) + kk * LD + d4 * 4;
                        threadgroup DT* vd = Vs + buf * (BK * LD) + kk * LD + d4 * 4;
                        UNR for (int c4 = 0; c4 < 4; c4++) { kd[c4] = (DT)kreg[u][c4]; vd[c4] = (DT)vreg[u][c4]; }
                    }
                }
            } else {
                UNR for (int u = 0; u < NLS; u++) {
                    uint e = tid + u * NT;
                    if (e < BK * C) {
                        uint kk = e / C, d = e % C;
                        Ks[buf * (BK * LD) + kk * LD + d] = (DT)ksr[u];
                        Vs[buf * (BK * LD) + kk * LD + d] = (DT)vsr[u];
                    }
                }
            }
            if (HAS_MASK) {
                UNR for (int u = 0; u < NLM; u++) {
                    uint e = tid + u * NT;
                    if (e < BK) Ms[buf * BK + e] = mreg[u];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // epilogue: normalise, gate, store
    UNR for (int rb = 0; rb < RB; rb++) {
        float l = l_i[rb];
        l += simd_shuffle_xor(l, (ushort)1);
        l += simd_shuffle_xor(l, (ushort)8);
        float inv = 1.0f / l;
        uint qi = q0 + rb * 8 + fm;
        if (qi < J_DIM) {
            UNR for (int db = 0; db < NDB; db++) {
                const device HT* gp = hrow + (size_t)qi * RS + 3 * HC + hh * C + db * 8 + fn;
                device OT* op = out + ((size_t)row * J_DIM + qi) * HC + hh * C + db * 8 + fn;
                float g0 = 1.0f / (1.0f + metal::exp(-float(gp[0])));
                float g1 = 1.0f / (1.0f + metal::exp(-float(gp[1])));
                op[0] = OT(Of[rb][db].thread_elements()[0] * inv * g0);
                op[1] = OT(Of[rb][db].thread_elements()[1] * inv * g1);
            }
        }
    }
"""

# MMA variant configurations: (NSG simdgroups/threadgroup, RQ query rows/simdgroup, BK keys/tile, PAD, half operands).
# Tuned on M5 Pro (N=117..400): RQ=16 rows/simdgroup and BK=16 keys/tile keep every fragment array in registers
# (bigger tiles spill and get 2-10x slower); 4 simdgroups/TG for small J (fills the GPU), 2 for J > 256.
_MMA_CFG = {"mma": (4, 16, 16, 4, False), "mma_half": (4, 16, 16, 8, True)}
_MMA_CFG_LARGE_J = {"mma": (2, 16, 16, 4, False), "mma_half": (2, 16, 16, 8, True)}

TRIATT_IMPL = os.environ.get("BOLTZ_MLX_TRIATT_IMPL", "mma")   # scalar | mma | mma_half (read at call time)


def _tri_attention_mma(h, bias, m, scale, inf_a, n_heads, c, has_mask, B, I, J, F, impl, out_dtype=mx.float32):
    nsg, rq, bk, pad, half = (_MMA_CFG_LARGE_J if J > 256 else _MMA_CFG)[impl]
    assert c % 8 == 0
    HC = n_heads * c
    vec4 = int(F % 4 == 0 and c % 4 == 0)
    key = ("tri_att_mma", impl, n_heads, c, has_mask, nsg, rq, bk, pad, half, vec4)  # HT/OT are template args
    if key not in _kernels:
        _kernels[key] = metal_kernel(
            name=f"boltz_tri_att_{impl}_h{n_heads}_c{c}_m{int(has_mask)}",
            input_names=["h", "bias", "mask", "scale", "inf_val"],
            output_names=["out"],
            source=_TRI_ATT_MMA_SRC,
            header=_TRI_ATT_MMA_HEADER,
        )
    qpt = nsg * rq
    (o,) = _kernels[key](
        inputs=[h, bias, m, scale, inf_a],
        template=[("I_DIM", I), ("J_DIM", J), ("NH", n_heads), ("C", c), ("HC", HC), ("ROW_STRIDE", F),
                  ("NSG", nsg), ("RQ", rq), ("BK", bk), ("PAD", pad), ("HAS_MASK", int(has_mask)),
                  ("VEC4", vec4), ("DT", mx.float16 if half else mx.float32), ("HT", h.dtype), ("OT", out_dtype)],
        grid=(((J + qpt - 1) // qpt) * nsg * 32, n_heads, B * I),
        threadgroup=(nsg * 32, 1, 1),
        output_shapes=[(B, I, J, HC)],
        output_dtypes=[out_dtype],
    )
    return o


def tri_attention_fused(h: mx.array, n_heads: int, c: int, mask: mx.array | None, inf: float,
                        bias: mx.array | None = None, tq: int = 64, tk: int = 32, impl: str | None = None,
                        out_dtype=mx.float32):
    """h: [B, I, J, F] with q|k|v|g in the first 4*H*C columns (F may include the tri-bias columns).
    bias: [B, H, K, Q] tri bias (bias(b,h,q,k) at [b,h,k,q]); if None it is taken from h[..., 4HC:]
    (only valid when I == J, i.e. not row-chunked). mask: [B, I, J] key mask or None.

    Returns sigmoid(g) * softmax(q k^T / sqrt(C) + mask_bias + tri_bias) v as [B, I, J, H*C].
    impl: None -> module attribute TRIATT_IMPL ("scalar" | "mma" | "mma_half").
    """
    B, I, J, F = h.shape
    HC = n_heads * c
    if bias is None:
        assert F == 4 * HC + n_heads and I == J
        bias = h[..., 4 * HC :].transpose(0, 3, 2, 1)  # [b, h, k, q] = h[b, q, k, h]
    impl = TRIATT_IMPL if impl is None else impl
    if bias.dtype != mx.float32:
        bias = bias.astype(mx.float32)  # tri bias is small ([B, H, N, N]); the softmax adds it in fp32
    if h.dtype != mx.float32 and impl == "scalar":
        impl = "mma"  # only the MMA kernel reads fp16 h
    if impl != "scalar":
        m = mask.reshape(B * I, J).astype(mx.float32) if mask is not None else mx.zeros((1,), mx.float32)
        return _tri_attention_mma(h, bias, m, mx.array(c**-0.5, mx.float32), mx.array(inf, mx.float32),
                                  n_heads, c, mask is not None, B, I, J, F, impl, out_dtype)
    key = ("tri_att", n_heads, c, mask is not None, tq, tk)
    if key not in _kernels:
        _kernels[key] = metal_kernel(
            name=f"boltz_tri_att_h{n_heads}_c{c}_m{int(mask is not None)}",
            input_names=["h", "bias", "mask", "scale", "inf_val"],
            output_names=["out"],
            source=_TRI_ATT_SRC,
        )
    k = _kernels[key]
    m = mask.reshape(B * I, J).astype(mx.float32) if mask is not None else mx.zeros((1,), mx.float32)
    (o,) = k(
        inputs=[h, bias, m, mx.array(c**-0.5, mx.float32), mx.array(inf, mx.float32)],
        template=[("I_DIM", I), ("J_DIM", J), ("NH", n_heads), ("C", c), ("HC", HC), ("ROW_STRIDE", F),
                  ("TQ", tq), ("TK", tk), ("HAS_MASK", int(mask is not None))],
        grid=(((J + tq - 1) // tq) * tq, n_heads, B * I),
        threadgroup=(tq, 1, 1),
        output_shapes=[(B, I, J, HC)],
        output_dtypes=[mx.float32],
    )
    return o


# ----------------------------------------------------------------------------------------------
# Fused Transition MLP (pair rep D=128, H=512): y = W3 (silu(LN(x) W1^T) * (LN(x) W2^T)), fp32.
# One threadgroup = BM rows; NSG simdgroups split the *columns* of every GEMM so each packed weight
# fragment is read by exactly one simdgroup per threadgroup:
#   LN        : rows -> threadgroup xs[BM][D]
#   per hidden chunk of HC units (SG g owns HS=HC/NSG of them):
#     GEMM1   : a,b = xs @ W1c^T, xs @ W2c^T      (A fragments from xs, B fragments straight from device)
#     SwiGLU  : h = silu(a)*b on the accumulator fragments -> threadgroup hs[BM][HC]  (double buffered)
#     GEMM2   : O[:, g*OS:(g+1)*OS] += hs @ W3c^T  (O stays in registers for all chunks)
# Weights are pre-packed (pack_transition_weights) into the simdgroup fragment layout, [frag][lane][2], so
# a B fragment is one coalesced 256 B float2-per-lane load - no threadgroup staging or barriers for weights.
# The [rows, 1024] / [rows, 512] intermediates never exist in device memory.
# ----------------------------------------------------------------------------------------------
_TRANS_HEADER = "#include <metal_simdgroup_matrix>\n#define UNR _Pragma(\"clang loop unroll(full)\")\n"
_TRANS_SRC = """
    constexpr int RB = BM / 8;
    constexpr int HS = HC / NSG;            // hidden units per simdgroup per chunk
    constexpr int NB1 = HS / 8;
    constexpr int OS = D / NSG;             // output columns per simdgroup
    constexpr int NB2 = OS / 8;
    constexpr int KS1 = D / 8;
    constexpr int NCH = H / HC;
    constexpr int KB2 = HC / 8;
    constexpr int NOB = D / 8;              // 8-wide output column blocks
    constexpr int LDX = D + PADX;
    constexpr int LDH = HC + PADH;

    uint sg = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint r0 = threadgroup_position_in_grid.x * BM;
    const uint nrows = n_rows[0];

    threadgroup float xs[BM * LDX];
    threadgroup float hs[2 * BM * LDH];

    // ---- LayerNorm (one simdgroup per row, rows strided over simdgroups) ----
    for (int r = sg; r < BM; r += NSG) {
        uint row = r0 + r;
        float v[D / 32];
        if (row < nrows) {
            const device float* xr = x + (size_t)row * D;
            float s = 0.0f;
            for (int i = 0; i < D / 32; i++) { v[i] = xr[lane + 32 * i]; s += v[i]; }
            float mean = simd_sum(s) / D;
            float ss = 0.0f;
            for (int i = 0; i < D / 32; i++) { float d = v[i] - mean; ss += d * d; }
            float rstd = metal::precise::rsqrt(simd_sum(ss) / D + eps[0]);
            for (int i = 0; i < D / 32; i++) {
                uint c = lane + 32 * i;
                xs[r * LDX + c] = (v[i] - mean) * rstd * ln_w[c] + ln_b[c];
            }
        } else {
            for (int i = 0; i < D / 32; i++) xs[r * LDX + lane + 32 * i] = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_float8x8 O[RB][NB2];
    UNR for (int rb = 0; rb < RB; rb++)
        UNR for (int nb = 0; nb < NB2; nb++) O[rb][nb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (int c = 0; c < NCH; c++) {
        simdgroup_float8x8 Sa[RB][NB1], Sb[RB][NB1];
        UNR for (int rb = 0; rb < RB; rb++)
            UNR for (int nb = 0; nb < NB1; nb++) {
                Sa[rb][nb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                Sb[rb][nb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            }
        // packed W1/W2: [2][H/8][KS1][32][2]
        const uint hb0 = (uint)(c * HC + sg * HS) / 8;
        UNR for (int ks = 0; ks < KS1; ks++) {
            simdgroup_float8x8 B1[NB1], B2[NB1];
            UNR for (int nb = 0; nb < NB1; nb++) {
                const device float* p1 = w12p + (((size_t)(hb0 + nb) * KS1 + ks) * 32 + lane) * 2;
                const device float* p2 = p1 + (size_t)(H / 8) * KS1 * 64;
                float2 v1 = *(const device float2*)p1;
                float2 v2 = *(const device float2*)p2;
                B1[nb].thread_elements()[0] = v1.x; B1[nb].thread_elements()[1] = v1.y;
                B2[nb].thread_elements()[0] = v2.x; B2[nb].thread_elements()[1] = v2.y;
            }
            UNR for (int rb = 0; rb < RB; rb++) {
                simdgroup_float8x8 Af;
                simdgroup_load(Af, xs + rb * 8 * LDX + ks * 8, LDX);
                UNR for (int nb = 0; nb < NB1; nb++) {
                    simdgroup_multiply_accumulate(Sa[rb][nb], Af, B1[nb], Sa[rb][nb]);
                    simdgroup_multiply_accumulate(Sb[rb][nb], Af, B2[nb], Sb[rb][nb]);
                }
            }
        }
        threadgroup float* hb = hs + (c & 1) * (BM * LDH);
        UNR for (int rb = 0; rb < RB; rb++)
            UNR for (int nb = 0; nb < NB1; nb++) {
                UNR for (int e = 0; e < 2; e++) {
                    float a = Sa[rb][nb].thread_elements()[e];
                    float bb = Sb[rb][nb].thread_elements()[e];
                    Sa[rb][nb].thread_elements()[e] = a / (1.0f + metal::exp(-a)) * bb;
                }
                simdgroup_store(Sa[rb][nb], hb + rb * 8 * LDH + sg * HS + nb * 8, LDH);
            }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // packed W3: [H/8][NOB][32][2]
        UNR for (int kb = 0; kb < KB2; kb++) {
            simdgroup_float8x8 B3[NB2];
            UNR for (int nb = 0; nb < NB2; nb++) {
                const device float* p = w3p + (((size_t)(c * KB2 + kb) * NOB + sg * NB2 + nb) * 32 + lane) * 2;
                float2 v = *(const device float2*)p;
                B3[nb].thread_elements()[0] = v.x; B3[nb].thread_elements()[1] = v.y;
            }
            UNR for (int rb = 0; rb < RB; rb++) {
                simdgroup_float8x8 Af;
                simdgroup_load(Af, hb + rb * 8 * LDH + kb * 8, LDH);
                UNR for (int nb = 0; nb < NB2; nb++)
                    simdgroup_multiply_accumulate(O[rb][nb], Af, B3[nb], O[rb][nb]);
            }
        }
    }

    // ---- epilogue ----
    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;
    UNR for (int rb = 0; rb < RB; rb++) {
        uint row = r0 + rb * 8 + fm;
        if (row < nrows) {
            UNR for (int nb = 0; nb < NB2; nb++) {
                device float* op = out + (size_t)row * D + sg * OS + nb * 8 + fn;
                *(device float2*)op = float2(O[rb][nb].thread_elements()[0], O[rb][nb].thread_elements()[1]);
            }
        }
    }
"""

# (BM rows/threadgroup, NSG simdgroups, HC hidden chunk)
TRANS_CFG = (24, 8, 64)
# fused fp32 kernel wins in the strict fp32 build; with TF32 enabled MLX's TF32 GEMMs are faster
TRANSITION_IMPL = os.environ.get("BOLTZ_MLX_TRANSITION_IMPL", "mlx" if os.environ.get("BOLTZ_MLX_TF32") == "1" else "fused")   # mlx | fused (read at call time)
TRANS_D, TRANS_H = 128, 512
_trans_pack_cache: dict = {}


def pack_transition_weights(w12: mx.array, w3: mx.array):
    """w12 [2H, D] (fc1 rows then fc2 rows), w3 [D, H] -> fragment-packed (w12p, w3p), both flat fp32."""
    import numpy as np

    H2, D = w12.shape
    H = H2 // 2
    assert w3.shape == (D, H) and D % 8 == 0 and H % 8 == 0
    lane = np.arange(32)
    qid = lane // 4
    fm = (qid & 4) + ((lane // 2) % 4)
    fn = (qid & 2) * 2 + (lane % 2) * 2
    j = np.arange(2)
    # B1[k, n] = W[n, k]: lane holds (row k = fm, cols n = fn, fn+1)  ->  W[hb*8 + fn + j, ks*8 + fm]
    hb = np.arange(H2 // 8)[:, None, None, None]
    ks = np.arange(D // 8)[None, :, None, None]
    idx1 = (hb * 8 + fn[None, None, :, None] + j) * D + ks * 8 + fm[None, None, :, None]
    # B3[k, n] = W3[n, k]: k = hidden block kb, n = output block nb  ->  W3[nb*8 + fn + j, kb*8 + fm]
    kb = np.arange(H // 8)[:, None, None, None]
    nb = np.arange(D // 8)[None, :, None, None]
    idx3 = (nb * 8 + fn[None, None, :, None] + j) * H + kb * 8 + fm[None, None, :, None]
    w12p = mx.take(w12.reshape(-1), mx.array(idx1.reshape(-1).astype(np.int32)))
    w3p = mx.take(w3.reshape(-1), mx.array(idx3.reshape(-1).astype(np.int32)))
    mx.eval(w12p, w3p)
    return w12p, w3p


def transition_supported(x: mx.array, w12: mx.array, w3: mx.array) -> bool:
    # fp32 weights only (an fp16 parameter of the mixed checkpoint, reached by the f16 guard fallback, takes the MLX path)
    return bool(USE_KERNELS and x.dtype == mx.float32 and w12.dtype == mx.float32 and w3.dtype == mx.float32
                and w12.shape == (2 * TRANS_H, TRANS_D)
                and w3.shape == (TRANS_D, TRANS_H) and x.shape[-1] == TRANS_D)


def transition_fused(x: mx.array, ln_w: mx.array, ln_b: mx.array, w12: mx.array, w3: mx.array,
                     eps: float = 1e-5, packed=None) -> mx.array:
    """Fused LayerNorm -> SwiGLU MLP (D=128, H=512, fp32). x [..., 128] -> [..., 128].
    `packed` = pack_transition_weights(w12, w3) (cached by the caller; computed and cached here if None)."""
    D, H = TRANS_D, TRANS_H
    BM, NSG, HC = TRANS_CFG
    assert transition_supported(x, w12, w3)
    if packed is None:
        key = (id(w12), id(w3))
        ent = _trans_pack_cache.get(key)
        if ent is None or ent[0] is not w12 or ent[1] is not w3:
            ent = (w12, w3, pack_transition_weights(w12, w3))
            _trans_pack_cache[key] = ent
        packed = ent[2]
    w12p, w3p = packed
    n_rows = x.size // D
    k = _kernels.get("transition")
    if k is None:
        k = _kernels["transition"] = metal_kernel(
            name="boltz_transition_fused",
            input_names=["x", "ln_w", "ln_b", "w12p", "w3p", "eps", "n_rows"],
            output_names=["out"],
            source=_TRANS_SRC,
            header=_TRANS_HEADER,
        )
    (y,) = k(
        inputs=[x, ln_w, ln_b, w12p, w3p, mx.array([eps], mx.float32), mx.array([n_rows], mx.uint32)],
        template=[("D", D), ("H", H), ("BM", BM), ("NSG", NSG), ("HC", HC), ("PADX", 4), ("PADH", 4)],
        grid=(((n_rows + BM - 1) // BM) * NSG * 32, 1, 1),
        threadgroup=(NSG * 32, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[mx.float32],
    )
    return y


# ----------------------------------------------------------------------------------------------
# Memory: chunk pair-representation ops above CHUNK_N tokens (results identical to unchunked).
# ----------------------------------------------------------------------------------------------
CHUNK_N = int(os.environ.get("BOLTZ_MLX_CHUNK_N", "512"))
CHUNK_ROWS = int(os.environ.get("BOLTZ_MLX_CHUNK_ROWS", "128"))
CHUNK_CH = int(os.environ.get("BOLTZ_MLX_CHUNK_CH", "32"))


def chunking(n: int) -> bool:
    return n > CHUNK_N


# ----------------------------------------------------------------------------------------------
# Pipelined submission (R6-prof; numerics-neutral). Without it a whole recycle (MSA module + 64 pairformer layers,
# est. 0.5-1 ms of Python graph building per layer) is built lazily and only then evaluated, so the GPU idles for the
# entire build of every recycle (est. 0.02-0.08 s per recycle, independent of N; tools/prof/pipeline_profile.py
# measures it). With BOLTZ_MLX_PIPE_LAYERS=1 (default) the layer
# stacks hand each finished layer to mx.async_eval: the GPU runs layer k while Python builds layer k+1 (MLX's
# scheduler keeps the host at most ~10 command buffers ahead, so memory is bounded as before). Same kernels on the
# same inputs in the same order: bitwise-identical results (fingerprints). The per-recycle guard sync is unchanged.
# ----------------------------------------------------------------------------------------------
PIPE_LAYERS = os.environ.get("BOLTZ_MLX_PIPE_LAYERS", "1") == "1"


def submit(*arrays) -> None:
    """Start evaluating `arrays` on the GPU without waiting (no-op when off or inside a function mx.compile may
    trace). In per-op guard mode the pending guard count goes first, as in guard_first."""
    if not PIPE_LAYERS or GUARD_SUSPEND:
        return
    mx.async_eval(*guard_first(*arrays))


# ----------------------------------------------------------------------------------------------
# Activation-range recorder (tools/prof/act_ranges.py). No-op unless a hook is installed; never on in normal runs.
# record(name, x) reports x under the current phase ("trunk.msa", "trunk.pairformer", "confidence", ...).
# PHASE_HOOK (tools/prof/pipeline_profile.py): called as PHASE_HOOK(name) at every phase change (host timeline).
# ----------------------------------------------------------------------------------------------
ACT_HOOK = None
ACT_PHASE = "?"
PHASE_HOOK = None


def set_phase(name: str) -> None:
    global ACT_PHASE
    ACT_PHASE = name
    if PHASE_HOOK is not None:
        PHASE_HOOK(name)


def record(name: str, x) -> None:
    if ACT_HOOK is not None:
        ACT_HOOK(ACT_PHASE, name, x)


# ----------------------------------------------------------------------------------------------
# Round-2 precision mode. BOLTZ_MLX_PRECISION = fp32 (default: strict / round-1 modes)
# | f16 (fp16 GEMM operands + outputs where ACT_RANGES allows; everything in "always fp32" stays fp32).
# ----------------------------------------------------------------------------------------------
PRECISION = os.environ.get("BOLTZ_MLX_PRECISION", "fp32")


def f16() -> bool:
    return PRECISION == "f16" and _use_kernels()


def h16(x: mx.array) -> mx.array:
    return x if x.dtype == mx.float16 else x.astype(mx.float16)


# Overflow guard: lazily accumulated count of non-finite values in fp16 GEMM outputs (checked per recycle / step).
_GUARD = {"count": None, "by_op": {}}
GUARD_BY_OP = os.environ.get("BOLTZ_MLX_GUARD_BY_OP", "0") == "1"
# Guard modes: "residual" (default, shipped): inf/nan from any fp16 output propagates into the fp32 residual streams,
# so z and s are checked once per recycle (one reduction each). "op": a lazy count on every fp16 GEMM output
# (validation / diagnostics; numerically identical, measured ~80% slower at prot117 because the pending counts keep
# every guarded tensor alive). "off": cost measurement only.
GUARD_MODE = os.environ.get("BOLTZ_MLX_GUARD_MODE", "residual")
GUARDS = os.environ.get("BOLTZ_MLX_GUARDS", "1") == "1" and GUARD_MODE != "off"
GUARD_SUSPEND = 0  # > 0 inside functions that mx.compile may trace (diffusion step): non-finite coords are checked instead


def guard(name: str, x: mx.array) -> mx.array:
    """Adds the number of inf/nan in x to the running guard count (lazy, no sync); returns x."""
    if not GUARDS or GUARD_MODE != "op" or GUARD_SUSPEND or not f16():
        return x
    c = mx.sum(~mx.isfinite(x)).astype(mx.int32)
    _GUARD["count"] = c if _GUARD["count"] is None else _GUARD["count"] + c
    _GUARD["n"] = _GUARD.get("n", 0) + 1
    if _GUARD["n"] % 16 == 0:  # bound the tensors a pending count keeps alive (outside the per-recycle evals too)
        mx.eval(_GUARD["count"])
    if GUARD_BY_OP:
        prev = _GUARD["by_op"].get(name)
        _GUARD["by_op"][name] = c if prev is None else prev + c
    return x


def guard_pending():
    """The lazy guard count (or None). Put it FIRST in an mx.eval list so each fp16 output's count is computed
    right after that output (otherwise the counts keep every guarded tensor alive until the end)."""
    return _GUARD["count"]


def guard_residual(*arrays) -> None:
    """Residual-mode guard: add the inf/nan count of the given fp32 streams (lazy)."""
    if not GUARDS or GUARD_MODE != "residual" or not f16():
        return
    for x in arrays:
        c = mx.sum(~mx.isfinite(x)).astype(mx.int32)
        _GUARD["count"] = c if _GUARD["count"] is None else _GUARD["count"] + c


def guard_first(*arrays):
    g = _GUARD["count"]
    return ([g] if g is not None else []) + list(arrays)


def guard_take() -> int:
    """Evaluate and reset the guard count (one sync)."""
    c = _GUARD["count"]
    _GUARD["count"] = None
    return 0 if c is None else int(c.item())


def guard_by_op_take() -> dict:
    d = {k: int(v.item()) for k, v in _GUARD["by_op"].items()}
    _GUARD["by_op"] = {}
    return d


@mx.compile
def _radd_cast(z, y):
    return z + y.astype(z.dtype)


@mx.compile
def _radd_cast_scale(z, y, s):
    return z + y.astype(z.dtype) * s


class Scaled:
    """An fp16 module output that carries an exact power-of-two scale (true value = y * scale), undone in fp32
    inside the residual add."""

    def __init__(self, y: mx.array, scale: float):
        self.y, self.scale = y, scale


def radd(z: mx.array, y) -> mx.array:
    """Residual add; an fp16 update is cast (and a Scaled update unscaled) inside one fused (compiled) kernel.
    fp32 + fp32 is a plain add."""
    if isinstance(y, Scaled):
        return _radd_cast_scale(z, y.y, mx.array(y.scale, mx.float32))
    return z + y if y.dtype == z.dtype else _radd_cast(z, y)


def unscale(y) -> mx.array:
    """fp32 value of a possibly Scaled output (for callers that do not go through radd)."""
    if isinstance(y, Scaled):
        return y.y.astype(mx.float32) * y.scale
    return y


@mx.compile
def _swiglu16_scale(a, b, s):
    return (mx.sigmoid(a.astype(mx.float32)) * a.astype(mx.float32) * b.astype(mx.float32) * s).astype(mx.float16)


# Power-of-two down-scaling of the transition hidden activations (so fc3's fp16 output keeps >= 16x headroom), per
# input dim, chosen from the activation ranges (tools/prof/act_ranges.py); the output is returned as Scaled and unscaled in fp32 in radd.
TRANSITION_DOWN_K = {64: int(os.environ.get("BOLTZ_MLX_TRANS64_K", "2"))}


# Weights of fp16 GEMMs. Every fp16 GEMM weight goes through w16 / wcast with the
# parameters it is made of (`srcs`, concatenated along axis 0), never a pre-built fp32 concatenation, so that
#  - with the fp32 checkpoint, the fp16 copy is cast at first use (fp32 parameters + fp16 cache), and
#  - with the mixed checkpoint (boltz_mlx/mixed.py), the parameters already are fp16: no cast, no fp32 copy.
# cast(concat(srcs)) == concat(cast(srcs)) elementwise, so both give bit-identical fp16 weights.
# W16_TRACE (tools/make_mixed_checkpoint.py --derive): called as W16_TRACE(owner, name, srcs) for every fp16 weight
# built, which is how the list of fp16-consumed parameters (boltz_mlx/fp16_manifest.json) is derived from the code.
W16_TRACE = None


def wcast(srcs, dt=mx.float16, owner=None, name: str = "") -> mx.array:
    """concatenate(srcs, 0) in dtype dt; sources already in dt (mixed checkpoint) are used as they are (no copy for a
    single source)."""
    if W16_TRACE is not None and dt == mx.float16:
        W16_TRACE(owner, name, tuple(srcs))
    w = srcs[0] if len(srcs) == 1 else mx.concatenate(list(srcs), 0)
    return w if w.dtype == dt else w.astype(dt)


def w16(owner, name: str, *srcs: mx.array) -> mx.array:
    """fp16 weight concatenate(srcs, 0), cached on `owner` as attribute `name` (mixed checkpoint: boltz_mlx/mixed.py
    pre-fills these caches at load time)."""
    h = owner.__dict__.get(name)
    if h is None:
        h = wcast(srcs, mx.float16, owner, name)
        mx.eval(h)
        owner.__dict__[name] = h
    return h


@mx.compile
def _bias16(y, b):
    return (y.astype(mx.float32) + b).astype(mx.float16)


def lin16(x: mx.array, lin, name: str = "_w16") -> mx.array:
    """fp16 nn.Linear: fp16 x @ fp16 W^T (fp32 accumulate in the GEMM, fp16 out); bias added in fp32 in one
    fused op (never addmm in fp16 paths)."""
    y = h16(x) @ w16(lin, name, lin.weight).T
    b = lin.get("bias") if "bias" in lin else None
    return y if b is None else _bias16(y, b)


@mx.compile
def _gate16(p, g):
    """p * sigmoid(g) in fp32, fp16 out."""
    return (p.astype(mx.float32) * mx.sigmoid(g.astype(mx.float32))).astype(mx.float16)


@mx.compile
def _swiglu16(a, b):
    return (mx.sigmoid(a.astype(mx.float32)) * a.astype(mx.float32) * b.astype(mx.float32)).astype(mx.float16)


def swiglu16(h: mx.array, hidden: int) -> mx.array:
    return _swiglu16(h[..., :hidden], h[..., hidden:])


@mx.compile
def _gate_mask_scale16(p, g, m, s):
    return (p.astype(mx.float32) * mx.sigmoid(g.astype(mx.float32)) * m * s).astype(mx.float16)


# Power-of-two pre-scaling of the triangle-multiplication operands a, b (each scaled by 2^-TRIMULT_K, the
# contraction output by 2^-2K). K is chosen from the activation ranges (tools/prof/act_ranges.py) (contraction max over all cases <= 4096 * 2^2K);
# the LayerNorm that follows is scale-invariant and is given eps * 2^-4K, so the scale cancels exactly.
TRIMULT_K = int(os.environ.get("BOLTZ_MLX_TRIMULT_K", "4"))  # ACT_RANGES: max 7.9e5 (prot1000) / 2^8 = 3088


# ----------------------------------------------------------------------------------------------
# Fused residual add + LayerNorm (f16 pair stack): z_new = z + y (y fp16/fp32, optional exact power-of-two scale,
# optionally read transposed) and LN(z_new) written fp16 (optionally transposed) in one pass. Same fp32 arithmetic as
# radd followed by layer_norm, so results are bit-identical to the unfused path.
# ----------------------------------------------------------------------------------------------
_ADD_LN_SRC = """
    uint row = threadgroup_position_in_grid.x * ROWS_PER_TG + simdgroup_index_in_threadgroup;
    if (row >= n_rows) return;
    uint lane = thread_index_in_simdgroup;
    uint ri = row / NN, rj = row % NN;
    uint yrow = TRANS_Y ? (rj * NN + ri) : row;
    uint orow = TRANS_O ? (rj * NN + ri) : row;
    const device float* zr = z + (size_t)row * D;
    const device YT* yr = y + (size_t)yrow * D;
    device float* zo = z_out + (size_t)row * D;
    device OT* lo = ln_out + (size_t)orow * D;
    float v[D / 32];
    float s = 0.0f;
    for (int i = 0; i < D / 32; i++) {
        uint c = lane + 32 * i;
        float yv = float(yr[c]);
        if (HAS_SCALE) yv = yv * scale[0];
        v[i] = zr[c] + yv;
        zo[c] = v[i];
        s += v[i];
    }
    float mean = simd_sum(s) / D;
    float ss = 0.0f;
    for (int i = 0; i < D / 32; i++) {
        float d = v[i] - mean;
        ss += d * d;
    }
    float rstd = metal::precise::rsqrt(simd_sum(ss) / D + eps[0]);
    for (int i = 0; i < D / 32; i++) {
        uint c = lane + 32 * i;
        float o = (v[i] - mean) * rstd;
        if (HAS_W) o = o * w[c];
        if (HAS_B) o = o + b[c];
        lo[c] = OT(o);
    }
"""


_ADD_LN_DYN_SRC = _ADD_LN_SRC.replace(
    "    if (row >= n_rows) return;",
    "    const uint n_rows_v = (uint)dims[0];\n    const int NN = (int)dims[1];\n    if (row >= n_rows_v) return;")
assert _ADD_LN_DYN_SRC != _ADD_LN_SRC


def add_ln(z: mx.array, y, ln, trans_y: bool = False, trans_out: bool = False):
    """(z + y, LN(z + y) in fp16) for z [1, N, N, D] fp32. y: [1, N, N, D] (fp16/fp32 or Scaled); trans_y reads y[j, i];
    trans_out writes the LN output at [j, i] (the ending-node layout). ln: a LayerNorm module (weight/bias/eps)."""
    scale = None
    if isinstance(y, Scaled):
        y, scale = y.y, y.scale
    B, N, M, D = z.shape
    assert B == 1 and N == M and D % 32 == 0 and z.dtype == mx.float32
    w, b = ln.get("weight"), ln.get("bias")
    dyn = nax("DYNN")
    key = ("add_ln", w is not None, b is not None) + (("dyn",) if dyn else ())
    if key not in _kernels:
        _kernels[key] = metal_kernel(
            name=f"boltz_add_ln_w{int(w is not None)}_b{int(b is not None)}" + ("_dyn" if dyn else ""),
            input_names=["z", "y", "w", "b", "eps", "scale", "dims" if dyn else "n_rows"],
            output_names=["z_out", "ln_out"],
            source=_ADD_LN_DYN_SRC if dyn else _ADD_LN_SRC,
        )
    n_rows = N * N
    dummy = mx.zeros((1,), mx.float32)
    zo, lo = _kernels[key](
        inputs=[z, y, w if w is not None else dummy, b if b is not None else dummy, mx.array([ln.eps], mx.float32),
                mx.array([scale if scale is not None else 1.0], mx.float32),
                mx.array([n_rows, N], mx.uint32) if dyn else mx.array(n_rows, mx.uint32)],
        template=[("D", D), ("ROWS_PER_TG", ROWS_PER_TG), ("HAS_W", int(w is not None)), ("HAS_B", int(b is not None)),
                  ("YT", y.dtype), ("OT", mx.float16)] + ([] if dyn else [("NN", N)])
                 + [("TRANS_Y", int(trans_y)), ("TRANS_O", int(trans_out)), ("HAS_SCALE", int(scale is not None))],
        grid=(((n_rows + ROWS_PER_TG - 1) // ROWS_PER_TG) * 32 * ROWS_PER_TG, 1, 1),
        threadgroup=(32 * ROWS_PER_TG, 1, 1),
        output_shapes=[z.shape, z.shape],
        output_dtypes=[mx.float32, mx.float16],
    )
    return zo, lo


FUSE_ADD_LN = os.environ.get("BOLTZ_MLX_FUSE_ADD_LN", "1") == "1"


def pair_epi_ok(z: mx.array) -> bool:
    """Round-3 pair epilogues (kernels_nax3: residual add + next LN inside the producing GEMM) apply to z?"""
    if not nax("PAIREPI") or GUARD_MODE == "op":  # per-op guards need the fp16 module outputs
        return False
    from boltz_mlx import kernels_nax3 as _KN3

    return _KN3.pair_ok(z, 32)


def pair_lnx_ok(z: mx.array) -> bool:
    """R3-pair transition epilogue (next layer's norm_in LN + s-track pair bias from the new z rows) applies?"""
    if not nax("PAIRLNX") or not pair_epi_ok(z) or 128 not in NAX_TRANSITION_CFG or not nax("TRANSITION") \
            or TRANSITION_DOWN_K.get(128, 0):
        return False
    from boltz_mlx import kernels_nax3 as _KN3

    return _KN3.transition_lnx_ok(z, NAX_TRANSITION_CFG[128])


def pair_block_f16(layer, z: mx.array, pair_mask: mx.array, xn_in: mx.array | None = None, lnx=None):
    """z track of a pairformer layer in f16 with each residual add fused into the next module's LayerNorm.
    xn_in: this layer's tri_mul_out.norm_in(z) in fp16 if the previous layer's transition already wrote it.
    lnx = (next_ln | None, attention | None) (only with pair_lnx_ok): the transition also writes next_ln(z_new) and the
    attention's pair bias; returns (z, xn_next, pair_bias) instead of z."""
    if pair_epi_ok(z):
        return pair_block_f16_epi(layer, z, pair_mask, xn_in, lnx)
    assert lnx is None
    xn = layer.tri_mul_out.norm_in.f16(z) if xn_in is None else xn_in
    y = layer.tri_mul_out(z, mask=pair_mask, xn=xn)
    z, xn = add_ln(z, y, layer.tri_mul_in.norm_in)
    y = layer.tri_mul_in(z, mask=pair_mask, xn=xn)
    z, xn = add_ln(z, y, layer.tri_att_start.layer_norm)
    y = layer.tri_att_start(z, mask=pair_mask, xn=xn)
    z, xn_t = add_ln(z, y, layer.tri_att_end.layer_norm, trans_out=True)
    y_t = layer.tri_att_end(z, mask=pair_mask, xn=xn_t, keep_t=True)
    z, xn = add_ln(z, y_t, layer.transition_z.norm, trans_y=True)
    zr = layer.transition_z.f16_resid(z, xn)  # NAX transition with the residual add fused (bitwise == radd)
    return zr if zr is not None else radd(z, layer.transition_z(z, xn=xn))


def pair_block_f16_epi(layer, z: mx.array, pair_mask: mx.array, xn_in: mx.array | None = None, lnx=None):
    """pair_block_f16 with every residual add + next LayerNorm inside the producer's NAX GEMM epilogue
    (kernels_nax3.trimul_tail_addln / linear_addln): no fp16 module output is written and read back, no separate
    norm_out / gate / bias / add_ln passes. Residual arithmetic as add_ln (z + float(y16), y16 rounded as lin16 /
    _gate16); LN statistics summed in a different order (ulp-level differences in the fp16 LN outputs).
    R3-pair (flags): PAIRTB — the LN epilogues feeding the two triangle attentions also write their 4-head tri bias
    (no x @ W_b^T re-read of the LN output, no transposed copy); PAIRLNX — see pair_block_f16 (lnx)."""
    xn = layer.tri_mul_out.norm_in.f16(z) if xn_in is None else xn_in
    z, xn = layer.tri_mul_out.f16_addln(z, pair_mask, xn, layer.tri_mul_in.norm_in)
    if nax("PAIRTB") and layer.tri_att_start.tb_ok(xn) and layer.tri_att_end.tb_ok(xn):
        ts, te = layer.tri_att_start, layer.tri_att_end
        z, xn, tb = layer.tri_mul_in.f16_addln(z, pair_mask, xn, ts.layer_norm, w_tb=ts.tb_weight16())
        z, xn_t, tb_t = ts.f16_addln(z, pair_mask, xn, te.layer_norm, trans_out=True, tb=tb, w_tb=te.tb_weight16())
        z, xn = te.f16_addln(z, pair_mask, xn_t, layer.transition_z.norm, tb=tb_t)
    else:
        z, xn = layer.tri_mul_in.f16_addln(z, pair_mask, xn, layer.tri_att_start.layer_norm)
        z, xn_t = layer.tri_att_start.f16_addln(z, pair_mask, xn, layer.tri_att_end.layer_norm, trans_out=True)
        z, xn = layer.tri_att_end.f16_addln(z, pair_mask, xn_t, layer.transition_z.norm)
    if lnx is not None:
        return layer.transition_z.f16_resid_lnx(z, xn, *lnx)
    zr = layer.transition_z.f16_resid(z, xn)
    return zr if zr is not None else radd(z, layer.transition_z(z, xn=xn))


def use_pair_block_f16(z: mx.array) -> bool:
    return FUSE_ADD_LN and f16() and z.ndim == 4 and z.shape[0] == 1 and z.shape[1] == z.shape[2] \
        and not chunking(z.shape[1]) and z.dtype == mx.float32


# f16 triangle-attention core: "nax" = NAX (Metal 4 tensor-op) flash kernel (kernels_nax.py; L1-validated in
# logs/triatt_2n.jsonl), "mma" = the simdgroup-MMA kernel. Read at call time.
TRIATT_F16 = os.environ.get("BOLTZ_MLX_TRIATT_F16", "nax")
TRIATT_NAX_CFG = {"order": 1, "wq": 5}


# NAX kernels of round 2 (kernels_nax2.py). Env BOLTZ_MLX_NAX_TRANSITION / _TRIMUL /
# _DIFFATTN = 1 | 0; unset = on in --precision f16 only, so strict / fast / turbo never reach them. Read at call time.
#   TRANSITION: fused SwiGLU transition (pair D=128, MSA D=64), + fused residual add in pair_block_f16
#   TRIMUL    : fused tri-mult projection + gate + mask + 2^-K + channel-major store (the contraction stays the MLX
#               bmm: the NAX contraction was bit-identical but slower, 1.60 vs 1.10 ms at N=400)
#   DIFFATTN  : diffusion token-transformer attention (16 heads x 48) on NAX (fp16 diffusion matmuls only)
# Round 3 (kernels_nax3.py; speed-first round). ON in f16 for this round's measurement (round_check / bench strip
# BOLTZ_MLX_* variables), PENDING the queued GPU validation (tests/test_nax3_kernels.py, tools/prof/nax3_ab.py gates):
# if that fails, put the group back into NAX_DEFAULT_OFF (or revert the commit that emptied it).
#   GEMM      : token transformer on fused kernels (adaln16 from the raw fp16 conditioning, NAX GEMMs with gated-residual
#               / SwiGLU epilogues, pair bias shared across samples); per-site config BOLTZ_MLX_NAX_GEMM_CFG
#   PAIREPI   : pair_block_f16 producers with the residual add + next LayerNorm in the GEMM epilogue (tri-att linear_o,
#               tri-mult norm_out -> p_out * sigmoid(g_out)); per-site config BOLTZ_MLX_NAX_PAIR_CFG
# R3-pair (pair stack, kernels_nax3) — validated 2026-10-28 19:05 (strict + flags-off fingerprints, 149 tests,
# pair_profile gates, R3-pair round check PASS):
#   PAIRLNX   : the pair transition's epilogue writes the next pairformer layer's tri_mul_out.norm_in output and the
#               s-track attention's pair bias (proj_z LN + projection) from the new z rows (PairformerModule /
#               PairformerNoSeqModule carry it to the next layer)
#   PAIRPROJ  : triangle-attention q|k|v|g projection (N^2 x 128 -> 512) on kernels_nax3.gemm16 instead of MLX
#   PAIRTB    : the LN epilogues feeding the triangle attentions also write their 4-head tri bias
# Atom round (kernels_atom.py) — validated 2026-10-28 19:27 (strict + atom-off fingerprints, 91 tests, atom_ab accuracy
# gate, R3-atom round check PASS):
#   ATOM      : diffusion atom encoder / decoder: AtomTransformers on adaln16 / gemm16 (+ epilogues) and a NAX windowed
#               attention reading the keys in place (no key-window gather, no fp32 SDPA fallback), conditioning shared
#               across samples; atom -> token mean as a segment kernel, token -> atom broadcast as a gather, s_to_a on
#               the shared single conditioning. Per-site config boltz_mlx/atom_cfg.json / BOLTZ_MLX_ATOM_CFG.
# R4-pair (measured negative, not merged into round2: in-model 0.1-0.2 s slower at prot400 together with the dropped
# transition variant): TAIL2 = the split tri-mult tail (transpose_ln16 + gemm2_gate_addln), kept as an off-by-default
# option (alone it saved 0.13 ms per tail in isolation).
# R5-tail: TAIL3 = the fused tri-mult tail with lower register pressure (p / g in column chunks, fp16 p overwritten in
#   place by y16, max_total_threads_per_threadgroup declared, configurable g-loop unrolling): same arithmetic and
#   accumulation order -> outputs bitwise equal to the shipped fused tail; shipped only on an in-model A/B win
#   (tools/prof/pair_ab_inmodel.py). Config tail3= in nax3_cfg.json (index, or off).
# R6-trans: TRANS3 = pair transition implementation chosen per nax3_cfg.json trans3= (kernels_nax3.TRANS3_CFGS index:
#   register-fragment NAX kernel with an in-register next-LN / pair-bias epilogue, other MPP-kernel configs, or MLX fp16
#   GEMMs + SwiGLU; or off). Not bitwise (ulp-level vs the shipped kernel); shipped only on an in-model A/B win.
# R8-coldN: DYNN = size-only constants (token counts N, padded key counts) of the per-layer kernels passed at run time
#   in a small int buffer instead of baked into each kernel's template, so a never-seen N does not compile new Metal
#   libraries (tri-att NAX core, diffusion NAX attention, linear_addln / tri-mult tail (+ tri bias), the register
#   transition, add_ln). Same source text otherwise: tile shapes, loops, accumulation order and integer types are kept
#   (`const int` locals with the template names), so outputs are bitwise identical.
NAX_FLAGS = {k: os.environ.get(f"BOLTZ_MLX_NAX_{k}")
             for k in ("TRANSITION", "TRIMUL", "DIFFATTN", "GEMM", "PAIREPI", "PAIRLNX", "PAIRPROJ", "PAIRTB", "ATOM",
                       "TAIL2", "TAIL3", "TRANS3", "DYNN")}
NAX_DEFAULT_OFF: set = {"TAIL2"}  # default-off groups (DYNN on for the R8-coldN round check, pending the in-model A/B)
# f16: atom-attention keys by index gather instead of the one-hot contraction (encodersv2.ToKeys.gather; same values up
# to the sign of zeros in masked key slots). Env BOLTZ_MLX_TOKEYS_GATHER=0 restores the contraction.
TOKEYS_GATHER = os.environ.get("BOLTZ_MLX_TOKEYS_GATHER", "1") == "1"
NAX_TRANSITION_CFG = {128: dict(tm=16, hb=32, nsg=8, tgw=1), 64: dict(tm=16, hb=32, nsg=4, tgw=0)}  # by input dim
NAX_TRIMUL_CFG = dict(tp=32, tc=16, nsg=4, xreg=1)
NAX_DIFFATTN_CFG = dict(wq=4, nkf=2, order=1)

# Per-device launch shapes of the two NAX attention kernels above (nax3_cfg.json entry key "kern", e.g.
# "triatt.wq=4,triatt@160.wq=8,diffattn@256.wq=1"): only keys that regroup independent work are honoured — WQ (query
# blocks of 16 per threadgroup), HPT (heads per threadgroup, tri-att) and ORDER (grid order). Each simdgroup computes
# its 16 queries x 1 head exactly as before (same key blocking NKF, same online-softmax order), so every choice gives
# bitwise the same output (tests/test_cfg_bitwise.py). "site@MAX.key=v": a size bucket (MAX = queries per row J for
# triatt, rows B * N for diffattn), the first bucket with size <= MAX wins over the site's plain entry.
_KERN_NEUTRAL = {"triatt": ("wq", "hpt", "order"), "diffattn": ("wq", "order")}
KERN_BUCKETS: dict = {"triatt": [], "diffattn": []}


def _kern_tables() -> dict:
    return {"triatt": TRIATT_NAX_CFG, "diffattn": NAX_DIFFATTN_CFG}


def apply_kern_cfg(spec: str, tables: dict | None = None, kb: dict | None = None) -> None:
    """Apply a "kern" spec (see above); unknown sites / keys are ignored (never a numerics-relevant key). A spec with
    plain items for a site but no buckets for it clears that site's buckets (as kernels_nax3 site specs)."""
    tables = _kern_tables() if tables is None else tables
    kb = KERN_BUCKETS if kb is None else kb
    buckets: dict = {}
    plain: set = set()
    for item in filter(None, spec.split(",")):
        name, v = item.split("=")
        site, key = name.rsplit(".", 1)
        site, _, mx_ = site.partition("@")
        if site not in _KERN_NEUTRAL or key not in _KERN_NEUTRAL[site]:
            continue
        if mx_:
            buckets.setdefault(site, {}).setdefault(int(mx_), {})[key] = int(v)
        else:
            tables[site][key] = int(v)
            plain.add(site)
    for site in plain - buckets.keys():
        kb[site] = []
    for site, b in buckets.items():
        kb[site] = sorted(b.items())


def _kern_pick(site: str, size: int, tables: dict, kb: dict) -> dict:
    for mx_, over in kb.get(site, ()):
        if size <= mx_:
            return {**tables[site], **over}
    return tables[site]


def _kern_cfg(site: str, size: int) -> dict:
    return _kern_pick(site, size, _kern_tables(), KERN_BUCKETS)


def resolve_kern(spec: str):
    """(site, size) -> launch config under a "kern" spec applied to the module defaults (live tables untouched)."""
    tables = {k: dict(v) for k, v in _KERN_BASE.items()}
    kb: dict = {}
    apply_kern_cfg(spec, tables, kb)
    return lambda site, size: _kern_pick(site, size, tables, kb)


def triatt_cfg(J: int) -> dict:
    """tri_attention_nax launch config for J queries per row (TRIATT_NAX_CFG or this chip's size bucket)."""
    return _kern_cfg("triatt", J)


def diffattn_cfg(rows: int) -> dict:
    """attention_nax launch config of the diffusion token transformer for B * N query rows."""
    return _kern_cfg("diffattn", rows)


def _load_kern_cfg() -> None:
    import json

    path = os.path.join(os.path.dirname(__file__), "nax3_cfg.json")
    if not os.path.exists(path):
        return
    with open(path) as f:
        d = json.load(f)
    e = d.get(cfg_chip(), d.get("default", {})) if "default" in d else d   # the note is kernels_nax3's (same file)
    apply_kern_cfg(e.get("kern", ""))
    apply_kern_cfg(os.environ.get("BOLTZ_MLX_NAX_KERN_CFG", ""))


_KERN_BASE = {k: dict(v) for k, v in _kern_tables().items()}   # module defaults (before any device entry)
_load_kern_cfg()


# Groups in NAX_FLAGS that select a kernel variant, not MPP kernels: DYNN (run-time sizes) also applies to the non-MPP
# add_ln kernel of the portable path (bitwise identical either way), so it does not require mpp_enabled().
NAX_INDEPENDENT = {"DYNN"}


def nax(kind: str) -> bool:
    """Is MPP kernel group `kind` (TRANSITION | TRIMUL | DIFFATTN | GEMM | PAIREPI | ATOM | ...) on? Never on the
    portable kernel path (mpp_enabled() is checked last, so strict / CPU calls never reach detection)."""
    v = NAX_FLAGS.get(kind)
    on = (v == "1" and _use_kernels()) if v is not None else (f16() and kind not in NAX_DEFAULT_OFF)
    on = on and (kind in NAX_INDEPENDENT or mpp_enabled())
    if on and _CFG_NOTES["nax"]:
        _cfg_note("nax")
    return on


def triatt_nax() -> bool:
    """f16 triangle-attention core on the MPP flash kernel (else the simdgroup-MMA kernel, out_dtype fp16)."""
    return TRIATT_F16 == "nax" and mpp_enabled()


def msa_opm_nax() -> bool:
    """OPM proj_o inside the MSA kernel's epilogue on MPP (else kernels_msa.opm_outer + the MLX fp16 proj_o GEMM)."""
    return MSA_OPM_NAX and mpp_enabled()


# f16 OPM: re-associated (proj_o folded into b) when S * c_hidden <= this (crossover ~S=42 at c=32, c_out=128)
OPM_REASSOC_MAX_K = int(os.environ.get("BOLTZ_MLX_OPM_REASSOC_MAX_K", "1024"))


# MSA-module kernels (kernels_msa.py, fp32 simdgroup MMA for the MSA-row / token reductions). Env BOLTZ_MLX_MSA_OPM /
# BOLTZ_MLX_MSA_PWA = 1 | 0; unset = MSA_DEFAULT_ON in --precision f16. f16 paths only (the call sites also require
# f16()), so strict / fast / turbo never reach them. Read at call time.
#   OPM: outer sum over MSA rows (fp32 MMA) with the / num_mask + fp16 cast fused in its epilogue, written pair-major
#        (the fp16 proj_o GEMM input); replaces the MLX fp32 GEMM + transposed copy + divide/cast pass. Used when the
#        re-associated path does not apply (S * c_hidden > OPM_REASSOC_MAX_K).
#   PWA: softmax weights straight from z (LN + proj_z + mask + softmax, fp32) and the weighted average over tokens
#        (fp32 MMA) with the sigmoid gate fused; replaces the fp16 LN/proj_z, transposes, batched matmul and gate pass.
MSA_FLAGS = {k: os.environ.get(f"BOLTZ_MLX_MSA_{k}") for k in ("OPM", "PWA")}
# Owner brief: default on in f16 only once validated. Off until the msa-agent GPU queue items (tests/test_msa_kernels.py,
# tools/prof/msa_kernels.py, L1 msa384 / prot400 with BOLTZ_MLX_MSA_OPM=1 BOLTZ_MLX_MSA_PWA=1) pass; then set True.
MSA_DEFAULT_ON = True  # validated 2026-10-28 18:35 (77 GPU tests; L1 5 inputs; msa384 MSALayer 58.0 -> 46.0 ms)
MSA_OPM_CFG = dict(ti=4, tj=4, si=1, sj=2, bk=8, swz=0, at=1)   # A/B best (module 17.7 ms at msa384)
# Experimental (off unless BOLTZ_MLX_MSA_OPM_NAX=1): proj_o inside the OPM kernel on NAX (kernels_msa.opm_outer_proj),
# so the fp16 z / num tensor never reaches device memory. Enable only after tools/prof/msa_kernels.py shows a gain.
MSA_OPM_NAX = os.environ.get("BOLTZ_MLX_MSA_OPM_NAX", "1") == "1"  # A/B: 17.0 vs 17.7 ms, lower error, no z tensor
MSA_OPM_NAX_CFG = dict(ti=4, tj=4, si=1, sj=2, bk=8, at=0, kc=256)   # A/B best core 16.98 ms
MSA_PWA_CFG = dict(bm=64, ts=2, wm=2, wn=2, bk=16)         # A/B best (module 5.54 ms at msa384)
# Per-device tuned values (boltz_mlx/msa_cfg.json: "opm" / "opm_nax" / "pwa" per chip, written by
# `tools/prof/msa_kernels.py suggest --write`) replace the defaults above; env BOLTZ_MLX_MSA_CFG_FILE=<json> for A/B runs.
MSA_CFG_FILE = os.environ.get("BOLTZ_MLX_MSA_CFG_FILE", os.path.join(os.path.dirname(__file__), "msa_cfg.json"))


def _load_msa_cfg() -> None:
    e = device_cfg(MSA_CFG_FILE, "msa", "tools/prof/msa_kernels.py && tools/prof/msa_kernels.py suggest --write")
    for key, table in (("opm", MSA_OPM_CFG), ("opm_nax", MSA_OPM_NAX_CFG), ("pwa", MSA_PWA_CFG)):
        if isinstance(e.get(key), dict):
            table.clear()
            table.update(e[key])


_load_msa_cfg()


def msa_kernel(kind: str) -> bool:
    """Is MSA-module kernel `kind` (OPM | PWA) on? Never on the CPU device (also not under mixed.cpu_f16, which
    patches f16() on for CPU traces: the CPU then takes the MLX f16 path, which builds the same w16 weights)."""
    v = MSA_FLAGS.get(kind)
    on = (v == "1" and _use_kernels()) if v is not None else (MSA_DEFAULT_ON and f16() and _use_kernels())
    if on and _CFG_NOTES["msa"]:
        _cfg_note("msa")
    return on
