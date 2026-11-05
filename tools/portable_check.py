"""Portability test kit: one command on any Apple-silicon Mac (macOS 26+) -> one results JSON + a short markdown summary.

  python tools/portable_check.py files                 # the files to copy to another Mac (+ sizes) and rsync lines
  python tools/portable_check.py run [--out DIR] [--runs 3] [--skip-install-check] [--skip-tests] [--skip-mpp-tests]
                                     [--paths tensorops,portable] [--no-portable-ref] [--quick]
  tools/portable_check.sh [DIR]                         # the same `run`, from a shell 

Shipped kernel paths (owner decision 2026-10-28 22:30; boltz_mlx/kernels.py "Hardware capability"): M5 = nax,
M1-M4 = tensorops (the MPP kernels on the shader cores), both gated by the on-device tensor-op check; portable =
internal fallback only. The kit's job on an M1: run the layout / correctness checks, then time and accuracy-check this
Mac's path against strict (with one cheap portable reference row).
Steps of `run` (every model run in a fresh child process with its kernel path set before MLX initialises; nothing
needs the network or the 67 GB caches; data from $BOLTZ_MLX_DATA, default <checkout>/data; weights from
$BOLTZ_MLX_WEIGHTS / the checkout's weights/ / the dev location, as boltz_mlx/paths.py):
  1 info       chip, GPU architecture, macOS, MLX, the tensor-op check (layout + correctness; cached per device + OS)
               and the kernel path this Mac takes.
  2 install    tools/install_check.sh into OUT/install (skipped with a reason if uv or the upstream mols/ cache is
               missing; never fatal).
  3 tests      GPU pytest: the MPP per-kernel tests vs fp32 references on this Mac's path (+ the layout diagnostics)
               BEFORE that path is timed (a failure drops it from timing); then the simdgroup kernels strict uses.
  4 predict    strict + this Mac's f16 path x 2 runs (+ portable x 1): the fingerprint cases (tools/fingerprint.py
               run_once: prot117 structure pass 3 recycles x 200 steps; lig_aff affinity pass 5 x 200 x 5): hashes
               (determinism), fingerprint, per-case seconds, Metal kernels built (and how many used MPP), arrays.
  5 bench      prot117 @50 (main + portable reference), strict prot117 @200, prot400 @50 if >= 16 GB, lig_aff affinity
               pass (5 recycles, 200 steps, 5 samples); median of runs 2..N, peak memory.
  6 accuracy   f16 paths vs strict (and main vs portable): prot117 coordinates RMSD after superposition, pLDDT / pTM
               deltas; lig_aff per-sample RMSD, affinity value / probability / ipTM deltas. A smoke check of end-to-end
               agreement, not the L1 module gates (their captures are not portable).
  7 decide     is this Mac's shipped path confirmed (check + tests + accuracy + determinism), or is the fallback needed?
Outputs: OUT/results.json, OUT/summary.md, OUT/arrays/<label>/*.npz, OUT/logs/*.txt. Copy OUT back.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # this checkout's boltz_mlx first: the editable install maps boltz_mlx to the main checkout
DATA = Path(os.environ.get("BOLTZ_MLX_DATA", Path(__file__).resolve().parents[1] / "data")).expanduser()
DATA_FILES = [  # (path relative to DATA, used by)
    ("cache/features/prot117/s0.npz", "predict: prot117 structure pass (the fingerprint input)"),
    ("cache/features/prot117/s0.meta.pkl", "predict: its metadata"),
    ("lig_aff/feats_affinity.npz", "predict + bench: lig_aff affinity pass"),
    ("prot117/feats.npz", "bench: prot117"),
    ("prot400/feats.npz", "bench: prot400 (Macs with >= 16 GB)"),
]
WEIGHT_FILES = ["boltz2_conf.json", "boltz2_aff.json", "boltz2_conf.safetensors", "boltz2_aff.safetensors",
                "boltz2_conf_mixed.safetensors", "boltz2_aff_mixed.safetensors"]
KEEP_ENV = {"BOLTZ_MLX_DATA", "BOLTZ_MLX_WEIGHTS", "BOLTZ_MLX_CACHE", "BOLTZ_MLX_CAP_CACHE"}
PORTABLE_TESTS = ["tests/test_portable.py", "tests/test_nax_layout.py", "tests/test_triatt_mma.py",
                  "tests/test_msa_kernels.py", "tests/test_transition_fused.py", "tests/test_difftx_fused.py"]
MPP_TESTS = ["tests/test_nax_layout.py", "tests/test_triatt_nax.py", "tests/test_nax2_kernels.py",
             "tests/test_nax3_kernels.py", "tests/test_nax3_pair.py", "tests/test_atom_kernels.py",
             "tests/test_msa_kernels.py", "tests/test_dynn_gpu.py"]
MIN_GB_PROT400 = 16.0


def weights_dir() -> Path:
    sys.path.insert(0, str(ROOT))
    from boltz_mlx.paths import weights_dir as wd

    return wd()


def _size(p: Path) -> int:
    return p.stat().st_size if p.exists() else -1


def _gb(n: int) -> str:
    return "MISSING" if n < 0 else (f"{n / 1e9:.2f} GB" if n >= 1e8 else f"{n / 1e6:.1f} MB")


def files_cmd() -> int:
    wd = weights_dir()
    rows = [(DATA / f, why) for f, why in DATA_FILES] + [(wd / f, "weights") for f in WEIGHT_FILES]
    total = 0
    print(f"data dir {DATA} (env BOLTZ_MLX_DATA), weights dir {wd} (env BOLTZ_MLX_WEIGHTS)")
    for p, why in rows:
        n = _size(p)
        total += max(n, 0)
        print(f"  {_gb(n):>10}  {p}  - {why}")
    print(f"  {_gb(total):>10}  total (+ the repository checkout: `git archive HEAD` ~50 MB)")
    print("\nrsync lines (run on this Mac; OTHER_MAC = ssh name or a mounted volume path):")
    print("  rsync -aR --progress " + " ".join(f"{DATA}/./{f}" for f, _ in DATA_FILES) + " OTHER_MAC:boltz-mlx/data/")
    print("  rsync -a --progress " + " ".join(str(wd / f) for f in WEIGHT_FILES) + " OTHER_MAC:boltz-mlx/weights/")
    return 0


# --------------------------------------------------------------------------------------------------------- children
def child_env(precision: str | None, path: str | None) -> dict:
    env = {k: v for k, v in os.environ.items()
           if not (k.startswith("BOLTZ_MLX_") and k not in KEEP_ENV) and k != "MLX_ENABLE_TF32"}
    env["PYTHONPATH"] = str(ROOT)
    env["BOLTZ_MLX_DATA"] = str(DATA)
    if precision:
        env["BOLTZ_MLX_PRECISION"] = precision
    if path:
        env["BOLTZ_MLX_KERNEL_PATH"] = path
    return env


def run_child(task: dict, env: dict, log: Path, timeout: int) -> dict:
    t0 = time.time()
    cmd = [sys.executable, str(Path(__file__).resolve()), "_child", json.dumps(task)]
    try:
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
        out, err, rc = r.stdout, r.stderr, r.returncode
    except subprocess.TimeoutExpired as e:
        out, err, rc = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or ""), "TIMEOUT", -9
    with open(log, "a") as f:
        f.write(f"\n=== {time.strftime('%H:%M:%S')} {task}\n{out}\n--- stderr\n{err[-20000:]}\n")
    res = None
    for line in reversed(out.strip().splitlines()):
        if line.startswith("{"):
            try:
                res = json.loads(line)
                break
            except json.JSONDecodeError:
                pass
    if res is None:
        res = {"error": f"rc={rc}: " + (err.strip().splitlines()[-1][:300] if err.strip() else "no output")}
    res["wall_s"] = round(time.time() - t0, 1)
    return res


def child(task: dict) -> None:
    import boltz_mlx  # noqa: F401  (TF32 env before MLX init)
    import mlx.core as mx
    import numpy as np

    from boltz_mlx import kernels as K

    mx.set_cache_limit(int(1e9))
    kind = task["kind"]
    if kind == "info":
        from boltz_mlx import capability

        s = capability.summary()
        s["lines"] = capability.lines(s)
        print(json.dumps(s))
        return
    st = K.nax_status()
    if task.get("path") and st["path"] != task["path"]:  # requested path refused (e.g. the tensor-op check failed)
        print(json.dumps({"error": f"kernel path {task['path']} not active: {st['reason']}", "device": K.device_record()}))
        return
    if kind == "predict":
        sys.path.insert(0, str(ROOT / "tools"))
        import fingerprint as FP

        K.KERNEL_LOG = []
        save = Path(task["save"]) if task.get("save") else None
        if save is not None:
            save.mkdir(parents=True, exist_ok=True)
        timings: dict = {}
        mx.reset_peak_memory()
        hashes = FP.run_once(task["mode"], save=save, timings=timings)
        built = {n: m for n, m in K.KERNEL_LOG}
        import hashlib

        print(json.dumps({"hashes": hashes, "weights": FP.WEIGHTS_USED, "timings": timings,
                          "fingerprint": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:12],
                          "mode_label": K.mode_label(task["mode"] if task["mode"] == "strict" else "f16"),
                          "kernels_built": len(built), "mpp_kernels_built": sum(built.values()),
                          "peak_mem_gb": round(mx.get_peak_memory() / 1e9, 2), "device": K.device_record()}))
        return
    if kind == "bench":
        from boltz_mlx.model import PredictArgs, load_model
        from boltz_mlx.sampler import record_noise

        aff = task["affinity"]
        t = time.perf_counter()
        model = load_model("aff" if aff else "conf")
        load_s = time.perf_counter() - t
        z = np.load(DATA / task["case"] / ("feats_affinity.npz" if aff else "feats.npz"))
        feats = {k: mx.array(z[k]) for k in z.files}
        rec, samples = (5, 5) if aff else (3, 1)
        runs = []
        for r in range(task["runs"]):
            mx.reset_peak_memory()
            ph: dict = {}
            t0 = time.perf_counter()
            out = model(feats, PredictArgs(rec, task["steps"], samples),
                        record_noise(r, task["case"], "affinity" if aff else "structure", samples), ph)
            mx.eval({k: v for k, v in out.items() if isinstance(v, mx.array)})
            runs.append({"predict_s": round(time.perf_counter() - t0, 3), **{k: round(v, 3) for k, v in ph.items()},
                         "peak_mem_gb": round(mx.get_peak_memory() / 1e9, 2)})
            del out
            mx.clear_cache()
        warm = [x["predict_s"] for x in runs[1:]] or [runs[0]["predict_s"]]
        print(json.dumps({"median_runs2N_s": round(statistics.median(warm), 3), "min_s": min(x["predict_s"] for x in runs),
                          "runs": runs, "load_s": round(load_s, 2), "weights": getattr(model, "weights_kind", "fp32"),
                          "tokens": int(z["token_pad_mask"].sum()), "mode_label": K.mode_label(task["mode"]),
                          "device": K.device_record()}))
        return
    raise SystemExit(f"unknown child task {kind}")


# --------------------------------------------------------------------------------------------------------- accuracy
def _kabsch_rmsd(a, b, w):
    import numpy as np

    w = w.astype(np.float64)
    a, b = a.astype(np.float64), b.astype(np.float64)
    ca, cb = (a * w[:, None]).sum(0) / w.sum(), (b * w[:, None]).sum(0) / w.sum()
    a0, b0 = a - ca, b - cb
    h = (a0 * w[:, None]).T @ b0
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    diff = (a0 @ r.T) - b0
    return float(np.sqrt(((diff ** 2).sum(1) * w).sum() / w.sum()))


def compare(a_dir: Path, b_dir: Path) -> dict:
    """Outputs of run a vs run b (b = reference): RMSD after superposition and scalar / per-token deltas."""
    import numpy as np

    res = {}
    for case in ("prot117", "lig_aff_affinity"):
        fa, fb = a_dir / f"{case}.npz", b_dir / f"{case}.npz"
        if not (fa.exists() and fb.exists()):
            continue
        A, B = np.load(fa), np.load(fb)
        r: dict = {}
        m = B["atom_pad_mask"].reshape(-1).astype(bool)
        ca, cb = A["sample_atom_coords"], B["sample_atom_coords"]
        ca, cb = ca.reshape(-1, ca.shape[-2], 3), cb.reshape(-1, cb.shape[-2], 3)
        rms = [_kabsch_rmsd(x[: m.size][m], y[: m.size][m], np.ones(int(m.sum()))) for x, y in zip(ca, cb)]
        r["coords_rmsd_A"] = round(float(np.mean(rms)), 4)
        r["coords_rmsd_max_A"] = round(float(np.max(rms)), 4)
        r["finite"] = bool(all(np.isfinite(A[k]).all() for k in A.files))
        for k in ("plddt",):
            if k in A.files and k in B.files:
                r["plddt_mean_abs_delta"] = round(float(np.abs(A[k].astype(np.float64) - B[k]).mean()), 5)
        for k in ("ptm", "iptm", "complex_plddt", "affinity_pred_value", "affinity_probability_binary"):
            if k in A.files and k in B.files:
                r[f"{k}_delta"] = round(float(np.mean(A[k].astype(np.float64)) - np.mean(B[k].astype(np.float64))), 5)
        res[case] = r
    return res


# --------------------------------------------------------------------------------------------------------- steps
def pytest_session(files: list[str], path: str, out: Path, tag: str, timeout: int) -> dict:
    files = [f for f in dict.fromkeys(files) if (ROOT / f).exists()]
    junit = out / "logs" / f"pytest_{tag}.xml"
    env = child_env(None, path)
    env["BOLTZ_MLX_DEVICE"] = "gpu"
    if path in ("nax", "tensorops"):
        env["BOLTZ_MLX_MPP_REQUIRED"] = "1"  # a refused MPP kernel is a failure here, not a skip (tests/conftest.py)
    t0 = time.time()
    try:
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={junit}", *files],
                           cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout)
        text, rc = r.stdout + r.stderr, r.returncode
    except subprocess.TimeoutExpired:
        text, rc = "TIMEOUT", -9
    (out / "logs" / f"pytest_{tag}.txt").write_text(text)
    counts = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}
    failed = []
    if junit.exists():
        import xml.etree.ElementTree as ET

        for tc in ET.parse(junit).getroot().iter("testcase"):
            name = f"{tc.get('classname')}::{tc.get('name')}"
            if tc.find("failure") is not None:
                counts["failed"] += 1
                failed.append(name)
            elif tc.find("error") is not None:
                counts["errors"] += 1
                failed.append(name)
            elif tc.find("skipped") is not None:
                counts["skipped"] += 1
            else:
                counts["passed"] += 1
    return {"files": files, "kernel_path": path, "rc": rc, **counts, "failed_tests": failed[:50],
            "ok": rc == 0 and counts["failed"] == 0 and counts["errors"] == 0, "seconds": round(time.time() - t0, 1),
            "tail": text.strip().splitlines()[-1][:300] if text.strip() else ""}


def install_step(out: Path) -> dict:
    if shutil.which("uv") is None:
        return {"status": "skipped", "reason": "uv not installed (install_check builds a fresh venv with uv)"}
    cache = Path(os.environ.get("BOLTZ_MLX_CACHE", os.environ.get("BOLTZ_CACHE", "~/.boltz"))).expanduser()
    if not (cache / "mols").is_dir():
        return {"status": "skipped", "reason": f"{cache}/mols missing (upstream molecule dictionary needed by screen)"}
    t0 = time.time()
    try:
        r = subprocess.run(["bash", str(ROOT / "tools" / "install_check.sh"), str(out / "install")], capture_output=True,
                           text=True, timeout=3600)
        text, rc = r.stdout + r.stderr, r.returncode
    except subprocess.TimeoutExpired:
        text, rc = "TIMEOUT", -9
    (out / "logs" / "install_check.txt").write_text(text)
    return {"status": "ok" if rc == 0 else "failed", "rc": rc, "seconds": round(time.time() - t0, 1),
            "tail": "\n".join(text.strip().splitlines()[-3:])}


def decide(res: dict) -> dict:
    """Owner decision 2026-10-28 22:30: M5 = nax, M1-M4 = tensorops (MPP kernels on the shader cores), each gated by the
    on-device check; portable = fallback only. The kit confirms the shipped path for this chip or says why it must fall
    back. Accuracy rule (smoke): finite outputs, deterministic, and RMSD vs strict not worse than the portable
    reference's by more than max(1.5x, +0.25 A). Speed vs the portable reference is reported, not a criterion."""
    info, b, acc = res["info"], res.get("bench", {}), res.get("accuracy", {})
    main = res.get("main_path", info.get("path"))
    lab = "f16" if main == "nax" else f"f16-{main}"
    d = {"chip": info.get("chip"), "nax_hw": info.get("nax_hw"), "tensor_op_check": info.get("tensorops"),
         "main_path": main}
    if main == "portable":
        d["recommendation"] = f"FALLBACK: portable path ({info.get('reason')}) - the shipped MPP path is not usable here"
        return d
    tests = res.get("tests", {}).get(main, {})
    d["mpp_tests_ok"] = bool(tests.get("ok")) if tests else None
    pred = res.get("predict", {}).get(lab, {})
    d["deterministic"] = pred.get("deterministic")
    speed = {}
    for case in ("prot117_f16@50",):
        tp, pp = b.get(f"{case}_{main}", {}), b.get(f"{case}_portable", {})
        if "median_runs2N_s" in tp and "median_runs2N_s" in pp:
            speed[case] = round(pp["median_runs2N_s"] / tp["median_runs2N_s"], 3)  # > 1: the MPP path is faster
    d[f"speedup_{main}_vs_portable_ref"] = speed
    a_m, a_p = acc.get(f"{lab}_vs_strict", {}), acc.get("f16-portable_vs_strict", {})
    concerns = []
    for case in ("prot117", "lig_aff_affinity"):
        rm, rp = a_m.get(case, {}), a_p.get(case, {})
        if rm and not rm.get("finite", False):
            concerns.append(f"{case}: non-finite outputs")
        x, y = rm.get("coords_rmsd_A"), rp.get("coords_rmsd_A")
        if x is not None and y is not None and x > max(1.5 * y, y + 0.25):
            concerns.append(f"{case}: RMSD vs strict {x} A vs portable reference {y} A")
    d["accuracy_concerns"] = concerns
    if d["mpp_tests_ok"] is False:
        why = "MPP per-kernel tests FAILED on this device (the check passed: extend the check, fall back meanwhile)"
    elif concerns:
        why = "accuracy concern: " + "; ".join(concerns)
    elif d["deterministic"] is False:
        why = "not deterministic run to run"
    else:
        why = ""
    slow = [k for k, v in speed.items() if v < 1.0]
    d["recommendation"] = (f"FALLBACK needed ({why})" if why else
                           f"{main} confirmed as this chip's path" + ("" if d["mpp_tests_ok"] else " (MPP tests skipped)")
                           + (f"; NOTE slower than the portable reference on {slow} - review" if slow else ""))
    return d


def summary_md(res: dict) -> str:
    i = res["info"]
    L = [f"# Portability check: {i.get('chip')} ({i.get('architecture')}), macOS {i.get('macos')} ({i.get('os_build')}), "
         f"MLX {i.get('mlx')}", "",
         f"- date {res['date']}, git {res['git']}, kit runtime {res.get('seconds', '?')} s",
         f"- memory {i.get('memory_gb')} GB; NAX hardware: {i.get('nax_hw')}; tensor-op check: {i.get('tensorops')}",
         f"- auto kernel path: **{i.get('path')}** ({i.get('reason')}); paths measured: {', '.join(res['paths'])}",
         f"- install_check: {res.get('install', {}).get('status')} {res.get('install', {}).get('reason', '')}", ""]
    L += ["## Tests (GPU)", "| session | path | passed | failed | errors | skipped | ok |", "|---|---|---|---|---|---|---|"]
    for k, t in res.get("tests", {}).items():
        L.append(f"| {k} | {t.get('kernel_path')} | {t.get('passed')} | {t.get('failed')} | {t.get('errors')} | "
                 f"{t.get('skipped')} | {t.get('ok')} |")
    L += ["", "## Fingerprints (prot117 structure 200 steps + lig_aff affinity 200 x 5; 2 runs)",
          "| label | fingerprint | deterministic | prot117 s | affinity s | kernels (MPP) | peak GB |", "|---|---|---|---|---|---|---|"]
    for lab, p in res.get("predict", {}).items():
        r0 = (p.get("runs") or [{}])[0]
        L.append(f"| {lab} | {r0.get('fingerprint', r0.get('error', '?'))} | {p.get('deterministic')} | "
                 f"{r0.get('timings', {}).get('prot117_s')} | {r0.get('timings', {}).get('lig_aff_affinity_s')} | "
                 f"{r0.get('kernels_built')} ({r0.get('mpp_kernels_built')}) | {r0.get('peak_mem_gb')} |")
    L += ["", "## Benchmarks (median of runs 2..N, seconds; model predict only)", "| config | median s | min s | peak GB |",
          "|---|---|---|---|"]
    for k, b in res.get("bench", {}).items():
        pk = max((r.get("peak_mem_gb", 0) for r in b.get("runs", [])), default=None)
        L.append(f"| {k} | {b.get('median_runs2N_s', b.get('error', b.get('skipped', '?')))} | {b.get('min_s', '')} | {pk or ''} |")
    L += ["", "## Accuracy smoke (vs strict, same inputs and noise)", "| comparison | case | RMSD A (max) | pLDDT abs | "
          "affinity value d | prob d | finite |", "|---|---|---|---|---|---|---|"]
    for k, a in res.get("accuracy", {}).items():
        for case, r in a.items():
            L.append(f"| {k} | {case} | {r.get('coords_rmsd_A')} ({r.get('coords_rmsd_max_A')}) | "
                     f"{r.get('plddt_mean_abs_delta', '')} | {r.get('affinity_pred_value_delta', '')} | "
                     f"{r.get('affinity_probability_binary_delta', '')} | {r.get('finite')} |")
    d = res.get("decision", {})
    L += ["", "## Decision", f"- recommendation: **{d.get('recommendation')}**",
          f"- details: {json.dumps({k: v for k, v in d.items() if k != 'recommendation'})}", "",
          "Copy this whole folder back (results.json has every number; arrays/ the outputs; logs/ the raw output)."]
    return "\n".join(L) + "\n"


def run_cmd(a) -> int:
    t_start = time.time()
    info_env = child_env(None, None)
    for k in ("BOLTZ_MLX_KERNEL_PATH", "BOLTZ_MLX_NAX"):  # the caller's choice of path applies to detection too
        if k in os.environ:
            info_env[k] = os.environ[k]
    git = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    tmp_log = Path(os.environ.get("TMPDIR", "/tmp")) / f"portable_check_{os.getpid()}.txt"
    info = run_child({"kind": "info"}, info_env, tmp_log, 600)
    if "error" in info:
        print(f"capability detection failed: {info['error']}", file=sys.stderr)
        return 1
    slug = info["chip"].replace("Apple ", "").replace(" ", "")
    out = Path(a.out or ROOT / f"portable_check_{slug}_{time.strftime('%Y%m%d-%H%M')}").expanduser().resolve()
    (out / "logs").mkdir(parents=True, exist_ok=True)
    shutil.move(str(tmp_log), out / "logs" / "children.txt")
    log = out / "logs" / "children.txt"
    print("\n".join(info["lines"]), flush=True)
    res = {"kit": "tools/portable_check.py", "date": time.strftime("%Y-%m-%d %H:%M:%S"), "git": git, "info": info,
           "argv": sys.argv[1:], "data": str(DATA), "weights": str(weights_dir()), "python": platform.python_version()}

    # f16 kernel paths: the shipped one this Mac takes (nax on M5, tensorops on M1-M4; portable only as the fallback
    # when the check fails), plus the portable path as a cheap reference row (1 predict run, prot117 @50 bench)
    main = info["path"]
    ref = [] if (a.no_portable_ref or main == "portable") else ["portable"]
    paths = a.paths.split(",") if a.paths else [main] + ref
    res["main_path"], res["paths"] = paths[0], paths

    def save():
        res["seconds"] = round(time.time() - t_start, 1)
        (out / "results.json").write_text(json.dumps(res, indent=1))
        (out / "summary.md").write_text(summary_md(res))

    # 2 install
    if a.skip_install_check:
        res["install"] = {"status": "skipped", "reason": "--skip-install-check"}
    else:
        print("== install_check", flush=True)
        res["install"] = install_step(out)
    save()

    # 3 tests: first the MPP kernels on this Mac's path (layout diagnostics + every kernel vs fp32 references), BEFORE
    # anything on that path is timed; then the simdgroup kernels strict uses (+ the fallback path's)
    res["tests"] = {}
    if not a.skip_tests:
        mp = res["main_path"]
        if mp in ("tensorops", "nax") and not a.skip_mpp_tests:
            print(f"== tests: MPP kernels on the {mp} path (layout + correctness vs fp32 references)", flush=True)
            res["tests"][mp] = pytest_session(MPP_TESTS, mp, out, mp, a.test_timeout)
            if not res["tests"][mp]["ok"]:
                print(f"MPP tests FAILED on the {mp} path: it is not timed (results.json tests.{mp})", flush=True)
                paths.remove(mp)
                res["mpp_dropped"] = f"MPP per-kernel tests failed on the {mp} path"
        print("== tests: simdgroup kernels (strict + fallback path)", flush=True)
        res["tests"]["portable"] = pytest_session(PORTABLE_TESTS, "portable", out, "portable", a.test_timeout)
    save()

    # 4 predict (fingerprints x 2, arrays of run 0)
    res["predict"] = {}
    labels = [("strict", "strict", None)] + [(("f16" if p == "nax" else f"f16-{p}"), "f16", p) for p in paths]
    for lab, mode, path in labels:
        n_runs = 1 if (path == "portable" and path != res["main_path"]) else a.fp_runs  # reference row: 1 run
        print(f"== predict {lab} x {n_runs}", flush=True)
        runs = []
        for r in range(n_runs):
            task = {"kind": "predict", "mode": mode, "path": path,
                    "save": str(out / "arrays" / lab) if r == 0 else None}
            runs.append(run_child(task, child_env("f16" if mode == "f16" else None, path), log, a.timeout))
            print(f"   run {r}: {runs[-1].get('fingerprint', runs[-1].get('error'))} {runs[-1].get('timings', '')}",
                  flush=True)
        ok = [x for x in runs if "hashes" in x]
        res["predict"][lab] = {"runs": runs, "deterministic": None if n_runs < 2 else (
            len(ok) == len(runs) and len({json.dumps(x["hashes"]) for x in ok}) == 1)}
        save()

    # 5 bench
    res["bench"] = {}
    mem = float(info.get("memory_gb") or 0)
    plan = [("prot117_strict@200", "prot117", False, 200, "strict", None)]
    for p in paths:
        plan.append((f"prot117_f16@50_{p}", "prot117", False, 50, "f16", p))
        if p == "portable" and p != res["main_path"]:
            continue  # reference row only
        plan.append((f"lig_aff_affinity_f16_{p}", "lig_aff", True, 200, "f16", p))
        plan.append((f"prot400_f16@50_{p}", "prot400", False, 50, "f16", p))
    for name, case, aff, steps, mode, path in plan:
        if a.quick and case == "prot400":
            res["bench"][name] = {"skipped": "--quick"}
            continue
        if case == "prot400" and mem < MIN_GB_PROT400:
            res["bench"][name] = {"skipped": f"{mem} GB < {MIN_GB_PROT400} GB"}
            continue
        if not (DATA / case / ("feats_affinity.npz" if aff else "feats.npz")).exists():
            res["bench"][name] = {"skipped": f"missing {case} features"}
            continue
        print(f"== bench {name}", flush=True)
        task = {"kind": "bench", "case": case, "affinity": aff, "steps": steps, "runs": a.runs, "mode": mode, "path": path}
        res["bench"][name] = run_child(task, child_env("f16" if mode == "f16" else None, path), log, a.timeout)
        print(f"   {res['bench'][name].get('median_runs2N_s', res['bench'][name].get('error'))} s", flush=True)
        save()

    # 6 accuracy
    res["accuracy"] = {}
    arr = out / "arrays"
    for lab, _, _ in labels[1:]:
        res["accuracy"][f"{lab}_vs_strict"] = compare(arr / lab, arr / "strict")
    labs = [x[0] for x in labels]
    for main_lab in ("f16-tensorops", "f16"):
        if main_lab in labs and "f16-portable" in labs:
            res["accuracy"][f"{main_lab}_vs_f16-portable"] = compare(arr / main_lab, arr / "f16-portable")

    # 7 decide
    res["decision"] = decide(res)
    save()
    print(f"\nwrote {out / 'results.json'} and {out / 'summary.md'}\n")
    print(summary_md(res))
    return 0


def main() -> int:
    if len(sys.argv) > 2 and sys.argv[1] == "_child":
        sys.path.insert(0, str(ROOT))
        child(json.loads(sys.argv[2]))
        return 0
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["files", "run"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--runs", type=int, default=3, help="bench runs per config (median of runs 2..N)")
    ap.add_argument("--fp-runs", type=int, default=2, help="fingerprint runs per label (determinism)")
    ap.add_argument("--paths", default=None, help="f16 kernel paths to measure, first = main (default: this Mac's path "
                                                  "+ the portable reference)")
    ap.add_argument("--no-portable-ref", action="store_true", help="skip the portable reference rows")
    ap.add_argument("--skip-install-check", action="store_true")
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--skip-mpp-tests", action="store_true", help="skip the MPP per-kernel tests (then the MPP path "
                                                                   "is timed unvalidated - avoid on M1-M4)")
    ap.add_argument("--quick", action="store_true", help="no prot400 benches")
    ap.add_argument("--timeout", type=int, default=3600, help="seconds per child process")
    ap.add_argument("--test-timeout", type=int, default=3600)
    a = ap.parse_args()
    return files_cmd() if a.cmd == "files" else run_cmd(a)


if __name__ == "__main__":
    sys.exit(main())
