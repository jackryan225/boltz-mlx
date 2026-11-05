"""Time MLX Boltz-2 model predict on featurized inputs; warm runs + peak memory.

Usage: python tools/bench_mlx.py prot400 --mode turbo --steps 50 --runs 5 [--affinity] [--samples 1] [--sampler heun11]
       [--kernel-path auto|nax|tensorops|portable] [--mem-limit-gb 18] [--cache-limit-gb G|policy] [--note "..."]   (--mode f16-portable == --mode f16
       --kernel-path portable; rows carry mode_label = the path that ran: f16 | f16-tensorops | f16-portable)
- --sampler: a boltz_mlx.sampler preset (sets steps / rho / step_scale / solver of the timed pass; an affinity preset such as
  aff25 with --affinity). The row records `sampler` and `sampler_cfg` (incl. NFE).
- Refuses to run if the code tree is dirty (boltz_mlx/, tools/, engine/, tests/, pyproject.toml).
- Each run of the model happens in this process after one mode-setting re-exec (env applied before MLX init).
- Reports median of runs 2..N (round 2 statistic) and min (round 1 statistic).
- Row fields: git hash, fingerprint (from logs/fingerprints.jsonl for this hash+mode label, else "none"), mode,
  mode_label (f16 without NAX kernels = f16-portable), device (chip, architecture, nax flag), note, env, MLX version, other GPU-capable processes, top non-loop CPU%, swap, thermal, power.
"""

import argparse
import json
import os
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # this checkout's boltz_mlx first: the editable install maps boltz_mlx to the main checkout
DATA = Path(os.environ.get("BOLTZ_MLX_DATA", Path(__file__).resolve().parents[1] / "data")).expanduser()
MODES = {
    "strict": {},
    "fast": {"BOLTZ_MLX_DIFF_DTYPE": "fp16"},
    "turbo": {"BOLTZ_MLX_DIFF_DTYPE": "fp16", "BOLTZ_MLX_TF32": "1"},
    "f16": {"BOLTZ_MLX_PRECISION": "f16"},  # shipped guard mode (residual); "f16-portable" on chips without NAX
    "f16-portable": {"BOLTZ_MLX_PRECISION": "f16", "BOLTZ_MLX_KERNEL_PATH": "portable"},  # no MPP kernels
    "f16-tensorops": {"BOLTZ_MLX_PRECISION": "f16", "BOLTZ_MLX_KERNEL_PATH": "tensorops"},  # MPP kernels w/o NAX
}
CODE_PATHS = ["boltz_mlx", "tools", "engine", "tests", "pyproject.toml"]


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


def tree_dirty() -> str:
    return sh(f"git -C {ROOT} status --porcelain -- " + " ".join(CODE_PATHS))


def fingerprint_for(pkg: str, mode: str) -> str:
    """Fingerprint recorded for this boltz_mlx/ tree hash and mode (logs/fingerprints.jsonl), else 'none'."""
    fp = "none"
    p = ROOT / "logs" / "fingerprints.jsonl"
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("pkg_tree") == pkg and r.get("mode") == mode:
                fp = r.get("fingerprint", fp)
    return fp


def _ancestors() -> set:
    """This process and its ancestors (a launcher such as tools/bench_r2/o6_sweep.py is not other GPU work)."""
    ppid = {}
    for line in sh("ps -axo pid=,ppid=").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            ppid[int(parts[0])] = int(parts[1])
    out, pid = {os.getpid(), os.getppid()}, os.getppid()
    while pid in ppid and pid > 1 and len(out) < 64:
        pid = ppid[pid]
        out.add(pid)
    return out


def machine_state() -> dict:
    skip = _ancestors()
    procs = sh("ps -axo pid=,%cpu=,command=")
    others, gpu_like = [], []
    for line in procs.splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        pid, cpu, cmd = int(parts[0]), float(parts[1]), parts[2]
        if pid in skip:
            continue
        if cpu >= 5.0:
            others.append({"pid": pid, "cpu": cpu, "cmd": cmd[:80]})
        if "python" in cmd and any(t in cmd for t in ("mlx", "boltz", "bench", "e2e_parity", "torch", "predict")):
            gpu_like.append({"pid": pid, "cmd": cmd[:100]})
    therm = sh("pmset -g therm")
    return {
        "other_cpu_ge5pct": sorted(others, key=lambda r: -r["cpu"])[:6],
        "other_gpu_capable_procs": gpu_like,
        "swap": sh("sysctl -n vm.swapusage"),
        "thermal": "no warning" if "No thermal warning" in therm else therm.replace("\n", " | ")[:200],
        "power": sh("pmset -g batt").splitlines()[0] if sh("pmset -g batt") else "",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--mode", default="strict", choices=list(MODES))
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--steps", type=int, default=None, help="sampling steps (default 200, or the --sampler preset's)")
    ap.add_argument("--sampler", default=None, help="sampler preset (boltz_mlx.sampler.SAMPLER_PRESETS) for the timed pass")
    ap.add_argument("--recycling", type=int, default=3)
    ap.add_argument("--mem-limit-gb", type=float, default=18)
    ap.add_argument("--cache-limit-gb", default=None, metavar="G|policy",
                    help="mx.set_cache_limit for the runs (default: MLX's own, which follows the memory limit); the CLI "
                         "caps the buffer cache at 4 GB (BOLTZ_MLX_CACHE_GB), so 4 gives the shipped process footprint. A number is "
                         "an explicit limit (it wins over the policy). 'policy' = what the shipped CLI does: 4 GB, then "
                         "boltz_mlx.cachelimit.adapt() once the token count is known (the row records the limit applied)")
    ap.add_argument("--affinity", action="store_true", help="affinity checkpoint on feats_affinity.npz (5 recycles, 5 samples)")
    ap.add_argument("--weights", default="auto", choices=["auto", "fp32", "mixed"],
                    help="checkpoint (load_model): auto = mixed in f16 mode when built, fp32 otherwise")
    ap.add_argument("--kernel-path", default="auto", choices=["auto", "nax", "tensorops", "portable"],
                    help="BOLTZ_MLX_KERNEL_PATH for the child (BOLTZ_MLX_* variables are stripped)")
    ap.add_argument("--cfg-chip", default=None,
                    help="BOLTZ_MLX_CFG_CHIP for the child: per-device config entry to use (e.g. 'default'; A/B of tuned "
                         "launch configs, bitwise-neutral); recorded in the row's env")
    ap.add_argument("--note", default="", help="free-text label recorded in the row")
    ap.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()

    if not a._child:
        dirty = tree_dirty()
        if dirty:
            sys.exit(f"bench_mlx: refusing to run on a dirty tree:\n{dirty}")
        env = {k: v for k, v in os.environ.items() if not k.startswith("BOLTZ_MLX_")}
        env.update(MODES[a.mode])
        if a.kernel_path != "auto":
            env["BOLTZ_MLX_KERNEL_PATH"] = a.kernel_path
        if a.cfg_chip:
            env["BOLTZ_MLX_CFG_CHIP"] = a.cfg_chip
        os.execve(sys.executable, [sys.executable, __file__, *sys.argv[1:], "--_child"], env)

    import boltz_mlx  # noqa: F401  (TF32 env handling before MLX init)
    import mlx.core as mx
    import numpy as np

    from boltz_mlx import kernels as K
    from boltz_mlx.model import PredictArgs, load_model
    from boltz_mlx.sampler import SamplerConfig, affinity_config, record_noise, resolve_sampler

    if a.sampler:
        cfg = affinity_config(a.sampler) if a.affinity else resolve_sampler(a.sampler).structure
        if cfg is None:
            sys.exit(f"bench_mlx: --sampler {a.sampler} has no structure-pass config")
        if a.steps is not None and a.steps != cfg.steps:
            sys.exit(f"bench_mlx: --sampler {a.sampler} sets {cfg.steps} steps; --steps {a.steps} conflicts")
    else:
        cfg = SamplerConfig(200 if a.steps is None else a.steps)
    a.steps = cfg.steps

    git = sh(f"git -C {ROOT} rev-parse --short HEAD")
    pkg = sh(f"git -C {ROOT} rev-parse --short HEAD:boltz_mlx")
    state_start = machine_state()
    mx.set_memory_limit(int(a.mem_limit_gb * 1e9))
    from boltz_mlx import cachelimit

    cache_policy = a.cache_limit_gb is not None and str(a.cache_limit_gb).strip().lower() == "policy"
    if cache_policy:
        cachelimit.install_startup(environ={})  # 4 GB, policy armed (the child env has no BOLTZ_MLX_CACHE_GB: tools strip it)
    elif a.cache_limit_gb is not None:
        a.cache_limit_gb = float(a.cache_limit_gb)
        cachelimit.set_explicit(a.cache_limit_gb)  # explicit limit: wins over the policy
    if a.affinity:
        a.recycling, a.samples = 5, 5
    t_load = time.perf_counter()
    model = load_model("aff" if a.affinity else "conf", weights=a.weights)
    t_load = time.perf_counter() - t_load
    load_mem_gb = round(mx.get_active_memory() / 1e9, 2)
    z = np.load(DATA / a.case / ("feats_affinity.npz" if a.affinity else "feats.npz"))
    feats = {k: mx.array(z[k]) for k in z.files}
    cachelimit.adapt_feats(feats)  # no-op unless --cache-limit-gb policy; N is known here, before any run
    runs = []
    for r in range(a.runs):
        mx.reset_peak_memory()
        t = {}
        t0 = time.perf_counter()
        noise = record_noise(r, a.case, "affinity" if a.affinity else "structure", a.samples)
        out = model(feats, PredictArgs.from_sampler(cfg, a.recycling, a.samples), noise, t)
        runs.append({"predict_s": round(time.perf_counter() - t0, 3), **{k: round(v, 3) for k, v in t.items()},
                     "peak_mem_gb": round(mx.get_peak_memory() / 1e9, 2)})
        del out
        mx.clear_cache()
    warm = [x["predict_s"] for x in runs[1:]] or [runs[0]["predict_s"]]
    phases = {k: round(statistics.median([x[k] for x in (runs[1:] or runs)]), 3)
              for k in runs[0] if k.endswith("_s") and k != "predict_s"}
    label = K.mode_label("f16" if a.mode.startswith("f16") else a.mode)
    res = {
        "command": "python tools/bench_mlx.py " + " ".join(x for x in sys.argv[1:] if x != "--_child"),
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git": git,
        "pkg_tree": pkg,
        "fingerprint": fingerprint_for(pkg, label),
        "mode": a.mode,
        "mode_label": label,  # f16 without NAX kernels -> f16-portable
        "device": K.device_record(),  # chip, architecture, nax flag (+ reason), macOS, MLX
        "note": a.note,
        "weights": getattr(model, "weights_kind", "fp32"),
        "load_s": round(t_load, 2),
        "load_active_gb": load_mem_gb,
        "case": a.case,
        "affinity": a.affinity,
        "steps": a.steps,
        "sampler": a.sampler,
        "sampler_cfg": cfg.describe(),
        "recycling": a.recycling,
        "samples": a.samples,
        "mem_limit_gb": a.mem_limit_gb,
        "cache_limit_gb": a.cache_limit_gb,
        "cache_limit_applied_gb": cachelimit.current_gb(),  # None when --cache-limit-gb was not given
        "tokens": int(z["token_pad_mask"].sum()),
        "atoms": int(z["atom_pad_mask"].sum()),
        "env": {k: v for k, v in os.environ.items() if k.startswith("BOLTZ_MLX") or k == "MLX_METAL_GPU_ARCH"},
        "mlx": mx.__version__,
        "median_runs2N_s": round(statistics.median(warm), 3),
        "min_s": round(min(x["predict_s"] for x in runs), 3),
        "median_phases_runs2N": phases,
        "runs": runs,
        "max_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2),
        "machine_start": state_start,
        "machine_end": machine_state(),
    }
    print(json.dumps(res))
    with open(ROOT / "logs" / "bench_mlx.jsonl", "a") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
