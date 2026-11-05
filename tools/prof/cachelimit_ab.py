"""A/B of the MLX buffer-cache limit inside ONE process: same model, same features, same noise, alternating arms.

  python tools/prof/cachelimit_ab.py prot1600 --order ABBAABBA --steps 50 --mode f16 --mem-limit-gb 100 --warm \
         [--arm-a 4] [--arm-b policy] [--hash] [--label cl-c-1600] [--note "..."] [--out logs/cachelimit_ab.jsonl]

Arms (--arm-a / --arm-b): a number = explicit cache limit in GB (boltz_mlx.cachelimit.set_explicit); 'policy' = what the
shipped CLI does: cachelimit.install_startup(environ={}) (4 GB, policy armed) then cachelimit.adapt_feats(feats) once the
token count is known, i.e. the production code path, not a copy of its formula. Arm A = 4 is the old behaviour: install_startup
without the adapt call.
--order: one letter per timed prediction ('ABBAABBA' = 4 per arm, ABBA blocks cancel a linear drift). Before every run:
mx.clear_cache() (each run starts with an empty buffer cache, like a one-shot CLI process), then the arm's limit is applied.
--warm: one untimed warm-up prediction of the same inputs (0 recycles, 2 steps, arm A's limit) so the first timed run does not
pay one-time costs (Metal pipeline builds, first page faults) that would land on arm A.
--hash: after each timed run (outside the timed region) sha256 of every mx.array in the output (coordinates, confidence
outputs, pdistogram, s, z; dtype + shape + raw bytes). Arms must give identical digests: the limit changes allocator reuse only.
Per run: predict_s and the trunk / diffusion / confidence split (model timings), user / system CPU seconds of the process during
the run, MLX peak, cache memory held after the run (shows the limit took effect), fp16 guard counts.
The row is appended to --out (default logs/cachelimit_ab.jsonl) and printed as one CACHELIMIT_AB_JSON line.
Refuses to run on a dirty code tree (boltz_mlx/, tools/, engine/, tests/, pyproject.toml), like tools/bench_mlx.py.
"""

import argparse
import hashlib
import json
import os
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # this checkout's boltz_mlx first (the editable install maps boltz_mlx to the main checkout)
DATA = Path(os.environ.get("BOLTZ_MLX_DATA", Path(__file__).resolve().parents[2] / "data")).expanduser()
MODES = {  # same environments as tools/bench_mlx.py
    "strict": {},
    "f16": {"BOLTZ_MLX_PRECISION": "f16"},
    "f16-portable": {"BOLTZ_MLX_PRECISION": "f16", "BOLTZ_MLX_KERNEL_PATH": "portable"},
}
CODE_PATHS = ["boltz_mlx", "tools", "engine", "tests", "pyproject.toml"]


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


def digest(x) -> str:
    """sha256 of dtype + shape + raw bytes of an mx array (native dtype; float32 view when numpy cannot hold the dtype)."""
    import mlx.core as mx
    import numpy as np

    try:
        a = np.array(x)
    except (RuntimeError, TypeError, ValueError):
        a = np.array(x.astype(mx.float32))  # exact widening (bf16 -> f32)
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def digests(out: dict) -> dict:
    import mlx.core as mx

    return {k: digest(v) for k, v in sorted(out.items()) if isinstance(v, mx.array)}


def combined(d: dict) -> str:
    return hashlib.sha256("".join(f"{k}:{v};" for k, v in sorted(d.items())).encode()).hexdigest()[:16]


def spread(xs):
    med = statistics.median(xs)
    return {"n": len(xs), "median": round(med, 3), "min": round(min(xs), 3), "max": round(max(xs), 3),
            "mad": round(statistics.median([abs(x - med) for x in xs]), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--order", default="ABBAABBA")
    ap.add_argument("--arm-a", default="4")
    ap.add_argument("--arm-b", default="policy")
    ap.add_argument("--mode", default="f16", choices=list(MODES))
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--recycling", type=int, default=3)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--mem-limit-gb", type=float, default=100)
    ap.add_argument("--weights", default="auto", choices=["auto", "fp32", "mixed"])
    ap.add_argument("--warm", action="store_true")
    ap.add_argument("--hash", action="store_true")
    ap.add_argument("--seed", type=int, default=0, help="record_noise seed, the same for every run (outputs are comparable)")
    ap.add_argument("--idle-s", type=float, default=0, help="sleep before loading (the queue jobs idle in the shell instead)")
    ap.add_argument("--label", default="")
    ap.add_argument("--note", default="")
    ap.add_argument("--out", default=str(ROOT / "logs" / "cachelimit_ab.jsonl"))
    ap.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if set(a.order) - {"A", "B"} or not a.order:
        sys.exit("cachelimit_ab: --order is a string of A and B")

    if not a._child:
        dirty = sh(f"git -C {ROOT} status --porcelain -- " + " ".join(CODE_PATHS))
        if dirty:
            sys.exit(f"cachelimit_ab: refusing to run on a dirty tree:\n{dirty}")
        env = {k: v for k, v in os.environ.items() if not k.startswith("BOLTZ_MLX_")}  # a row is the listed settings and nothing else
        env.update(MODES[a.mode])
        os.execve(sys.executable, [sys.executable, __file__, *sys.argv[1:], "--_child"], env)

    if a.idle_s > 0:
        print(f"CACHELIMIT_AB idle {a.idle_s:.0f} s", flush=True)
        time.sleep(a.idle_s)

    import boltz_mlx  # noqa: F401  (TF32 env handling before MLX init)
    import mlx.core as mx
    import numpy as np

    from boltz_mlx import cachelimit
    from boltz_mlx import kernels as K
    from boltz_mlx.model import PredictArgs, load_model
    from boltz_mlx.sampler import SamplerConfig, record_noise

    git = sh(f"git -C {ROOT} rev-parse --short HEAD")
    pkg = sh(f"git -C {ROOT} rev-parse --short HEAD:boltz_mlx")
    mx.set_memory_limit(int(a.mem_limit_gb * 1e9))
    t_load = time.perf_counter()
    model = load_model("conf", weights=a.weights)
    t_load = time.perf_counter() - t_load
    z = np.load(DATA / a.case / "feats.npz")
    feats = {k: mx.array(z[k]) for k in z.files}
    n_tokens = cachelimit.n_tokens_of(feats)
    cfg = SamplerConfig(a.steps)
    pargs = PredictArgs.from_sampler(cfg, a.recycling, a.samples)
    arms = {"A": a.arm_a, "B": a.arm_b}
    total = cachelimit.total_memory_gb()
    print(f"CACHELIMIT_AB case {a.case} tokens {n_tokens} total_mem_gb {total:.1f} policy value "
          f"{cachelimit.cache_limit_gb(n_tokens, total):.2f} GB; arms A={a.arm_a} B={a.arm_b}; order {a.order}; mode {a.mode}; "
          f"{a.steps} steps; mem limit {a.mem_limit_gb} GB; kernel path {K.device_record().get('path')}", flush=True)

    def apply(arm: str) -> float:
        """Empty cache, then the arm's limit. Returns the limit in force in GB."""
        mx.clear_cache()
        spec = str(arms[arm]).strip().lower()
        if spec == "policy":
            cachelimit.install_startup(environ={})
            cachelimit.adapt_feats(feats)  # the production call: N is known, trunk not started
            return cachelimit.current_gb()
        return cachelimit.set_explicit(float(spec))

    if a.warm:
        apply("A")
        t0 = time.perf_counter()
        o = model(feats, PredictArgs(0, 2, 1), record_noise(a.seed, a.case, "structure", a.samples))
        mx.eval(o["sample_atom_coords"])
        print(f"CACHELIMIT_AB warm-up (untimed) {time.perf_counter() - t0:.1f} s", flush=True)
        del o

    runs = []
    for i, arm in enumerate(a.order):
        limit = apply(arm)
        mx.reset_peak_memory()
        r0, t0 = resource.getrusage(resource.RUSAGE_SELF), time.perf_counter()
        tm = {}
        out = model(feats, pargs, record_noise(a.seed, a.case, "structure", a.samples), tm)
        mx.eval(out["sample_atom_coords"])
        wall = time.perf_counter() - t0
        r1 = resource.getrusage(resource.RUSAGE_SELF)
        row = {"i": i, "arm": arm, "limit_gb": limit, "predict_s": round(wall, 3),
               **{k: round(v, 3) for k, v in tm.items()},
               "user_cpu_s": round(r1.ru_utime - r0.ru_utime, 2), "sys_cpu_s": round(r1.ru_stime - r0.ru_stime, 2),
               "sys_share": round((r1.ru_stime - r0.ru_stime) / wall, 3),
               "peak_mem_gb": round(mx.get_peak_memory() / 1e9, 2), "cache_after_gb": round(mx.get_cache_memory() / 1e9, 2),
               "guard_counts": out.get("guard_counts")}
        if a.hash:
            d = digests(out)
            row["digest"], row["digests"] = combined(d), {k: v[:16] for k, v in d.items()}
        runs.append(row)
        print("CACHELIMIT_AB run " + json.dumps(row), flush=True)
        del out
    mx.clear_cache()

    summary = {}
    for arm in "AB":
        xs = [r for r in runs if r["arm"] == arm]
        if xs:
            summary[arm] = {"predict_s": spread([r["predict_s"] for r in xs]),
                            "trunk_s": spread([r["trunk_s"] for r in xs]), "diffusion_s": spread([r["diffusion_s"] for r in xs]),
                            "sys_share": spread([r["sys_share"] for r in xs]), "limit_gb": sorted({r["limit_gb"] for r in xs})}
            if a.hash:
                summary[arm]["digests"] = sorted({r["digest"] for r in xs})
    if "A" in summary and "B" in summary:
        summary["B_vs_A_median_pct"] = round(100 * (summary["B"]["predict_s"]["median"] / summary["A"]["predict_s"]["median"] - 1), 2)
    res = {"command": "python tools/prof/cachelimit_ab.py " + " ".join(x for x in sys.argv[1:] if x != "--_child"),
           "date": time.strftime("%Y-%m-%d %H:%M:%S"), "git": git, "pkg_tree": pkg, "label": a.label, "note": a.note,
           "device": K.device_record(), "mlx": mx.__version__, "case": a.case, "tokens": n_tokens, "total_mem_gb": round(total, 1),
           "mode": a.mode, "steps": a.steps, "recycling": a.recycling, "samples": a.samples, "mem_limit_gb": a.mem_limit_gb,
           "order": a.order, "arms": arms, "warm": a.warm, "hash": a.hash, "load_s": round(t_load, 2),
           "env": {k: v for k, v in os.environ.items() if k.startswith("BOLTZ_MLX")}, "runs": runs, "summary": summary,
           "max_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)}
    print("CACHELIMIT_AB_JSON " + json.dumps(res), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
