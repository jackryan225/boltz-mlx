"""R8-coldN: cold Metal-compile cost of new token counts N, kernels.NAX_FLAGS DYNN on vs off (GPU, ~2 min).

Per arm a fresh f16 process (BOLTZ_MLX_NAX_DYNN=1 / 0; order on, off [, off, on with --reps 2]) that counts every
custom-kernel instantiation (unique (kernel name, template args) = one Metal library compile) by wrapping
mx.fast.metal_kernel, and times
  model      3 cached cases of different N in one process (default prot117 N=117, ex_multimer N=228, prot400 N=400):
             f16 predict (1 recycle, --steps diffusion steps, 1 sample, confidence) twice per case; cold - warm = the
             first-call overhead at that N (case 1 also pays every N-independent compile and the lazy weight casts)
  synthetic  3 random never-seen N (seeded per process, recorded): the first 2 pairformer layers of the loaded trunk
             (every pair-stack kernel variant: layer 0 carries the next LN, layer 1 is the last) on random s / z +
             the diffusion-transformer NAX attention, first call vs second call. Never-seen N matters because macOS
             caches compiled Metal libraries by source hash: a familiar N can hit that cache in the static arm.
Rows -> logs/cold_compile.jsonl (exp cold_compile / cold_compile_summary).

  python tools/prof/cold_compile.py [--reps 2] [--steps 10] [--cases prot117,ex_multimer,prot400]
"""

from __future__ import annotations

import json
import os
import random
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # this checkout's boltz_mlx first: the editable install maps boltz_mlx to the main checkout
OUT = ROOT / "logs" / "cold_compile.jsonl"
FLAG = "BOLTZ_MLX_NAX_DYNN"


def _arg(name: str, default: str) -> str:
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def emit(row: dict):
    git = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    row = {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "git": git, **row}
    print(json.dumps(row), flush=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(row) + "\n")


# --------------------------------------------------------------------------------------------------------- child
def child():
    import mlx.core as mx

    seen, log = set(), []
    orig = mx.fast.metal_kernel

    def patched(*a, **kw):
        k = orig(*a, **kw)
        name = kw.get("name", a[0] if a else "?")

        def call(*ca, **ckw):
            key = (name, tuple((str(t[0]), str(t[1])) for t in (ckw.get("template") or [])))
            if key not in seen:
                seen.add(key)
                log.append(key)
            return k(*ca, **ckw)

        return call

    mx.fast.metal_kernel = patched
    sys.path.insert(0, str(ROOT / "tools"))
    import feature_cache
    from boltz_mlx import kernels as K
    from boltz_mlx import kernels_nax2 as KN2
    from boltz_mlx.model import PredictArgs, load_model
    from boltz_mlx.sampler import record_noise

    arm = os.environ.get(FLAG, "")
    steps = int(_arg("--steps", "10"))
    cases = _arg("--cases", "prot117,ex_multimer,prot400").split(",")
    seed = int(_arg("--seed", "0"))
    out = {"f16": K.f16(), "dynn": K.nax("DYNN"), "model": [], "synthetic": []}

    def delta(mark):
        new = log[mark:]
        return len(new), dict(Counter(n for n, _ in new))

    t0 = time.perf_counter()
    m = load_model("conf")
    out["load_s"] = round(time.perf_counter() - t0, 3)
    for case in cases:
        arr, _ = feature_cache.load(case, 0, build_if_missing=False)
        feats = {k: mx.array(v) for k, v in arr.items()}
        n_tok = int(arr["token_pad_mask"].sum())
        rec = {"case": case, "N": n_tok}
        for r in ("cold", "warm"):
            mark = len(log)
            t = time.perf_counter()
            o = m(feats, PredictArgs(1, steps, 1), record_noise(0, case, "structure", 1), {})
            mx.eval(o["sample_atom_coords"])
            rec[f"{r}_s"] = round(time.perf_counter() - t, 3)
            rec[f"{r}_new"], names = delta(mark)
            if r == "cold":
                rec["new_by_kernel"] = names
        rec["overhead_s"] = round(rec["cold_s"] - rec["warm_s"], 3)
        out["model"].append(rec)
        del feats, o
        mx.clear_cache()

    # synthetic never-seen N: first 2 pairformer layers + diffusion NAX attention
    pm = m.pairformer_module
    full = pm.layers
    pm.layers = full[:2]
    cs, cz = m.s_recycle.weight.shape[0], pm.token_z
    rng = random.Random(seed)
    used = {r["N"] for r in out["model"]}
    ns = []
    while len(ns) < 3:
        n = rng.randrange(150, 451)
        if n not in used and n not in ns:
            ns.append(n)
    for n in ns:
        key = mx.random.key(n)
        s = mx.random.normal((1, n, cs), key=key)
        z = mx.random.normal((1, n, n, cz), key=key)
        mask, pmask = mx.ones((1, n)), mx.ones((1, n, n))
        h = (0.1 * mx.random.normal((n, 4 * 768), key=key)).astype(mx.float16)
        bias = mx.zeros((1, 16, n, n))
        rec = {"N": n}
        for r in ("cold", "warm"):
            mark = len(log)
            t = time.perf_counter()
            so, zo = pm(s, z, mask, pmask)
            ao = KN2.attention_nax(h, bias, B=1, N=n, heads=16, c=48, qbias=mx.zeros((768,)))
            mx.eval(so, zo, ao)
            rec[f"{r}_s"] = round(time.perf_counter() - t, 3)
            rec[f"{r}_new"], names = delta(mark)
            if r == "cold":
                rec["new_by_kernel"] = names
        rec["overhead_s"] = round(rec["cold_s"] - rec["warm_s"], 3)
        out["synthetic"].append(rec)
    pm.layers = full
    out["instantiations_total"] = len(log)
    out["peak_gb"] = round(mx.get_peak_memory() / 1e9, 2)
    print("COLDJSON " + json.dumps(out), flush=True)


# -------------------------------------------------------------------------------------------------------- parent
def main():
    reps = int(_arg("--reps", "1"))
    order = ("1", "0", "0", "1")[: 2 * reps] if reps <= 2 else ("1", "0") * reps
    passthru = [x for a in ("--steps", "--cases") if a in sys.argv for x in (a, _arg(a, ""))]
    res = {"1": [], "0": []}
    for k, v in enumerate(order):
        seed = random.SystemRandom().randrange(1 << 30)
        env = {**os.environ, "BOLTZ_MLX_PRECISION": "f16", FLAG: v, "PYTHONPATH": str(ROOT)}
        r = subprocess.run([sys.executable, __file__, "--child", "--seed", str(seed), *passthru], cwd=ROOT, env=env,
                           capture_output=True, text=True)
        line = next((x for x in r.stdout.splitlines() if x.startswith("COLDJSON ")), None)
        row = {"exp": "cold_compile", "flag": FLAG, "value": v, "rep": k, "seed": seed, "rc": r.returncode}
        if line:
            row.update(json.loads(line[len("COLDJSON "):]))
            res[v].append(row)
        else:
            row["tail"] = (r.stdout + r.stderr)[-2000:]
        emit(row)

    def agg(rows, part, idx):
        recs = [rec for row in rows for i, rec in enumerate(row[part]) if i in idx]
        if not recs:
            return None
        return {"new_inst_mean": round(statistics.mean(r["cold_new"] for r in recs), 2),
                "overhead_s_median": round(statistics.median(r["overhead_s"] for r in recs), 3),
                "cold_s_median": round(statistics.median(r["cold_s"] for r in recs), 3),
                "warm_s_median": round(statistics.median(r["warm_s"] for r in recs), 3)}

    summ = {"exp": "cold_compile_summary", "flag": FLAG}
    for v, lab in (("1", "on"), ("0", "off")):
        rows = res[v]
        summ[lab] = {"first_case": agg(rows, "model", {0}), "later_cases": agg(rows, "model", {1, 2, 3, 4}),
                     "synthetic_new_N": agg(rows, "synthetic", {0, 1, 2}),
                     "instantiations_total": [r["instantiations_total"] for r in rows],
                     "load_s": [r["load_s"] for r in rows]}
    emit(summ)


if __name__ == "__main__":
    if "--child" in sys.argv:
        child()
    else:
        main()
