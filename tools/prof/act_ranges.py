"""Activation ranges per module / GEMM output in the strict fp32 build -> logs/act_ranges.jsonl, ACT_RANGES.md.

Usage:
  python tools/prof/act_ranges.py run <case> [--affinity] [--steps 200]   # one case, fresh process
  python tools/prof/act_ranges.py all                                       # the PREREG-A ACT_RANGES case list
  python tools/prof/act_ranges.py report                                    # aggregate jsonl -> ACT_RANGES.md
Records, per (phase, op): max |x|, ~99.99th percentile of |x| (from a 2M-element seeded subsample), min nonzero |x|,
the number of calls, and the exact number of elements with 0 < |x| < 6.1e-5 (fp16 subnormal range) over all calls.
The percentile sample is uniform with replacement (GPU randint, keyed by a call counter, so it is deterministic) and is
taken on the first 4 calls of each op and every 10th call after that; max / min-nonzero / subnormal counts use every
element of every call.
Strict fp32; MLX paths for the transition / diffusion transformer (so hidden GEMM outputs are visible);
diffusion uncompiled (hooks cannot run inside mx.compile).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # this checkout's boltz_mlx first: the editable install maps boltz_mlx to the main checkout
DATA = Path(os.environ.get("BOLTZ_MLX_DATA", Path(__file__).resolve().parents[2] / "data")).expanduser()
OUT = Path(os.environ.get("ACT_RANGES_OUT", ROOT / "logs" / "act_ranges.jsonl"))
FP16_MAX = 65504.0
FP16_MIN_NORMAL = 6.103515625e-05

V36_10 = ["9eyb", "8oxk", "8k9n", "8t98", "9eqb", "7h67", "9rfg", "9g8v", "13xs", "9ssd"]  # PREREG-A section C
CASES = (["prot117", "prot400", "msa384", "ex_ligand", "lig_aff"] + V36_10
         + ['cdk2_17', 'tyk2_ejm_31', 'jnk1_17124-1', 'bace_cat-13a']  # first ligand by id per target (PREREG-A section C)
         + ["prot1000", "ex_pocket", "msa4096"])


def child(case: str, affinity: bool, steps: int):
    os.environ["BOLTZ_MLX_TRANSITION_IMPL"] = "mlx"
    os.environ["BOLTZ_MLX_DIFFTX_IMPL"] = "mlx"
    import boltz_mlx  # noqa: F401
    import mlx.core as mx
    import numpy as np

    from boltz_mlx import kernels as K
    from boltz_mlx.model import PredictArgs, load_model
    from boltz_mlx.sampler import record_noise

    stats: dict = {}
    counter = [0]
    SAMPLE = 2_000_000

    def hook(phase, name, x):
        a = mx.abs(x.astype(mx.float32)).reshape(-1)
        n = a.size
        key = (phase, name)
        r = stats.setdefault(key, {"maxabs": 0.0, "p9999": 0.0, "minnz": np.inf, "calls": 0, "n_subnormal": 0,
                                   "n_elems": 0, "n_sampled": 0, "shape": list(x.shape)})
        vals = [mx.max(a), mx.min(mx.where(a > 0, a, mx.array(np.inf, mx.float32))),
                mx.sum((a > 0) & (a < FP16_MIN_NORMAL))]
        take_sample = r["calls"] < 4 or r["calls"] % 10 == 0
        if take_sample:
            counter[0] += 1
            if n > SAMPLE:
                idx = mx.random.randint(0, n, (SAMPLE,), dtype=mx.uint32, key=mx.random.key(counter[0]))
                vals.append(mx.take(a, idx))
            else:
                vals.append(a)
        mx.eval(vals)
        r["maxabs"] = max(r["maxabs"], float(vals[0].item()))
        r["minnz"] = min(r["minnz"], float(vals[1].item()))
        r["n_subnormal"] += int(vals[2].item())
        r["n_elems"] += int(n)
        r["calls"] += 1
        if take_sample:
            subn = np.array(vals[3])
            r["p9999"] = max(r["p9999"], float(np.quantile(subn, 0.9999)))
            r["n_sampled"] += int(subn.size)

    K.ACT_HOOK = hook
    mx.set_cache_limit(int(4e9))
    if affinity:
        model = load_model("aff", compile_diffusion=False)
        z = np.load(DATA / case / "feats_affinity.npz")
        args = PredictArgs(5, steps, 5)
        noise = record_noise(0, case, "affinity", 5)
    else:
        model = load_model("conf", compile_diffusion=False)
        p = DATA / case / "feats.npz"
        z = np.load(p)
        args = PredictArgs(3, steps, 1)
        noise = record_noise(0, case, "structure", 1)
    feats = {k: mx.array(z[k]) for k in z.files}
    t0 = time.perf_counter()
    out = model(feats, args, noise)
    mx.eval(out["sample_atom_coords"])
    rows = []
    for (phase, name), r in stats.items():
        r.update(case=case, affinity=affinity, phase=phase, name=name, steps=steps, tokens=int(z["token_pad_mask"].sum()),
                 msa_rows=int(z["msa"].shape[1]) if "msa" in z.files else None, seconds=round(time.perf_counter() - t0, 1),
                 peak_mem_gb=round(mx.get_peak_memory() / 1e9, 2))
        r["minnz"] = None if r["minnz"] == np.inf else r["minnz"]
        rows.append(r)
    for r in rows:
        print("ROW " + json.dumps(r))


def run(case: str, affinity: bool, steps: int) -> int:
    git = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"]).decode().strip()
    res = subprocess.run([sys.executable, __file__, "_child", case, str(int(affinity)), str(steps)],
                         capture_output=True, text=True)
    n = 0
    with open(OUT, "a") as f:
        for line in res.stdout.splitlines():
            if line.startswith("ROW "):
                r = json.loads(line[4:])
                r["git"] = git
                f.write(json.dumps(r) + "\n")
                n += 1
    if res.returncode != 0:
        print(res.stderr[-2000:])
    print(f"{case}{' (affinity)' if affinity else ''}: {n} ops recorded, rc={res.returncode}", flush=True)
    return res.returncode


def report():
    rows = [json.loads(l) for l in open(OUT) if l.strip()]
    agg: dict = {}
    for r in rows:
        k = (r["phase"], r["name"])
        a = agg.setdefault(k, {"maxabs": 0, "p9999": 0, "minnz": None, "worst": "", "cases": set(), "sub": 0, "subn": 0})
        if r["maxabs"] >= a["maxabs"]:
            a["maxabs"], a["worst"] = r["maxabs"], r["case"] + (" (aff)" if r["affinity"] else "")
        a["p9999"] = max(a["p9999"], r["p9999"])
        if r["minnz"] is not None:
            a["minnz"] = r["minnz"] if a["minnz"] is None else min(a["minnz"], r["minnz"])
        a["cases"].add(r["case"])
        a["sub"] += r["n_elems"]
        a["subn"] += r["n_subnormal"]
    lines = ["# Activation ranges (round 2, strict fp32 build)", "",
             f"Generated by `python tools/prof/act_ranges.py report` from logs/act_ranges.jsonl ({len(rows)} rows, "
             f"{len({r['case'] for r in rows})} cases). Headroom = 65504 / max|x|. Verdict: **fp16-OK** if max ≤ 4096 "
             "(headroom ≥ 16×) on every measured case; **scale** if 4096 < max (needs power-of-two pre-scaling, undone in fp32, "
             "or TF32); **overflow** if max > 65504 (fp16 output impossible without scaling). Subnormal share = fraction of "
             "all elements (all calls) with 0 < |x| < 6.1e-5 (fp16 subnormal range: reduced precision, not lost). "
             "p99.99 is the max over sampled calls of a 2M-element uniform sample's 99.99th percentile.", "",
             "| Phase | Op | max \\|x\\| | worst case | p99.99 | min nonzero | headroom | subnormal share | verdict | cases |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for (phase, name), a in sorted(agg.items(), key=lambda kv: -kv[1]["maxabs"]):
        head = FP16_MAX / a["maxabs"] if a["maxabs"] else float("inf")
        verdict = "overflow" if a["maxabs"] > FP16_MAX else ("scale" if a["maxabs"] > 4096 else "fp16-OK")
        mnz = "—" if a["minnz"] is None else f"{a['minnz']:.2e}"
        lines.append(f"| {phase} | {name} | {a['maxabs']:.4g} | {a['worst']} | {a['p9999']:.4g} | {mnz} | "
                     f"{head:.3g}× | {a['subn'] / max(a['sub'], 1):.2e} | **{verdict}** | {len(a['cases'])} |")
    (ROOT / "ACT_RANGES.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:40]))


def main():
    cmd = sys.argv[1]
    if cmd == "_child":
        child(sys.argv[2], sys.argv[3] == "1", int(sys.argv[4]))
    elif cmd == "run":
        case = sys.argv[2]
        run(case, "--affinity" in sys.argv, int(sys.argv[sys.argv.index("--steps") + 1]) if "--steps" in sys.argv else 200)
    elif cmd == "all":
        for c in CASES:
            if not (DATA / c / "feats.npz").exists():
                print(f"{c}: no feats.npz (featurize first) — skipped", flush=True)
                continue
            run(c, False, 200)
        for c in ["lig_aff"]:
            run(c, True, 200)
    elif cmd == "report":
        report()


if __name__ == "__main__":
    main()
