"""GEMM rates at Boltz-2's real shapes: fp32 / TF32 / fp16 / bf16, plus unfused fp32->fp16 cast cost.

Usage: python tools/prof/gemm_rates.py            (spawns one child per precision setting; writes logs/gemm_rates.jsonl)
TF32 is a process-level MLX setting (MLX_ENABLE_TF32), so fp32 and TF32 run in separate processes.
Timing: pipelined (n calls, one eval), min over 3 rounds; TFLOP/s = 2*M*K*N / t.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # this checkout's boltz_mlx first: the editable install maps boltz_mlx to the main checkout


def shapes():
    out = []
    for n in (117, 400):
        r = n * n
        out += [
            (f"N{n} tri-mult proj W·xT (512x128 @ 128x{r})", 512, 128, r),
            (f"N{n} tri-mult g_out / p_out ({r}x128 @ 128x128)", r, 128, 128),
            (f"N{n} tri-mult contraction (bmm 128 x {n}x{n}x{n})", ("bmm", 128, n), None, None),
            (f"N{n} tri-att proj ({r}x128 @ 128x516)", r, 128, 516),
            (f"N{n} transition up ({r}x128 @ 128x1024)", r, 128, 1024),
            (f"N{n} transition down ({r}x512 @ 512x128)", r, 512, 128),
            (f"N{n} pair-bias attn qkvg (s: {n}x384 @ 384x1536)", n, 384, 1536),
        ]
    out += [
        ("diff qkvg (117x768 @ 768x3072)", 117, 768, 3072),
        ("diff ff up (117x768 @ 768x4608)", 117, 768, 4608),
        ("diff ff down (117x1536 @ 1536x768)", 117, 1536, 768),
        ("diff out proj (117x768 @ 768x768)", 117, 768, 768),
        ("diff cond (117x768 @ 768x110592)", 117, 768, 110592),
        ("diff qkvg N400 (400x768 @ 768x3072)", 400, 768, 3072),
    ]
    return out


def child(setting: str):
    import boltz_mlx  # noqa: F401
    import mlx.core as mx

    dt = {"fp32": mx.float32, "tf32": mx.float32, "fp16": mx.float16, "bf16": mx.bfloat16}[setting]

    def bench(f, reps):
        mx.eval(f())
        best = 1e9
        for _ in range(3):
            t = time.perf_counter()
            mx.eval([f() for _ in range(reps)])
            best = min(best, (time.perf_counter() - t) / reps)
        return best

    rows = []
    for name, M, K, N in shapes():
        if isinstance(M, tuple):  # batched contraction
            _, B, n = M
            a = mx.random.normal((B, n, n)).astype(dt)
            b = mx.random.normal((B, n, n)).astype(dt)
            mx.eval(a, b)
            flops = 2 * B * n ** 3
            reps = max(2, int(2e10 / flops))
            t = bench(lambda: a @ b.transpose(0, 2, 1), reps)
            cast_src, cast_shape = a, (B, n, n)
        else:
            a = mx.random.normal((M, K)).astype(dt)
            b = mx.random.normal((K, N)).astype(dt)
            mx.eval(a, b)
            flops = 2 * M * K * N
            reps = max(2, int(2e10 / flops))
            t = bench(lambda: a @ b, reps)
            cast_shape = (M, K)
        row = {"setting": setting, "shape": name, "ms": round(t * 1e3, 4), "tflops": round(flops / t / 1e12, 2)}
        if setting == "fp32":  # cost of an unfused fp32->fp16 cast of the A operand (and of the output back)
            x = mx.random.normal(cast_shape)
            mx.eval(x)
            row["cast_A_to_fp16_ms"] = round(bench(lambda: x.astype(mx.float16), 20) * 1e3, 4)
        rows.append(row)
        print(json.dumps(row), flush=True)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(sys.argv[2])
        return
    git = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"]).decode().strip()
    out = ROOT / "logs" / "gemm_rates.jsonl"
    for setting in ("fp32", "tf32", "fp16", "bf16"):
        env = {k: v for k, v in os.environ.items() if not k.startswith("BOLTZ_MLX_") and k != "MLX_ENABLE_TF32"}
        if setting == "tf32":
            env["BOLTZ_MLX_TF32"] = "1"
        res = subprocess.run([sys.executable, __file__, "--child", setting], env=env, capture_output=True, text=True, check=True)
        with open(out, "a") as f:
            for line in res.stdout.splitlines():
                if line.startswith("{"):
                    r = json.loads(line)
                    r.update(git=git, date=time.strftime("%Y-%m-%d %H:%M"))
                    f.write(json.dumps(r) + "\n")
        print(res.stdout)


if __name__ == "__main__":
    main()
