"""Print the detected GPU capability and which kernel path each precision mode takes on this Mac.

  python -m boltz_mlx.capability [--json]

Kernel paths (boltz_mlx/kernels.py "Hardware capability"; macOS 26 required): nax (M5-class GPU: the
MetalPerformancePrimitives kernels on the neural accelerators), tensorops (M1-M4: the same kernels, Metal 4 tensor ops
on the shader cores), both only after the on-device tensor-op check (a few one-simdgroup kernels, cached on disk per
device + OS); portable = internal fallback when the check fails (loud warning) or on request (BOLTZ_MLX_KERNEL_PATH).
"""

from __future__ import annotations

import json
import sys

PATH_TEXT = {
    "nax": "f16: fp16 GEMMs + the MPP kernels on the M5 neural accelerators (NAX)",
    "tensorops": "f16-tensorops: fp16 GEMMs + the MPP kernels as Metal 4 tensor ops on the shader cores (M1-M4 path)",
    "portable": "f16-portable: FALLBACK path (internal, unvalidated): fp16 MLX GEMMs + simdgroup-MMA kernels, no MPP",
}


def summary() -> dict:
    from boltz_mlx import kernels as K

    s = K.nax_status()
    s["paths"] = {
        "strict": "fp32 everywhere; simdgroup-MMA / simd-reduction kernels (same code on every Apple-silicon GPU)",
        "f16": PATH_TEXT[s["path"]],
    }
    s["f16_label"] = K.mode_label("f16")
    return s


def lines(s: dict) -> list[str]:
    return [
        f"chip {s['chip']} | GPU architecture {s['architecture'] or '?'} (generation {s['gpu_gen']}) | "
        f"memory {s['memory_gb']} GB | macOS {s['macos']} ({s['os_build']}) | MLX {s['mlx']}",
        f"NAX hardware (M5 neural accelerators): {'yes' if s['nax_hw'] else 'no'} | tensor-op check: {s['tensorops']}"
        + (" (cached)" if s.get("tensorops_cached") else ""),
        f"kernel path: {s['path']} (requested {s['requested']}) - {s['reason']}",
        f"--precision f16 / presets f16, screen run as: {s['paths']['f16']}",
        f"strict runs as: {s['paths']['strict']}",
    ]


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    s = summary()
    print(json.dumps(s) if "--json" in argv else "\n".join(lines(s)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
