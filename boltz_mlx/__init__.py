"""Boltz-2 ported to MLX."""

import os

# M5-class GPUs run fp32 matmul/attention at TF32 precision by default (ml-explore/mlx #3860).
# Measured on this M5 Pro, 512^3 matmul max rel err: 7.7e-4 default vs 5.1e-7 with TF32 off.
# Parity mode keeps true fp32 unless the caller opts in (BOLTZ_MLX_TF32=1).
if os.environ.get("BOLTZ_MLX_TF32", "0") != "1":
    os.environ.setdefault("MLX_ENABLE_TF32", "0")

MIN_MACOS = 26  # owner decision 2026-10-28: macOS 26 (Metal 4: tensor ops on every Apple-silicon GPU) is required


def macos_version() -> str:
    """The real macOS product version ('26.6.2'), '' off macOS. sysctl kern.osproductversion first: a Python built
    against an older SDK may see the compatibility version (macOS 26 reported as '16.x', like Big Sur's '10.16')."""
    import platform
    import subprocess

    if platform.system() != "Darwin":
        return ""
    try:
        v = subprocess.run(["sysctl", "-n", "kern.osproductversion"], capture_output=True, text=True,
                           timeout=5).stdout.strip()
    except Exception:
        v = ""
    v = v or platform.mac_ver()[0]
    if v.split(".")[0] == "16":  # compatibility numbering of macOS 26
        v = "26" + v[2:]
    return v


def require_macos() -> None:
    """Refuse to run below macOS 26 (BOLTZ_MLX_ALLOW_OLD_MACOS=1: unsupported, internal debugging only)."""
    v = macos_version()
    if not v or os.environ.get("BOLTZ_MLX_ALLOW_OLD_MACOS") == "1":
        return
    try:
        major = int(v.split(".")[0])
    except ValueError:
        return
    if major < MIN_MACOS:
        raise RuntimeError(f"boltz-mlx requires macOS {MIN_MACOS} or later (Metal 4); this Mac runs macOS {v}. "
                           f"Update macOS (every Apple-silicon Mac, M1 and later, runs macOS {MIN_MACOS}).")


require_macos()
