"""Shared parity-test helpers.

Tests run on the MLX CPU device by default (BOLTZ_MLX_DEVICE=gpu to override) so that
parallel test runs never contend for the GPU; run again with BOLTZ_MLX_DEVICE=gpu for the GPU kernels.
"""

import os
import re
from pathlib import Path

import boltz_mlx  # noqa: F401  (sets MLX_ENABLE_TF32=0 before mlx initializes)
import mlx.core as mx
import numpy as np
import pytest

DATA = Path(os.environ.get("BOLTZ_MLX_DATA", Path(__file__).resolve().parents[1] / "data")).expanduser()

mx.set_default_device(mx.gpu if os.environ.get("BOLTZ_MLX_DEVICE", "cpu") == "gpu" else mx.cpu)

from boltz_mlx import kernels as _K  # noqa: E402

if mx.default_device() == mx.cpu:
    # CPU sessions never run the kernel-path detection (its tensor-op check is a GPU dispatch): pinned to "nax" (the
    # M5 dev machine's path, so the existing CPU wiring tests see the MPP groups), or to the requested path
    # (BOLTZ_MLX_KERNEL_PATH / BOLTZ_MLX_NAX=0). Tests of other paths monkeypatch kernels._CAP themselves.
    _req = _K._requested()
    _K._CAP.clear()
    _K._CAP.update(_K.pinned_status(_req if _req in _K.KERNEL_PATHS else "nax", "CPU test session (pinned, no probe)"))
else:
    _K.nax_status()  # GPU session: detect once, here, outside any mx.compile trace


# Local resources that are not part of the repository. A test that stops because one of these files is missing is
# reported as SKIPPED (naming the file and how to get it), so a fresh clone runs the suite with 0 failures.
from boltz_mlx.paths import default_cache as _default_cache, weights_dir as _weights_dir  # noqa: E402

_RESOURCES = [
    (DATA, "parity data (BOLTZ_MLX_DATA, default <checkout>/data; made by tools/prep_case.py + tools/make_reference.py)"),
    (_weights_dir(), "converted weights (BOLTZ_MLX_WEIGHTS; run `boltz-mlx setup`)"),
    (_default_cache(), "the Boltz cache with mols/ (BOLTZ_MLX_CACHE, default ~/.boltz; run `boltz-mlx setup`)"),
]
_MISSING_FILE = re.compile(r"(?:No such file or directory: |Failed to open file )'?([^'\s]+)")


def missing_resource(exc: BaseException):
    """(path, label) when exc is a missing-file error for a path under one of _RESOURCES that really is absent."""
    if not isinstance(exc, (OSError, RuntimeError)):
        return None
    m = _MISSING_FILE.search(str(exc))
    if not m:
        return None
    path = Path(m.group(1)).expanduser()
    if path.exists():
        return None
    for root, label in _RESOURCES:
        try:
            path.resolve().relative_to(Path(root).expanduser().resolve())
        except ValueError:
            continue
        return path, label
    return None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Reported as skipped, with the reason, instead of failed:
    - on the portable kernel path a test that builds an MPP kernel stops with kernels.NaxUnavailableError (the guard
      worked). BOLTZ_MLX_MPP_REQUIRED=1 (the portable_check kit's MPP session): left as a failure;
    - a test (or fixture) that stops because a local resource outside the repository is missing (_RESOURCES)."""
    outcome = yield
    rep = outcome.get_result()
    if call.excinfo is None or not rep.failed:
        return
    if call.excinfo.errisinstance(_K.NaxUnavailableError) and os.environ.get("BOLTZ_MLX_MPP_REQUIRED") != "1":
        rep.outcome = "skipped"
        rep.longrepr = (str(item.fspath), item.location[1] or 0, f"Skipped: portable kernel path: {call.excinfo.value}")
        return
    miss = missing_resource(call.excinfo.value)
    if miss:
        rep.outcome = "skipped"
        rep.longrepr = (str(item.fspath), item.location[1] or 0, f"Skipped: needs {miss[1]}: {miss[0]} is missing")


def load_dump(name: str, case: str = "prot117", ref: str = "ref") -> dict:
    """Module dump: in_<i>, kw_<name>, out / out_<i>. Value 'FEATS' means pass the feats dict."""
    path = DATA / case / ref / "modules" / f"{name}.npz"
    if not path.exists():
        pytest.skip(f"missing dump {path}")
    z = np.load(path)
    return {k: z[k] for k in z.files}


def load_feats(case: str = "prot117", affinity: bool = False) -> dict:
    path = DATA / case / ("feats_affinity.npz" if affinity else "feats.npz")
    if not path.exists():
        pytest.skip(f"missing featurized input {path} (parity data: tools/prep_case.py {case})")
    z = np.load(path)
    return {k: mx.array(z[k]) for k in z.files}


def mxa(x):
    """numpy -> mx.array (float64 -> float32; int64 kept)."""
    x = np.asarray(x)
    if x.dtype == np.float64:
        x = x.astype(np.float32)
    return mx.array(x)


def pytest_addoption(parser):
    parser.addoption("--e2e", action="store_true", help="run end-to-end parity tests")
