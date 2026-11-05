"""End-to-end (trunk + diffusion) parity on prot117 with replayed noise. Slow (~10 s GPU, longer on CPU)."""

import os
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif("not config.getoption('--e2e')")
def test_e2e_prot117():
    subprocess.run([sys.executable, str(ROOT / "tools/e2e_parity.py"), "prot117"], check=True)
    res = json.loads((Path(os.environ.get("BOLTZ_MLX_DATA", Path(__file__).resolve().parents[1] / "data")).expanduser() / "prot117" / "mlx" / "e2e.json").read_text())
    assert res["coords_rmsd"] < 0.1
    assert res["s"]["max_rel"] < 1e-4 and res["z"]["max_rel"] < 1e-4
