import numpy as np

from boltz_mlx.compare import kabsch_rmsd, metrics
from conftest import load_dump


def test_metrics_identity():
    x = np.random.default_rng(0).normal(size=(4, 5)).astype(np.float32)
    m = metrics(x, x)
    assert m["max_abs"] == 0 and abs(m["cos"] - 1) < 1e-12


def test_kabsch_rotation_invariant():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 3))
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    assert kabsch_rmsd(x @ q + 3.0, x) < 1e-9


def test_dump_present():
    d = load_dump("pairformer_module.layers.0.tri_mul_out")
    assert d["in_0"].shape == d["out"].shape
