"""Per-record seeding: noise and features depend only on (global seed, record, pass, sample)."""

import os
import pickle
import shutil
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from boltz_mlx.sampler import SeededNoise, record_noise
from boltz_mlx.seeding import SeededFeatureDataset, record_seed

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("BOLTZ_MLX_DATA", Path(__file__).resolve().parents[1] / "data")).expanduser()


def test_record_seed_deterministic_and_distinct():
    a = record_seed(0, "x", "structure", 0)
    assert a == record_seed(0, "x", "structure", 0)
    others = {record_seed(1, "x", "structure", 0), record_seed(0, "y", "structure", 0),
              record_seed(0, "x", "affinity", 0), record_seed(0, "x", "structure", 1), record_seed(0, "x", "features", 0)}
    assert a not in others and len(others) == 5
    assert 0 <= a < 2**32
    with pytest.raises(ValueError):
        record_seed(0, "x", "bogus")


def _draws(noise, M, n=4, shape=(7, 3)):
    return [np.array(noise.randn((M, *shape))) for _ in range(n)]


def test_noise_bit_identical_and_independent_of_sample_count():
    d1 = _draws(record_noise(0, "rec", "structure", 1), 1)
    d3 = _draws(record_noise(0, "rec", "structure", 3), 3)
    d1b = _draws(record_noise(0, "rec", "structure", 1), 1)
    for a, b, c in zip(d1, d3, d1b):
        assert np.array_equal(a, c)            # same (seed, record, pass): bit-identical
        assert np.array_equal(a[0], b[0])      # sample 0 does not depend on how many samples run
    other = _draws(record_noise(0, "rec", "affinity", 1), 1)
    assert not np.array_equal(d1[0], other[0])
    with pytest.raises(ValueError):
        record_noise(0, "rec", "structure", 2).randn((3, 5, 3))


def test_noise_independent_of_mode():
    """Noise generation does not depend on the precision settings of the build."""
    from boltz_mlx import kernels as K

    old = K.DIFF_DTYPE
    try:
        a = _draws(record_noise(5, "r", "structure", 2), 2)
        K.set_diff_dtype("fp16")
        b = _draws(record_noise(5, "r", "structure", 2), 2)
    finally:
        K.DIFF_DTYPE = old
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


# ---------------------------------------------------------------- features
CASES = ["8j9r", "8fj8"]  # two small validation chains (processed by tools/prep_case.py)


@pytest.fixture(scope="module")
def two_record_dataset(tmp_path_factory):
    """One upstream PredictionDataset holding two records (order and subsets are test parameters)."""
    for c in CASES:
        if not (DATA / c / "processed" / "manifest.json").exists():
            pytest.skip(f"missing processed case {c}")
    sys.path.insert(0, str(ROOT / "tools"))
    from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
    from boltz.data.types import Manifest

    tmp = tmp_path_factory.mktemp("two_records")
    records = []
    for sub in ("structures", "msa"):
        (tmp / sub).mkdir()
    for c in CASES:
        m = Manifest.load(DATA / c / "processed" / "manifest.json")
        records += m.records
        for sub in ("structures", "msa"):
            for f in (DATA / c / "processed" / sub).glob("*"):
                shutil.copy(f, tmp / sub / f.name)

    def make(order):
        man = Manifest(records=[records[i] for i in order])
        dm = Boltz2InferenceDataModule(manifest=man, target_dir=tmp / "structures", msa_dir=tmp / "msa",
                                       mol_dir=Path(os.environ.get("BOLTZ_MLX_CACHE", Path.home() / ".boltz")).expanduser() / "mols", num_workers=0)
        return dm.predict_dataloader().dataset

    return make, [r.id for r in records]


def _ref_pos(ds, global_seed):
    sds = SeededFeatureDataset(ds, global_seed)
    return {sds.record_id(i): sds[i]["ref_pos"].numpy().copy() for i in range(len(sds))}


def test_features_identical_across_order_resume_and_repeat(two_record_dataset):
    make, ids = two_record_dataset
    fwd = _ref_pos(make([0, 1]), 0)
    rev = _ref_pos(make([1, 0]), 0)            # input order reversed
    resume = _ref_pos(make([1]), 0)            # resume: the first record already finished
    again = _ref_pos(make([0, 1]), 0)
    for rid in ids:
        assert np.array_equal(fwd[rid], rev[rid])
        assert np.array_equal(fwd[rid], again[rid])
    assert np.array_equal(fwd[ids[1]], resume[ids[1]])
    other = _ref_pos(make([0, 1]), 1)          # a different global seed rotates the conformers differently
    assert not np.array_equal(fwd[ids[0]], other[ids[0]])


def test_feature_cache_roundtrip_matches_seeded_featurizer(tmp_path, monkeypatch):
    """Cached features (shared by every mode) equal a fresh seeded featurization, bit for bit."""
    if not (DATA / "8j9r" / "processed" / "manifest.json").exists():
        pytest.skip("missing processed case 8j9r")
    sys.path.insert(0, str(ROOT / "tools"))
    import feature_cache
    from prep_case import load_batch_dataset

    monkeypatch.setattr(feature_cache, "CACHE_DIR", tmp_path)
    a1, _ = feature_cache.load("8j9r", 3)
    a2, _ = feature_cache.load("8j9r", 3)          # second call reads the cache
    fresh = SeededFeatureDataset(load_batch_dataset(DATA / "8j9r", False), 3)[0]
    assert set(a1) == set(a2)
    for k in a1:
        assert np.array_equal(a1[k], a2[k])
    assert np.array_equal(a1["ref_pos"][0], fresh["ref_pos"].numpy())
    with pytest.raises(ValueError):
        feature_cache.build("8j9r", 3, affinity=True)   # affinity features need a run tag


def test_cli_batch_keeps_affinity_mw():
    """--affinity_mw_correction needs feats['affinity_mw'], which the collate step leaves as a list of floats."""
    import torch

    from boltz_mlx.cli import _batch_to_mx

    out = _batch_to_mx({"x": torch.zeros(2), "affinity_mw": [347.26], "record": ["r"]})
    assert "affinity_mw" in out and abs(float(out["affinity_mw"][0]) - 347.26) < 1e-3
    assert "record" not in out
