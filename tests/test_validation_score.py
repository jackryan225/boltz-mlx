"""Unit tests for tools/validation/score.py (CA-lDDT, TM-score, RMSD, mmCIF residue mapping)."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "validation"))
import score  # noqa: E402

DATA = Path(os.environ.get("BOLTZ_MLX_DATA", Path(__file__).resolve().parents[1] / "data")).expanduser()


def chain(n=120, seed=0, step=3.8):
    """Compact-ish random-walk 'CA trace' with ~3.8 A steps (self-avoiding enough for scoring tests)."""
    rng = np.random.default_rng(seed)
    x = np.zeros((n, 3))
    d = np.array([1.0, 0, 0])
    for i in range(1, n):
        d = d + rng.normal(0, 0.9, 3)
        d /= np.linalg.norm(d)
        x[i] = x[i - 1] + step * d
    return x


def rigid(x, seed=1):
    rng = np.random.default_rng(seed)
    R = Rotation.random(random_state=seed).as_matrix()
    return x @ R.T + rng.normal(0, 30, 3)


def naive_lddt(pred, ref, radius=15.0, thr=(0.5, 1, 2, 4)):
    """Straight-from-the-definition lDDT with explicit loops (independent of the vectorized implementation)."""
    n = len(ref)
    res = []
    for i in range(n):
        if np.isnan(ref[i]).any():
            continue
        tot, cnt = 0.0, 0
        for j in range(n):
            if i == j or np.isnan(ref[j]).any():
                continue
            dr = np.linalg.norm(ref[i] - ref[j])
            if dr >= radius:
                continue
            cnt += 1
            if np.isnan(pred[i]).any() or np.isnan(pred[j]).any():
                continue  # missing in prediction -> all thresholds failed
            dp = np.linalg.norm(pred[i] - pred[j])
            tot += np.mean([abs(dp - dr) < t for t in thr])
        if cnt:
            res.append(tot / cnt)
    return float(np.mean(res))


# ------------------------------------------------------------------ lDDT


def test_lddt_identical_and_rigid_motion():
    x = chain()
    assert score.lddt_ca(x, x) == 1.0
    assert score.lddt_ca(rigid(x), x) == pytest.approx(1.0, abs=1e-12)


def test_lddt_hand_computed():
    # only the (0,1) pair is within 15 A; residue 2 has no neighbours and is excluded from the mean
    ref = np.array([[0, 0, 0], [3, 0, 0], [20, 0, 0]], float)
    for shift, expected in [(0.3, 1.0), (0.7, 0.75), (1.5, 0.5), (3.0, 0.25), (5.0, 0.0)]:
        pred = np.array([[0, 0, 0], [3 + shift, 0, 0], [20, 0, 0]], float)
        assert score.lddt_ca(pred, ref) == pytest.approx(expected), shift


def test_lddt_matches_naive_and_decreases_with_noise():
    x = chain(150)
    rng = np.random.default_rng(3)
    vals = []
    for sigma in [0.1, 0.3, 0.6, 1.0, 2.0, 4.0]:
        p = x + rng.normal(0, sigma, x.shape)
        v = score.lddt_ca(p, x)
        assert v == pytest.approx(naive_lddt(p, x), abs=1e-12)
        vals.append(v)
    assert all(a > b for a, b in zip(vals, vals[1:])), vals
    assert vals[0] > 0.9 and vals[-1] < 0.5


def test_lddt_uniform_scaling_matches_naive():
    x = chain(100)
    vals = []
    for s in (1.04, 1.08, 1.15, 1.3):
        v = score.lddt_ca(x * s, x)
        assert v == pytest.approx(naive_lddt(x * s, x), abs=1e-12)
        vals.append(v)
    assert all(a > b for a, b in zip(vals, vals[1:]))


def test_lddt_only_experimentally_modeled_residues_and_missing_pred():
    x = chain(80)
    ref = x.copy()
    ref[10:20] = np.nan  # unmodeled in experiment: ignored entirely
    assert score.lddt_ca(x, ref) == 1.0
    pred = x.copy()
    pred[40:45] = np.nan  # missing in prediction: pairs involving them fail
    v = score.lddt_ca(pred, ref)
    assert v == pytest.approx(naive_lddt(pred, ref), abs=1e-12)
    assert 0.8 < v < 1.0


# ------------------------------------------------------------------ TM-score


def test_tm_d0_values():
    assert score.tm_d0(10) == 0.5 and score.tm_d0(21) == 0.5
    assert score.tm_d0(100) == pytest.approx(1.24 * 85 ** (1 / 3) - 1.8)
    assert score.tm_d0(100) == pytest.approx(3.6522, abs=1e-3)
    assert score.tm_d0(300) == pytest.approx(1.24 * 285 ** (1 / 3) - 1.8)


def test_tm_identical_and_rigid_motion():
    x = chain()
    assert score.tm_score(x, x) == pytest.approx(1.0, abs=1e-12)
    assert score.tm_score(rigid(x), x) == pytest.approx(1.0, abs=1e-9)
    assert score.tm_score(x, rigid(x, 5)) == pytest.approx(1.0, abs=1e-9)
    r = score.score_arrays(rigid(x), x)
    assert r["lddt_ca"] == pytest.approx(1.0, abs=1e-12)
    assert r["tm_score"] == pytest.approx(1.0, abs=1e-9)
    assert r["rmsd_ca"] < 1e-9


def test_tm_missing_residues_normalised_by_reference_length():
    x = chain(100)
    pred = x.copy()
    pred[:10] = np.nan  # 10 of 100 reference residues absent in the prediction
    assert score.tm_score(rigid(pred), x) == pytest.approx(0.9, abs=1e-9)
    ref = x.copy()
    ref[50:60] = np.nan  # unmodeled in experiment -> L_norm = 90 and perfect prediction scores 1
    assert score.tm_score(rigid(x), ref) == pytest.approx(1.0, abs=1e-9)


def test_tm_decreases_with_noise_and_bounded():
    x = chain(150)
    rng = np.random.default_rng(7)
    vals = [score.tm_score(rigid(x + rng.normal(0, s, x.shape)), x) for s in (0.2, 0.5, 1, 2, 4, 8)]
    assert all(a > b for a, b in zip(vals, vals[1:])), vals
    assert 0 < vals[-1] < vals[0] <= 1


def _opt_tm(P, Q, d0, L, starts):
    def neg(p):
        R = Rotation.from_rotvec(p[:3]).as_matrix()
        d2 = (((P @ R.T + p[3:]) - Q) ** 2).sum(1)
        return -(1 / (1 + d2 / d0**2)).sum() / L

    best = 0.0
    for p0 in starts:
        r = minimize(neg, p0, method="Powell", options={"xtol": 1e-4, "ftol": 1e-8, "maxiter": 4000})
        best = max(best, -r.fun)
    return best


def test_tm_search_matches_independent_optimiser_on_two_domain_rearrangement():
    """A hinge-bent copy: TM must (a) be >= the oracle superposition of either domain and (b) not be beatable by a
    generic 6-parameter optimiser started from many places (including the oracle)."""
    x = chain(140, seed=4)
    pred = x.copy()
    k = 85
    hinge = x[k]
    Rb = Rotation.from_rotvec(np.array([0.3, -0.5, 0.4])).as_matrix()
    pred[k:] = (x[k:] - hinge) @ Rb.T + hinge + np.array([3.0, -2.0, 1.0])
    L = len(x)
    d0 = score.tm_d0(L)

    tm = score.tm_score(pred, x)
    # oracle: superpose on the larger (unbent) domain
    R, t = score.kabsch(pred[:k], x[:k])
    d2 = (((pred @ R.T + t) - x) ** 2).sum(1)
    oracle = (1 / (1 + d2 / d0**2)).sum() / L
    assert tm >= oracle - 1e-9
    rng = np.random.default_rng(0)
    starts = [np.zeros(6)] + [np.r_[Rotation.random(random_state=int(s)).as_rotvec(), rng.normal(0, 5, 3)] for s in range(12)]
    # centre both so rotvec starts act about the centroid-ish region
    c = x.mean(0)
    opt = _opt_tm(pred - c, x - c, d0, L, starts)
    assert tm >= opt - 2e-3, (tm, opt)
    assert 0.5 < tm < 1.0


def test_tm_better_than_single_kabsch_when_outlier_domain():
    """A single global Kabsch fit is dragged by a displaced tail; the TM search must do clearly better."""
    x = chain(120, seed=9)
    pred = x.copy()
    pred[90:] += np.array([25.0, 0, 0])  # displace the last 30 residues
    R, t = score.kabsch(pred, x)
    d2 = (((pred @ R.T + t) - x) ** 2).sum(1)
    single = (1 / (1 + d2 / score.tm_d0(120) ** 2)).sum() / 120
    tm = score.tm_score(pred, x)
    assert tm > single + 0.02
    assert tm == pytest.approx(90 / 120, abs=0.02)  # 90 perfect residues + 30 far-off ones (each contributing ~0)


# ------------------------------------------------------------------ mmCIF I/O


def _write_cif(path, coords, seqres_len, auth_offset=1000, name="A"):
    import gemmi

    st = gemmi.Structure()
    st.cell = gemmi.UnitCell(1, 1, 1, 90, 90, 90)
    m = gemmi.Model("1")
    ch = gemmi.Chain(name)
    for i, xyz in enumerate(coords):
        if np.isnan(xyz).any():
            continue
        r = gemmi.Residue()
        r.name = "ALA"
        r.seqid = gemmi.SeqId(auth_offset + i + (3 if i > 20 else 0), " ")  # auth numbering != SEQRES index
        r.label_seq = i + 1
        r.entity_type = gemmi.EntityType.Polymer
        a = gemmi.Atom()
        a.name = "CA"
        a.element = gemmi.Element("C")
        a.pos = gemmi.Position(*xyz)
        r.add_atom(a)
        ch.add_residue(r)
    m.add_chain(ch)
    st.add_model(m)
    ent = gemmi.Entity("1")
    ent.entity_type = gemmi.EntityType.Polymer
    ent.polymer_type = gemmi.PolymerType.PeptideL
    ent.subchains = [ch.name]
    ent.full_sequence = ["ALA"] * seqres_len
    st.entities.append(ent)
    st.make_mmcif_document().write_file(str(path))


def test_mmcif_roundtrip_label_seq_mapping(tmp_path):
    x = chain(60)
    ref = x.copy()
    ref[5:9] = np.nan  # gap in experimental numbering
    ref_path, pred_path = tmp_path / "ref.cif", tmp_path / "pred.cif"
    _write_cif(ref_path, ref, 60, auth_offset=1000)
    _write_cif(pred_path, rigid(x), 60, auth_offset=1)  # different auth numbering, same SEQRES indexing
    got = score.load_ca(ref_path)
    assert got.shape == (60, 3)
    assert np.isnan(got[5:9]).all() and np.isnan(got).sum() == 12
    np.testing.assert_allclose(got[np.isfinite(got[:, 0])], ref[np.isfinite(ref[:, 0])], atol=1e-3)
    res = score.score_files(pred_path, ref_path)
    assert res["lddt_ca"] == pytest.approx(1.0, abs=1e-6)
    assert res["tm_score"] == pytest.approx(1.0, abs=1e-5)
    assert res["n_ref_modeled"] == 56 and res["n_scored"] == 56 and res["n_seqres"] == 60
    assert score.main([str(pred_path), str(ref_path)]) == 0


@pytest.mark.skipif(not (DATA / "validation" / "cif" / "8j9r.cif").exists(), reason="validation cifs not downloaded")
def test_real_experimental_cifs_self_score():
    for f in sorted((DATA / "validation" / "cif").glob("*.cif"))[:3]:
        r = score.score_files(f, f)
        assert r["lddt_ca"] == 1.0
        assert r["tm_score"] == pytest.approx(1.0, abs=1e-12)
        assert r["rmsd_ca"] < 1e-6
        assert r["n_ref_modeled"] <= r["n_seqres"]
