"""tools/validation/run_l3.py: row building, resume, collect-only, MW correction, crop hash, mode presets (CPU, no model)."""

import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "validation"))
sys.path.insert(0, str(ROOT / "tools"))

import gates  # noqa: E402
import run_l3 as r  # noqa: E402


# ------------------------------------------------------------------------------------------------------------- fixtures


def _protein_case(**kw):
    d = dict(id="9ssd", kind="protein", target="9ssd", exp_cif=Path("/nonexistent/9ssd.cif"), exp_chain="A", n_res=494, msa="empty")
    d.update(kw)
    return r.Case(**d)


def _ligand_case(**kw):
    d = dict(id="tyk2_ejm_31", kind="ligand", target="tyk2", smiles="CC(=O)Nc1cc(NC(=O)c2c(Cl)cccc2Cl)ccn1", exp_value=-9.1,
             ligand_chain="L", has_affinity=True, msa="empty")
    d.update(kw)
    return r.Case(**d)


def _sinfo(tmp_path, case_id="tyk2_ejm_31", **kw):
    cdir = tmp_path / "seed0" / case_id
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / f"{case_id}_model_0.cif").write_text("data_x\n")
    (cdir / f"pre_affinity_{case_id}.npz").write_bytes(b"x")
    d = dict(case=case_id, seed=0, mode="strict", steps=200, recycling=3, device="gpu", env={}, pkg_tree="abc1234", git="deadbee",
             n_tokens=250, ptm=0.8, iptm=0.7, ligand_iptm=0.65, complex_plddt=0.85, nan=False, inf=False, guard_counts=None,
             seconds=12.5, model_cif=str(cdir / f"{case_id}_model_0.cif"), pre_affinity=str(cdir / f"pre_affinity_{case_id}.npz"),
             date="2026-10-27 20:00:00")
    d.update(kw)
    (cdir / "l3_struct.json").write_text(json.dumps(d))
    return d


def _ainfo(**kw):
    d = dict(case="tyk2_ejm_31", seed=0, mode="strict", steps=200, affinity_steps=200, recycling=5, pkg_tree="abc1234",
             affinity_pred_value=0.5, affinity_probability_binary=0.8, affinity_pred_value1=0.4, affinity_pred_value2=0.6,
             affinity_probability_binary1=0.7, affinity_probability_binary2=0.9, mw=300.0,
             value_mw_corrected=r.mw_correct(0.5, 300.0), crop_hash="0123456789ab", crop_tokens=200, best_sample=2,
             iptm_samples=[0.5, 0.6, 0.7, 0.6, 0.5], nan=False, inf=False, guard_counts=None, seconds=7.25)
    d.update(kw)
    return d


# --------------------------------------------------------------------------------------------------------- mode presets


def test_mode_presets_and_scrub():
    assert r.MODES["strict"] == {}
    assert r.MODES["fast"] == {"BOLTZ_MLX_DIFF_DTYPE": "fp16"}
    assert r.MODES["turbo"] == {"BOLTZ_MLX_DIFF_DTYPE": "fp16", "BOLTZ_MLX_TF32": "1"}
    base = {"PATH": "/bin", "BOLTZ_MLX_DIFF_DTYPE": "fp16", "BOLTZ_MLX_TF32": "1", "MLX_ENABLE_TF32": "1", "BOLTZ_MLX_CACHE_GB": "6"}
    strict = r.mode_environ(base, r.MODES["strict"])
    assert strict == {"PATH": "/bin", "BOLTZ_MLX_CACHE_GB": "6"}  # strict never inherits numerics switches from the shell
    assert r.mode_environ({"PATH": "/bin"}, r.MODES["turbo"])["BOLTZ_MLX_TF32"] == "1"


def test_modes_file_extends(tmp_path):
    f = tmp_path / "modes.json"
    f.write_text(json.dumps({"f16": {"BOLTZ_MLX_GEMM_DTYPE": "fp16"}, "fast": {"BOLTZ_MLX_DIFF_DTYPE": "bf16"}}))
    m = r.load_modes(f)
    assert m["f16"] == {"BOLTZ_MLX_GEMM_DTYPE": "fp16"} and m["fast"] == {"BOLTZ_MLX_DIFF_DTYPE": "bf16"}
    assert r.MODES["fast"] == {"BOLTZ_MLX_DIFF_DTYPE": "fp16"}  # module table untouched
    assert r.load_modes()["strict"] == {}


# ------------------------------------------------------------------------------------------------------- resume + jsonl


def test_load_done_ignores_comments_torn_lines_and_pairs(tmp_path):
    p = tmp_path / "o.jsonl"
    rows = [
        {"case": "a", "seed": 0, "mode": "strict", "steps": 200},
        {"case": "a", "seed": 1, "mode": "strict", "steps": 200},
        {"case": "l", "seed": 0, "mode": "strict", "steps": 200, "affinity_steps": 200},
        {"type": "ligand_rmsd_pair", "case": "l", "a_mode": "x", "a_seed": 0, "b_mode": "y", "b_seed": 0, "rmsd": 1.0},
    ]
    p.write_text("# comment\n" + "\n".join(json.dumps(x) for x in rows) + '\n{"case": "torn", "seed"')
    done = r.load_done(p)
    assert done == {("a", 0, "strict", 200, None), ("a", 1, "strict", 200, None), ("l", 0, "strict", 200, 200)}
    assert r.load_done(tmp_path / "missing.jsonl") == set()


def test_append_row_flushes_each_row_and_is_resumable(tmp_path):
    p = tmp_path / "sub" / "o.jsonl"
    r.append_row(p, {"case": "a", "seed": np.int64(0), "mode": "strict", "steps": 200, "x": np.float32(0.5), "ok": np.bool_(True)})
    assert len(p.read_text().splitlines()) == 1  # on disk immediately
    r.append_row(p, {"case": "b", "seed": 0, "mode": "strict", "steps": 200})
    assert [json.loads(x)["case"] for x in p.read_text().splitlines()] == ["a", "b"]
    assert r.load_done(p) == {("a", 0, "strict", 200, None), ("b", 0, "strict", 200, None)}


def test_affinity_steps_is_part_of_the_resume_key():
    a = {"case": "l", "seed": 0, "mode": "strict", "steps": 200, "affinity_steps": 200}
    b = dict(a, affinity_steps=50)
    assert r.row_key(a) != r.row_key(b)


def test_row_keys_match_between_planner_and_built_rows(tmp_path):
    # the planner's key (row_key_for) must equal the key of the row that run/collect later writes, or resume would loop
    pc = _protein_case(exp_cif=None)
    prow = r.build_protein_row(pc, 3, "strict", 200, _sinfo(tmp_path, "9ssd", seed=3))
    assert r.row_key(prow) == r.row_key_for(pc, 3, "strict", 200, 200)
    lc = _ligand_case()
    lrow = r.build_ligand_row(lc, 1, "turbo", 50, 25, _sinfo(tmp_path, steps=50), _ainfo())
    assert r.row_key(lrow) == r.row_key_for(lc, 1, "turbo", 50, 25)
    nc = _ligand_case(id="ex_ligand", has_affinity=False)
    nrow = r.build_ligand_row(nc, 0, "strict", 200, None, _sinfo(tmp_path, "ex_ligand"), None)
    assert r.row_key(nrow) == r.row_key_for(nc, 0, "strict", 200, 200) and nrow["affinity_steps"] is None


def test_budget_stops_at_limit():
    t = [0.0]
    b = r.Budget(2.0, clock=lambda: t[0])
    assert not b.exhausted()
    t[0] = 119.9
    assert not b.exhausted()
    t[0] = 120.0
    assert b.exhausted() and b.elapsed_min() == pytest.approx(2.0)
    assert not r.Budget(0, clock=lambda: 1e9).exhausted()  # 0 = unlimited


# ------------------------------------------------------------------------------------------------------ numerics helpers


def test_mw_correction_formula_and_upstream_equivalence():
    # upstream Boltz2.forward: 1.03525938 * v - 0.59992683 * MW**0.3 + 2.83288489
    v, mw = 1.25, 347.264
    assert r.mw_correct(v, mw) == pytest.approx(1.03525938 * v - 0.59992683 * mw**0.3 + 2.83288489, rel=1e-12)
    from boltz_mlx.modules.affinity import affinity_ensemble

    o1 = {"affinity_pred_value": mx.array([1.0]), "affinity_logits_binary": mx.array([0.2])}
    o2 = {"affinity_pred_value": mx.array([1.5]), "affinity_logits_binary": mx.array([-0.1])}
    ens = affinity_ensemble(o1, o2, {"affinity_mw": [mw]}, mw_correction=True)
    assert r.mw_correct(v, mw) == pytest.approx(float(ens["affinity_pred_value"][0]), abs=1e-5)
    off = affinity_ensemble(o1, o2, {"affinity_mw": [mw]}, mw_correction=False)
    assert float(off["affinity_pred_value"][0]) == pytest.approx(1.25)


def test_crop_hash_is_order_insensitive_and_set_sensitive():
    asym = np.array([[0, 0, 0, 1, 1, 0]])
    res = np.array([[0, 1, 2, 0, 0, 0]])
    pad = np.array([[1, 1, 1, 1, 1, 0.0]])  # last token is padding
    h = r.crop_hash(asym, res, pad)
    assert len(h) == 12
    perm = [3, 1, 4, 0, 2, 5]
    assert r.crop_hash(asym[:, perm], res[:, perm], pad[:, perm]) == h  # token order does not matter
    assert r.crop_hash(asym, res, np.array([[1, 1, 0, 1, 1, 0.0]])) != h  # different residue set
    assert r.crop_hash(asym[:, :5], res[:, :5]) == h  # padding tokens are not part of the crop


def test_nonfinite_flags_and_counts():
    ok = np.zeros((2, 3), np.float32)
    assert r.nonfinite_flags([ok, None, np.arange(3)]) == (False, False)
    assert r.nonfinite_flags([ok, np.array([1.0, np.nan], np.float32)]) == (True, False)
    assert r.nonfinite_flags([np.array([np.inf], np.float32), np.array([np.nan], np.float32)]) == (True, True)
    assert r.to_counts(None) is None
    assert r.to_counts(mx.array([1, 2])) == 3
    assert r.to_counts({"tri": mx.array([0, 1]), "ffn": 0}) == {"tri": 1, "ffn": 0}
    assert r.merge_counts(structure=None, affinity=None) is None
    g = r.merge_counts(structure=2, affinity={"tri": 1, "ffn": 0})
    assert g == {"structure": 2, "affinity.tri": 1, "affinity.ffn": 0} and gates.guard_total({"guard_counts": g}) == 3


# ---------------------------------------------------------------------------------------------------------------- rows


def test_protein_row_schema(tmp_path):
    fp = tmp_path / "fp.jsonl"
    fp.write_text(json.dumps({"pkg_tree": "abc1234", "mode": "strict", "fingerprint": "f00dfeedf00d"}) + "\n")
    info = _sinfo(tmp_path, "9ssd")
    calls = []

    def scorer(pred, ref, exp_chain=None):
        calls.append((pred, str(ref), exp_chain))
        return {"lddt_ca": 0.71, "tm_score": 0.8, "rmsd_ca": 3.1}

    row = r.build_protein_row(_protein_case(), 2, "strict", 200, info, scorer, fp)
    for k in ("case", "target", "seed", "mode", "steps", "n_res", "large_chain", "ca_lddt", "tm", "ptm", "plddt", "nan", "inf",
              "fingerprint", "model_cif", "seconds"):
        assert k in row, k
    assert row["case"] == row["target"] == "9ssd" and row["seed"] == 2 and row["large_chain"] is True and row["n_res"] == 494
    assert (row["ca_lddt"], row["tm"], row["plddt"], row["ptm"]) == (0.71, 0.8, 0.85, 0.8)
    assert row["fingerprint"] == "f00dfeedf00d" and row["nan"] is False and row["inf"] is False
    assert "guard_counts" not in row  # model returned none
    assert calls == [(info["model_cif"], "/nonexistent/9ssd.cif", "A")]
    json.dumps(row)
    small = r.build_protein_row(_protein_case(id="9eyb", n_res=64), 0, "strict", 200, _sinfo(tmp_path, "9eyb"), scorer, fp)
    assert small["large_chain"] is False
    # unknown build/mode -> fingerprint "none"; guard counts are carried when present
    row2 = r.build_protein_row(_protein_case(), 0, "f16", 200, _sinfo(tmp_path, "9ssd", guard_counts={"a": 0, "b": 2}), scorer, fp)
    assert row2["fingerprint"] == "none" and row2["guard_counts"] == {"a": 0, "b": 2} and gates.guard_total(row2) == 2


def test_protein_row_without_reference_and_scoring_error(tmp_path):
    row = r.build_protein_row(_protein_case(exp_cif=None, id="prot117", n_res=None), 0, "strict", 200, _sinfo(tmp_path, "prot117"), None)
    assert "ca_lddt" not in row and row["n_res"] == 250 and row["large_chain"] is False

    def bad(*a, **k):
        raise ValueError("SEQRES length mismatch")

    row = r.build_protein_row(_protein_case(), 0, "strict", 200, _sinfo(tmp_path, "9ssd"), bad)
    assert "ca_lddt" not in row and "SEQRES" in row["score_error"]


def test_ligand_row_schema(tmp_path):
    fp = tmp_path / "fp.jsonl"
    fp.write_text(json.dumps({"pkg_tree": "abc1234", "mode": "strict", "fingerprint": "f00dfeedf00d"}) + "\n")
    sinfo = _sinfo(tmp_path)
    row = r.build_ligand_row(_ligand_case(), 1, "strict", 200, 200, sinfo, _ainfo(), fp)
    for k in ("case", "target", "seed", "mode", "steps", "affinity_steps", "affinity_pred_value", "value_mw_corrected",
              "affinity_probability_binary", "affinity_pred_value1", "affinity_pred_value2", "affinity_probability_binary1",
              "affinity_probability_binary2", "exp_value", "msa", "pose_cif", "ligand_chain", "smiles", "iptm", "ligand_iptm",
              "complex_plddt", "affinity_crop_hash", "nan", "inf", "fingerprint", "seconds_structure", "seconds_affinity", "seconds"):
        assert k in row, k
    assert row["mw_correction"] is False and row["affinity_pred_value"] == row["affinity_pred_value_mw_off"] == 0.5
    assert row["value_mw_corrected"] == pytest.approx(1.03525938 * 0.5 - 0.59992683 * 300.0**0.3 + 2.83288489)
    assert row["pose_cif"] == sinfo["model_cif"] and row["ligand_chain"] == "L" and row["exp_value"] == -9.1
    assert row["seconds"] == pytest.approx(12.5 + 7.25) and row["affinity_best_sample"] == 2 and row["fingerprint"] == "f00dfeedf00d"
    assert row["nan"] is False and "guard_counts" not in row
    json.dumps(row)
    # NaN/inf/guard counts from either phase propagate
    bad = r.build_ligand_row(_ligand_case(), 1, "strict", 200, 200, _sinfo(tmp_path, guard_counts=1), _ainfo(nan=True, guard_counts={"t": 3}))
    assert bad["nan"] is True and bad["inf"] is False and gates.guard_total(bad) == 4
    # chymase-style row without a label
    nolabel = r.build_ligand_row(_ligand_case(exp_value=None), 0, "strict", 200, 200, sinfo, _ainfo())
    assert "exp_value" not in nolabel


def test_gates_can_read_rows(tmp_path):
    """Rows emitted by build_protein_row feed the L3-P arrays used by gates (smoke: field names agree)."""
    rows = []
    for mode, d in (("strict", 0.0), ("f16", -0.001)):
        for cid in ("a", "b", "c"):
            info = _sinfo(tmp_path, cid, mode=mode)
            rows.append(r.build_protein_row(_protein_case(id=cid, target=cid, n_res=100), 0, mode, 200, info,
                                            lambda *a, **k: {"lddt_ca": 0.7 + d, "tm_score": 0.8 + d, "rmsd_ca": 2.0}))
    assert {x["mode"] for x in gates.select(rows, "f16", 200)} == {"f16"}


# ------------------------------------------------------------------------------------------------- reuse + collect-only


def test_struct_reusable(tmp_path):
    ident = {"pkg_tree": "abc1234", "git": "deadbee"}
    case = _ligand_case()
    info = _sinfo(tmp_path)
    assert r.struct_reusable(info, case, "strict", 200, 3, ident)
    assert not r.struct_reusable(None, case, "strict", 200, 3, ident)
    assert not r.struct_reusable(info, case, "turbo", 200, 3, ident)          # other mode
    assert not r.struct_reusable(info, case, "strict", 50, 3, ident)          # other steps
    assert not r.struct_reusable(info, case, "strict", 200, 0, ident)         # other recycling
    assert not r.struct_reusable(info, case, "strict", 200, 3, {"pkg_tree": "zzz", "git": "x"})  # other build
    assert not r.struct_reusable(info, case, "strict", 200, 3, {"pkg_tree": "abc1234-dirty", "git": "x"})
    assert not r.struct_reusable(dict(info, model_cif=str(tmp_path / "gone.cif")), case, "strict", 200, 3, ident)
    Path(info["pre_affinity"]).unlink()
    assert not r.struct_reusable(info, case, "strict", 200, 3, ident)          # affinity needs pre_affinity
    assert r.struct_reusable(info, _ligand_case(has_affinity=False), "strict", 200, 3, ident)


def test_collect_rows_rebuilds_from_sidecars(tmp_path):
    run_root = tmp_path / "run"
    sdir = r.seed_dir(run_root, 0)
    pc, lc, lc2, nc = _protein_case(id="9eyb", n_res=64), _ligand_case(), _ligand_case(id="tyk2_ejm_42"), _ligand_case(id="ex_ligand", has_affinity=False)
    for c in (pc, lc, lc2, nc):
        cdir = sdir / c.id
        cdir.mkdir(parents=True, exist_ok=True)
        info = {"case": c.id, "seed": 0, "mode": "strict", "steps": 200, "recycling": 3, "device": "gpu", "env": {}, "pkg_tree": "abc1234",
                "git": "g", "n_tokens": 10, "ptm": 0.5, "iptm": 0.4, "ligand_iptm": 0.3, "complex_plddt": 0.6, "nan": False, "inf": False,
                "guard_counts": None, "seconds": 1.0, "model_cif": str(cdir / f"{c.id}_model_0.cif"), "pre_affinity": ""}
        r.struct_sidecar(sdir, c.id).write_text(json.dumps(info))
    a_path = r.aff_sidecar(sdir, lc.id, 200)
    a_path.parent.mkdir(parents=True, exist_ok=True)
    a_path.write_text(json.dumps(_ainfo()))  # tyk2_ejm_42 has no affinity sidecar yet
    got = list(r.collect_rows([pc, lc, lc2, nc], [0, 1], "strict", 200, 200, run_root, lambda *a, **k: {"lddt_ca": 0.5, "tm_score": 0.5, "rmsd_ca": 1.0}))
    rows = {(c.id, s): (row, why) for row, c, s, why in got}
    assert rows[("9eyb", 0)][0]["ca_lddt"] == 0.5
    assert rows[("tyk2_ejm_31", 0)][0]["affinity_steps"] == 200
    assert rows[("tyk2_ejm_42", 0)] == (None, "no affinity sidecar")
    assert rows[("ex_ligand", 0)][0]["affinity_steps"] is None
    assert rows[("9eyb", 1)] == (None, "no structure sidecar")
    for (row, _) in rows.values():
        if row is not None:
            assert r.row_key(row) in {r.row_key_for(c, 0, "strict", 200, 200) for c in (pc, lc, lc2, nc)}


# ----------------------------------------------------------------------------------------------------------- case lists


def test_resolve_cases_sets():
    v36 = r.resolve_cases("v36")
    assert len(v36) == 36 and all(c.kind == "protein" and c.target == c.id for c in v36)
    large = {c.id for c in v36 if c.n_res >= r.LARGE_CHAIN_MIN}
    assert large == {"9y35", "8x3e", "13xs", "8jdp", "9ns4", "9ssd"}
    sub = r.resolve_cases("v36", cases=["9eyb"])
    assert [c.id for c in sub] == ["9eyb"] and sub[0].exp_chain == "AAA"
    with pytest.raises(ValueError):
        r.resolve_cases("v36", cases=["nope"])
    par = r.resolve_cases("parity")
    assert [c.id for c in par] == list(r.PARITY_CASES)
    byid = {c.id: c for c in par}
    assert byid["lig_aff"].kind == "ligand" and byid["lig_aff"].has_affinity and byid["lig_aff"].ligand_chain == "B"
    assert byid["ex_ligand"].kind == "ligand" and not byid["ex_ligand"].has_affinity and byid["msa384"].msa == "a3m"
    assert byid["prot117"].kind == "protein" and byid["prot117"].exp_cif is None


@pytest.mark.skipif(not (r.FEP / "labels.csv").exists(), reason="FEP+ data not present")
def test_resolve_fep_targets_and_labels():
    tyk2 = r.resolve_cases("fep", targets=["tyk2"])
    assert len(tyk2) == 16 and {c.target for c in tyk2} == {"tyk2"} and tyk2[0].id == "tyk2_ejm_31"
    c = tyk2[0]
    assert c.ligand_chain == "L" and c.has_affinity and c.msa == "empty" and c.exp_value is not None and c.smiles
    bace = r.resolve_cases("fep", targets=["bace1"])
    assert len(bace) == 36 and {c.target for c in bace} == {"bace1"}  # labels.csv "bace" -> gates "bace1"
    allfep = r.resolve_cases("fep")
    assert len(allfep) == 123 and [c.target for c in allfep][0] == "cdk2"
    with pytest.raises(ValueError):
        r.resolve_cases("fep", targets=["mcl1"])
    assert len(r.resolve_cases("thrombin")) == 11


@pytest.mark.skipif(not (r.CASP / "labels.csv").exists(), reason="CASP16 chymase data not present")
def test_resolve_chymase():
    ch = r.resolve_cases("chymase")
    assert len(ch) == 17 and all(c.target == "chymase" and c.has_affinity for c in ch)
    # exp_value sign: dG-like, stronger binder = more negative (same convention as the FEP+ labels: higher = weaker)
    byid = {c.id: c for c in ch}
    assert byid["chymase_L1001"].exp_value < byid["chymase_L1002"].exp_value


def test_structure_only_rows_have_their_own_key(tmp_path):
    # --structure-only (sampler tuning stage A): ligand rows without the affinity pass, keyed apart from the full rows
    lc = _ligand_case()
    srow = r.build_ligand_row(lc, 0, "f16+s25", 25, None, _sinfo(tmp_path, steps=25), None)
    assert r.row_key(srow) == r.row_key_for(lc, 0, "f16+s25", 25, None) and srow["affinity_steps"] is None
    assert r.row_key(srow) != r.row_key_for(lc, 0, "f16+s25", 25, 200)


# ------------------------------------------------------------------------------------------------------------ --data-root


@pytest.fixture
def restore_data_root(monkeypatch):
    """set_data_root rebinds module globals of run_l3 and feature_cache: restore them after the test."""
    import feature_cache

    for mod, names in ((r, ("DATA", "WORK")), (feature_cache, ("DATA", "CACHE_DIR"))):
        for n in names:
            monkeypatch.setattr(mod, n, getattr(mod, n))
    return feature_cache


def test_data_root_moves_processed_inputs_feature_cache_and_work(tmp_path, restore_data_root):
    fc = restore_data_root
    assert r.set_data_root(None) == r.DATA  # no option: the root in use is kept
    root = r.set_data_root(tmp_path / "dr")
    assert root == (tmp_path / "dr").resolve() and r.DATA == root
    assert r.Case(id="cdk2_17", kind="ligand", target="cdk2").data_dir == root / "cdk2_17"
    assert fc._path("cdk2_17", 0, False) == root / "cache" / "features" / "cdk2_17" / "s0.npz"
    assert fc._path("cdk2_17", 0, True, "strict_s200") == root / "cache" / "features" / "cdk2_17" / "s0_aff_strict_s200.npz"
    assert r.WORK == root / "validation" / "runs_l3"
    # reference structures are read from the default root, whatever the data root
    assert r.REF_DATA == r.DEFAULT_DATA
    v36 = r.resolve_cases("v36")
    assert v36 and all(c.exp_cif.is_relative_to(r.DEFAULT_DATA / "validation" / "cif") for c in v36)
    a = r.build_parser().parse_args(["--set", "fep", "--out", str(tmp_path / "o.jsonl"), "--data-root", str(root)])
    assert a.work is None and a.data_root == root  # main() resolves --work under the data root


@pytest.mark.skipif(not (r.FEP / "labels.csv").exists(), reason="FEP+ data not present")
def test_data_root_dry_run_plans_under_the_new_root(tmp_path, restore_data_root, capsys):
    root = (tmp_path / "dr").resolve()
    rc = r.main(["--set", "fep", "--targets", "cdk2", "--seeds", "0", "--mode", "strict", "--out", str(tmp_path / "o.jsonl"),
                 "--data-root", str(root), "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0 and f"data_root={root} work={root / 'validation' / 'runs_l3'}" in out
    assert out.count("  todo: cdk2_") == 16 and not root.exists()  # a dry run writes nothing


def _files_state(d: Path) -> dict:
    return {str(p.relative_to(d)): (p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(d.rglob("*")) if p.is_file()}


@pytest.mark.skipif(not (r.FEP / "yaml" / "cdk2" / "cdk2_17.yaml").exists() or not (r.CACHE / "mols").is_dir(),
                    reason="FEP+ inputs / CCD mols cache not present")
def test_data_root_prefeaturize_writes_only_under_the_new_root(tmp_path):
    """CPU, upstream process_inputs + featurizer (no model): --prefeaturize with a new data root processes the case into
    that root and builds its feature cache there; the default root's processed inputs and feature cache are untouched.
    cdk2_17's cached conformer is a first-in-process (seed-42) draw, so the new root's conformer equals it."""
    import pickle
    import subprocess

    case, root = "cdk2_17", (tmp_path / "dr").resolve()
    canon = [r.DEFAULT_DATA / case, r.DEFAULT_DATA / "cache" / "features" / case]
    before = [_files_state(d) for d in canon]
    p = subprocess.run([sys.executable, str(ROOT / "tools" / "validation" / "run_l3.py"), "--set", "fep", "--targets", "cdk2",
                        "--cases", case, "--seeds", "0", "--data-root", str(root), "--out", str(tmp_path / "o.jsonl"),
                        "--prefeaturize"], capture_output=True, text=True, timeout=900, cwd=tmp_path)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    assert f"data_root={root} feature_cache={root / 'cache' / 'features'}" in p.stdout
    assert (root / case / "processed" / "manifest.json").exists()
    assert (root / "cache" / "features" / case / "s0.npz").exists()
    assert [_files_state(d) for d in canon] == before
    assert not (tmp_path / "o.jsonl").exists()
    old_pkl = r.DEFAULT_DATA / case / "processed" / "mols" / f"{case}.pkl"
    if old_pkl.exists():
        old, new = (pickle.loads(f.read_bytes()) for f in (old_pkl, root / case / "processed" / "mols" / f"{case}.pkl"))
        assert sorted(old) == sorted(new)
        for k in old:
            assert np.array_equal(old[k].GetConformer().GetPositions(), new[k].GetConformer().GetPositions())
