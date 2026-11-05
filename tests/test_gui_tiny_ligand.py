"""Fast preset, tiny-ligand rule (boltz_mlx/tiny_ligand.py), module level and in the GUI: a ligand with fewer than 10 heavy atoms
makes Fast run 200 structure steps (affinity stays 25) unless the user set the structure steps. CPU only; no prediction runs
(validate / estimate / argv building only)."""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest
from rdkit import Chem

from boltz_mlx import tiny_ligand as TL
from boltz_mlx.gui import predict as PR
from test_gui_targets import PROT, app, env  # noqa: F401  (fixtures)

BCT, CEL = "OC(=O)[O-]", "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1"
SIN = "OC(=O)CCC(=O)O"        # succinic acid, 8 heavy atoms (8eil)


def _pkl(d, code, smi):
    (d / f"{code}.pkl").write_bytes(pickle.dumps(Chem.AddHs(Chem.MolFromSmiles(smi))))   # hydrogens included: not counted


def job(lig, kind="smiles", **opts):
    return {"kind": "predict", "name": "t", "chains": [{"type": "protein", "id": "A", "sequence": PROT},
                                                      {"type": "ligand", "id": "B", kind: lig}],
            "affinity": {"binder": "B"}, "options": {"preset": "standard", **opts}}


# ------------------------------------------------------------------------------------------------------ the module
@pytest.mark.parametrize("smi,n", [(BCT, 4), (SIN, 8), (CEL, 26), ("[H]OC([H])([H])C([H])([H])[H]", 3), ("C" * 9, 9), ("C" * 10, 10),
                                   ("[Na+].[Cl-]", 2), ("[2H]C([2H])([2H])O", 2)])
def test_heavy_atoms_are_atoms_with_atomic_number_above_one(smi, n):
    assert TL.heavy_atoms_smiles(smi) == n


def test_unreadable_smiles_has_no_count_and_is_not_tiny():
    assert TL.heavy_atoms_smiles("xx(") is None
    assert not TL.size_of("ligand B", "smiles", "xx(").tiny


def test_ccd_counts_come_from_the_mols_dictionary(tmp_path):
    d = tmp_path / "mols"
    d.mkdir()
    _pkl(d, "BCT", BCT)
    _pkl(d, "CEL", CEL)
    assert TL.heavy_atoms_ccd("BCT", d) == 4 and TL.heavy_atoms_ccd("CEL", d) == 26
    assert TL.heavy_atoms_ccd("NOPE", d) is None and TL.heavy_atoms_ccd("BCT", None) is None
    (d / "BAD.pkl").write_bytes(b"not a pickle")
    assert TL.heavy_atoms_ccd("BAD", d) is None


def test_decide_and_reason_wording():
    s = [TL.LigandSize("ligand B", "smiles", BCT, 4), TL.LigandSize("ligand C", "smiles", CEL, 26)]
    assert TL.decide(s, 20) == (200, "ligand B has 4 heavy atoms (< 10)")
    assert TL.decide(s[1:], 20) == (20, None)
    assert TL.decide([], 20) == (20, None)
    many = [TL.LigandSize(f"ligand {c}", "smiles", "C", 1) for c in "BCDE"]
    assert TL.decide(many, 20)[1] == "ligand B has 1 heavy atom, ligand C has 1 heavy atom, ligand D has 1 heavy atom, 1 more (< 10)"


def test_sizes_from_yaml_and_fasta(tmp_path):
    d = tmp_path / "mols"
    d.mkdir()
    _pkl(d, "BCT", BCT)
    y = tmp_path / "a.yaml"
    y.write_text(f"version: 1\nsequences:\n  - protein: {{id: A, sequence: {PROT}, msa: empty}}\n"
                 f"  - ligand: {{id: [B, C], smiles: '{SIN}'}}\n  - ligand: {{id: D, ccd: bct}}\n")
    got = [(s.label, s.heavy) for s in TL.sizes_from_path(y, d)]
    assert got == [("ligand B", 8), ("ligand D", 4)]                      # one entry with two ids counts once; ccd upper-cased
    f = tmp_path / "b.fa"
    f.write_text(f">A|protein|empty\n{PROT}\n>B|smiles|\n{BCT}\n>C|ccd|\nbct\n")
    assert [(s.label, s.heavy) for s in TL.sizes_from_path(f, d)] == [("ligand B", 4), ("ligand C", 4)]


# -------------------------------------------------------------------------------------------------- the GUI: predict
def test_gui_predict_tiny_smiles_runs_200_steps(app):
    v = app.validate_predict(job(BCT))
    o = v["options"]
    assert v["ok"] and (o["steps"], o["affinity_steps"], o["preset"]) == (200, 25, "screen")
    assert o["steps_reason"] == "ligand B has 4 heavy atoms (< 10)" and o["steps_auto"] and "steps" not in o["changed"]
    e = v["estimate"]
    assert e["validated"] and e["steps_note"].startswith("Fast runs 200 structure steps here, not 20: ligand B has 4 heavy atoms")
    argv = PR.predict_argv(Path("/r"), {"yaml": {"file": "i/job.yaml"}}, o)
    assert "--sampling_steps=200" in argv and "--sampling_steps_affinity=25" in argv


def test_gui_predict_26_atom_ligand_stays_at_20(app):
    o = app.validate_predict(job(CEL))["options"]
    assert o["steps"] == 20 and o["steps_reason"] is None and not o["steps_auto"]


def test_gui_predict_explicit_steps_win(app):
    o = app.validate_predict(job(BCT, steps=20))["options"]
    assert o["steps"] == 20 and o["steps_reason"] is None and o["steps_explicit"]
    o = app.validate_predict(job(BCT, steps=75))["options"]
    assert o["steps"] == 75 and "steps" in o["changed"] and o["steps_reason"] is None


def test_gui_predict_full_is_unchanged(app):
    o = app.validate_predict(job(BCT, preset="reference"))["options"]
    assert o["preset"] == "strict" and o["steps"] == 200 and o["steps_reason"] is None


def test_gui_predict_ccd_ligand(app, env):
    mols = Path(env["cache"]) / "mols"
    _pkl(mols, "BCT", BCT)
    _pkl(mols, "CEL", CEL)
    v = app.validate_predict(job("bct", kind="ccd"))
    assert v["ok"] and v["options"]["steps"] == 200 and v["options"]["steps_reason"] == "ligand B has 4 heavy atoms (< 10)"
    assert app.validate_predict(job("CEL", kind="ccd"))["options"]["steps"] == 20


def test_gui_predict_yaml_input_and_run_record(app):
    import time

    y = f"version: 1\nsequences:\n  - protein: {{id: A, sequence: {PROT}, msa: empty}}\n  - ligand: {{id: B, smiles: '{BCT}'}}\n"
    v = app.validate_predict({"kind": "predict", "yaml": {"content": y, "files": []}, "options": {"preset": "standard"}})
    assert v["ok"] and v["options"]["steps"] == 200
    r = app.create_predict_run({"kind": "predict", "name": "tiny", "chains": job(BCT)["chains"], "options": {"preset": "standard"}})
    meta = app.runs.read_meta(r["id"])
    assert meta["options"]["steps"] == 200 and meta["options"]["steps_reason"].startswith("ligand B has 4 heavy atoms")
    assert meta["summary"]["estimate"]["steps_note"]
    for _ in range(100):   # the fake predict subprocess finishes
        if app.runs.state(r["id"]) != "running":
            break
        time.sleep(0.1)


# --------------------------------------------------------------------------------------------------- the GUI: screen
LIGS = f"id,smiles\ntiny,{BCT}\nbig,{CEL}\nethanol,CCO\nmid,CC(=O)Nc1ccc(O)cc1\n"


def _screen(**opts):
    return {"target": {"kind": "text", "content": PROT}, "msa": {"mode": "empty"},
            "ligands": {"kind": "file", "filename": "l.csv", "content": LIGS}, "options": {"preset": "standard", **opts}}


def test_gui_screen_estimate_names_the_tiny_ligands(app):
    est = app.estimate(_screen())
    e = est["estimate"]
    assert e["steps_note"].startswith("Fast preset: 2 of 4 ligands have < 10 heavy atoms (tiny, ethanol) and run 200 structure steps")
    assert any(l.strip().startswith("tiny ligands: 2 of 4 valid ligands") for l in est["lines"])
    assert not e.get("custom")                                    # the rule is not a custom setting


def test_gui_screen_no_note_with_explicit_steps_or_full_or_no_tiny_ligand(app):
    assert app.estimate(_screen(steps=20))["estimate"]["steps_note"] is None
    assert app.estimate(_screen(preset="reference"))["estimate"]["steps_note"] is None
    p = _screen()
    p["ligands"]["content"] = f"id,smiles\nbig,{CEL}\n"
    assert app.estimate(p)["estimate"]["steps_note"] is None


def test_gui_screen_passes_no_steps_so_the_cli_rule_decides(app):
    from boltz_mlx.gui import inputs as GI

    o = GI.normalize_options({"preset": "standard"}, app.presets)
    assert o["steps"] is None
    a = GI.screen_argv(Path("/r"), {"target": {"file": "t"}, "ligands": {"file": "l"}}, o, Path("/o"))
    assert not any(x.startswith("--steps") for x in a)
    assert any(x.startswith("--steps=20") for x in GI.screen_argv(
        Path("/r"), {"target": {"file": "t"}, "ligands": {"file": "l"}}, GI.normalize_options({"preset": "standard", "steps": 20}, app.presets), Path("/o")))
