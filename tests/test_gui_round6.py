"""GUI round 6 (answers to the round-5 audit): a YAML input survives 'Use inputs' with no companion files, attached MSAs are
checked against their chain, colliding file names are refused, exported YAML runs from the extracted folder, estimates do not
ignore recycles / affinity samples, screen YAML targets take companion files, CSV by id, changed sites in the lead-opt drawing.

CPU only (the fake screen / predict subprocesses of tests/gui_fake_*.py).
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest
import yaml
from test_gui_targets import LIGS, PROT, STRUCT, app, call, env, http_gui, wait_state  # noqa: F401  (fixtures)

from boltz_mlx.gui import predict as PR
from boltz_mlx.gui.common import ApiError

BARE = f"""version: 1
sequences:
  - protein:
      id: A
      sequence: {PROT}
      msa: empty
  - ligand:
      id: B
      smiles: CC(=O)Nc1ccc(O)cc1
constraints:
  - pocket:
      binder: B
      contacts: [[A, 10]]
properties:
  - affinity:
      binder: B
"""
WITH_MSA = BARE.replace("msa: empty", "msa: receptors/msa.a3m")
MSA = f">q\n{PROT}\n>h\n{PROT}\n"
OPTS = {"preset": "reference"}


def _predict(app, doc, files=None, **o):
    return {"kind": "predict", "yaml": {"content": doc, "files": files or []}, "options": {**OPTS, **o}}


def test_yaml_without_companions_survives_use_inputs(app):
    p = _predict(app, BARE)
    r = app.create_run(p)
    assert wait_state(app, r["id"], ("complete", "failed")) == "complete"
    back = app.run_inputs(r["id"])
    assert back["yaml"]["files"] == [] and "constraints" in back["yaml"]["content"] and "pocket" in back["yaml"]["content"]
    again = app.validate_predict(_predict(app, back["yaml"]["content"], back["yaml"]["files"]))
    assert again["ok"]
    form = app.create_run({"kind": "predict", "chains": [{"type": "protein", "id": "A", "sequence": PROT}], "options": OPTS})
    wait_state(app, form["id"], ("complete", "failed"))
    assert "yaml" not in app.run_inputs(form["id"])    # a form run comes back as a form


def test_attached_msa_must_be_this_chains_alignment(app):
    other = ">q\nMKTAYIAKQR\n"
    out = app.validate_predict(_predict(app, WITH_MSA, [{"filename": "msa.a3m", "content": other}]))
    assert out["ok"] is False and out["error"].startswith("chain A: MSA query")
    ok = app.validate_predict(_predict(app, WITH_MSA, [{"filename": "msa.a3m", "content": MSA}]))
    assert ok["ok"]


def test_two_paths_with_one_file_name_are_refused(app):
    two = f"""version: 1
sequences:
  - protein: {{id: A, sequence: {PROT}, msa: a/msa.a3m}}
  - protein: {{id: C, sequence: {PROT}, msa: b/msa.a3m}}
"""
    out = app.validate_predict(_predict(app, two, [{"filename": "msa.a3m", "content": MSA}]))
    assert out["ok"] is False and "2 different paths" in out["error"]


def test_constraint_naming_an_unknown_chain_is_refused(app):
    out = app.validate_predict(_predict(app, BARE.replace("contacts: [[A, 10]]", "contacts: [[Q, 10]]")))
    assert out["ok"] is False and "chain Q" in out["error"]


def test_exported_yaml_runs_from_the_extracted_folder(http_gui, app, tmp_path):
    r = app.create_run(_predict(app, WITH_MSA, [{"filename": "msa.a3m", "content": MSA}]))
    assert wait_state(app, r["id"], ("complete", "failed")) == "complete"
    s, _, b = call(http_gui, "GET", f"/api/runs/{r['id']}/export.zip")
    assert s == 200
    z = zipfile.ZipFile(io.BytesIO(b))
    z.extractall(tmp_path / "x")
    root = next((tmp_path / "x").iterdir())
    doc = yaml.safe_load((root / "inputs" / "job.yaml").read_text())
    ref = doc["sequences"][0]["protein"]["msa"]
    assert not Path(ref).is_absolute() and (root / "inputs" / ref).resolve().is_file()
    assert str(app.runs.run_dir(r["id"])) not in (root / "inputs" / "job.yaml").read_text()
    man = json.loads((root / "manifest.json").read_text())
    assert man["rerun"].startswith("cd inputs && boltz-mlx predict job.yaml --out_dir=out") and man["rerun"] in (root / "README.txt").read_text()


def test_form_run_export_is_portable_too(http_gui, app, tmp_path):
    chains = [{"type": "protein", "id": "A", "sequence": PROT, "msa": {"mode": "file", "filename": "m.a3m", "content": MSA}}]
    r = app.create_run({"kind": "predict", "chains": chains, "options": OPTS})
    assert wait_state(app, r["id"], ("complete", "failed")) == "complete"
    _, _, b = call(http_gui, "GET", f"/api/runs/{r['id']}/export.zip")
    z = zipfile.ZipFile(io.BytesIO(b))
    y = next(n for n in z.namelist() if n.endswith("inputs/job.yaml"))
    ref = yaml.safe_load(z.read(y))["sequences"][0]["protein"]["msa"]
    assert ref == "msa_A.a3m" and y.replace("job.yaml", ref) in z.namelist()


def test_estimate_has_no_number_for_custom_recycles_or_affinity_samples(app):
    chains = [{"type": "protein", "id": "A", "sequence": PROT}, {"type": "ligand", "id": "B", "smiles": "CCO"}]
    base = {"kind": "predict", "chains": chains, "affinity": {"binder": "B"}}
    e0 = app.validate_predict({**base, "options": {"preset": "reference"}})["estimate"]
    assert e0["available"] and e0["total_s"] > 0
    for o in ({"recycling": 5}, {"affinity_samples": 2}):
        e = app.validate_predict({**base, "options": {"preset": "reference", **o}})["estimate"]
        assert e["available"] is False and "custom" in e["caveat"]
    noaff = app.validate_predict({"kind": "predict", "chains": chains[:1], "options": {"preset": "reference", "affinity_samples": 2}})["estimate"]
    assert noaff["available"]    # affinity samples do not matter without the affinity pass


def test_screen_yaml_target_takes_companion_files(app):
    doc = f"version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: {PROT}\n      msa: ../x/rec.a3m\n"
    p = {"target": {"kind": "file", "filename": "t.yaml", "content": doc, "files": [{"filename": "rec.a3m", "content": MSA}]},
         "ligands": {"kind": "file", "filename": "l.csv", "content": LIGS}, "options": {"preset": "standard"}}
    v = app.validate(p)
    assert v["ok"] and v["target"]["ok"] and v["target"]["chains"][0]["msa"] != "empty"
    bad = app.validate({**p, "target": {**p["target"], "files": [{"filename": "rec.a3m", "content": ">q\nMKT\n"}]}})
    assert not bad["target"]["ok"] and "MSA query" in bad["target"]["error"]
    r = app.create_run(p)
    assert wait_state(app, r["id"], ("complete", "failed")) == "complete"
    assert (app.runs.run_dir(r["id"]) / "inputs" / "tf_rec.a3m").is_file()
    assert app.run_inputs(r["id"])["target"]["bundle"] is True


def test_saved_target_from_yaml_with_files(app):
    doc = f"version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: {PROT}\n      msa: rec.a3m\n"
    with pytest.raises(ApiError) as ex:
        app.prepare_target({"name": "Y", "source": {"kind": "file", "filename": "t.yaml", "content": doc}})
    assert "add rec.a3m" in ex.value.message
    t = app.prepare_target({"name": "Y", "source": {"kind": "file", "filename": "t.yaml", "content": doc},
                            "files": [{"filename": "rec.a3m", "content": MSA}]})
    assert t["msa"]["mode"] == "file" and t["n_residues"] == len(PROT)
    r = app.create_run({"target": {"kind": "saved", "id": t["id"]}, "ligands": {"kind": "file", "filename": "l.csv", "content": LIGS},
                        "options": {"preset": "standard"}})
    assert wait_state(app, r["id"], ("complete", "failed")) == "complete"


def test_csv_by_ids_keeps_whole_run_ranks(http_gui, app):
    t = app.prepare_target({"name": "K", "source": {"kind": "file", "filename": "x.cif", "content": STRUCT.read_text()}})
    r = app.create_run({"target": {"kind": "saved", "id": t["id"]}, "ligands": {"kind": "file", "filename": "l.csv", "content": LIGS},
                        "options": {"preset": "standard"}})
    assert wait_state(app, r["id"], ("complete", "failed")) == "complete"
    rows = app.runs.results(r["id"], "prob")["rows"]
    keep = rows[1]["record_id"]
    s, _, b = call(http_gui, "GET", f"/api/runs/{r['id']}/results.csv?rank_by=prob&ids={keep}")
    lines = b.decode().strip().splitlines()
    assert s == 200 and len(lines) == 2 and lines[0].startswith("rank,") and lines[1].split(",")[0] == str(rows[1]["rank"])
    whole = call(http_gui, "GET", f"/api/runs/{r['id']}/results.csv?rank_by=prob")[2].decode().strip().splitlines()
    assert len(whole) == len(rows) + 1
    cmp_run = app.recheck(r["id"], {"ligand_ids": [rows[0]["ligand_id"]]})
    assert wait_state(app, cmp_run["id"], ("complete", "failed")) == "complete"
    c = app.compare(cmp_run["id"])
    assert (c["n_original"], c["n_recheck"]) == (len(rows), 1)
    assert app.runs.read_meta(cmp_run["id"])["name"].startswith("Recheck · ")


def test_leadopt_marks_charge_and_stereo_changes_inside_the_common_part():
    from boltz_mlx.gui.leadopt import _changed_sites
    from rdkit import Chem

    ref, charged = Chem.MolFromSmiles("CC[C@H](N)C(=O)O"), Chem.MolFromSmiles("CC[C@@H]([NH3+])C(=O)O")
    match = charged.GetSubstructMatch(Chem.MolFromSmarts("CCC(N)C(=O)O"))
    rmatch = ref.GetSubstructMatch(Chem.MolFromSmarts("CCC(N)C(=O)O"))
    sites = _changed_sites(charged, ref, list(zip(match, rmatch)))
    assert sites["atoms"] >= {2, 3}    # the stereocentre (R/S) and the charged nitrogen
    same = _changed_sites(ref, Chem.MolFromSmiles("CC[C@H](N)C(=O)O"), list(zip(range(7), range(7))))
    assert not same["atoms"] and not same["bonds"]
