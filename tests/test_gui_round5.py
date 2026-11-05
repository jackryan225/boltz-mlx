"""GUI round 5: pose availability per result row, per-ligand raw PAE (route and zip), format-true pose downloads, a run's own
target bundle ('Use inputs' after the saved target was deleted), and advanced YAML input for structure jobs.

CPU only (the fake screen / predict subprocesses of tests/gui_fake_*.py).
"""

from __future__ import annotations

import io
import json
import zipfile

import gemmi
import numpy as np
import pytest
from test_gui_targets import (LIGS, PROT, STRUCT, app, call, env, http_gui, wait_state)  # noqa: F401  (fixtures)

from boltz_mlx.gui import predict as PR
from boltz_mlx.gui.common import ApiError
from boltz_mlx.gui.exports import pose_bytes


def _screen(app, **opts):
    t = app.prepare_target({"name": "K", "source": {"kind": "file", "filename": "x.cif", "content": STRUCT.read_text()}})
    r = app.create_run({"target": {"kind": "saved", "id": t["id"]}, "ligands": {"kind": "file", "filename": "l.csv", "content": LIGS},
                        "options": {"preset": "standard", "save_structures": "all", **opts}})
    assert wait_state(app, r["id"], ("complete", "failed")) == "complete"
    return t, r["id"]


def test_rows_say_whether_a_pose_was_retained(app):
    _, rid = _screen(app)
    rows = app.runs.results(rid, "prob")["rows"]
    assert len(rows) == 3 and all(r["pose"] is True for r in rows)
    rec, out = rows[0]["record_id"], app.runs.out_dir(rid)
    for f in [*(out / "structures").glob(f"[0-9][0-9][0-9][0-9]_{rec}.*"), *(out / "work" / "predictions" / rec).glob("*_model_0.*")]:   # the top-N copy and the work file
        f.unlink()
    again = {r["record_id"]: r["pose"] for r in app.runs.results(rid, "prob")["rows"]}
    assert again[rows[0]["record_id"]] is False and sum(again.values()) == 2


def test_ligand_pae_npz_route_and_zip(http_gui, app):
    _, rid = _screen(app)
    rec = app.runs.results(rid, "prob")["rows"][0]["record_id"]
    assert call(http_gui, "GET", f"/api/runs/{rid}/ligand/{rec}/pae.npz")[0] == 404   # not kept
    d = app.runs.out_dir(rid) / "work" / "predictions" / rec
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / f"pae_{rec}_model_0.npz", pae=np.arange(16, dtype=np.float32).reshape(4, 4))
    s, h, b = call(http_gui, "GET", f"/api/runs/{rid}/ligand/{rec}/pae.npz")
    assert s == 200 and h["content-type"] == "application/octet-stream" and ".npz" in h["content-disposition"]
    assert float(np.load(io.BytesIO(b))["pae"][3, 3]) == 15.0
    s, _, b = call(http_gui, "GET", f"/api/runs/{rid}/export.zip")
    z = zipfile.ZipFile(io.BytesIO(b))
    raw = [n for n in z.namelist() if "/raw/" in n]
    assert len(raw) == 1 and raw[0].endswith(f"pae_{rec}.npz")
    man = json.loads(z.read([n for n in z.namelist() if n.endswith("manifest.json")][0]))
    assert man["files"]["raw_arrays"] == [f"raw/pae_{rec}.npz"] and man["files"]["model_files"][rec]["pae"] == f"raw/pae_{rec}.npz"


def test_pose_download_is_the_format_asked_for(tmp_path):
    pdb = tmp_path / "p.pdb"
    pdb.write_text(gemmi.read_structure(str(STRUCT)).make_pdb_string())
    data, ext = pose_bytes(pdb, "mmcif")
    assert ext == "cif" and len(gemmi.make_structure_from_block(gemmi.cif.read_string(data.decode()).sole_block())[0]) == 1
    assert pose_bytes(pdb, None)[1] == "pdb" and pose_bytes(STRUCT, "pdb")[1] == "pdb" and pose_bytes(STRUCT, "mmcif")[1] == "cif"
    with pytest.raises(ApiError):
        pose_bytes(STRUCT, "xyz")


def test_a_runs_own_target_survives_the_saved_target(app):
    t, rid = _screen(app)
    app.delete_target(t["id"])
    p = {"target": {"kind": "run", "id": rid}, "ligands": {"kind": "file", "filename": "l.csv", "content": LIGS},
         "options": {"preset": "standard", "save_structures": "all"}}
    v = app.validate(p)
    assert v["ok"] and v["target"]["n_residues"] == 74
    assert app.run_inputs(rid)["target"]["bundle"] is True
    r = app.create_run({**p, "name": "again"})
    assert wait_state(app, r["id"], ("complete", "failed")) == "complete"
    names = {x.name for x in (app.runs.run_dir(r["id"]) / "inputs").iterdir()}
    assert "target_template.cif" in names and any(n.startswith("ligands") for n in names)
    with pytest.raises(ApiError):   # another run's id that does not exist
        app.validate({**p, "target": {"kind": "run", "id": "no-such-run"}}) and None
        app.create_run({**p, "target": {"kind": "run", "id": "no-such-run"}})


YAML = f"""version: 1
sequences:
  - protein:
      id: A
      sequence: {PROT}
      msa: ../somewhere/foo.a3m
  - ligand:
      id: B
      smiles: CC(=O)Nc1ccc(O)cc1
templates:
  - cif: /abs/path/tmpl.cif
constraints:
  - pocket:
      binder: B
      contacts: [[A, 10]]
properties:
  - affinity:
      binder: B
"""


def test_import_lists_what_a_yaml_needs():
    r = PR.import_yaml(YAML)
    assert r["needs"] == ["foo.a3m", "tmpl.cif"] and r["binder"] == "B"
    assert r["constraints"] and r["templates"] == [{"file": "tmpl.cif", "chain_id": "A", "template_id": None}]
    assert not any("constraints" in x or "templates" in x for x in r["skipped"])   # the form holds them


def test_yaml_input_runs_as_written(app):
    files = [{"filename": "foo.a3m", "content": f">q\n{PROT}\n>h\n{PROT}\n"}, {"filename": "tmpl.cif", "content": STRUCT.read_text()}]
    bad = app.validate_predict({"kind": "predict", "yaml": {"content": YAML, "files": files[:1]}, "options": {"preset": "reference"}})
    assert bad["ok"] is False and "tmpl.cif" in bad["error"]
    p = {"kind": "predict", "yaml": {"content": YAML, "files": files}, "options": {"preset": "reference"}}
    v = app.validate_predict(p)
    assert v["ok"] and v["affinity"] and v["binder"] == "B" and "inputs/files/tmpl.cif" in v["yaml"] and "constraints" in v["yaml"]
    assert not any("no MSA" in w for w in v["warnings"])
    r = app.create_run(p)
    assert wait_state(app, r["id"], ("complete", "failed")) == "complete"
    rd = app.runs.run_dir(r["id"])
    y = (rd / "inputs" / "job.yaml").read_text()
    assert str(rd / "inputs" / "files" / "tmpl.cif") in y and "pocket" in y and (rd / "inputs" / "files" / "foo.a3m").is_file()
    back = app.run_inputs(r["id"])
    assert back["yaml"]["filename"] == "job.yaml" and sorted(f["filename"] for f in back["yaml"]["files"]) == ["foo.a3m", "tmpl.cif"]
    again = app.validate_predict({"kind": "predict", "yaml": {"content": back["yaml"]["content"], "files": back["yaml"]["files"]}, "options": {"preset": "reference"}})
    assert again["ok"]


@pytest.mark.parametrize("spec,needle", [
    ({"content": ""}, "choose a Boltz YAML"),
    ({"content": ">a\nMKT\n"}, "choose a Boltz YAML"),
    ({"content": YAML, "files": [{"filename": "foo.a3m", "content": "x"}, {"filename": "evil.sh", "content": "x"}]}, "use .a3m"),
])
def test_yaml_input_rejects_bad_specs(app, spec, needle):
    out = app.validate_predict({"kind": "predict", "yaml": spec, "options": {"preset": "reference"}})
    assert out["ok"] is False and needle in out["error"]
