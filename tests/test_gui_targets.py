"""GUI affinity-screening backend: the three GUI presets (Standard / Reference (strict) / Fast, disabled), the target library
(prepare / list / get / delete, a saved target as a screen's target), measured per-ligand times, the results CSV of a
structure job, the export zip and the PDB conversion of poses.

CPU only: the screen and predict subprocesses are the fake ones (tests/gui_fake_screen.py, tests/gui_fake_predict.py).
"""

from __future__ import annotations

import http.client
import io
import json
import sys
import tempfile
import threading
import time
import zipfile
from pathlib import Path

import gemmi
import pytest

from boltz_mlx.gui import presets as GP
from boltz_mlx.gui.api import GuiApp
from boltz_mlx.gui.common import ApiError
from boltz_mlx.gui.exports import conversions, to_pdb
from boltz_mlx.gui.server import make_server

ROOT = Path(__file__).resolve().parents[1]
FAKE_SCREEN = ROOT / "tests" / "gui_fake_screen.py"
FAKE_PREDICT = ROOT / "tests" / "gui_fake_predict.py"
STRUCT = ROOT / "docs" / "viewer" / "structures" / "8j9r_mlx.cif"  # one protein chain, 74 residues (an MLX prediction)
PROT = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQW"
LIGS = "id,smiles\na,CCO\nb,c1ccccc1O\nc,CC(=O)Nc1ccc(O)cc1\n"


@pytest.fixture
def env(tmp_path, monkeypatch):
    w, c = tmp_path / "weights", tmp_path / "cache"
    w.mkdir()
    (c / "mols").mkdir(parents=True)
    for code in ("SAH", "SEP", "TPO"):   # empty stand-ins: the GUI checks only that a code is in the dictionary
        (c / "mols" / f"{code}.pkl").write_bytes(b"")
    for n in ("conf", "aff"):
        (w / f"boltz2_{n}.safetensors").write_bytes(b"")
        (w / f"boltz2_{n}.json").write_text("{}")
    (c / "ccd.pkl").write_bytes(b"")
    monkeypatch.setenv("GUI_FAKE_DELAY", "0")
    monkeypatch.setenv("BOLTZ_MLX_GUI_CHIP", "Apple M5 Pro")
    return {"weights": str(w), "cache": str(c)}


@pytest.fixture
def app(tmp_path, env):
    a = GuiApp(tmp_path / "runs", screen_cmd=[sys.executable, str(FAKE_SCREEN), "screen"],
               predict_cmd=[sys.executable, str(FAKE_PREDICT), "predict"], token="t0k-targets", **env)
    yield a
    for rid in a.runs.adopt_note():
        try:
            a.runs.stop(rid, force=True)
        except ApiError:
            pass


def wait_state(app, rid, states, timeout=60.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = app.runs.state(rid)
        if s in states:
            return s
        time.sleep(0.1)
    raise AssertionError(f"{rid} never reached {states} (now {app.runs.state(rid)})")


def a3m(seq, n=3):
    return "".join(f">s{i}\n{seq}\n" for i in range(n))


# --------------------------------------------------------------------------------------------------------- presets
def test_gui_presets_map_onto_the_cli_presets():
    assert [p["key"] for p in GP.GUI_PRESETS] == ["standard", "reference"]  # Fast is gone: its composition is Standard
    assert GP.to_cli("standard", "strict") == "screen" and GP.to_cli("reference", "strict") == "strict"  # NAX (test pin)
    assert GP.to_cli("turbo", "strict") == "turbo" and GP.to_cli(None, "strict") == "strict"  # CLI names pass through
    assert GP.to_cli("fast", "strict") == "fast"  # no longer a GUI key: a CLI name, rejected by the caller (unknown)
    std, ref = GP.payload("Apple M5 Max")
    assert std["status"] == "validated" and std["label"] == "Fast" and std["tooltip"] == "Matches 200-step output in our accuracy checks."
    assert std["default_screen"] and std["default_predict"] and not ref["default_predict"] and std["cli"] == "screen" and "20 structure steps" in std["summary"]
    assert "10 vs 3" in std["note"] and "non_nax" not in std and std["line"].startswith("Fast:")  # the two untested differences, plainly
    assert ref["label"] == "Full" and ref["status"] == "validated"
    assert ref["time_note"] == "~30 s per ligand (measured, ~360-residue target)"
    assert ref["measured"]["lo"] == 29.7 and "boltz-val-strict-trial-20261029a.log" in ref["measured"]["source"]
    # Standard's time = what a full screen takes (rung 5 G1: 4 cold 199-ligand screens), the 5-ligand measure secondary
    assert (std["measured"]["lo"], std["measured"]["hi"], std["measured"]["pooled"]) == (6.32, 7.21, 6.84)
    assert "g1_t400_20261029a" in std["measured"]["source"] and std["time_note"].startswith("~6.8 s per ligand (measured")
    assert "AC power" in std["time_note"] and "5.4 s" in std["time_note"] and std["measured"]["warm"]["lo"] == 5.38
    assert GP.payload("Apple M1")[0]["measured"] is None and GP.payload("Apple M1")[0]["time_note"] == "not measured on this chip yet"
    assert GP.measured("Apple M5 Pro", "screen") is None  # the 15.5 s there was the earlier 50 + 200 settings


def test_run_label_names_a_screen_by_what_it_ran():
    cfg = {"preset": "screen", "steps": 20, "recycling": 3, "affinity_steps": 25}
    assert GP.run_label(cfg) == "Fast"
    assert GP.run_label({**cfg, "steps": 50, "affinity_steps": 200}) == "screen, 50 + 200 steps (Fast is 20 + 25)"
    assert GP.run_label({**cfg, "recycling": 1}) == "screen, 20 + 25 steps, 1 recycle (Fast is 20 + 25, 3 recycles)"
    assert GP.run_label({"preset": "strict", "steps": 200, "recycling": 3, "affinity_steps": 200}) == "Full"
    assert GP.run_label({"preset": "strict", "steps": 100, "recycling": 3, "affinity_steps": 200}) == (
        "strict, 100 + 200 steps (Full is 200 + 200)")
    assert GP.run_label({}, "screen") == "Fast" and GP.run_label(None) is None  # not started yet: the preset chosen
    assert GP.run_label({"preset": "turbo", "steps": 50, "recycling": 3, "affinity_steps": 200}) == "turbo"
    assert GP.run_label({**cfg, "preset": "screen-tensorops"}) == "screen-tensorops"  # the M1-M4 f16 path: not Standard


def test_standard_runs_strict_without_nax():
    # project decision: on M1-M4 (tensorops / portable kernel path) the GUI's Standard runs strict, and says so
    for path in ("tensorops", "portable"):
        assert GP.to_cli("standard", "strict", path=path) == "strict" and GP.to_cli(None, "standard", path=path) == "strict"
        std = GP.payload("Apple M1", path=path)[0]
        assert std["key"] == "standard" and std["cli"] == "strict" and std["status"] == "validated"
        assert not std["default_screen"] and not std["default_predict"]   # M1-M4 start on Full; Fast stays selectable
        assert GP.default_key(path) == "reference" and GP.default_key("nax") == "standard"
        assert "Validated on M5" in std["tooltip"] and std["summary"].startswith("on this Mac: fp32") and "--preset screen" in std["note"]
    assert GP.to_cli("screen", "standard", path="tensorops") == "screen"  # the f16 path stays available by name


def test_info_and_options_use_the_gui_presets(app):
    info = app.info()
    assert info["default_screen_preset"] == "standard" and info["default_predict_preset"] == "standard"  # Fast: the default on NAX Macs
    assert [p["key"] for p in info["gui_presets"]] == ["standard", "reference"]
    assert info["default_preset"] == "screen"  # the CLI default on this kernel path (NAX, test pin)
    p = {"target": {"kind": "text", "content": PROT}, "msa": {"mode": "empty"},
         "ligands": {"kind": "file", "filename": "l.csv", "content": LIGS}, "options": {"preset": "standard"}}
    assert app.estimate(p)["estimate"]["preset"] == "screen"
    with pytest.raises(ApiError) as ex:
        app.create_run({**p, "options": {"preset": "fast"}})
    assert ex.value.status == 400


def test_measured_estimate_on_the_m5_max(app, monkeypatch):
    monkeypatch.setenv("BOLTZ_MLX_GUI_CHIP", "Apple M5 Max")
    p = {"target": {"kind": "text", "content": PROT}, "msa": {"mode": "empty"},
         "ligands": {"kind": "file", "filename": "l.csv", "content": LIGS}, "options": {"preset": "standard"}}
    est = app.estimate(p)
    e = est["estimate"]
    assert e["available"] and e["measured"]["runs"] == [6.68, 7.12, 7.21, 6.32]
    assert (e["per_ligand_min_s"], e["per_ligand_max_s"], e["total_lo_s"], e["total_hi_s"]) == (6.32, 7.21, 19, 22)
    assert [l for l in est["lines"] if l.startswith("ESTIMATE")][0].endswith("total ~19-22 s for 3 ligands")
    assert "measured on this chip" in e["caveat"]
    r = app.estimate({**p, "options": {"preset": "reference"}})["estimate"]
    assert r["available"] and r["measured"]["runs"] == [31.5] and (r["total_lo_s"], r["total_hi_s"]) == (89, 94)
    # said once: it read "From whole ligands measured on this chip (measured on this chip: 4 full cold screens ...)"
    assert not any(m["what"].startswith("measured") for m in GP.MEASURED.values())
    assert e["caveat"].startswith("From whole ligands measured on this chip (4 full cold screens of 199 ligands")
    for text in (e["caveat"], r["caveat"], *est["lines"]):
        assert text.count("on this chip") <= 1, text
    monkeypatch.setattr(GP, "MEASURING", {("Apple M5 Max", "turbo"): "measuring (a trial is running)"})
    t = app.estimate({**p, "options": {"preset": "turbo"}})["estimate"]
    assert not t["available"] and t["measuring"].startswith("measuring") and "measuring" in t["caveat"]


# --------------------------------------------------------------------------------------------------- target library
def test_prepare_from_sequence_with_msa(app):
    t = app.prepare_target({"name": "Kinase X", "source": {"kind": "text", "content": ">kx\n" + PROT},
                            "msa": {"mode": "file", "filename": "../kx.a3m", "content": a3m(PROT)}})
    assert t["name"] == "Kinase X" and t["n_residues"] == len(PROT) and t["msa"]["rows"] == 3 and t["template"] is None
    d = app.targets.root / t["id"]
    assert sorted(p.name for p in d.iterdir()) == ["msa.a3m", "target.json", "target.yaml"]
    assert app.get_target(t["id"])["sequence_sha"] == t["sequence_sha"]
    assert [x["id"] for x in app.list_targets()] == [t["id"]]
    assert app.runs.list_runs() == []  # the library (runs/.targets) is not listed as a run


def test_prepare_from_structure_keeps_a_template(app):
    t = app.prepare_target({"name": "from cif", "source": {"kind": "file", "filename": "x.cif", "content": STRUCT.read_text()}})
    assert t["n_residues"] == 74 and t["template"]["chain"] == "A" and t["source"]["chain"] == "A"
    d = app.targets.root / t["id"]
    st = gemmi.read_structure(str(d / "template.cif"))
    assert len(st) == 1 and [c.name for c in st[0]] == ["A"]
    y = (d / "target.yaml").read_text()
    assert "templates:" in y and "cif: template.cif" in y and "msa: empty" in y
    pdb = gemmi.read_structure(str(STRUCT)).make_pdb_string()
    t2 = app.prepare_target({"name": "from pdb", "template": False, "source": {"kind": "file", "filename": "x.pdb", "content": pdb}})
    assert t2["template"] is None and t2["n_residues"] == 74 and not (app.targets.root / t2["id"] / "template.cif").exists()


@pytest.mark.parametrize("payload,needle", [
    ({"source": {"kind": "text", "content": "MKVJJ"}}, "not an amino-acid sequence"),
    ({"source": {"kind": "text", "content": ""}}, "no content"),
    ({"source": {"kind": "text", "content": PROT}, "msa": {"mode": "server"}}, "no MSA server"),
    ({"source": {"kind": "text", "content": f">a\n{PROT}\n>b\n{PROT}\n"}, "msa": {"mode": "file", "filename": "m.a3m",
      "content": a3m(PROT)}}, "single protein chain"),
    ({"source": {"kind": "file", "filename": "t.yaml", "content": f"sequences:\n  - protein:\n      id: A\n      sequence: {PROT}\n"
                 "      msa: /etc/passwd\n"}}, "add passwd"),   # a YAML naming a file needs it attached, never read from disk
    ({"source": {"kind": "file", "filename": "x.cif", "content": STRUCT.read_text()}, "chain": "Z"}, "no protein chain 'Z'"),
])
def test_prepare_rejects_bad_input(app, payload, needle):
    with pytest.raises(ApiError) as ex:
        app.prepare_target(payload)
    assert ex.value.status == 400 and needle in ex.value.message
    assert app.list_targets() == [] and not any(p.name.startswith(".prep-") for p in app.targets.root.glob(".*"))


def test_delete_and_bad_ids(app):
    t = app.prepare_target({"source": {"kind": "text", "content": PROT}})
    for bad in ("../x", "A", "", ".targets"):
        with pytest.raises(ApiError):
            app.get_target(bad)
    assert app.delete_target(t["id"]) == {"deleted": t["id"]} and app.list_targets() == []
    with pytest.raises(ApiError) as ex:
        app.get_target(t["id"])
    assert ex.value.status == 404


def test_screen_from_a_saved_target(app):
    t = app.prepare_target({"name": "Kinase X", "source": {"kind": "file", "filename": "x.cif", "content": STRUCT.read_text()}})
    p = {"name": "saved", "target": {"kind": "saved", "id": t["id"]},
         "ligands": {"kind": "file", "filename": "l.csv", "content": LIGS}, "options": {"preset": "standard", "save_structures": "all"}}
    v = app.validate(p)
    assert v["ok"] and v["target"]["n_residues"] == 74
    r = app.create_run(p)
    assert wait_state(app, r["id"], ("complete", "failed", "interrupted")) == "complete"
    rd = app.runs.run_dir(r["id"])
    names = sorted(x.name for x in (rd / "inputs").iterdir())
    assert "target.yaml" in names and "target_template.cif" in names
    assert "cif: target_template.cif" in (rd / "inputs" / "target.yaml").read_text()
    meta = app.runs.read_meta(r["id"])
    assert meta["inputs"]["target"]["saved"] == t["id"] and "Kinase X" in meta["inputs"]["target"]["name"] and meta["options"]["preset"] == "screen"
    app.delete_target(t["id"])  # the run keeps its own copy
    assert (rd / "inputs" / "target_template.cif").is_file()


# ------------------------------------------------------------------------------------------------------------ exports
def test_conversions_follow_the_upstream_definition():
    c = conversions(-1.0)
    assert c["ic50_um_est"] == pytest.approx(0.1) and c["pic50_est"] == pytest.approx(7.0)
    assert c["upstream_6_minus_y_x1364_kcal_per_mol"] == pytest.approx(7.0 * 1.364)  # as upstream writes it, no sign added
    assert conversions(None)["pic50_est"] is None


def test_to_pdb_converts_mmcif():
    text = to_pdb(STRUCT)
    assert text.count("\nATOM") > 70 and gemmi.read_pdb_string(text)[0]["A"][0].name


@pytest.fixture
def http_gui(app):
    srv = make_server(app, port=0)
    th = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    th.start()
    yield srv
    srv.shutdown()
    srv.server_close()
    th.join(5)


def call(srv, method, path, body=None):
    c = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=60)
    h = {"Host": f"127.0.0.1:{srv.port}", "X-Boltz-Token": srv.app.token}
    raw = None
    if body is not None:
        raw = json.dumps(body).encode()
        h["Content-Type"] = "application/json"
    c.request(method, path, body=raw, headers=h)
    r = c.getresponse()
    out = (r.status, {k.lower(): v for k, v in r.getheaders()}, r.read())
    c.close()
    return out


def test_http_targets_and_exports(http_gui, app):
    s, h, b = call(http_gui, "POST", "/api/targets", {"name": "T", "source": {"kind": "text", "content": PROT}})
    assert s == 201
    tid = json.loads(b)["id"]
    s, _, b = call(http_gui, "GET", "/api/targets")
    assert s == 200 and [t["id"] for t in json.loads(b)["targets"]] == [tid]
    assert call(http_gui, "GET", f"/api/targets/{tid}")[0] == 200
    s, _, b = call(http_gui, "GET", "/api/presets")
    assert [p["key"] for p in json.loads(b)["gui"]] == ["standard", "reference"]
    # screen from the saved target, then its CSV and zip
    r = app.create_run({"target": {"kind": "saved", "id": tid}, "ligands": {"kind": "file", "filename": "l.csv", "content": LIGS},
                        "options": {"preset": "standard", "save_structures": "all"}})
    assert wait_state(app, r["id"], ("complete", "failed")) == "complete"
    s, h, b = call(http_gui, "GET", f"/api/runs/{r['id']}/export.zip")
    assert s == 200 and h["content-type"] == "application/zip" and "attachment" in h["content-disposition"]
    z = zipfile.ZipFile(io.BytesIO(b))
    names = z.namelist()
    assert any(n.endswith("/results.csv") for n in names) and any(n.endswith("/README.txt") for n in names)
    assert sum("/structures/" in n for n in names) == 3
    assert not list(Path(tempfile.gettempdir()).glob(".export-*.zip"))  # the server deletes the temporary zip
    # a structure job: one-row CSV with the conversions, pose as PDB on request, zip with model + JSON
    pj = app.create_run({"kind": "predict", "name": "u", "chains": [{"type": "protein", "id": "A", "sequence": PROT},
                         {"type": "ligand", "id": "B", "smiles": "CC(=O)Nc1ccc(O)cc1"}], "affinity": {"binder": "B"},
                         "options": {"preset": "reference"}})
    assert wait_state(app, pj["id"], ("complete", "failed")) == "complete"
    assert app.runs.read_meta(pj["id"])["options"]["preset"] == "strict"
    s, h, b = call(http_gui, "GET", f"/api/runs/{pj['id']}/results.csv")
    lines = b.decode().splitlines()
    assert s == 200 and lines[0].startswith("# affinity conversions are model estimates")
    row = dict(zip(lines[1].split(","), lines[2].split(",")))
    y = float(row["affinity_pred_value"])
    assert float(row["pic50_est"]) == pytest.approx(6 - y, abs=1e-4)
    assert float(row["upstream_6_minus_y_x1364_kcal_per_mol"]) == pytest.approx((6 - y) * 1.364, rel=1e-4)
    model = next(app.runs.predict_out(pj["id"]).rglob("*_model_0.cif"))
    model.write_text(STRUCT.read_text())  # the fake writes a minimal CIF; convert a real model file
    s, h, b = call(http_gui, "GET", f"/api/runs/{pj['id']}/prediction/pose?format=pdb")
    assert s == 200 and h["content-type"] == "chemical/x-pdb" and b"ATOM" in b and h["content-disposition"].endswith('.pdb"')
    assert call(http_gui, "GET", f"/api/runs/{pj['id']}/prediction/pose?format=xyz")[0] == 400
    s, h, b = call(http_gui, "GET", f"/api/runs/{pj['id']}/export.zip?format=pdb")
    names = zipfile.ZipFile(io.BytesIO(b)).namelist()
    assert s == 200 and any(n.endswith("_model_0.pdb") for n in names) and any("affinity_" in n for n in names)
    assert call(http_gui, "POST", f"/api/targets/{tid}/delete", {})[0] == 200
    assert call(http_gui, "GET", f"/api/targets/{tid}")[0] == 404


# ------------------------------------------------------------------------------------------------- ligand import checks
def _lig_payload(spec):
    return {"target": {"kind": "text", "content": PROT}, "msa": {"mode": "empty"}, "ligands": spec, "options": {"preset": "standard"}}


def test_ligand_import_sdf_smi_and_pasted(app):
    from rdkit import Chem

    buf = io.StringIO()
    w = Chem.SDWriter(buf)
    for name, smi in (("phenol", "c1ccccc1O"), ("big", "C" * 70), ("huge", "C" * 140)):
        m = Chem.MolFromSmiles(smi)
        m.SetProp("_Name", name)
        w.write(m)
    w.close()
    for spec in ({"kind": "file", "filename": "set.sdf", "content": buf.getvalue()},
                 {"kind": "file", "filename": "set.smi", "content": f"c1ccccc1O phenol\n{'C' * 70} big\n{'C' * 140} huge\n"},
                 {"kind": "text", "content": f"c1ccccc1O phenol\n{'C' * 70} big\n{'C' * 140} huge\n"}):
        res = app.validate(_lig_payload(spec))
        by = {r["id"]: r for r in res["ligands"]["rows"]}
        assert res["ok"] and set(by) == {"phenol", "big", "huge"}, spec["kind"]
        assert by["phenol"]["status"] == "ok"
        assert by["big"]["status"] == "warn" and "56" in by["big"]["warnings"][0]
        assert by["huge"]["status"] == "invalid" and "128" in by["huge"]["error"]
    bad = app.validate(_lig_payload({"kind": "file", "filename": "set.xlsx", "content": "x"}))
    assert not bad["ok"] and not bad["ligands"]["ok"] and ".sdf" in bad["ligands"]["error"]


def test_affinity_binder_size_checks(app):
    chains = lambda smi: [{"type": "protein", "id": "A", "sequence": PROT}, {"type": "ligand", "id": "B", "smiles": smi}]  # noqa: E731
    v = app.validate_predict({"kind": "predict", "chains": chains("C" * 70), "affinity": {"binder": "B"}})
    assert any("56" in w for w in v["warnings"])
    v = app.validate_predict({"kind": "predict", "chains": chains("C" * 140), "affinity": {"binder": "B"}})
    assert not v["ok"] and "128" in v["error"]


# ---------------------------------------------------------------------------------------------- re-check shortlist
def test_screen_presets_offer_reference_and_recheck_reruns_with_strict(app):
    keys = [g["key"] for g in GP.payload("Apple M5 Max") if g.get("screen") is not False]
    assert keys == ["standard", "reference"]  # Reference is a setup choice as well as the results table's Re-check action
    assert GP.BY_KEY["reference"]["predict_label"] == "Full"
    t = app.prepare_target({"name": "K", "source": {"kind": "file", "filename": "x.cif", "content": STRUCT.read_text()}})
    r = app.create_run({"name": "first pass", "target": {"kind": "saved", "id": t["id"]},
                        "ligands": {"kind": "file", "filename": "l.csv", "content": LIGS}, "options": {"preset": "standard", "seed": 5}})
    assert wait_state(app, r["id"], ("complete", "failed")) == "complete"
    for bad in ({}, {"ligand_ids": []}, {"ligand_ids": ["nope"]}, {"ligand_ids": "a"}):
        with pytest.raises(ApiError) as ex:
            app.recheck(r["id"], bad)
        assert ex.value.status == 400
    rc = app.recheck(r["id"], {"ligand_ids": ["c", "a", "a"]})
    assert wait_state(app, rc["id"], ("complete", "failed")) == "complete"
    meta = app.runs.read_meta(rc["id"])
    assert meta["recheck_of"] == r["id"] and meta["recheck_of_preset"] == "screen" and meta["name"] == "Recheck · first pass"
    assert meta["options"]["preset"] == "strict" and meta["options"]["seed"] == 5 and meta["options"]["steps"] is None
    rd = app.runs.run_dir(rc["id"]) / "inputs"
    assert (rd / "target_template.cif").is_file() and (rd / "ligands.csv").read_text().splitlines() == ["id,smiles", "c,CC(=O)Nc1ccc(O)cc1", "a,CCO"]
    cmp = app.compare(rc["id"])
    assert cmp["original"] == r["id"] and cmp["original_preset"] == "screen" and cmp["preset"] == "strict"
    assert app.runs.detail(rc["id"])["preset_label"] == "Full"
    # the label is what the run ran (its screen_config.json), not only the preset asked for. (The fake screen subprocess is
    # not pinned to the NAX path, so it records "screen-tensorops"; on the M5 Max a Standard run records "screen".)
    sc = app.runs.out_dir(r["id"]) / "screen_config.json"
    cfg = json.loads(sc.read_text())
    assert cfg["preset"] in ("screen", "screen-tensorops") and cmp["original_label"] == GP.run_label(cfg)
    sc.write_text(json.dumps({**cfg, "preset": "screen", "steps": 20, "recycling": 3, "affinity_steps": 25}))
    assert app.compare(rc["id"])["original_label"] == "Fast" and app.runs.detail(r["id"])["preset_label"] == "Fast"
    # a screen run from before 2026-10-29 (CLI "screen" was f16 + 50 + 200 then) is not today's validated Standard
    sc.write_text(json.dumps({**cfg, "preset": "screen", "steps": 50, "recycling": 3, "affinity_steps": 200}))
    assert app.compare(rc["id"])["original_label"] == "screen, 50 + 200 steps (Fast is 20 + 25)"
    assert app.runs.detail(r["id"])["preset_label"] == "screen, 50 + 200 steps (Fast is 20 + 25)"
    assert sorted(x["ligand_id"] for x in cmp["rows"]) == ["a", "c"]
    for x in cmp["rows"]:
        assert x["orig_rank"] is not None and x["delta_affinity_probability_binary"] == pytest.approx(
            x["affinity_probability_binary"] - x["orig_affinity_probability_binary"])
    assert app.runs.detail(r["id"])["rechecks"] == [rc["id"]] and app.runs.detail(rc["id"])["recheck_of"] == r["id"]
    with pytest.raises(ApiError) as ex:
        app.compare(r["id"])
    assert ex.value.status == 404
    with pytest.raises(ApiError):  # a request body cannot make stage_inputs copy another run's files
        app.create_run({"target": {"kind": "rerun", "id": r["id"]}, "ligands": {"kind": "file", "filename": "l.csv", "content": LIGS}})


def test_recheck_refuses_runs_without_gui_inputs(app):
    rd = app.runs.root / "cli_run"
    (rd / "out").mkdir(parents=True)  # a command-line screen: no gui_run.json, nothing staged here
    with pytest.raises(ApiError) as ex:
        app.recheck("cli_run", {"ligand_ids": ["a"]})
    assert ex.value.status in (400, 404)


# ------------------------------------------------------------------------------------------------ lead optimisation
def _moved_copy(dst: Path) -> None:
    """STRUCT rotated 90 degrees about z and shifted, as an mmCIF with every category kept."""
    doc = gemmi.cif.read(str(STRUCT))
    tab = doc.sole_block().find("_atom_site.", ["Cartn_x", "Cartn_y", "Cartn_z"])
    for row in tab:
        x, y, z = float(row[0]), float(row[1]), float(row[2])
        row[0], row[1], row[2] = f"{-y + 11:.3f}", f"{x - 7:.3f}", f"{z + 3:.3f}"
    doc.write_file(str(dst))


def test_aligned_pose_and_mcs_depiction(tmp_path):
    from boltz_mlx.gui.leadopt import aligned_pose, depict_vs

    moved = tmp_path / "moved.cif"
    _moved_copy(moved)
    text, fmt_, rmsd, n = aligned_pose(moved, STRUCT)
    assert fmt_ == "mmcif" and rmsd < 0.01 and n == 74
    a = gemmi.make_structure_from_block(gemmi.cif.read_string(text).sole_block())
    b = gemmi.read_structure(str(STRUCT))
    assert a[0][0][5][0].pos.dist(b[0][0][5][0].pos) < 0.01
    assert gemmi.cif.read_string(text).sole_block().get_mmcif_category_names() == gemmi.cif.read(str(STRUCT)).sole_block().get_mmcif_category_names()
    svg, info = depict_vs("Cc1ccc(O)cc1Cl", "Cc1ccc(O)cc1")
    assert "<svg" in svg and info["changed_atoms"] == 1 and info["mcs_atoms"] == 8
    assert depict_vs("CCO", "CCO")[1]["changed_atoms"] == 0
    with pytest.raises(ApiError):
        depict_vs("not a smiles (", "CCO")


def test_http_leadopt_endpoints(http_gui, app):
    t = app.prepare_target({"name": "K", "source": {"kind": "file", "filename": "x.cif", "content": STRUCT.read_text()}})
    r = app.create_run({"target": {"kind": "saved", "id": t["id"]}, "ligands": {"kind": "file", "filename": "l.csv",
                        "content": "id,smiles\np,Cc1ccc(O)cc1\nq,Cc1ccc(O)cc1Cl\n"}, "options": {"preset": "standard", "save_structures": "all"}})
    assert wait_state(app, r["id"], ("complete", "failed")) == "complete"
    rows = {x["ligand_id"]: x["record_id"] for x in app.runs.results(r["id"], "prob")["rows"]}
    p, q = rows["p"], rows["q"]
    app.runs.pose_file(r["id"], p).write_text(STRUCT.read_text())  # the fake screen writes placeholder poses
    _moved_copy(app.runs.pose_file(r["id"], q))
    s, h, b = call(http_gui, "GET", f"/api/runs/{r['id']}/ligand/{q}/pose?align_to={p}")
    assert s == 200 and float(h["x-align-rmsd"]) < 0.01 and h["x-align-n"] == "74" and h["x-pose-format"] == "mmcif"
    s, h, b = call(http_gui, "GET", f"/api/runs/{r['id']}/ligand/{q}/2d.svg?ref={p}&theme=dark")
    assert s == 200 and h["content-type"] == "image/svg+xml" and h["x-changed-atoms"] == "1" and b"<svg" in b
    assert call(http_gui, "GET", f"/api/runs/{r['id']}/ligand/{q}/pose?align_to=nope")[0] == 404
    assert call(http_gui, "GET", f"/api/runs/{r['id']}/ligand/{q}/2d.svg?ref=..%2Fx")[0] in (400, 404)
