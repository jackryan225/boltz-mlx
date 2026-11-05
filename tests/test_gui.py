"""`boltz-mlx gui`: API, run lifecycle (start / progress / stop / resume / results / past runs), input validation, path safety,
bind address and request guards.

CPU only. The screen subprocess is tests/gui_fake_screen.py: the real `screen` parser and runner with the fake engine
(tests/screen_fakes.py), so start / stop (real SIGTERM) / resume run for real, without checkpoints and without the GPU.
"""

from __future__ import annotations

import csv
import hashlib
import http.client
import io
import json
import os
import re
import signal
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

from boltz_mlx.gui import progress as P
from boltz_mlx.gui.api import GuiApp
from boltz_mlx.gui.common import LIMITS, PRIVACY, ApiError, is_within, safe_filename, valid_record_id, valid_run_id
from boltz_mlx.gui.runs import META
from boltz_mlx.gui.server import STATIC, make_server
from boltz_mlx.screen import results as R
from boltz_mlx.screen.inputs import Ligand, load_target, validate_ligand
from boltz_mlx.screen.main import add_parser, run as screen_run
from boltz_mlx.screen.presets import PRESETS
from boltz_mlx.screen.runner import ScreenConfig, ScreenRunner
from screen_fakes import FakeEngine, FakeModel

ROOT = Path(__file__).resolve().parents[1]
FAKE = ROOT / "tests" / "gui_fake_screen.py"
PROT = ("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQW")
SMILES = ["CCO", "c1ccccc1O", "CC(=O)Nc1ccc(O)cc1", "CCN(CC)CC", "OC(=O)c1ccccc1", "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "CCOC(=O)C", "c1ccncc1"]


def ligand_csv(n=6, extra=()):
    rows = [(f"cpd{i}", SMILES[i % len(SMILES)]) for i in range(n)] + list(extra)
    return "id,smiles\n" + "".join(f"{a},{b}\n" for a, b in rows)


def payload(n=6, extra=(), **opt):
    o = {"preset": "strict", "chunk": 3, "save_structures": "2"}
    o.update(opt)
    return {"name": "unit test", "target": {"kind": "text", "content": PROT}, "msa": {"mode": "empty"},
            "ligands": {"kind": "file", "filename": "ligs.csv", "content": ligand_csv(n, extra)}, "options": o}


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    FakeModel.reset()
    # time estimates come from a per-chip table (boltz_mlx/gui/estimates.py): pin the chip so the tests do not depend on
    # the Mac they run on (tests/test_gui_casona.py covers the other chips)
    monkeypatch.setenv("BOLTZ_MLX_GUI_CHIP", "Apple M5 Pro")
    yield
    FakeModel.reset()


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Dummy weights + cache so the pre-flight check passes; the fake screen needs neither."""
    w, c = tmp_path / "weights", tmp_path / "cache"
    w.mkdir()
    (c / "mols").mkdir(parents=True)
    for n in ("conf", "aff"):
        (w / f"boltz2_{n}.safetensors").write_bytes(b"")
        (w / f"boltz2_{n}.json").write_text("{}")
    (c / "ccd.pkl").write_bytes(b"")
    monkeypatch.setenv("GUI_FAKE_DELAY", "0")
    return {"weights": str(w), "cache": str(c)}


def make_app(tmp_path, env, name="runs", **kw):
    return GuiApp(tmp_path / name, screen_cmd=[sys.executable, str(FAKE), "screen"], token="t0k-9f3a1c7e5b", **env, **kw)


@pytest.fixture
def app(tmp_path, env):
    a = make_app(tmp_path, env)
    yield a
    for rid in a.runs.adopt_note():  # never leave a screen behind
        try:
            a.runs.stop(rid, force=True)
        except ApiError:
            pass


def wait_for(cond, timeout=60.0, every=0.1, what="condition"):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = cond()
        if v:
            return v
        time.sleep(every)
    raise AssertionError(f"timed out waiting for {what}")


def wait_state(app, rid, states, timeout=90.0):
    states = (states,) if isinstance(states, str) else tuple(states)
    return wait_for(lambda: app.runs.state(rid) in states and app.runs.state(rid), timeout, what=f"{rid} in {states}")


# ---------------------------------------------------------------------------------------------- common helpers
def test_ids_and_names():
    assert valid_run_id("20261028-101500-tyk2") and not valid_run_id("..") and not valid_run_id("a/b") and not valid_run_id(".hidden")
    assert not valid_run_id("a..b") and not valid_run_id("") and not valid_run_id(None)
    assert valid_record_id("cpd_1-a.b") and not valid_record_id("..") and not valid_record_id(".x") and not valid_record_id("a/b")
    assert not valid_record_id("a b") and not valid_record_id("")
    assert safe_filename("../../etc/passwd") == ("passwd", "")
    assert safe_filename("C:\\Users\\x\\lig.CSV") == ("lig.CSV", ".csv")
    assert safe_filename("a b$%.sdf")[1] == ".sdf" and "$" not in safe_filename("a b$%.sdf")[0]
    assert safe_filename(None) == ("upload", "")


def test_privacy_statement_matches_researcher_doc():
    doc = (ROOT / "docs" / "researcher.md").read_text()
    assert PRIVACY.split(". ")[0] in doc and "Ligand structures never leave the Mac" in PRIVACY


# ------------------------------------------------------------------------------------------------ presets / info
def test_presets_payload(app):
    ps = {p["name"]: p for p in app.presets_payload()}
    assert set(ps) >= {"strict", "turbo", "f16", "screen"}
    for n in ("strict", "screen"):
        assert ps[n]["validated"] and ps[n]["status"] == "validated" and ps[n]["caution"] is None
    assert ps["screen"]["default"] and not ps["strict"]["default"] and ps["strict"]["env"] == {}  # NAX path (test pin)
    for n in ("turbo", "f16"):
        assert not ps[n]["validated"] and ps[n]["status"] == "provisional" and "Not a gated preset" in ps[n]["caution"]
        assert ps[n]["note"] == PRESETS[n].note
    assert [p["name"] for p in app.presets_payload()][:2] == ["screen", "strict"]  # validated first, the default first
    assert (ps["screen"]["steps"], ps["screen"]["affinity_steps"], ps["screen"]["affinity_samples"]) == (20, 25, 5)


def test_info_checks_and_problems(tmp_path, env):
    i = make_app(tmp_path, env).info()
    assert i["problems"] == [] and i["checks"]["weights"]["conf"] and i["checks"]["cache"]["mols"]
    assert i["privacy"] == PRIVACY and i["molstar"]["version"] == "4.5.0" and i["molstar"]["license"] == "MIT"
    bad = GuiApp(tmp_path / "r2", screen_cmd=[sys.executable, str(FAKE)], cache=str(tmp_path / "nocache"), weights=str(tmp_path / "nowt"))
    pr = bad.info()["problems"]
    assert [x.split(" missing")[0] for x in pr] == ["Structure weights", "Affinity weights", "Molecule dictionary"]
    with pytest.raises(ApiError) as ex:
        bad.create_run(payload())
    assert ex.value.status == 400 and "Structure weights missing" in ex.value.message


def test_setup_problems_follow_the_workflow(tmp_path, env):
    """Structure-only jobs need no affinity weights; the model JSON files are checked, each named; a missing ccd.pkl
    (Boltz-1 only) is not a problem."""
    app = make_app(tmp_path, env)
    (Path(env["weights"]) / "boltz2_aff.safetensors").unlink()
    (Path(env["weights"]) / "boltz2_conf.json").unlink()
    (Path(env["cache"]) / "ccd.pkl").unlink()
    by = app.info()["problems_by"]
    assert [x.split(" missing")[0] for x in by["predict"]] == ["Structure weights"] and "boltz2_conf.json" in by["predict"][0]
    assert [x.split(" missing")[0] for x in by["affinity"]] == ["Structure weights", "Affinity weights"]
    assert [x.split(" missing")[0] for x in by["screen"]] == ["Structure weights", "Affinity weights"]
    (Path(env["weights"]) / "boltz2_conf.json").write_text("{}")
    assert app.info()["problems_by"]["predict"] == [] and "boltz2_aff.safetensors" in app.info()["problems_by"]["screen"][0]


# ------------------------------------------------------------------------------------------------- validation
def test_validate_flags_invalid_oversize_and_salts(app):
    extra = [("bad_smiles", "C1CC"), ("huge", "C" * 140), ("big", "C" * 70), ("salt", "CCN.Cl"), ("cpd0", "CCCC")]
    res = app.validate(payload(4, extra))
    assert res["ok"] and res["target"]["ok"] and res["ligands"]["ok"]
    t = res["target"]
    assert t["n_residues"] == len(PROT) and t["ligand_chain"] == "L" and t["msa_source"] == "empty"
    assert t["chains"][0]["type"] == "protein" and t["chains"][0]["length"] == len(PROT)
    c = res["ligands"]["counts"]
    assert c["total"] == 9 and c["invalid"] == 3 and c["duplicates"] == 1 and c["large"] == 1 and c["warn"] >= 2
    assert c["valid"] == 6 and not c["truncated"]
    by = {r["id"]: r for r in res["ligands"]["rows"]}
    assert by["bad_smiles"]["status"] == "invalid" and "parse" in by["bad_smiles"]["error"]
    assert by["huge"]["status"] == "invalid" and "128" in by["huge"]["error"] and by["huge"]["size"] == "too_large"
    assert by["big"]["status"] == "warn" and by["big"]["size"] == "large" and "56" in by["big"]["warnings"][0]
    assert by["salt"]["status"] == "warn" and "fragments" in by["salt"]["warnings"][0]
    assert by["cpd1"]["status"] == "ok" and by["cpd1"]["n_atoms"] == 7
    # per-preset time estimate for this target + median ligand
    pt = res["preset_times"]
    assert set(pt) >= {"strict", "screen"} and pt["strict"] > pt["screen"] > 0


def test_validate_sections_are_independent(app):
    bad_t = {"target": {"kind": "text", "content": "not a sequence 123 !!"}, "ligands": payload()["ligands"]}
    r = app.validate(bad_t)
    assert not r["target"]["ok"] and "amino-acid" in r["target"]["error"] and r["ligands"]["ok"] and not r["ok"]
    only_t = app.validate({"target": payload()["target"], "msa": {"mode": "empty"}})
    assert only_t["target"]["ok"] and only_t["ligands"] is None and not only_t["ok"]
    bad_l = app.validate({"ligands": {"kind": "file", "filename": "x.csv", "content": "foo,bar\n1,2\n"}})
    assert not bad_l["ligands"]["ok"] and "SMILES column" in bad_l["ligands"]["error"]
    pasted = app.validate({"ligands": {"kind": "text", "content": "CCO ethanol\nc1ccccc1 benzene\n"}})
    assert [r["id"] for r in pasted["ligands"]["rows"]] == ["ethanol", "benzene"]
    csv_text = app.validate({"ligands": {"kind": "text", "content": "name,smiles\nx,CCO\n"}})
    assert csv_text["ligands"]["rows"][0]["id"] == "x"
    fasta = app.validate({"target": {"kind": "text", "content": ">a\nMKTAYIAK\n>b\nGGGS\n"}})
    assert fasta["target"]["ok"] and [c["ids"] for c in fasta["target"]["chains"]] == [["A"], ["B"]]
    yaml_t = app.validate({"target": {"kind": "file", "filename": "t.yaml", "content": "version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: MKTAYIAK\n      msa: empty\n"}})
    assert yaml_t["target"]["ok"] and yaml_t["target"]["msa_source"] == "empty"
    broken = app.validate({"target": {"kind": "file", "filename": "t.yaml", "content": "sequences: [unclosed"}})
    assert not broken["target"]["ok"]
    pocket = app.validate({"target": payload()["target"], "msa": {"mode": "empty"}, "options": {"pocket": ["A:5,A:9"]}})
    assert pocket["target"]["pocket"][0]["binder"] == "L"
    with pytest.raises(ApiError) as ex:
        app.validate({"target": payload()["target"], "options": {"pocket": ["Z:1; rm -rf"]}})
    assert ex.value.status == 400


def test_validate_rejects_bad_uploads(app):
    p = payload()
    for kind, fn in (("target", "t.exe"), ("ligands", "l.pdf")):
        q = json.loads(json.dumps(p))
        q[kind] = {"kind": "file", "filename": fn, "content": "x"}
        res = app.validate(q)
        assert not res[kind]["ok"] and "unsupported file type" in res[kind]["error"]
    q = json.loads(json.dumps(p))
    q["msa"] = {"mode": "file", "filename": "m.txt", "content": ">a\nA\n"}
    assert not app.validate(q)["target"]["ok"]
    big = {"ligands": {"kind": "file", "filename": "l.csv", "content": "x" * (LIMITS["ligands"] + 1)}}
    assert not app.validate(big)["ligands"]["ok"] and "too large" in app.validate(big)["ligands"]["error"]
    binary = {"ligands": {"kind": "file", "filename": "l.csv", "content": "a,b\n\x00\x01"}}
    assert "binary" in app.validate(binary)["ligands"]["error"]
    with pytest.raises(ApiError):
        app.validate({"options": {"preset": "nope"}})
    for bad in ({"chunk": 0}, {"chunk": "x"}, {"rank_by": "z"}, {"save_structures": -2}, {"seed": 1e12}, {"max_minutes": -1},
                {"ligand_chain": "--evil"}, {"id_col": "a\nb"}):
        with pytest.raises(ApiError):
            app.validate({"target": p["target"], "options": bad})


def test_uploaded_filename_cannot_choose_where_files_go(app):
    p = payload()
    p["ligands"]["filename"] = "../../../../../tmp/evil.csv"
    p["target"] = {"kind": "file", "filename": "..\\..\\x/../target.yaml", "content": "version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: MKTAYIAK\n      msa: empty\n"}
    r = app.create_run(p)
    rd = app.runs.run_dir(r["id"])
    assert sorted(x.name for x in (rd / "inputs").iterdir()) == ["ligands.csv", "target.yaml"]
    assert not Path("/tmp/evil.csv").exists()
    meta = json.loads((rd / META).read_text())
    assert meta["inputs"]["ligands"] == {"file": "inputs/ligands.csv", "name": "evil.csv"}
    wait_state(app, r["id"], "complete")


def test_estimate_matches_screen_dry_run(app, tmp_path, capsys):
    p = payload(5, [("bad", "xx(")], preset="screen", chunk=4)
    est = app.estimate(p)
    lig = tmp_path / "l.csv"
    lig.write_text(ligand_csv(5, [("bad", "xx(")]))
    ap = __import__("argparse").ArgumentParser()
    add_parser(ap.add_subparsers(dest="cmd"))
    a = ap.parse_args(["screen", f"--target={PROT}", f"--ligands={lig}", f"--out={tmp_path / 'o'}", "--preset=screen", "--chunk=4",
                       "--msa=empty", "--save-structures=2", "--dry-run"])
    lines = []
    assert screen_run(a, [], log=lines.append) == 0
    norm = lambda ls: [re.sub(r", (\d+) already done in [^,]+,", r", \1 already done,", x) for x in ls]  # noqa: E731
    assert est["lines"][0].startswith("target: pasted sequence  (117 polymer residues") and lines[0].startswith("target: sequence  (117 polymer residues")
    # the screen's own --dry-run plan, line for line, except the ESTIMATE line: the GUI's numbers come from the chip table
    ours, theirs = est["lines"][1:], norm(lines)[1:]
    assert [x for x in ours if not x.startswith("ESTIMATE")] == [x for x in theirs if not x.startswith("ESTIMATE")]
    eline = [x for x in ours if x.startswith("ESTIMATE")]
    assert len(eline) == 1 and "M5 Pro 24 GB, f16" in eline[0] and "s/ligand" in eline[0]
    e = est["estimate"]
    assert e["n_to_run"] == 5 and e["n_chunks"] == 2 and e["validated"] and e["total_s"] > 0 and "ESTIMATE" in e["caveat"]  # screen: gated
    assert 0 < e["w_struct"] < 0.5 and e["per_ligand_min_s"] <= e["per_ligand_median_s"] <= e["per_ligand_max_s"]
    assert not (tmp_path / "o").exists()  # nothing written
    with pytest.raises(ApiError):
        app.estimate({**payload(0), "ligands": {"kind": "text", "content": "xx(\n"}})


# ------------------------------------------------------------------------------------------------ run lifecycle
def test_run_lifecycle_with_fake_screen(app):
    r = app.create_run(payload(7, [("bad_smiles", "C1CC")]))
    rid = r["id"]
    assert r["state"] in ("running", "complete") and re.match(r"^\d{8}-\d{6}-unit-test", rid)
    assert wait_state(app, rid, ("complete", "failed", "interrupted")) == "complete"
    d = app.runs.detail(rid)
    pr = d["progress"]
    assert d["state"] == "complete" and pr["total"] == 8 and pr["done"] == 8 and pr["ok"] == 7 and pr["invalid"] == 1
    assert pr["remaining"] == 0 and pr["fraction"] == 1.0 and pr["phase"] == "finished" and d["can_resume"] is False
    assert pr["errors_total"] == 1 and pr["errors"][0]["ligand_id"] == "bad_smiles" and "parse" in pr["errors"][0]["error"]
    assert d["plan"]["to_run"] == 7 and d["plan"]["chunks"] == 3 and d["status"]["complete"] is True
    assert d["ligand_chain"] == "L" and d["config"]["preset"] == "strict" and d["meta"]["options"]["chunk"] == 3
    assert any("[status]" in ln for ln in d["log_tail"])
    # results: ranked, slim rows
    res = app.runs.results(rid, "prob")
    assert res["n"] == 8 and [r["rank"] for r in res["rows"]][:7] == list(range(1, 8)) and res["rows"][-1]["rank"] is None
    probs = [r["affinity_probability_binary"] for r in res["rows"][:7]]
    assert probs == sorted(probs, reverse=True)
    assert {"ligand_id", "smiles", "affinity_pred_value", "affinity_probability_binary", "iptm", "ligand_iptm", "complex_plddt", "status"} <= set(res["rows"][0])
    again = app.runs.results(rid, "prob", known_version=res["version"])
    assert again == {"version": res["version"], "unchanged": True}
    by_val = app.runs.results(rid, "value")
    vals = [r["affinity_pred_value"] for r in by_val["rows"][:7]]
    assert vals == sorted(vals) and by_val["version"] != res["version"]
    with pytest.raises(ApiError):
        app.runs.results(rid, "bogus")
    # csv: same columns as the screen's results.csv
    rows = list(csv.DictReader(io.StringIO(app.runs.csv(rid, "prob"))))
    assert list(rows[0]) == R.CSV_COLUMNS and len(rows) == 8
    disk = list(csv.DictReader(open(app.runs.out_dir(rid) / "results.csv")))
    assert [x["ligand_id"] for x in rows] == [x["ligand_id"] for x in disk]
    # poses: top-K copy in structures/, others from work/
    top = res["rows"][0]
    assert app.runs.pose_file(rid, top["record_id"]).parent.name == "structures"
    last_ok = res["rows"][6]
    assert app.runs.pose_file(rid, last_ok["record_id"]).parent.name == last_ok["record_id"]
    with pytest.raises(ApiError) as ex:
        app.runs.pose_file(rid, "bad_smiles")  # invalid ligand: no structure
    assert ex.value.status == 404
    det = app.runs.ligand_detail(rid, top["record_id"])
    assert det["pose"]["format"] == "mmcif" and det["row"]["ligand_id"] == top["ligand_id"] and "env" not in det["row"]
    assert det["affinity"] is None or "v" in det["affinity"]
    # 2D depiction
    svg = app.depict_ligand(rid, top["record_id"])
    assert svg.lstrip().startswith("<?xml") and "<svg" in svg and "</svg>" in svg
    assert "<svg" in app.depict_ligand(rid, top["record_id"], dark=True)
    # past runs
    runs = app.runs.list_runs()
    assert [x["id"] for x in runs] == [rid]
    s = runs[0]
    assert s["state"] == "complete" and s["n_ok"] == 7 and s["n_invalid"] == 1 and s["n_done"] == 7 and s["n_ligands"] == 7 and s["name"] == "unit test"
    assert s["preset"] == "strict" and s["rank_by"] == "prob" and s["target"] == "pasted sequence" and not s["external"]
    with pytest.raises(ApiError) as ex:
        app.start_existing(rid)
    assert ex.value.status == 409 and "complete" in ex.value.message
    # the subprocess log has both the header and the screen's lines; the launch records its pid and exit code
    meta = json.loads((app.runs.run_dir(rid) / META).read_text())
    assert meta["launches"][0]["exit_code"] == 0 and meta["launches"][0]["ended_at"] is not None
    assert "[plan]" in (app.runs.run_dir(rid) / "screen.log").read_text()


def test_stop_resume_is_graceful_and_exact(app, monkeypatch):
    monkeypatch.setenv("GUI_FAKE_DELAY", "0.25")
    # reference run (no stop), then a stopped + resumed run of the same inputs
    ref = app.create_run(payload(8))
    wait_state(app, ref["id"], "complete")
    want = {r["ligand_id"]: (r["affinity_pred_value"], r["affinity_probability_binary"]) for r in app.runs.results(ref["id"], "prob")["rows"]}
    r = app.create_run(payload(8))
    rid = r["id"]
    # wait for the first rows, then stop
    wait_for(lambda: app.runs.detail(rid)["progress"]["done"] >= 1, 60, what="first result")
    d = app.runs.detail(rid)
    assert d["state"] == "running" and d["can_stop"] and d["pid"] and d["progress"]["phase"] in ("structure", "affinity", "starting")
    assert d["progress"]["current"] is None or d["progress"]["current"]["phase"] in ("structure", "affinity")
    with pytest.raises(ApiError) as ex:  # one GPU job at a time
        app.create_run(payload(2))
    assert ex.value.status == 409 and rid in ex.value.message
    assert app.stop(rid)["state"] == "stopping"
    assert app.runs.detail(rid)["can_force_stop"]
    assert wait_state(app, rid, ("stopped", "complete", "failed", "interrupted")) == "stopped"
    d = app.runs.detail(rid)
    assert d["status"]["stopped_by_signal"] is True and d["status"]["complete"] is False and d["can_resume"] and not d["can_stop"]
    assert 0 < d["progress"]["done"] < 8 and d["progress"]["remaining"] > 0
    assert "[stop]" in d["stop_note"]
    n_before = d["progress"]["done"]
    # resume: only the remaining ligands run; results identical to the uninterrupted run
    assert app.start_existing(rid)["state"] in ("running", "complete")
    assert wait_state(app, rid, ("complete", "failed", "interrupted")) == "complete"
    d = app.runs.detail(rid)
    assert d["progress"]["done"] == 8 and d["launches"] == 2
    assert d["plan"]["already_done"] == n_before and d["plan"]["to_run"] == 8 - n_before
    rows = R.read_rows(app.runs.out_dir(rid) / "results.jsonl")
    ids = [x["ligand_id"] for x in rows]
    assert len(ids) == len(set(ids)) == 8
    got = {r["ligand_id"]: (r["affinity_pred_value"], r["affinity_probability_binary"]) for r in app.runs.results(rid, "prob")["rows"]}
    assert got == want
    # the progress of the second launch is read from its own part of the log
    assert "[stop]" not in " ".join(d["log_tail"])


def test_stop_when_idle_and_resume_when_running_are_409(app):
    r = app.create_run(payload(2))
    wait_state(app, r["id"], "complete")
    with pytest.raises(ApiError) as ex:
        app.stop(r["id"])
    assert ex.value.status == 409
    with pytest.raises(ApiError) as ex:
        app.start_existing("20990101-000000-nothing")
    assert ex.value.status == 404


def test_force_stop_keeps_rows_and_run_resumes(app, monkeypatch):
    monkeypatch.setenv("GUI_FAKE_DELAY", "0.3")
    rid = app.create_run(payload(6))["id"]
    wait_for(lambda: app.runs.detail(rid)["progress"]["done"] >= 1, 60, what="first result")
    app.stop(rid)
    time.sleep(0.15)
    try:
        app.stop(rid, force=True)  # 2nd SIGTERM: stop now (a no-op if the run already ended)
    except ApiError:
        pass
    wait_state(app, rid, ("stopped", "interrupted", "complete"))
    done = app.runs.detail(rid)["progress"]["done"]
    assert done >= 1
    monkeypatch.setenv("GUI_FAKE_DELAY", "0")
    if app.runs.state(rid) != "complete":
        assert app.runs.detail(rid)["can_resume"]
        app.start_existing(rid)
        assert wait_state(app, rid, ("complete", "failed")) == "complete"
    assert app.runs.detail(rid)["progress"]["ok"] == 6


def test_failed_run_shows_log_and_can_be_resumed(tmp_path, env):
    """A screen that crashes (here: a command that prints and exits 3) -> state failed, the log tail is shown, Resume allowed."""
    bad = GuiApp(tmp_path / "runs2", screen_cmd=[sys.executable, "-c", "import sys; print('boom: no weights'); sys.exit(3)"], token="t", **env)
    r = bad.create_run(payload(2))
    assert wait_state(bad, r["id"], ("failed", "interrupted", "complete")) == "failed"
    d = bad.runs.detail(r["id"])
    assert "boom: no weights" in d["run_error"] and d["can_resume"] and d["progress"]["done"] == 0
    assert bad.runs.list_runs()[0]["state"] == "failed"
    # the cause is fixed (a working command): Resume runs the same inputs to completion
    good = make_app(tmp_path, env, name="runs2")
    assert good.runs.state(r["id"]) == "failed"
    good.start_existing(r["id"])
    assert wait_state(good, r["id"], ("complete", "failed")) == "complete"
    assert good.runs.detail(r["id"])["progress"]["ok"] == 2 and good.runs.detail(r["id"])["launches"] == 2


def test_killed_screen_is_interrupted_and_resumes(app, monkeypatch):
    """kill -9 / a crash / a sleeping laptop: no final status is written; the run shows 'interrupted' and Resume finishes it."""
    monkeypatch.setenv("GUI_FAKE_DELAY", "0.3")
    rid = app.create_run(payload(8))["id"]
    wait_for(lambda: app.runs.detail(rid)["progress"]["done"] >= 1, 60, what="first result")
    os.kill(app.runs.detail(rid)["pid"], signal.SIGKILL)
    assert wait_state(app, rid, ("interrupted", "stopped", "failed", "complete")) == "interrupted"
    d = app.runs.detail(rid)
    assert d["can_resume"] and not d["can_stop"] and 1 <= d["progress"]["done"] < 8 and d["run_error"]
    monkeypatch.setenv("GUI_FAKE_DELAY", "0")
    app.start_existing(rid)
    assert wait_state(app, rid, ("complete", "failed")) == "complete"
    rows = R.read_rows(app.runs.out_dir(rid) / "results.jsonl")
    ids = [x["ligand_id"] for x in R.latest_rows(rows).values()]
    assert len(ids) == 8 and app.runs.detail(rid)["progress"]["ok"] == 8


def test_adopt_running_screen_after_gui_restart(tmp_path, env, monkeypatch):
    monkeypatch.setenv("GUI_FAKE_DELAY", "0.3")
    a = make_app(tmp_path, env)
    rid = a.create_run(payload(8))["id"]
    wait_for(lambda: a.runs.detail(rid)["progress"]["done"] >= 1, 60, what="first result")
    b = make_app(tmp_path, env)                       # a new server process on the same runs dir (no Popen object)
    assert b.runs.state(rid) == "running" and b.runs.adopt_note() == [rid]
    with pytest.raises(ApiError) as ex:
        b.create_run(payload(2))
    assert ex.value.status == 409
    assert b.stop(rid)["state"] == "stopping"
    wait_for(lambda: b.runs.state(rid) in ("stopped", "complete"), 60, what="stop")
    a.runs.procs[rid].wait(timeout=30)
    assert b.runs.state(rid) in ("stopped", "complete")
    monkeypatch.setenv("GUI_FAKE_DELAY", "0")
    if b.runs.state(rid) == "stopped":
        b.start_existing(rid)
        assert wait_state(b, rid, ("complete", "failed")) == "complete"


def test_recycled_pid_is_not_ours(tmp_path, env):
    a = make_app(tmp_path, env)
    rid = a.create_run(payload(2))["id"]
    wait_state(a, rid, "complete")
    meta = a.runs.read_meta(rid)
    meta["launches"][-1].update(pid=os.getpid(), ended_at=None, exit_code=None)   # some live process that is not a screen
    a.runs.write_meta(rid, meta)
    b = make_app(tmp_path, env)
    assert b.runs.state(rid) == "complete"                                        # command line does not name the run dir
    assert b.runs.adopt_note() == []


def test_external_cli_run_is_listed_read_only(tmp_path, env):
    out = tmp_path / "runs" / "cli_run"
    ligs = [validate_ligand(Ligand(f"c{i}", SMILES[i], i, f"c{i}")) for i in range(4)]
    cfg = ScreenConfig(
        out=out, preset=PRESETS["strict"], steps=200, recycling=3, affinity_steps=200, chunk=2, device="cpu", save_structures=1)
    ScreenRunner(cfg, load_target(PROT[:60], msa="empty"), ligs, FakeEngine(out / "work"), log=lambda *a: None).run()
    a = make_app(tmp_path, env)
    (a.runs.root / "not_a_run").mkdir()
    (a.runs.root / ".hidden").mkdir()
    runs = a.runs.list_runs()
    assert [r["id"] for r in runs] == ["cli_run"]
    s = runs[0]
    assert s["external"] and s["state"] == "complete" and s["n_ok"] == 4 and s["preset"] == "strict"
    d = a.runs.detail("cli_run")
    assert d["external"] and d["progress"]["ok"] == 4 and not d["can_resume"] and d["ligand_chain"] == "L"
    res = a.runs.results("cli_run", "value")
    assert res["n"] == 4
    assert a.runs.pose_file("cli_run", res["rows"][0]["record_id"]).is_file()
    with pytest.raises(ApiError) as ex:
        a.start_existing("cli_run")
    assert ex.value.status == 400


def test_export_zip_names_poses_by_the_requested_ranking(tmp_path, env):
    """The screen saves its poses as <rank>_<record> under its own ranking (here probability). Re-ranked by affinity value,
    the zip's results.csv changes order and the pose files must follow it: 0001_* is the CSV's first row (they kept the
    probability ranks before)."""
    import zipfile

    from boltz_mlx.gui.exports import build_zip

    out = tmp_path / "runs" / "cli_run"
    ligs = [validate_ligand(Ligand(f"c{i}", SMILES[i], i, f"c{i}")) for i in range(4)]
    cfg = ScreenConfig(out=out, preset=PRESETS["strict"], steps=200, recycling=3, affinity_steps=200, chunk=2, device="cpu",
                       save_structures=-1, rank_by="prob")
    ScreenRunner(cfg, load_target(PROT[:60], msa="empty"), ligs, FakeEngine(out / "work"), log=lambda *a: None).run()
    a = make_app(tmp_path, env)
    saved = sorted(p.name for p in (out / "structures").iterdir())
    names = {}
    for rb in ("prob", "value"):  # (the PDB variant names files the same way; the fake's placeholder poses do not convert)
        path, _ = build_zip(a.runs, "cli_run", None, rb)
        try:
            with zipfile.ZipFile(path) as z:
                rows = list(csv.DictReader(io.StringIO(z.read("cli_run/results.csv").decode())))
                poses = sorted(n.split("/")[-1] for n in z.namelist() if "/structures/" in n)
                readme = z.read("cli_run/README.txt").decode()
        finally:
            path.unlink()
        rank = {r["ligand_id"]: int(r["rank"]) for r in rows}
        assert len(poses) == 4 and all(p.endswith(".cif") for p in poses)
        for p in poses:
            rk, rec = p.split(".")[0].split("_", 1)
            assert int(rk) == rank[rec], (rb, p, rank)
        assert poses[0].split(".")[0] == f"0001_{rows[0]['ligand_id']}"
        assert ("ranked by probability" if rb == "prob" else "ranked by affinity value") in readme
        names[rb] = poses
    assert names["prob"] == saved                             # the screen's own ranking: names unchanged
    assert names["value"] != saved                            # the fake ranks value opposite to probability
    assert sorted(p.split("_", 1)[1] for p in names["value"]) == sorted(p.split("_", 1)[1] for p in saved)


# -------------------------------------------------------------------------------------------- progress parsing
def test_row_store_incremental_and_torn_lines(tmp_path):
    p = tmp_path / "results.jsonl"
    rs = P.RowStore(p)
    assert not rs.refresh() and rs.rows() == []
    R.append_row(p, {"ligand_id": "a", "status": "ok"})
    assert rs.refresh() and rs.counts()["ok"] == 1
    with open(p, "a") as f:
        f.write('{"ligand_id": "b", "status": "ok", "affinity')  # a writer in the middle of a line
    assert not rs.refresh() and len(rs.rows()) == 1
    with open(p, "a") as f:
        f.write('_pred_value": 1.5}\n{"ligand_id": "c", "status": "error"}\nnot json\n')
    assert rs.refresh() and rs.counts() == {"ok": 2, "error": 1, "invalid": 0}
    R.append_row(p, {"ligand_id": "c", "status": "ok"})  # a retried error row wins
    rs.refresh()
    assert rs.counts() == {"ok": 3, "error": 0, "invalid": 0}
    v = rs.version
    assert not rs.refresh() and rs.version == v
    p.write_text("")  # truncated / replaced
    rs.refresh()
    assert rs.rows() == [] and rs.version != v or rs.rows() == []


def test_parse_log_and_phase():
    lines = [
        "[target] warning: protein chain(s) ['A'] have no MSA: running single-sequence",
        "[plan] 100 ligands: 20 already done, 3 invalid, 1 id conflicts, 76 to run in 2 chunk(s) of <= 50; mode screen (NOT a gated preset)",
        "[chunk 2/2] 26 ligands", "[load] conf model 1.2s", "[start] structure my ligand 7/26", "[ok] x affinity_pred_value=1.0 0.0s",
        "[start] affinity LIG-9 3/26",
    ]
    info = P.parse_log(lines)
    assert info["plan"] == {"ligands": 100, "already_done": 20, "invalid": 3, "conflicts": 1, "to_run": 76, "chunks": 2, "chunk_size": 50,
                            "mode": "screen (NOT a gated preset)"}
    assert info["chunk"] == {"index": 2, "of": 2, "size": 26} and info["load"] == {"model": "conf", "seconds": 1.2}
    assert info["marker"] == {"phase": "affinity", "ligand_id": "LIG-9", "k": 3, "n": 26} and len(info["target_warnings"]) == 1
    assert P.parse_log(lines[:5])["marker"]["ligand_id"] == "my ligand"  # ids may contain spaces
    assert P.phase_of(info, True) == "affinity" and P.phase_of(P.parse_log([]), True) == "starting" and P.phase_of(info, False) == "idle"
    done = P.parse_log(lines + ['[status] {"complete": true}'])
    assert done["status"] == {"complete": True} and P.phase_of(done, False) == "finished"


def test_build_progress_eta_and_fraction():
    rows = [{"ligand_id": f"c{i}", "status": "ok", "seconds": 10.0, "t_done": 1000.0 + i} for i in range(10)]
    rows += [{"ligand_id": "e", "status": "error", "index": 3, "error": "boom", "error_phase": "affinity"},
             {"ligand_id": "i", "status": "invalid", "index": 4, "error": "bad", "error_phase": "input"}]
    launch = {"started_at": 100.0, "computed_at_start": 0}
    log = P.parse_log(["[start] structure c11 3/10"])
    pr = P.build_progress(total=40, n_valid=38, rows=rows, launch=launch, alive=True, log=log, struct_pending=5, w_struct=0.4,
                          est_total_s=999.0, now=400.0)
    assert pr["done"] == 12 and pr["ok"] == 10 and pr["error"] == 1 and pr["invalid"] == 1 and pr["remaining"] == 28
    assert pr["fraction_done"] == round(11 / 38, 4) and pr["fraction"] == round((11 + 0.4 * 5) / 38, 4)
    assert pr["s_per_ligand"] == pytest.approx(300 / 13, abs=0.01) and pr["eta_source"] == "measured"
    assert pr["eta_s"] == round((38 - 11 - 2) * 300 / 13) and pr["current"]["ligand_id"] == "c11" and pr["median_model_s"] == 10.0
    assert [e["ligand_id"] for e in pr["errors"]] == ["e", "i"] and pr["errors_total"] == 2
    early = P.build_progress(total=40, n_valid=38, rows=[], launch=launch, alive=True, log=log, struct_pending=0, w_struct=0.4,
                             est_total_s=500.0, now=101.0)
    assert early["eta_source"] == "estimate" and early["eta_s"] == 500 and early["s_per_ligand"] is None
    idle = P.build_progress(total=40, n_valid=38, rows=rows, launch=launch, alive=False, log=log, struct_pending=5, w_struct=0.4,
                            est_total_s=500.0, now=400.0)
    assert idle["eta_s"] is None and idle["current"] is None and idle["phase"] == "idle" and idle["structures_pending"] == 0


def test_struct_scan_counts_unfinished_structures(tmp_path):
    root = tmp_path / "predictions"
    for n in ("a", "b", "c", "d"):
        (root / n).mkdir(parents=True)
    for n in ("a", "b", "c"):
        (root / n / "screen_struct.json").write_text("{}")
    sc = P.StructScan()
    assert sc.pending(root, {"a"}) == 2           # b, c: structure done, no row yet (d has no sidecar)
    (root / "d" / "screen_struct.json").write_text("{}")
    assert sc.pending(root, {"a", "b"}) == 2      # c, d
    assert sc.pending(tmp_path / "missing", set()) == 0


# ------------------------------------------------------------------------------------------------------- HTTP
@pytest.fixture
def http_gui(tmp_path, env):
    app = make_app(tmp_path, env, name="hruns")
    srv = make_server(app, port=0)
    th = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    th.start()
    yield srv
    srv.shutdown()
    srv.server_close()
    th.join(5)
    for rid in app.runs.adopt_note():
        try:
            app.runs.stop(rid, force=True)
        except ApiError:
            pass


def call(srv, method, path, body=None, headers=None, token=True, raw=None, host=None):
    c = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=60)
    h = {"Host": host or f"127.0.0.1:{srv.port}"}
    if token:
        h["X-Boltz-Token"] = srv.app.token
    if body is not None and raw is None:
        raw = json.dumps(body).encode()
        h["Content-Type"] = "application/json"
    h.update(headers or {})
    c.request(method, path, body=raw, headers=h)
    r = c.getresponse()
    data = r.read()
    out = (r.status, dict((k.lower(), v) for k, v in r.getheaders()), data)
    c.close()
    return out


def jcall(*a, **kw):
    s, h, b = call(*a, **kw)
    try:
        return s, json.loads(b)
    except ValueError:
        return s, b


def test_http_end_to_end(http_gui):
    s, info = jcall(http_gui, "GET", "/api/info")
    assert s == 200 and info["runs_dir"].endswith("hruns") and info["privacy"] == PRIVACY and info["problems"] == []
    s, ps = jcall(http_gui, "GET", "/api/presets")
    assert s == 200 and ps["presets"][0]["name"] == "screen"  # validated and the default on the NAX path (test pin)
    s, v = jcall(http_gui, "POST", "/api/validate", payload(3))
    assert s == 200 and v["ok"] and v["ligands"]["counts"]["valid"] == 3
    s, e = jcall(http_gui, "POST", "/api/estimate", payload(3))
    assert s == 200 and e["estimate"]["n_to_run"] == 3
    s, r = jcall(http_gui, "POST", "/api/runs", payload(5))
    assert s == 201
    rid = r["id"]
    wait_for(lambda: jcall(http_gui, "GET", f"/api/runs/{rid}")[1]["state"] == "complete", 90, what="complete over http")
    s, d = jcall(http_gui, "GET", f"/api/runs/{rid}")
    assert s == 200 and d["progress"]["ok"] == 5
    s, res = jcall(http_gui, "GET", f"/api/runs/{rid}/results?rank_by=value")
    assert s == 200 and res["n"] == 5 and res["rank_by"] == "value"
    s, res2 = jcall(http_gui, "GET", f"/api/runs/{rid}/results?rank_by=value&v={res['version']}")
    assert res2 == {"version": res["version"], "unchanged": True}
    s, h, body = call(http_gui, "GET", f"/api/runs/{rid}/results.csv?rank_by=prob")
    assert s == 200 and h["content-type"].startswith("text/csv") and "attachment" in h["content-disposition"] and body.startswith(b"rank,ligand_id")
    rec = res["rows"][0]["record_id"]
    s, h, body = call(http_gui, "GET", f"/api/runs/{rid}/ligand/{rec}/pose")
    assert s == 200 and body.startswith(b"data_fake") and h["x-pose-format"] == "mmcif"
    s, h, body = call(http_gui, "GET", f"/api/runs/{rid}/ligand/{rec}/2d.svg?theme=dark")
    assert s == 200 and h["content-type"] == "image/svg+xml" and b"<svg" in body
    s, det = jcall(http_gui, "GET", f"/api/runs/{rid}/ligand/{rec}")
    assert s == 200 and det["row"]["record_id"] == rec
    s, h, body = call(http_gui, "GET", f"/api/runs/{rid}/log?lines=20")
    assert s == 200 and b"[status]" in body
    s, lst = jcall(http_gui, "GET", "/api/runs")
    assert s == 200 and [x["id"] for x in lst["runs"]] == [rid] and lst["active_run"] is None
    s, svg = call(http_gui, "POST", "/api/depict", {"smiles": "CCO", "dark": True})[:2]
    assert s == 200
    assert jcall(http_gui, "POST", "/api/depict", {"smiles": "not smiles ("})[0] == 400
    s, err = jcall(http_gui, "POST", f"/api/runs/{rid}/stop", {})
    assert s == 409 and "not running" in err["error"]
    s, err = jcall(http_gui, "POST", f"/api/runs/{rid}/resume", {})
    assert s == 409


def test_http_guards(http_gui):
    srv = http_gui
    assert call(srv, "GET", "/api/info", token=False)[0] == 403                                   # no token
    assert call(srv, "GET", "/api/info", headers={"X-Boltz-Token": "wrong"})[0] == 403
    assert call(srv, "GET", "/api/info", host="evil.example.com")[0] == 403                        # DNS rebinding
    assert call(srv, "GET", "/api/info", host=f"127.0.0.1.evil.com:{srv.port}")[0] == 403
    assert call(srv, "GET", "/", host="evil.example.com")[0] == 403
    assert call(srv, "GET", "/api/info", host=f"localhost:{srv.port}")[0] == 200
    assert call(srv, "GET", "/api/info", headers={"Origin": "http://evil.example.com"})[0] == 403    # cross-origin page
    assert call(srv, "GET", "/api/info", headers={"Origin": f"http://127.0.0.1:{srv.port}"})[0] == 200
    assert call(srv, "GET", "/api/info", headers={"Sec-Fetch-Site": "cross-site"})[0] == 403
    assert call(srv, "POST", "/api/validate", raw=b"{}", headers={"Content-Type": "text/plain"})[0] == 415   # simple-form CSRF
    assert call(srv, "POST", "/api/validate", raw=b"{}", headers={"Content-Type": "application/x-www-form-urlencoded"})[0] == 415
    assert call(srv, "POST", "/api/validate", raw=b"[1,2]", headers={"Content-Type": "application/json"})[0] == 400
    assert call(srv, "POST", "/api/validate", raw=b"{oops", headers={"Content-Type": "application/json"})[0] == 400
    assert call(srv, "POST", "/api/validate", raw=b"{}", headers={"Content-Type": "application/json", "Content-Length": str(LIMITS["body"] + 1)})[0] in (413, 400)
    for m in ("PUT", "DELETE", "PATCH"):
        assert call(srv, m, "/api/runs")[0] == 405
    assert call(srv, "POST", "/api/info", {})[0] == 405
    assert call(srv, "GET", "/api/nothing")[0] == 404
    s, h, _ = call(srv, "GET", "/api/info")
    assert h["cache-control"] == "no-store" and h["x-content-type-options"] == "nosniff" and h["x-frame-options"] == "DENY"
    assert "access-control-allow-origin" not in h                                                 # no CORS


def test_http_body_limit_is_checked_before_reading(http_gui, monkeypatch):
    monkeypatch.setitem(LIMITS, "body", 1000)
    s, err = jcall(http_gui, "POST", "/api/validate", {"target": {"content": "A" * 5000}})
    assert s == 413 and "too large" in err["error"]


def test_path_traversal_static_and_runs(http_gui, tmp_path):
    srv = http_gui
    app = srv.app
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP SECRET")
    for p in ("/static/../../../etc/passwd", "/static/%2e%2e/%2e%2e/etc/passwd", "/static/..%2f..%2fetc/passwd", "/static/%2e%2e%2fapi.py",
              "/static/..%5c..%5cetc", "/static//etc/passwd", "/static/vendor/../../api.py", "/static/css/../../server.py",
              "/static/%00", "/static/vendor/molstar/../../../common.py", "/static/."):
        s, h, b = call(srv, "GET", p)
        assert s in (404, 400) and b"SECRET" not in b and b"GuiApp" not in b, p
    # only files under static/ are served
    assert call(srv, "GET", "/static/index.html")[0] in (200, 404)
    # run ids and record ids
    rid = app.create_run(payload(3))["id"]
    wait_state(app, rid, "complete")
    rec = app.runs.results(rid, "prob")["rows"][0]["record_id"]
    (app.runs.run_dir(rid) / "out" / "structures" / "x.txt").write_text("x")
    for bad in ("..", "%2e%2e", "..%2f", "%2e%2e%2f%2e%2e", ".hidden", "a%2fb", "%00", "x" * 300):
        assert call(srv, "GET", f"/api/runs/{bad}")[0] == 404, bad
        assert call(srv, "GET", f"/api/runs/{bad}/results")[0] == 404, bad
        assert call(srv, "POST", f"/api/runs/{bad}/stop", {})[0] in (404, 409), bad
    for bad in ("..", "..%2f..%2fgui_run.json", "%2e%2e%2f..%2fscreen.log", "a%2f..%2f..", "cpd0%2f..", "not_a_record", ".hidden", "..%5c.."):
        for tail in ("", "/pose", "/2d.svg"):
            s, h, b = call(srv, "GET", f"/api/runs/{rid}/ligand/{bad}{tail}")
            assert s == 404 and b"launches" not in b and b"SECRET" not in b, (bad, tail)
    # a symlink planted in the run's structures/ must not leak files outside the run
    link = app.runs.run_dir(rid) / "out" / "structures" / f"0099_{rec}.cif"
    # the real record's own top-K copy is found first (glob order), so replace it by a symlink to the secret
    for f in (app.runs.run_dir(rid) / "out" / "structures").glob(f"*_{rec}.*"):
        f.unlink()
    shutil_target = app.runs.run_dir(rid) / "out" / "work" / "predictions" / rec / f"{rec}_model_0.cif"
    shutil_target.unlink()
    shutil_target.symlink_to(secret)
    link.symlink_to(secret)
    s, h, b = call(srv, "GET", f"/api/runs/{rid}/ligand/{rec}/pose")
    assert s == 404 and b"SECRET" not in b
    assert not is_within(secret, app.runs.run_dir(rid))
    # a symlinked run directory is not a run
    (app.runs.root / "linked").symlink_to(tmp_path)
    assert call(srv, "GET", "/api/runs/linked")[0] == 404
    assert [r["id"] for r in app.runs.list_runs()] == [rid]


def test_bind_address_is_loopback_only(tmp_path, env):
    app = make_app(tmp_path, env)
    for host in ("0.0.0.0", "", "192.168.1.5", "::", "::1", "example.com", "127.0.0.2", "0"):
        with pytest.raises(ValueError, match="refusing to bind"):
            make_server(app, host=host, port=0)
    for host in ("127.0.0.1", "localhost"):
        srv = make_server(app, host=host, port=0)
        try:
            assert srv.server_address[0] == "127.0.0.1" and srv.url == f"http://127.0.0.1:{srv.port}/"
            # not reachable through a non-loopback address of this machine (checked when the sandbox can resolve one)
            try:
                ext = [a for a in socket.gethostbyname_ex(socket.gethostname())[2] if not a.startswith("127.")]
            except OSError:
                ext = []
            for addr in ext[:1]:
                with pytest.raises(OSError):
                    socket.create_connection((addr, srv.port), timeout=2)
        finally:
            srv.server_close()


def test_port_default_fallback_and_explicit(tmp_path, env):
    app = make_app(tmp_path, env)
    first = make_server(app, port=None)            # 8765 or the next free port
    try:
        assert 8765 <= first.port < 8765 + 20 or first.port > 1024
        second = make_server(app, port=None)       # busy -> falls back to another one
        try:
            assert second.port != first.port
        finally:
            second.server_close()
        with pytest.raises(OSError, match="could not bind"):
            make_server(app, port=first.port)      # an explicit port that is busy is an error, not a silent fallback
    finally:
        first.server_close()


def test_static_index_and_token(http_gui):
    s, h, body = call(http_gui, "GET", "/", token=False)
    assert s == 200 and h["content-type"].startswith("text/html") and "content-security-policy" in h
    html = body.decode()
    assert http_gui.app.token in html and "__BOLTZ_TOKEN__" not in html
    csp = h["content-security-policy"]
    assert "default-src 'none'" in csp and "connect-src 'self'" in csp and "script-src 'self';" in csp and "unsafe-eval" not in csp
    assert h["x-frame-options"] == "DENY" and "frame-ancestors 'none'" in csp
    assert f"frame-src http://127.0.0.1:{http_gui.port} http://localhost:{http_gui.port}" in csp
    s, h, _ = call(http_gui, "GET", "/static/vendor/molstar/molstar.css", token=False)
    assert s == 200 and h["content-type"].startswith("text/css")
    etag = h["etag"]
    assert call(http_gui, "GET", "/static/vendor/molstar/molstar.css", token=False, headers={"If-None-Match": etag})[0] == 304
    assert call(http_gui, "GET", "/static/nope.js", token=False)[0] == 404
    assert call(http_gui, "GET", "/favicon.ico", token=False)[0] == 200


def test_viewer_frame_is_isolated_from_the_app(http_gui):
    """Mol* needs 'unsafe-eval' (Emscripten glue in the bundle). Only the viewer page gets it, and it is framed from the other
    loopback host name (another origin): no token, no API access, only postMessage."""
    srv = http_gui
    s, h, body = call(srv, "GET", "/static/viewer.html", token=False, host=f"localhost:{srv.port}")
    assert s == 200 and b"molstar.js" in body and b"viewer-frame.js" in body
    csp = h["content-security-policy"]
    assert "'unsafe-eval'" in csp and f"frame-ancestors http://127.0.0.1:{srv.port} http://localhost:{srv.port}" in csp
    assert "x-frame-options" not in h and h["cross-origin-resource-policy"] == "cross-origin"
    assert srv.app.token.encode() not in body                            # the viewer page never contains the token
    # the frame's own subresource requests carry a cross-site initiator but are public static files: served
    assert call(srv, "GET", "/static/js/viewer-frame.js", token=False, host=f"localhost:{srv.port}", headers={"Sec-Fetch-Site": "cross-site"})[0] == 200
    # ...but the API refuses them (and the frame has no token to send anyway)
    assert call(srv, "GET", "/api/info", host=f"localhost:{srv.port}", headers={"Sec-Fetch-Site": "cross-site"})[0] == 403
    assert call(srv, "GET", "/api/info", token=False, host=f"localhost:{srv.port}")[0] == 403
    js = (STATIC / "js" / "viewer-frame.js").read_text()
    assert "e.source !== window.parent" in js and "ALLOWED.has(e.origin)" in js and 'postMessage(m, "*")' not in js
    # dark mode: the frame marks its document and gives Mol*'s light-skin viewport buttons light icons on a dark square
    assert 'classList.toggle("dark", on)' in js
    assert b"html.dark .msp-plugin .msp-viewport-controls-buttons .msp-semi-transparent-background" in body
    page = (STATIC / "js" / "viewer.js").read_text()
    assert "e.origin !== this.origin" in page and 'postMessage({ ...msg, id }, "*")' not in page
    assert "unsafe-eval" not in call(srv, "GET", "/", token=False)[1]["content-security-policy"]


# ---------------------------------------------------------------------------------------- frontend / vendor assets
def test_vendored_molstar_matches_its_readme():
    d = STATIC / "vendor" / "molstar"
    data = (d / "README.md").read_text()
    for name in ("molstar.js", "molstar.css"):
        f = d / name
        sha = hashlib.sha256(f.read_bytes()).hexdigest()
        assert sha in data, f"{name}: sha256 {sha} is not recorded in vendor/molstar/README.md"
        assert f"{f.stat().st_size}" in data
    assert f"https://cdn.jsdelivr.net/npm/molstar@4.5.0/build/viewer/molstar.js" in data
    assert b"4.5.0" in (d / "molstar.js").read_bytes()
    assert (d / "LICENSE").read_text().startswith("MIT License") and "Mol*" in (d / "LICENSE").read_text()
    assert (d / "README.md").exists()


def test_frontend_loads_nothing_from_the_network():
    """No CDN or remote resource in our own pages / scripts / styles (the vendored Mol* files are checked separately)."""
    own = [p for p in STATIC.rglob("*") if p.is_file() and "vendor" not in p.relative_to(STATIC).parts and p.suffix in (".html", ".js", ".css", ".svg")]
    assert own, "no frontend files"
    for p in own:
        text = p.read_text(encoding="utf-8")
        for m in re.finditer(r"(?:https?:)?//[A-Za-z0-9.-]+\.[A-Za-z]{2,}[^\s\"')]*", text):
            url = m.group(0)
            assert url.startswith("http://www.w3.org/") or url.startswith("https://molstar.org"), f"{p.name}: remote URL {url}"
        assert not re.search(r"<(script|link|img|iframe)[^>]+(src|href)=[\"']https?:", text), p.name
        assert "@import" not in text and "fonts.googleapis" not in text
    css = (STATIC / "vendor" / "molstar" / "molstar.css").read_text()
    assert "http" not in css.replace("github.com", "")  # the vendored stylesheet makes no remote request either


def test_index_has_privacy_statement_and_attribution():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert "Local only" not in html and 'id="btn-about2"' in html  # the footer: About and the version; privacy is in About
    about = " ".join(p.read_text(encoding="utf-8") for p in STATIC.rglob("*.js") if "vendor" not in p.parts)
    assert "Privacy" in about
    assert "Mol*" in html + about and "MIT" in html + about
