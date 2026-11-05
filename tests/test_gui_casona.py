"""GUI additions of the gui-casona branch: per-chip time estimates, structure jobs (`boltz-mlx predict` as a subprocess),
PAE heatmap data, the screen's step / keep-PAE options, and the restyled pages.

CPU only. Structure jobs run tests/gui_fake_predict.py (writes the predict output layout with made-up numbers, no model).
"""

from __future__ import annotations

import base64
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from boltz_mlx.gui import estimates as EST
from boltz_mlx.gui import predict as PR
from boltz_mlx.gui.api import GuiApp
from boltz_mlx.gui.common import ApiError
from boltz_mlx.gui.inputs import normalize_options, screen_argv
from boltz_mlx.gui.pae import chain_spans, pae_payload
from boltz_mlx.gui.server import STATIC, make_server
from boltz_mlx.screen.presets import PRESETS, affinity_estimate, struct_estimate

ROOT = Path(__file__).resolve().parents[1]
FAKE_SCREEN = ROOT / "tests" / "gui_fake_screen.py"
FAKE_PREDICT = ROOT / "tests" / "gui_fake_predict.py"
PROT = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQW"


@pytest.fixture
def env(tmp_path, monkeypatch):
    w, c = tmp_path / "weights", tmp_path / "cache"
    w.mkdir()
    (c / "mols").mkdir(parents=True)
    (c / "mols" / "SAH.pkl").write_bytes(b"")
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
               predict_cmd=[sys.executable, str(FAKE_PREDICT), "predict"], token="t0k-casona", **env)
    yield a
    for rid in a.runs.adopt_note():
        try:
            a.runs.stop(rid, force=True)
        except ApiError:
            pass


def job(**kw):
    p = {"kind": "predict", "name": "unit", "chains": [{"type": "protein", "id": "A", "sequence": PROT},
                                                       {"type": "ligand", "id": "B", "smiles": "CC(=O)Nc1ccc(O)cc1"}],
         "affinity": {"binder": "B"}, "options": {"preset": "screen"}}
    p.update(kw)
    return p


def wait_state(app, rid, states, timeout=60.0):
    states = (states,) if isinstance(states, str) else tuple(states)
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = app.runs.state(rid)
        if s in states:
            return s
        time.sleep(0.1)
    raise AssertionError(f"{rid} never reached {states} (now {app.runs.state(rid)})")


# --------------------------------------------------------------------------------------------------- estimates
def test_chip_table_keeps_the_m5_pro_numbers():
    e = EST.CHIP_TABLE["Apple M5 Pro"]["presets"]
    assert (e["screen"]["struct_117"], e["screen"]["struct_400"], e["screen"]["steps"]) == (0.843, 6.0, 50)
    assert (e["screen"]["affinity"], e["screen"]["affinity_steps"]) == (7.63, 200)
    r = EST.CHIP_TABLE["Apple M5 Pro"]["reference"]
    assert (r["t400_s_per_ligand"], r["steps"], r["affinity_steps"]) == (15.5, 50, 200)  # the earlier screen settings
    assert EST.CHIP_TABLE["Apple M5 Max"]["presets"] == {}  # filled from the M5 Max benchmarks, not guessed


def test_t400_estimate_is_close_to_the_measured_rate():
    """The table reproduces the one whole-ligand measurement: T400 (384 aa + ~29-atom ligands), at the step counts it was
    measured at (the earlier screen settings, f16 + 50 + 200)."""
    r = EST.CHIP_TABLE["Apple M5 Pro"]["reference"]
    ref = EST.reference_preset(PRESETS["screen"], "Apple M5 Pro")
    per = (struct_estimate(ref, 384 + 29, r["steps"]) + affinity_estimate(ref, r["affinity_steps"])
           + EST.chip_constants("Apple M5 Pro")["overhead"])
    assert abs(per - 15.5) / 15.5 < 0.1, per


def test_unknown_or_unmeasured_chip_gets_no_number(monkeypatch):
    for chip in ("Apple M5 Max", "Apple M1"):
        monkeypatch.setenv("BOLTZ_MLX_GUI_CHIP", chip)
        assert EST.chip_name() == chip
        assert all(EST.reference_preset(p) is None for p in PRESETS.values())
        info = EST.chip_info()
        assert not info["has_estimates"] and EST.NO_ESTIMATE in info["note"]
    assert EST.entry("f16-portable", "Apple M5 Pro") is None  # another code path: no borrowed numbers


def test_steps_scale_from_the_entry_steps():
    ref = EST.reference_preset(PRESETS["f16"], "Apple M5 Pro")  # f16 runs 200 steps; the entry was measured at 50
    assert ref.steps == 50 and ref.est_struct_117 == 0.843
    assert struct_estimate(ref, 117, 50) == pytest.approx(0.843)
    assert struct_estimate(ref, 117, 200) == pytest.approx(0.843 * (0.6 + 0.4 * 4))
    assert affinity_estimate(ref, 25) < affinity_estimate(ref, 200) == pytest.approx(7.63)
    s = EST.reference_preset(PRESETS["screen"], "Apple M5 Pro")  # screen runs 20 + 25; the entry was measured at 50 + 200
    assert (s.est_steps, s.est_affinity_steps) == (50, 200)
    assert struct_estimate(s, 117, 20) == pytest.approx(0.843 * (0.6 + 0.4 * 20 / 50))
    assert affinity_estimate(s, 25) == pytest.approx(7.63 * (0.3 + 0.7 * 25 / 200))


def test_screen_estimate_on_a_chip_without_numbers(app, monkeypatch):
    # M5 Max: no bench rows, and no measured time for turbo (Standard / screen has one: see the next test)
    monkeypatch.setenv("BOLTZ_MLX_GUI_CHIP", "Apple M5 Max")
    lig = "id,smiles\n" + "".join(f"c{i},{s}\n" for i, s in enumerate(["CCO", "c1ccccc1O", "CCN(CC)CC"]))
    p = {"target": {"kind": "text", "content": PROT}, "msa": {"mode": "empty"},
         "ligands": {"kind": "file", "filename": "l.csv", "content": lig}, "options": {"preset": "turbo"}}
    est = app.estimate(p)
    e = est["estimate"]
    assert not e["available"] and "total_s" not in e and e["n_to_run"] == 3 and EST.NO_ESTIMATE in e["caveat"]
    eline = [l for l in est["lines"] if l.startswith("ESTIMATE")]
    assert eline == [f"ESTIMATE: {EST.NO_ESTIMATE} (Apple M5 Max, preset turbo); the first ligands of the run give the real rate"]
    assert all(v is None for v in app.validate(p)["preset_times"].values())
    assert all(x["est"] is None for x in app.presets_payload())
    assert app.info()["chip"]["name"] == "Apple M5 Max" and not app.info()["chip"]["has_estimates"]
    monkeypatch.setenv("BOLTZ_MLX_GUI_CHIP", "Apple M5 Pro")
    assert app.estimate(p)["estimate"]["total_s"] > 0
    # screen's own 20 steps, scaled from the entry's 50
    assert {x["name"]: x["est"]["struct_117_s"] for x in app.presets_payload()}["screen"] == pytest.approx(0.843 * 0.76, abs=0.01)


# ------------------------------------------------------------------------------------------------ screen options
def test_screen_step_and_keep_pae_options(tmp_path):
    o = normalize_options({"preset": "screen", "steps": "25", "affinity_steps": 100, "recycling": 3, "keep_work": True}, PRESETS)
    assert (o["steps"], o["affinity_steps"], o["recycling"], o["keep_work"]) == (25, 100, 3, True)
    a = screen_argv(tmp_path, {"target": {"file": "inputs/t.fasta"}, "ligands": {"file": "inputs/l.csv"}, "msa": {"mode": "empty"}},
                    o, tmp_path / "out")
    assert "--steps=25" in a and "--affinity-steps=100" in a and "--recycling=3" in a and "--keep-work" in a
    d = normalize_options({}, PRESETS)
    assert d["steps"] is None and not d["keep_work"]
    assert not any(x.startswith(("--steps", "--keep-work")) for x in screen_argv(tmp_path, {"target": {"file": "t"}, "ligands": {"file": "l"}}, d, tmp_path))
    for bad in ({"steps": 0}, {"steps": "x"}, {"recycling": 99}, {"affinity_steps": 5000}):
        with pytest.raises(ApiError):
            normalize_options(bad, PRESETS)


# ------------------------------------------------------------------------------------------------ structure jobs
def test_predict_validation(app):
    v = app.validate_predict(job())
    assert v["ok"] and v["affinity"] and v["binder"] == "B" and v["n_tokens"] == len(PROT) + 11
    assert "binder: B" in v["yaml"] and "msa: empty" in v["yaml"] and "smiles: CC(=O)Nc1ccc(O)cc1" in v["yaml"]
    assert v["options"]["steps"] == 20 and v["options"]["affinity_steps"] == 25 and v["estimate"]["total_s"] > 0  # screen preset
    # ids are filled in, FASTA headers and numbering are dropped, DNA / CCD chains are accepted
    v = app.validate_predict(job(chains=[{"type": "protein", "sequence": ">sp|X\n10 MKTAY IAKQR\n"}, {"type": "dna", "sequence": "acgt"},
                                         {"type": "ligand", "ccd": "sah"}], affinity=None))
    assert v["ok"] and [c["id"] for c in v["chains"]] == ["A", "B", "C"] and v["chains"][0]["sequence"] == "MKTAYIAKQR"
    assert v["chains"][2]["ccd"] == "SAH" and not v["affinity"]
    bad = [
        (job(chains=[]), "at least one chain"),
        (job(chains=[{"type": "protein", "sequence": "MKT1Z*"}]), "not an amino-acid"),
        (job(chains=[{"type": "rna", "sequence": "ACGT"}]), "not a RNA"),
        (job(chains=[{"type": "protein", "id": "A", "sequence": "MK"}, {"type": "ligand", "id": "a", "smiles": "CCO"}]), "used twice"),
        (job(chains=[{"type": "ligand", "id": "B", "smiles": "C1CC"}]), "parse"),
        (job(chains=[{"type": "ligand", "id": "B", "smiles": "CCO", "ccd": "SAH"}]), "one of the two"),
        (job(chains=[{"type": "ligand", "id": "B", "ccd": "ZZZ"}]), "not in the molecule dictionary"),
        (job(chains=[{"type": "protein", "id": "A", "sequence": PROT}], affinity={"binder": "A"}), "binder must be one of the ligand"),
        (job(chains=[{"type": "ligand", "id": "B", "smiles": "CCO"}], affinity={"binder": "B"}), "needs a protein"),
        (job(chains=[{"type": "protein", "id": "A", "sequence": PROT}, {"type": "ligand", "id": "B", "smiles": "C" * 140}]), "128"),
        (job(options={"preset": "nope"}), "unknown preset"),
        (job(options={"preset": "screen", "steps": 0}), "between"),
        (job(chains=[{"type": "protein", "id": "A", "sequence": PROT, "msa": {"mode": "file", "filename": "x.txt", "content": ">a\nMK\n"}}]),
         ".a3m"),
    ]
    for payload, msg in bad:
        r = app.validate_predict(payload)
        assert not r["ok"] and msg in r["error"], (msg, r)
    # a big non-binder ligand is fine for structure only
    assert app.validate_predict(job(chains=[{"type": "protein", "sequence": PROT}, {"type": "ligand", "smiles": "C" * 140}], affinity=None))["ok"]


def test_predict_job_lifecycle(app):
    msa = ">q\n" + PROT + "\n>h\n" + PROT + "\n"
    p = job(chains=[{"type": "protein", "id": "A", "sequence": PROT, "msa": {"mode": "file", "filename": "../../evil.a3m", "content": msa}},
                    {"type": "ligand", "id": "B", "smiles": "CC(=O)Nc1ccc(O)cc1"}], options={"preset": "strict", "steps": 20, "seed": 7})
    r = app.create_run(p)
    rid = r["id"]
    assert r["kind"] == "predict"
    assert wait_state(app, rid, ("complete", "failed", "interrupted")) == "complete"
    rd = app.runs.run_dir(rid)
    meta = json.loads((rd / "gui_run.json").read_text())
    assert meta["kind"] == "predict" and meta["options"]["steps"] == 20 and meta["options"]["changed"] == ["steps"]
    assert (rd / "inputs" / "job.yaml").is_file() and (rd / "inputs" / "msa_A.a3m").read_text() == msa
    assert not any("evil" in str(x) for x in rd.rglob("*"))  # the uploaded file name never decides where anything goes
    y = (rd / "inputs" / "job.yaml").read_text()
    assert f"msa: {rd / 'inputs' / 'msa_A.a3m'}" in y
    argv = meta["launches"][0]["argv"]
    assert argv[0] == str(rd / "inputs" / "job.yaml") and "--precision=fp32" in argv and "--sampling_steps=20" in argv
    assert "--seed=7" in argv and "--sampling_steps_affinity=200" in argv and any(a.startswith("--cache=") for a in argv)
    d = app.runs.detail(rid)
    assert d["kind"] == "predict" and d["progress"]["phase"] == "done" and d["progress"]["fraction"] == 1.0 and d["has_result"]
    assert d["timings"]["total_wall_s"] >= 0 and not d["can_resume"] and not d["can_stop"]
    s = app.runs.summary(rid)
    assert s["kind"] == "predict" and s["affinity"] and s["metrics"]["affinity_probability_binary"] == 0.87
    pr = app.runs.prediction(rid)
    assert pr["binder"] == "B" and pr["affinity"]["affinity_pred_value"] == 0.42 and pr["has_pae"] and pr["pose"]["format"] == "mmcif"
    assert pr["pocket"]["ligand_chain"] == "B" and pr["pocket"]["ligand_atoms"] == 11
    pae = app.runs.prediction_pae(rid)
    assert pae["n"] == len(PROT) + 11 and [c["id"] for c in pae["chains"]] == ["A", "B"] and pae["chains"][1]["n"] == 11
    assert len(base64.b64decode(pae["data"])) == pae["side"] ** 2 and {x["a"] for x in pae["pairs"]} == {"A", "B"}
    assert app.runs.prediction_pose(rid).name == "job_model_0.cif"
    with pytest.raises(ApiError) as ex:
        app.runs.launch(rid)
    assert ex.value.status == 409


def test_predict_fasta_records_are_never_merged(app):
    two = app.validate_predict(job(chains=[{"type": "protein", "sequence": ">a\nMKTAY\n>b\nGGGSA\n"}], affinity=None))
    assert not two["ok"] and "2 FASTA records" in two["error"] and "one chain per record" in two["error"]
    one = app.validate_predict(job(chains=[{"type": "protein", "sequence": ">a\nMKTAY\nIAKQR\n"}], affinity=None))
    assert one["ok"] and one["chains"][0]["sequence"] == "MKTAYIAKQR"
    bad = app.validate_predict(job(chains=[{"type": "protein", "sequence": "MK1!"}], affinity=None))
    assert not bad["ok"] and "YAML" not in bad["error"]  # the predict form has an Import action for YAML; the error does not point to pasting it


def test_predict_import_reads_yaml_and_fasta():
    y = ("version: 1\nsequences:\n  - protein:\n      id: [A, B]\n      sequence: MKTAYIAKQR\n      msa: /tmp/x.a3m\n"
         "  - ligand:\n      id: C\n      smiles: CCO\n  - dna:\n      id: D\n      sequence: ACGT\n"
         "properties:\n  - affinity:\n      binder: C\nconstraints:\n  - pocket: {binder: C, contacts: [[A, 3]]}\n")
    r = PR.import_yaml(y)
    assert [(c["type"], c["id"]) for c in r["chains"]] == [("protein", "A"), ("protein", "B"), ("ligand", "C"), ("dna", "D")]
    assert r["binder"] == "C" and "constraints" not in r["skipped"] and any("MSA" in x for x in r["skipped"])
    assert r["constraints"] == [{"pocket": {"binder": "C", "contacts": [["A", 3]]}}]   # the form holds constraints now
    f = PR.import_yaml(">A|protein|empty\nMKTAY\n>sp|P1|X\nGGGSA\n")
    assert [c["id"] for c in f["chains"]] == ["A", "B"] and [c["sequence"] for c in f["chains"]] == ["MKTAY", "GGGSA"]
    for text in ("", "not: [valid", "foo: 1"):
        with pytest.raises(ApiError):
            PR.import_yaml(text)


def test_predict_models_are_selectable(app):
    r = app.create_run(job(options={"preset": "strict", "samples": 3}))
    assert wait_state(app, r["id"], ("complete", "failed", "interrupted")) == "complete"
    p0, p2 = app.runs.prediction(r["id"]), app.runs.prediction(r["id"], 2)
    assert p0["model"] == 0 and [m["index"] for m in p0["models"]] == [0, 1, 2] and p2["model"] == 2
    assert p0["confidence"]["confidence_score"] > p2["confidence"]["confidence_score"] and p2["has_pae"]
    assert app.runs.prediction_pose(r["id"], 2).name == "job_model_2.cif"
    assert app.runs.prediction_pae(r["id"], 1)["n"] > 0
    for bad in (7, "x"):
        with pytest.raises(ApiError):
            app.runs.prediction(r["id"], bad)
    from boltz_mlx.gui import exports

    rows = list(__import__("csv").DictReader(line for line in exports.predict_csv(app.runs, r["id"]).splitlines() if not line.startswith("#")))
    assert [x["model"] for x in rows] == ["0", "1", "2"]


def test_saved_target_rename(app):
    t = app.prepare_target({"name": "old", "source": {"kind": "text", "content": PROT}, "msa": {"mode": "none"}})
    assert app.targets.rename(t["id"], "  new name ")["name"] == "new name" and app.get_target(t["id"])["name"] == "new name"
    with pytest.raises(ApiError):
        app.targets.rename(t["id"], "  ")


def test_predict_stop_resume_and_failure(app, monkeypatch, tmp_path, env):
    monkeypatch.setenv("GUI_FAKE_DELAY", "30")
    rid = app.create_run(job(affinity=None, chains=[{"type": "protein", "sequence": PROT}]))["id"]
    assert app.runs.state(rid) == "running" and app.runs.active_run_id() == rid
    with pytest.raises(ApiError) as ex:  # one job at a time, whatever its kind
        app.create_run(job())
    assert ex.value.status == 409
    app.stop(rid)
    assert wait_state(app, rid, ("stopped", "failed", "interrupted")) == "stopped"
    assert app.runs.detail(rid)["can_resume"] and app.runs.active_run_id() is None
    monkeypatch.setenv("GUI_FAKE_DELAY", "0")
    app.start_existing(rid)
    assert wait_state(app, rid, ("complete", "failed")) == "complete"
    assert len(app.runs.read_meta(rid)["launches"]) == 2
    monkeypatch.setenv("GUI_FAKE_FAIL", "1")
    rid2 = app.create_run(job())["id"]
    assert wait_state(app, rid2, ("failed", "complete", "interrupted")) == "failed"
    d = app.runs.detail(rid2)
    assert "fake failure" in d["run_error"] and d["can_resume"]
    with pytest.raises(ApiError):
        app.runs.prediction(rid2)
    lst = {r["id"]: r for r in app.runs.list_runs()}
    assert lst[rid]["kind"] == lst[rid2]["kind"] == "predict" and lst[rid2]["state"] == "failed" and lst[rid2]["metrics"] is None


def test_predict_env_drops_numerics_switches(app, monkeypatch):
    monkeypatch.setenv("BOLTZ_MLX_PRECISION", "f16")
    monkeypatch.setenv("BOLTZ_MLX_MEM_LIMIT_GB", "20")
    monkeypatch.setenv("MLX_ENABLE_TF32", "1")
    seen = {}

    class P:
        pid, returncode = 999999, 0

        def poll(self):
            return 0

    def fake_popen(cmd, env=None, **kw):
        seen.update(env)
        return P()

    monkeypatch.setattr("boltz_mlx.gui.runs.subprocess.Popen", fake_popen)
    app.create_run(job())
    assert "BOLTZ_MLX_PRECISION" not in seen and "MLX_ENABLE_TF32" not in seen and seen["BOLTZ_MLX_MEM_LIMIT_GB"] == "20"
    assert seen["BOLTZ_MLX_WEIGHTS"].endswith("weights")


# ---------------------------------------------------------------------------------------------------------- PAE
def test_pae_payload_downsamples_and_finds_chains(tmp_path):
    n = 1100
    m = np.tile(np.linspace(0, 30, n, dtype=np.float32), (n, 1))
    f = tmp_path / "pae_x_model_0.npz"
    np.savez(f, pae=m)
    p = pae_payload(f, None, max_side=512)
    assert p["n"] == n and p["factor"] == 3 and p["side"] == 367 and p["chains"] == []
    q = np.frombuffer(base64.b64decode(p["data"]), np.uint8).reshape(p["side"], p["side"])
    assert q[0, 0] < 4 and q[0, -1] > 230 and p["max"] == pytest.approx(30)
    cif = tmp_path / "x.cif"
    cif.write_text("data_x\nloop_\n_atom_site.group_PDB\n_atom_site.type_symbol\n_atom_site.label_comp_id\n_atom_site.auth_seq_id\n"
                   "_atom_site.Cartn_x\n_atom_site.Cartn_y\n_atom_site.Cartn_z\n_atom_site.auth_asym_id\n"
                   "ATOM N ALA 1 0 0 0 A\nATOM C ALA 1 1 0 0 A\nATOM N GLY 2 2 0 0 A\nHETATM C LIG 1 0 1 0 B\nHETATM O LIG 1 0 2 0 B\n"
                   "HETATM H LIG 1 0 3 0 B\n#\n")
    assert [(c["id"], c["n"], c["type"]) for c in chain_spans(cif)] == [("A", 2, "polymer"), ("B", 2, "ligand")]
    g = tmp_path / "pae_y.npz"
    np.savez(g, pae=np.ones((4, 4), np.float32))
    assert [c["end"] for c in pae_payload(g, cif)["chains"]] == [2, 4]
    np.savez(g, pae=np.ones((5, 5), np.float32))
    assert pae_payload(g, cif)["chains"] == []  # token count differs from the structure: no boundaries rather than wrong ones
    with pytest.raises(ApiError):
        pae_payload(tmp_path / "missing.npz")


def test_screen_ligand_pae_when_kept(app, monkeypatch):
    lig = "id,smiles\nc0,CCO\nc1,c1ccccc1O\n"
    p = {"name": "kept", "target": {"kind": "text", "content": PROT}, "msa": {"mode": "empty"},
         "ligands": {"kind": "file", "filename": "l.csv", "content": lig}, "options": {"preset": "strict", "keep_work": True}}
    rid = app.create_run(p)["id"]
    assert wait_state(app, rid, ("complete", "failed", "interrupted")) == "complete"
    assert "--keep-work" in app.runs.read_meta(rid)["launches"][0]["argv"]
    assert app.runs.detail(rid)["keep_work"]
    rec = app.runs.results(rid, "prob")["rows"][0]["record_id"]
    pd = app.runs.out_dir(rid) / "work" / "predictions" / rec
    if not (pd / f"pae_{rec}_model_0.npz").exists():  # the fake engine writes no npz: add one as the real screen would
        np.savez(pd / f"pae_{rec}_model_0.npz", pae=np.full((8, 8), 3.0, np.float32))
    assert app.runs.ligand_detail(rid, rec)["has_pae"]
    assert app.runs.ligand_pae(rid, rec)["n"] == 8
    other = app.runs.results(rid, "prob")["rows"][1]["record_id"]
    (app.runs.out_dir(rid) / "work" / "predictions" / other / f"pae_{other}_model_0.npz").unlink(missing_ok=True)
    with pytest.raises(ApiError) as ex:
        app.runs.ligand_pae(rid, other)
    assert ex.value.status == 404 and "Keep PAE" in ex.value.message


# ---------------------------------------------------------------------------------------------------------- HTTP
def test_http_new_endpoints(tmp_path, env):
    import http.client
    import threading

    app = GuiApp(tmp_path / "hr", screen_cmd=[sys.executable, str(FAKE_SCREEN), "screen"],
                 predict_cmd=[sys.executable, str(FAKE_PREDICT), "predict"], token="tok-http", **env)
    srv = make_server(app, port=0)
    th = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    th.start()

    def call(method, path, body=None, token=True):
        c = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=30)
        h = {"Host": f"127.0.0.1:{srv.port}"}
        if token:
            h["X-Boltz-Token"] = "tok-http"
        raw = None
        if body is not None:
            raw = json.dumps(body).encode()
            h["Content-Type"] = "application/json"
        c.request(method, path, body=raw, headers=h)
        r = c.getresponse()
        data = r.read()
        c.close()
        try:
            return r.status, json.loads(data)
        except ValueError:
            return r.status, data

    try:
        assert call("GET", "/api/info", token=False)[0] == 403
        assert call("GET", "/api/examples")[0] == 404  # the example prefills are gone: users bring their own inputs
        s, v = call("POST", "/api/predict/validate", job())
        assert s == 200 and v["ok"]
        s, r = call("POST", "/api/runs", job())
        assert s == 201 and r["kind"] == "predict"
        rid = r["id"]
        t0 = time.time()
        while call("GET", f"/api/runs/{rid}")[1]["state"] != "complete":
            assert time.time() - t0 < 60
            time.sleep(0.1)
        s, pr = call("GET", f"/api/runs/{rid}/prediction")
        assert s == 200 and pr["affinity"]["affinity_probability_binary"] == 0.87
        s, pose = call("GET", f"/api/runs/{rid}/prediction/pose")
        assert s == 200 and pose.startswith(b"data_model")
        s, pae = call("GET", f"/api/runs/{rid}/prediction/pae")
        assert s == 200 and pae["n"] > 0
        s, log = call("GET", f"/api/runs/{rid}/log")
        assert s == 200 and b"fake predict" in log
        s, info = call("GET", "/api/info")
        assert info["chip"]["name"] == "Apple M5 Pro" and info["chip"]["has_estimates"]
        assert call("GET", "/api/runs/../prediction")[0] == 404
        assert call("GET", f"/api/runs/{rid}/ligand/x/pae")[0] == 404
    finally:
        srv.shutdown()
        srv.server_close()


# ------------------------------------------------------------------------------------------------------- frontend
def test_branding_is_its_own_product():
    own = [p for p in STATIC.rglob("*") if p.is_file() and "vendor" not in p.relative_to(STATIC).parts
           and p.suffix in (".html", ".js", ".css", ".svg")]
    for p in own:
        assert "casona" not in p.read_text(encoding="utf-8").lower(), p.name
    html = (STATIC / "index.html").read_text()
    assert "<title>Boltz-2 MLX</title>" in html


def test_pages_use_only_bundled_fonts_and_scripts():
    css = (STATIC / "css" / "app.css").read_text()
    assert "@font-face" not in css or "url(/static/" in css  # system font stack (no vendored font needed)
    assert not re.search(r"url\((?!['\"]?(/static/|data:))", css)
    for js in (STATIC / "js").glob("*.js"):
        t = js.read_text()
        assert "fetch(\"http" not in t and "import(\"http" not in t and "https://" not in t.replace("https://molstar.org", ""), js.name


# ------------------------------------------------------------------------------------------------ round 3 (audit fixes)
def test_msa_is_parsed_not_just_named(app):
    bad_query = ">q\n" + PROT[:-3] + "AAA\n"
    r = app.validate_predict(job(chains=[{"type": "protein", "id": "A", "sequence": PROT, "msa": {"mode": "file", "filename": "a.a3m", "content": bad_query}}]))
    assert not r["ok"] and "not this chain's sequence" in r["error"]
    ragged = ">q\n" + PROT + "\n>h\n" + PROT[:-5] + "\n"
    r = app.validate_predict(job(chains=[{"type": "protein", "id": "A", "sequence": PROT, "msa": {"mode": "file", "filename": "a.a3m", "content": ragged}}]))
    assert not r["ok"] and "row 2" in r["error"]
    r = app.validate_predict(job(chains=[{"type": "protein", "id": "A", "sequence": PROT, "msa": {"mode": "file", "filename": "a.csv", "content": "a,b\n1,2\n"}}]))
    assert not r["ok"] and "key,sequence" in r["error"]
    ok = ">q\n" + PROT + "\n>h\n" + PROT[:10].lower() + PROT + "\n"  # lowercase = insertions, ignored for alignment width
    good = app.validate_predict(job(chains=[{"type": "protein", "id": "A", "sequence": PROT, "msa": {"mode": "file", "filename": "a.a3m", "content": ok}}], affinity=None))
    assert good["ok"], good
    # the reviewed YAML names the staged MSA file, as the run's YAML does
    assert "msa: inputs/msa_A.a3m" in good["yaml"]


def test_predict_affinity_samples_reach_the_command(app):
    o = PR.normalize_predict_options({"preset": "strict", "affinity_samples": 3}, PRESETS, "strict")
    assert o["affinity_samples"] == 3 and "affinity_samples" in o["changed"]
    assert PR.normalize_predict_options({"preset": "strict"}, PRESETS, "strict")["affinity_samples"] == 5
    with pytest.raises(ApiError):
        PR.normalize_predict_options({"preset": "strict", "affinity_samples": 99}, PRESETS, "strict")
    a = PR.predict_argv(Path("/r"), {"yaml": {"file": "inputs/job.yaml"}}, o)
    assert "--diffusion_samples_affinity=3" in a and "--diffusion_samples=1" in a


def test_yaml_import_lists_everything_it_drops():
    text = ("version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: MKTAYIAKQR\n      msa: ./a.a3m\n"
            "      modifications:\n        - position: 3\n          ccd: SEP\n  - ligand:\n      id: B\n      smiles: CCO\n"
            "constraints:\n  - pocket: {binder: B, contacts: [[A, 3]]}\ntemplates:\n  - cif: t.cif\n"
            "properties:\n  - affinity:\n      binder: B\n")
    r = PR.import_yaml(text)
    assert [c["id"] for c in r["chains"]] == ["A", "B"] and r["binder"] == "B"
    assert r["skipped"] == ["MSA file of chain A"]   # modifications, constraints and templates go into the form now
    assert r["chains"][0]["modifications"] == [{"position": 3, "ccd": "SEP"}] and len(r["constraints"]) == 1
    assert r["templates"] == [{"file": "t.cif", "chain_id": "A", "template_id": None}]
    assert PR.import_yaml("version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: MKTAYIAKQR\n")["skipped"] == []


def test_run_inputs_export_manifest_and_raw_arrays(app):
    msa = ">q\n" + PROT + "\n>h\n" + PROT + "\n"
    rid = app.create_run(job(chains=[{"type": "protein", "id": "A", "sequence": PROT, "msa": {"mode": "file", "filename": "m.a3m", "content": msa}},
                                      {"type": "ligand", "id": "B", "smiles": "CC(=O)Nc1ccc(O)cc1"}], options={"preset": "strict", "samples": 2}))["id"]
    assert wait_state(app, rid, ("complete", "failed", "interrupted")) == "complete"
    inp = app.run_inputs(rid)
    assert inp["kind"] == "predict" and inp["binder"] == "B" and inp["chains"][0]["msa"]["content"] == msa and inp["options"]["samples"] == 2
    import io
    import zipfile

    from boltz_mlx.gui.exports import build_zip

    path, _ = build_zip(app.runs, rid)
    try:
        z = zipfile.ZipFile(path)
        names = z.namelist()
        assert any(n.endswith("/manifest.json") for n in names) and any(n.endswith("/inputs/job.yaml") for n in names)
        assert any(n.endswith("/inputs/msa_A.a3m") for n in names) and any("/raw/pae_" in n for n in names)
        man = json.loads(z.read(next(n for n in names if n.endswith("/manifest.json"))))
        assert man["kind"] == "predict" and man["files"]["model_files"]["0"]["structure"].startswith("structures/") and man["options"]["samples"] == 2
    finally:
        path.unlink(missing_ok=True)


def test_pae_missing_cells_are_marked_not_zero(tmp_path):
    m = np.full((6, 6), 5.0, dtype=np.float32)
    m[2, 3] = np.nan
    f = tmp_path / "pae.npz"
    np.savez(f, pae=m)
    d = pae_payload(f)
    b = base64.b64decode(d["data"])
    assert b[2 * 6 + 3] == d["missing"] == 255 and d["n_missing"] == 1 and b[0] == 40


def test_custom_steps_get_no_measured_time(app, monkeypatch):
    monkeypatch.setenv("BOLTZ_MLX_GUI_CHIP", "Apple M5 Max")  # a chip with a measured preset time and no bench rows
    payload = {"target": {"kind": "text", "filename": "t.fasta", "content": ">A\n" + PROT + "\n"},
               "ligands": {"kind": "text", "content": "CCO a\nCCN b\n"}, "options": {"preset": "standard"}}
    base = app.estimate(payload)["estimate"]
    custom = app.estimate({**payload, "options": {"preset": "standard", "steps": 200}})["estimate"]
    assert base["available"] and not base.get("custom")
    assert custom.get("custom") and custom["available"] is False and "total_s" not in custom
