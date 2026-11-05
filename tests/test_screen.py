"""`boltz-mlx screen`: inputs, validation, ranking, results files, resume, error rows, seeding, dry-run.

CPU only; the model and the upstream featurizer are replaced by tests/screen_fakes.py (no checkpoints loaded).
"""

import argparse
import csv
import json
import random
from pathlib import Path

import pytest

from boltz_mlx.screen import results as R
from boltz_mlx.screen.inputs import InputError, Ligand, load_target, read_ligands, validate_ligand
from boltz_mlx.screen.main import add_parser, preset_environ, run
from boltz_mlx.screen.presets import DEFAULT_PRESET, PRESETS, affinity_estimate, default_preset, for_device, struct_estimate
from boltz_mlx.screen.runner import ScreenConfig, ScreenRunner, mw_correct
from screen_fakes import FakeEngine, FakeModel

SMILES = ["CCO", "c1ccccc1O", "CC(=O)Nc1ccc(O)cc1", "CCN(CC)CC", "OC(=O)c1ccccc1", "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "CCOC(=O)C", "c1ccncc1"]
PROT = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWUAEFSKFDALLELSDGSTPTHEVLEAE"


@pytest.fixture(autouse=True)
def _reset():
    FakeModel.reset()
    yield
    FakeModel.reset()


def _write_csv(path, rows, header=("id", "smiles")):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        if header:
            w.writerow(header)
        w.writerows(rows)
    return path


def _ligs(n=6, extra=()):
    out = []
    for i in range(n):
        out.append(validate_ligand(Ligand(ligand_id=f"cpd{i}", smiles=SMILES[i % len(SMILES)], index=i, record_id=f"cpd{i}")))
    for j, (lid, smi) in enumerate(extra):
        out.append(validate_ligand(Ligand(ligand_id=lid, smiles=smi, index=n + j, record_id=lid)))
    return out


def _target(tmp_path):
    return load_target(PROT[:60], msa="empty")


def _cfg(out, **kw):
    p = PRESETS["strict"]
    d = dict(out=Path(out), preset=p, steps=p.steps, recycling=p.recycling, affinity_steps=p.affinity_steps, chunk=3, device="cpu",
             save_structures=2)
    d.update(kw)
    return ScreenConfig(**d)


def _run(tmp_path, out="out", ligs=None, engine_kw=None, **cfg_kw):
    out = tmp_path / out
    eng = FakeEngine(out / "work", **(engine_kw or {}))
    r = ScreenRunner(_cfg(out, **cfg_kw), _target(tmp_path), ligs if ligs is not None else _ligs(), eng, log=lambda *a: None)
    return r, eng


def _values(out):
    return {r["ligand_id"]: (r["status"], r.get("affinity_pred_value"), r.get("affinity_probability_binary"), r.get("iptm"))
            for r in R.latest_rows(R.read_rows(Path(out) / "results.jsonl")).values()}


# ------------------------------------------------------------------------------------------------------------ inputs


def test_read_csv_header_variants_and_ids(tmp_path):
    p = _write_csv(tmp_path / "a.csv", [["x1", "CCO"], ["", "CCN"], ["x1", "CCC"], ["we ird/id", "c1ccccc1"]], header=("Compound_ID", "SMILES"))
    ligs = read_ligands(p)
    assert [l.ligand_id for l in ligs] == ["x1", "lig_00002", "x1", "we ird/id"]
    assert ligs[2].status == "invalid" and "duplicate" in ligs[2].error
    assert ligs[3].record_id == "we_ird_id"
    assert [l.smiles for l in ligs] == ["CCO", "CCN", "CCC", "c1ccccc1"]
    with pytest.raises(InputError):
        read_ligands(_write_csv(tmp_path / "b.csv", [["a", "b"]], header=("x", "y")))
    p = _write_csv(tmp_path / "c.csv", [["CCO", "m1"]], header=("structure", "label"))
    assert read_ligands(p, id_col="label", smiles_col="structure")[0].ligand_id == "m1"


def test_read_tsv_and_smi(tmp_path):
    t = tmp_path / "a.tsv"
    t.write_text("smiles\tname\nCCO\tethanol\n")
    assert read_ligands(t)[0].ligand_id == "ethanol"
    s = tmp_path / "a.smi"
    s.write_text("SMILES Name\n# comment\nCCO ethanol\n\nc1ccccc1 benzene ring\nCCN\n")
    ligs = read_ligands(s)
    assert [(l.ligand_id, l.smiles) for l in ligs] == [("ethanol", "CCO"), ("benzene ring", "c1ccccc1"), ("lig_00003", "CCN")]


def test_read_sdf(tmp_path):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    p = tmp_path / "a.sdf"
    w = Chem.SDWriter(str(p))
    for name, smi in (("lig_a", "C[C@H](O)CC"), ("", "c1ccncc1")):
        m = Chem.AddHs(Chem.MolFromSmiles(smi))
        AllChem.EmbedMolecule(m, randomSeed=1)
        m.SetProp("_Name", name)
        w.write(m)
    w.close()
    p.write_text(p.read_text() + "broken\n  RDKit\n\n  1  0  0  0  0  0  0  0  0  0999 V2000\nnot an atom line\nM  END\n$$$$\n")
    ligs = [validate_ligand(l) for l in read_ligands(p)]
    assert ligs[0].ligand_id == "lig_a" and ligs[0].status == "pending"
    assert Chem.MolToSmiles(Chem.MolFromSmiles(ligs[0].smiles)) == Chem.MolToSmiles(Chem.MolFromSmiles("C[C@H](O)CC"))  # 3D stereo kept
    assert ligs[1].ligand_id == "sdf_00002" and ligs[1].status == "pending"
    assert ligs[2].status == "invalid"


def test_validate_limits_and_warnings():
    v = lambda smi: validate_ligand(Ligand("x", smi, 0, "x"))  # noqa: E731
    assert v("not_a_smiles").status == "invalid"
    assert v("").status == "invalid"
    big = v("C" * 130)
    assert big.status == "invalid" and "130 atoms > 128" in big.error
    edge = v("C" * 128)
    assert edge.status == "pending" and edge.n_atoms == 128 and any("> 56" in w for w in edge.warnings)
    ok = v("C" * 56)
    assert ok.status == "pending" and not ok.warnings
    salt = v("CCN.Cl")
    assert salt.status == "pending" and salt.smiles_used == "CCN" and "fragments" in salt.warnings[0]
    st = v("CC(O)CC")
    assert any("stereocentre" in w for w in st.warnings)
    keep = v("OCC")
    assert keep.smiles_used == "OCC"  # single fragment: passed to upstream unchanged
    assert abs(keep.mw - 46.069) < 0.01 and keep.n_heavy == 3


def test_load_target_variants(tmp_path):
    t = load_target("mkta yiak", msa=None)
    assert t.doc["sequences"][0]["protein"]["sequence"] == "MKTAYIAK" and t.msa_source == "empty" and t.ligand_chain == "L"
    assert any("no MSA" in w for w in t.warnings)
    fa = tmp_path / "t.fasta"
    fa.write_text(">x\nMKTAY\nIAK\n>y\nGGGS\n")
    t = load_target(str(fa), msa="empty")
    assert [s["protein"]["id"] for s in t.doc["sequences"]] == ["A", "B"] and t.n_residues == 12
    a3m = tmp_path / "m.a3m"
    a3m.write_text(">q\nMKTAYIAK\n>h1\nMKTAYIAR\n")
    with pytest.raises(InputError):
        load_target(str(fa), msa=str(a3m))  # one MSA for two different sequences
    t = load_target("MKTAYIAK", msa=str(a3m), pocket=["A:3,A:5"])
    assert t.msa_paths == [str(a3m.resolve())] and t.msa_source == "a3m:m.a3m"
    assert t.doc["constraints"] == [{"pocket": {"binder": "L", "contacts": [["A", 3], ["A", 5]]}}]
    with pytest.raises(InputError):
        load_target("MKTAYIAK", msa="empty", pocket=["Z:3"])
    with pytest.raises(InputError):
        load_target("not a sequence 123")
    with pytest.raises(InputError):
        load_target("MKTAYIAK", msa=str(tmp_path / "missing.a3m"))
    # complex YAML (receptor + binder ligand + affinity property) as target: binder slot reused, msa path resolved from the YAML dir
    y = tmp_path / "cx.yaml"
    y.write_text("version: 1\nsequences:\n  - protein:\n      id: H\n      sequence: MKTAYIAK\n      msa: m.a3m\n"
                 "  - ligand:\n      id: Q\n      smiles: CCO\nproperties:\n  - affinity:\n      binder: Q\n")
    t = load_target(str(y))
    assert t.ligand_chain == "Q" and len(t.doc["sequences"]) == 1 and "properties" not in t.doc
    assert t.doc["sequences"][0]["protein"]["msa"] == str(a3m.resolve())
    lig_doc = t.ligand_yaml("CCN")
    assert lig_doc["sequences"][-1] == {"ligand": {"id": "Q", "smiles": "CCN"}}
    assert lig_doc["properties"] == [{"affinity": {"binder": "Q"}}]
    assert t.key() == load_target(str(y)).key()
    a3m.write_text(">q\nMKTAYIAK\n>h1\nMKTAYIAQ\n")
    assert t.key() != load_target(str(y)).key()  # MSA content is part of the resume key


# ----------------------------------------------------------------------------------------------------- ranking / files


def test_ranking_and_csv(tmp_path):
    rows = [
        {"ligand_id": "a", "status": "ok", "affinity_probability_binary": 0.2, "affinity_pred_value": -1.0, "index": 0},
        {"ligand_id": "b", "status": "ok", "affinity_probability_binary": 0.9, "affinity_pred_value": 1.5, "index": 1},
        {"ligand_id": "c", "status": "ok", "affinity_probability_binary": 0.9, "affinity_pred_value": 0.5, "index": 2},
        {"ligand_id": "d", "status": "invalid", "error": "bad", "index": 3},
        {"ligand_id": "e", "status": "error", "error": "boom", "index": 4},
        {"ligand_id": "e", "status": "ok", "affinity_probability_binary": 0.5, "affinity_pred_value": 0.0, "index": 4},  # retried
    ]
    by_prob = [(r["ligand_id"], r["rank"]) for r in R.rank_rows(rows, "prob")]
    assert by_prob == [("b", 1), ("c", 2), ("e", 3), ("a", 4), ("d", None)]
    by_val = [(r["ligand_id"], r["rank"]) for r in R.rank_rows(rows, "value")]
    assert by_val == [("a", 1), ("e", 2), ("c", 3), ("b", 4), ("d", None)]
    R.write_csv(tmp_path / "r.csv", rows, "prob")
    got = list(csv.DictReader(open(tmp_path / "r.csv")))
    assert list(got[0].keys()) == R.CSV_COLUMNS
    assert [g["ligand_id"] for g in got] == ["b", "c", "e", "a", "d"] and got[-1]["rank"] == "" and got[-1]["error"] == "bad"
    with pytest.raises(ValueError):
        R.rank_rows(rows, "bogus")


def test_append_row_survives_torn_line(tmp_path):
    p = tmp_path / "x.jsonl"
    R.append_row(p, {"ligand_id": "a"})
    with open(p, "a") as f:
        f.write('{"ligand_id": "b", "sta')  # killed mid-write
    R.append_row(p, {"ligand_id": "c"})
    assert [r["ligand_id"] for r in R.read_rows(p)] == ["a", "c"]


def test_mw_correct_matches_upstream_formula():
    assert mw_correct(0.5, 300.0) == pytest.approx(1.03525938 * 0.5 - 0.59992683 * 300.0 ** 0.3 + 2.83288489)


# ------------------------------------------------------------------------------------------------------------- runner


def test_end_to_end_two_phase_one_checkpoint(tmp_path):
    r, eng = _run(tmp_path, ligs=_ligs(7))
    st = r.run()
    assert st["complete"] and st["ok_total"] == 7 and st["error_total"] == 0
    assert FakeModel.max_alive == 1                       # never both checkpoints resident
    assert eng.loads == ["conf", "aff"] * 3                # 3 chunks of <= 3: conf then aff per chunk
    names = [c[0] for c in FakeModel.calls]
    assert names == ["conf"] * 3 + ["aff"] * 3 + ["conf"] * 3 + ["aff"] * 3 + ["conf"] + ["aff"]
    rows = list(csv.DictReader(open(tmp_path / "out" / "results.csv")))
    assert len(rows) == 7 and [int(x["rank"]) for x in rows] == list(range(1, 8))
    probs = [float(x["affinity_probability_binary"]) for x in rows]
    assert probs == sorted(probs, reverse=True)
    full = R.read_rows(tmp_path / "out" / "results.jsonl")
    one = full[0]
    for k in ("ligand_id", "smiles", "affinity_pred_value", "value_mw_corrected", "affinity_probability_binary", "iptm", "ligand_iptm",
              "complex_plddt", "ligand_plddt", "seconds", "mode", "msa_source", "seed", "status", "warnings", "t_done", "affinity_iptm_samples"):
        assert k in one, k
    assert one["value_mw_corrected"] == pytest.approx(mw_correct(one["affinity_pred_value"], one["affinity_mw"]))
    assert one["mode"] == "strict" and one["msa_source"] == "empty" and one["preset_validated"] is True
    assert len(list((tmp_path / "out" / "structures").iterdir())) == 2  # --save-structures 2
    assert json.loads((tmp_path / "out" / "screen_config.json").read_text())["seed"] == 0


def test_resume_after_kill_no_duplicates_identical_results(tmp_path):
    ref, _ = _run(tmp_path, out="ref")
    ref.run()
    # kill in the affinity phase of the second chunk (cpd4); structures of chunk 2 are already written
    r, eng = _run(tmp_path, out="killed", engine_kw={"interrupt_at": ("affinity", "cpd4")})
    with pytest.raises(KeyboardInterrupt):
        r.run()
    first = R.read_rows(tmp_path / "killed" / "results.jsonl")
    assert 3 <= len(first) < 6
    FakeModel.reset()
    r2, eng2 = _run(tmp_path, out="killed")
    st = r2.run()
    rows = R.read_rows(tmp_path / "killed" / "results.jsonl")
    ids = [x["ligand_id"] for x in rows]
    assert len(ids) == len(set(ids)) == 6 and st["complete"]
    assert _values(tmp_path / "killed") == _values(tmp_path / "ref")
    # the chunk-2 structures were reused, not recomputed: only the unfinished chunk's aff calls + nothing for done ligands
    assert "cpd3" not in eng2.prepared and "cpd4" not in eng2.prepared
    assert any(x.get("reused_structure") for x in rows if x["ligand_id"] in ("cpd4", "cpd5"))
    # a third run does nothing
    FakeModel.reset()
    r3, eng3 = _run(tmp_path, out="killed")
    st3 = r3.run()
    assert st3["rows_written_this_run"] == 0 and eng3.loads == [] and len(R.read_rows(tmp_path / "killed" / "results.jsonl")) == 6


def test_seeding_independent_of_order_chunk_and_prefetch(tmp_path):
    a, _ = _run(tmp_path, out="a", chunk=6, prefetch=False)
    a.run()
    ligs = _ligs()
    random.Random(3).shuffle(ligs)
    b, _ = _run(tmp_path, out="b", ligs=ligs, chunk=2, prefetch=True)  # prefetch is the default (both phases)
    b.run()
    c, _ = _run(tmp_path, out="c", seed=1)
    c.run()
    va, vb, vc = _values(tmp_path / "a"), _values(tmp_path / "b"), _values(tmp_path / "c")
    assert va == vb
    assert all(va[k][1] != vc[k][1] for k in va)  # another global seed gives other noise


def test_error_rows_do_not_stop_the_run(tmp_path):
    ligs = _ligs(5, extra=[("bad_smiles", "C1CC"), ("huge", "C" * 140)])
    r, eng = _run(tmp_path, ligs=ligs, engine_kw={"fail": {"cpd1": "prepare", "cpd2": "featurize", "cpd3": "affinity"}})
    st = r.run()
    v = _values(tmp_path / "out")
    assert v["cpd0"][0] == "ok" and v["cpd4"][0] == "ok"
    assert v["cpd1"][0] == v["cpd2"][0] == v["cpd3"][0] == "error"
    assert v["bad_smiles"][0] == v["huge"][0] == "invalid"
    rows = R.latest_rows(R.read_rows(tmp_path / "out" / "results.jsonl"))
    assert rows["cpd1"]["error_phase"] == "structure" and "prepare failed" in rows["cpd1"]["error"]
    assert rows["cpd3"]["error_phase"] == "affinity" and rows["cpd3"]["iptm"] is not None  # structure metrics kept
    assert "128" in rows["huge"]["error"]
    assert st["errors_this_run"] == 3 and st["complete"]
    got = list(csv.DictReader(open(tmp_path / "out" / "results.csv")))
    assert [g["status"] for g in got[:2]] == ["ok", "ok"] and all(g["rank"] == "" for g in got[2:])
    # resume: error rows stay (no retry) unless --retry-errors; invalid rows are never re-emitted
    r2, eng2 = _run(tmp_path, ligs=_ligs(5, extra=[("bad_smiles", "C1CC"), ("huge", "C" * 140)]))
    assert r2.run()["rows_written_this_run"] == 0
    r3, eng3 = _run(tmp_path, ligs=_ligs(5, extra=[("bad_smiles", "C1CC"), ("huge", "C" * 140)]), retry_errors=True)
    st3 = r3.run()
    assert st3["rows_written_this_run"] == 3 and st3["error_total"] == 0
    got = list(csv.DictReader(open(tmp_path / "out" / "results.csv")))
    assert len(got) == 7 and sum(g["status"] == "ok" for g in got) == 5  # latest row per ligand, no duplicates


def test_config_change_refused_and_id_conflict_skipped(tmp_path):
    r, _ = _run(tmp_path)
    r.run()
    r2, _ = _run(tmp_path, steps=50)
    with pytest.raises(SystemExit):
        r2.run()
    # chunk size / ranking / structures do not change predictions: allowed
    r3, _ = _run(tmp_path, chunk=1, rank_by="value", save_structures=0)
    assert r3.run()["rows_written_this_run"] == 0
    ligs = _ligs()
    ligs[0].smiles = "CCCCCC"  # same id, different molecule
    r4, eng4 = _run(tmp_path, ligs=ligs)
    assert r4.run()["rows_written_this_run"] == 0 and eng4.loads == []


def test_max_minutes_stops_cleanly(tmp_path):
    t = [0.0]

    def clock():
        t[0] += 7.0
        return t[0]

    out = tmp_path / "out"
    r = ScreenRunner(_cfg(out, max_minutes=1.0), _target(tmp_path), _ligs(), FakeEngine(out / "work"), log=lambda *a: None, clock=clock)
    st = r.run()
    assert st["stopped_by_budget"] and not st["complete"] and st["remaining"] > 0


# ------------------------------------------------------------------------------------------------------------ CLI / plan


def _args(argv):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    add_parser(sub)
    return ap.parse_args(["screen", *argv])


def test_cli_dry_run_plan_and_estimate(tmp_path, capsys):
    lig = _write_csv(tmp_path / "l.csv", [[f"m{i}", s] for i, s in enumerate(SMILES)] + [["bad", "xx("]])
    lines = []
    rc = run(_args(["--target", PROT[:400], "--ligands", str(lig), "--out", str(tmp_path / "o"), "--dry-run", "--chunk", "4"]),
             [], log=lines.append)
    text = "\n".join(lines)
    assert rc == 0
    assert "9 read, 8 valid, 1 invalid" in text and "8 to run" in text and "ESTIMATE" in text and "not a measurement" in text
    assert "chunks: 2 x <= 4" in text and "invalid: bad" in text
    assert not (tmp_path / "o" / "results.jsonl").exists()  # dry run writes nothing
    assert FakeModel.calls == []


def test_cli_run_with_fake_engine_and_preset_env(tmp_path):
    lig = _write_csv(tmp_path / "l.csv", [[f"m{i}", s] for i, s in enumerate(SMILES[:4])])
    out = tmp_path / "o"
    eng = FakeEngine(out / "work")
    rc = run(_args(["--target", PROT[:50], "--ligands", str(lig), "--out", str(out), "--rank-by", "value", "--save-structures", "all",
                    "--msa", "empty"]), [], engine=eng, log=lambda *a: None)
    assert rc == 0
    got = list(csv.DictReader(open(out / "results.csv")))
    vals = [float(g["affinity_pred_value"]) for g in got]
    assert vals == sorted(vals) and len(list((out / "structures").iterdir())) == 4
    assert got[0]["mode"] == DEFAULT_PRESET
    env = preset_environ({"BOLTZ_MLX_TF32": "1", "MLX_ENABLE_TF32": "1", "BOLTZ_MLX_CACHE_GB": "3", "HOME": "/h"}, PRESETS["f16"])
    assert env == {"BOLTZ_MLX_CACHE_GB": "3", "HOME": "/h", "BOLTZ_MLX_PRECISION": "f16"}


def test_presets_and_estimates():
    assert DEFAULT_PRESET in PRESETS and PRESETS["strict"].validated and PRESETS["strict"].env == {}
    for p in PRESETS.values():
        assert p.affinity_recycling == 5 and p.affinity_samples == 5 and p.recycling == 3  # never reduced
        # the est_* inputs are reproduced at the step counts they were measured at (est_steps, default the preset's)
        assert struct_estimate(p, 117, p.est_steps or p.steps) == pytest.approx(p.est_struct_117)
        assert struct_estimate(p, 400, p.est_steps or p.steps) == pytest.approx(p.est_struct_400)
        assert affinity_estimate(p, p.est_affinity_steps or p.affinity_steps) == pytest.approx(p.est_affinity)
        assert struct_estimate(p, 600) > struct_estimate(p, 400)
    assert PRESETS["f16"].env == {"BOLTZ_MLX_PRECISION": "f16"}
    s = PRESETS["screen"]  # the screen preset: f16 + 20 + 25 steps, scaled from the 50 / 200-step bench rows
    assert (s.steps, s.affinity_steps, s.recycling, s.affinity_samples, s.validated) == (20, 25, 3, 5, True)
    assert struct_estimate(s, 117) == pytest.approx(0.843 * (0.6 + 0.4 * 20 / 50))
    assert affinity_estimate(s) == pytest.approx(7.634 * (0.3 + 0.7 * 25 / 200))


def test_default_preset_by_kernel_path(tmp_path, monkeypatch):
    # project decision: screen on the NAX path (M5-class GPUs), strict on M1-M4 (tensorops / portable); f16 stays available by name
    from boltz_mlx import kernels as K

    assert DEFAULT_PRESET == "screen" and default_preset("nax") == "screen"
    assert default_preset("tensorops") == default_preset("portable") == "strict"
    off = for_device(PRESETS["screen"], "tensorops")
    assert off.name == "screen-tensorops" and not off.validated and off.steps == 20
    assert _args(["--target", "x", "--ligands", "l.csv", "--out", "o"]).preset is None  # resolved by run / build_config
    lig = _write_csv(tmp_path / "l.csv", [[f"m{i}", s] for i, s in enumerate(SMILES[:2])])
    for path, want in (("nax", "screen"), ("tensorops", "strict")):
        monkeypatch.setattr(K, "_CAP", K.pinned_status(path, f"test pin {path}"))
        lines = []
        assert run(_args(["--target", PROT[:60], "--ligands", str(lig), "--out", str(tmp_path / path), "--dry-run"]), [],
                   log=lines.append) == 0
        assert f"screen: preset {want} (the default for kernel path {path})" in lines
        assert any(l.startswith(f"preset: {want}  - ") for l in lines), lines
    lines = []
    run(_args(["--target", PROT[:60], "--ligands", str(lig), "--out", str(tmp_path / "x"), "--dry-run", "--preset", "screen"]),
        [], log=lines.append)  # explicit on M1-M4: the f16 path, labelled and not gated
    assert any(l.startswith("preset: screen-tensorops  - ") and l.endswith("[NOT a gated preset]") for l in lines), lines


def test_preset_environ_keeps_cfg_chip():
    # the per-device launch-config entry (speed only, bitwise-neutral) reaches the re-exec'd screen child: A/B runs of
    # tuned entries (BOLTZ_MLX_CFG_CHIP=default) used to run this chip's entry in both arms
    env = preset_environ({"BOLTZ_MLX_CFG_CHIP": "default", "BOLTZ_MLX_NAX_GEMM_CFG": "o=1"}, PRESETS["screen"])
    assert env == {"BOLTZ_MLX_CFG_CHIP": "default", "BOLTZ_MLX_PRECISION": "f16"}


@pytest.mark.parametrize("chip,steps,affinity_steps,measured", [
    ("Apple M5 Max", 50, 200, 8.0),    # the earlier screen settings (f16 + 50 + 200), --dry-run --measure 5
    ("Apple M5 Pro", 50, 200, 15.5)])  # (the screen preset @20 + 25: test_screen_estimate_t400_m5max_cold_screens)
def test_screen_estimate_t400_near_measure(tmp_path, chip, steps, affinity_steps, measured):
    # T400 (lig_aff target, 384 residues; ligands of 22-37 atoms), f16 screen: the dry-run ESTIMATE is within 15 % of
    # the --dry-run --measure 5 s/ligand of that chip at those step counts (the old inputs said ~36 s at 50 + 200)
    import re

    from boltz_mlx.screen.inputs import Target
    from boltz_mlx.screen.runner import plan_lines

    p = PRESETS["screen"]
    t = Target(doc={"sequences": []}, ligand_chain="B", n_residues=384, msa_source="a3m", msa_paths=[], source="t.yaml")
    ligs = [Ligand(ligand_id=f"l{n}", smiles="C" * n, index=i, record_id=f"l{n}", smiles_used="C" * n, n_atoms=n)
            for i, n in enumerate((22, 25, 30, 37))]
    hw = {"chip": chip, "architecture": "applegpu_g17", "path": "nax", "nax": True}
    cfg = ScreenConfig(out=tmp_path, preset=p, steps=steps, recycling=3, affinity_steps=affinity_steps, hw=hw)
    line = next(l for l in plan_lines(cfg, t, ligs, set()) if l.startswith("ESTIMATE"))
    assert f"from {chip} bench rows" in line and "not measured on this chip" not in line, line
    est = float(re.search(r"\): ~([0-9.]+) s/ligand \(range", line).group(1))
    assert abs(est / measured - 1) < 0.15, line


# Rung 5 G1 on the M5 Max (job boltz-val-g1-t400-20261029a, class 1, AC power): 4 full cold T400 screens with
# the screen preset (f16 + 20 + 25), 199 ligands each: MEASURED 6.683 / 7.123 / 7.207 / 6.323 s/ligand, pooled 6.835.
G1_POOLED_S, G1_PASSES_34_S = 6.835, (1434.2 + 1258.2) / 398


def test_screen_estimate_t400_m5max_cold_screens(tmp_path):
    # target fixed before G1 passes 3-4 (fixed 2026-10-29): the plan's T400 s/ligand within +-10 % of the G1 pooled cold
    # rate. NOT independent: the M5 Max s20 / aff25 estimate point was calibrated on passes 1-2, half of the pooled rate;
    # passes 3-4 alone are the out-of-sample check (also within 10 %).
    import re

    from boltz_mlx.screen.inputs import Target
    from boltz_mlx.screen.runner import plan_lines

    p = PRESETS["screen"]
    t = Target(doc={"sequences": []}, ligand_chain="B", n_residues=384, msa_source="a3m", msa_paths=[], source="t.yaml")
    ligs = [Ligand(ligand_id=f"l{n}", smiles="C" * n, index=i, record_id=f"l{n}", smiles_used="C" * n, n_atoms=n)
            for i, n in enumerate((22, 25, 30, 37))]
    hw = {"chip": "Apple M5 Max", "architecture": "applegpu_g17", "path": "nax", "nax": True}
    cfg = ScreenConfig(out=tmp_path, preset=p, steps=p.steps, recycling=3, affinity_steps=p.affinity_steps, hw=hw)
    line = next(l for l in plan_lines(cfg, t, ligs, set()) if l.startswith("ESTIMATE"))
    assert "G1 passes 1-2" in line and "not measured on this chip" not in line, line
    est = float(re.search(r"\): ~([0-9.]+) s/ligand \(range", line).group(1))
    assert abs(est / G1_POOLED_S - 1) <= 0.10, line
    assert abs(est / G1_PASSES_34_S - 1) <= 0.10, line


def test_estimate_points_by_steps():
    # the M5 Max screen entry holds two points: f16 + 20 + 25 from full cold screen rows (G1 passes 1-2) and the 50 / 200
    # bench rows; each phase starts from the point nearest its own step count, the overhead from the nearest point overall
    from boltz_mlx.screen.presets import est_overhead, est_point

    p, m = PRESETS["screen"], "Apple M5 Max"
    assert est_point(p, m)["steps"] == 20 and est_point(p, m)["struct_400"] == 4.25 and est_overhead(p, m) == 0.08
    assert est_point(p, m, 50, 200)["struct_400"] == 3.095 and est_overhead(p, m, 50, 200) == 1.0
    assert struct_estimate(p, 400, 20, m) == pytest.approx(4.25) and struct_estimate(p, 400, 50, m) == pytest.approx(3.095)
    assert affinity_estimate(p, 25, m) == pytest.approx(2.40) and affinity_estimate(p, 200, m) == pytest.approx(3.784)
    assert est_point(p, "Apple M5 Pro")["steps"] == 50 and est_overhead(p, "Apple M5 Pro") == 1.0  # the preset's own rows


def test_corrected_smiles_for_invalid_ligand_is_screened(tmp_path):
    r, _ = _run(tmp_path, ligs=_ligs(2, extra=[("fixme", "C1CC")]))
    r.run()
    assert _values(tmp_path / "out")["fixme"][0] == "invalid"
    r2, eng2 = _run(tmp_path, ligs=_ligs(2, extra=[("fixme", "C1CC1")]))
    assert r2.run()["rows_written_this_run"] == 1 and eng2.prepared == ["fixme"]
    got = list(csv.DictReader(open(tmp_path / "out" / "results.csv")))
    assert len(got) == 3 and all(g["status"] == "ok" for g in got)


# ------------------------------------------------------------------------- R6-prof: lean writer dict, prefetch default
def test_structure_pred_dict_lean_matches_full():
    """The screen's lean writer dict: same tensors for every key it keeps; no s / z (write_embeddings off), optionally
    no pae / pde; logits never exported."""
    import mlx.core as mx
    import numpy as np
    import torch

    from boltz_mlx.cli import _structure_pred_dict

    n, M = 6, 1
    rng = np.random.default_rng(0)

    def r(*s):
        return mx.array(rng.normal(size=s).astype(np.float32))

    out = {"sample_atom_coords": r(M, 12, 3), "s": r(1, n, 8), "z": r(1, n, n, 4), "s_inputs": r(1, n, 8),
           "pdistogram": r(1, n, n, 1, 64), "pae_logits": r(M, n, n, 64), "pde_logits": r(M, n, n, 64),
           "plddt_logits": r(M, n, 50), "plddt": r(M, n), "pae": r(M, n, n), "pde": r(M, n, n),
           "complex_plddt": r(M), "complex_iplddt": r(M), "complex_pde": r(M), "complex_ipde": r(M), "ptm": r(M),
           "iptm": r(M), "ligand_iptm": r(M), "protein_iptm": r(M), "pair_chains_iptm": {0: {0: r(M), 1: r(M)}},
           "fp16_fallbacks": 0}
    batch = {"atom_pad_mask": torch.ones(1, 12), "token_pad_mask": torch.ones(1, n)}
    full = _structure_pred_dict(out, batch)
    lean = _structure_pred_dict(out, batch, embeddings=False, pae_pde=False)
    assert {"s", "z", "pae", "pde"} <= set(full) and not ({"s", "z", "pae", "pde"} & set(lean))
    assert not any(k.endswith("_logits") for k in full)
    assert set(full) - set(lean) == {"s", "z", "pae", "pde"}
    for k in lean:
        if isinstance(lean[k], torch.Tensor):
            assert torch.equal(lean[k], full[k]), k
    assert torch.equal(full["confidence_score"], (4 * full["complex_plddt"] + full["iptm"]) / 5)


def test_screen_prefetch_cli_default():
    ap = argparse.ArgumentParser()
    add_parser(ap.add_subparsers())
    base = ["screen", "--target", "x.yaml", "--ligands", "l.csv", "--out", "o"]
    assert ap.parse_args(base).prefetch is True
    assert ap.parse_args(base + ["--prefetch"]).prefetch is True
    assert ap.parse_args(base + ["--no-prefetch"]).prefetch is False


def test_tiny_ligand_rule_is_per_ligand_in_the_config_and_the_estimate(tmp_path):
    """Fast: a ligand with < 10 heavy atoms runs 200 structure steps, the others keep the preset's; Full / explicit steps: off."""
    from boltz_mlx.screen.runner import estimate_seconds

    p = PRESETS["screen"]
    ligs = _ligs(0, extra=[("tiny", "OC(=O)[O-]"), ("nine", "C" * 9), ("ten", "C" * 10), ("big", "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1")])
    assert [l.n_heavy for l in ligs] == [4, 9, 10, 26]
    cfg = ScreenConfig(out=tmp_path, preset=p, steps=p.steps, recycling=3, affinity_steps=p.affinity_steps, tiny_steps=200)
    assert [cfg.steps_for(l) for l in ligs] == [200, 200, 20, 20]
    assert cfg.steps_reason(ligs[0]) == "ligand tiny has 4 heavy atoms (< 10)" and cfg.steps_reason(ligs[3]) is None
    off = ScreenConfig(out=tmp_path, preset=p, steps=p.steps, recycling=3, affinity_steps=p.affinity_steps)
    assert [off.steps_for(l) for l in ligs] == [20] * 4
    assert "tiny_ligand_steps" in cfg.result_config(_target(tmp_path)) and "tiny_ligand_steps" not in off.result_config(_target(tmp_path))
    assert cfg.mode == off.mode == "screen"          # the base steps are the preset's: the run is not "custom"
    t = _target(tmp_path)
    assert estimate_seconds(cfg, t, ligs[0]) > estimate_seconds(off, t, ligs[0])          # 200 steps cost more than 20
    assert estimate_seconds(cfg, t, ligs[3]) == pytest.approx(estimate_seconds(off, t, ligs[3]))


def test_tiny_ligand_rule_ligand_without_a_count_is_not_tiny(tmp_path):
    p = PRESETS["screen"]
    cfg = ScreenConfig(out=tmp_path, preset=p, steps=20, recycling=3, affinity_steps=25, tiny_steps=200)
    assert cfg.steps_for(Ligand(ligand_id="x", smiles="C", index=0)) == 20   # n_heavy unknown (invalid ligand)


def test_build_config_enables_the_rule_only_for_fast_without_steps():
    from boltz_mlx.screen.main import build_config

    def args(**kw):
        ap = argparse.ArgumentParser()
        add_parser(ap.add_subparsers(dest="cmd"))
        a = ap.parse_args(["screen", "--target", "x", "--ligands", "l.csv", "--out", "o", *kw.pop("argv", [])])
        return a

    presets = dict(PRESETS)
    assert build_config(args(argv=["--preset", "fast"]), presets).tiny_steps == 200
    assert build_config(args(argv=["--preset", "fast", "--steps", "20"]), presets).tiny_steps is None
    assert build_config(args(argv=["--preset", "full"]), presets).tiny_steps is None
    assert build_config(args(argv=["--preset", "turbo"]), presets).tiny_steps is None
