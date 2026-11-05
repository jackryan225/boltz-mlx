"""No user SMILES can take down the GUI server, a screen or predict (boltz_mlx/smiles_guard.py).

RDKit failures that are C-level aborts cannot be caught in Python, so unusual molecules are first tried in a throw-away
child interpreter. These tests cover: what is classified as unusual, what the child does when it dies or hangs (fake
workers that abort / hang / fail to start), that ordinary input never starts a child, and the three entry points: the
screen's validate_ligand(s), the GUI (check_chains, depict) and predict's input check (exit code 2).

SMILES: the fork's alkyne salt (boltz-community 7407b86), its overflow molecule (9cca9d6), and hypervalent / unreadable
ones (the fork's valence commit b555625 names no SMILES; 'organometallics, hypervalent atoms' are covered here)."""

import subprocess
import sys
import time

import pytest

from boltz_mlx import smiles_guard as G
from boltz_mlx.gui.common import ApiError

SALT = "C#CCN(C)CC(=C)c1ccc(Cl)c(Cl)c1.O=C(O)C(=O)O"
BR = "Br" + "C" * 50
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
UNREADABLE = ["C(C", "not a smiles", "CS(=O)(=O)(=O)(=O)C", "[CH5]", "B(F)(F)(F)(F)F"]
METALS = ["[Fe]", "C[Hg]C", "[Cu+2]", "C[Pt](Cl)(Cl)N"]


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    G._VERDICTS.clear()
    spawns = []
    real = subprocess.Popen

    def counting(cmd, *a, **k):
        spawns.append(cmd)
        return real(cmd, *a, **k)

    monkeypatch.setattr(G.subprocess, "Popen", counting)
    return spawns


# ------------------------------------------------------------------------------------------------ classification
def test_ordinary_molecules_are_not_suspicious_and_start_no_child(_fresh):
    smis = [ASPIRIN, "CCO", "c1ccccc1", "CC(C)[C@H](N)C(=O)O", "C[N+](C)(C)C", "CS(=O)(=O)N", "OC(=O)c1ccccc1Br", BR]
    for s in smis:
        assert G.suspicion(G.parse_checked(s)) is None, s
    assert all(v is None for v in G.check_many(smis, True).values())
    assert _fresh == []


def test_unusual_molecules_are_suspicious():
    for s in METALS:
        assert G.suspicion(G.parse_checked(s)), s
    assert G.suspicion(G.parse_checked("[Na+].[Cl-]")) and G.suspicion(G.parse_checked("[CH2]C"))     # element / radical
    assert G.suspicion(G.parse_checked(SALT)) is None and "fragments" in G.suspicion(G.parse_checked(SALT), affinity=True)


@pytest.mark.parametrize("bad", UNREADABLE)
def test_unreadable_smiles_are_refused_in_process(bad, _fresh):
    err = G.check_smiles(bad)
    assert err and "RDKit" in err
    assert _fresh == []


def test_empty_and_overlong():
    assert G.check_smiles("  ") == "empty SMILES"
    assert "longer than" in G.check_smiles("C" * (G.MAX_SMILES_CHARS + 1))


# ------------------------------------------------------------------------------------------------ the child process
def test_real_child_answers_for_suspicious_input_in_one_spawn(_fresh):
    r = G.check_many(["[Fe]", "C[Hg]C", ASPIRIN, "C[Pt](Cl)(Cl)N"])
    assert r == {"[Fe]": None, "C[Hg]C": None, ASPIRIN: None, "C[Pt](Cl)(Cl)N": None}
    assert len(_fresh) == 1
    G.check_many(["[Fe]"])           # remembered (the GUI redraws the same molecule)
    assert len(_fresh) == 1


def test_affinity_ligand_goes_through_upstreams_standardize_steps(_fresh):
    assert G.check_smiles(SALT, affinity=True) is None   # the fork's alkyne salt: fine with the valence cache filled in
    err = G.check_smiles("C[Pt](Cl)(Cl)N", affinity=True)
    assert err and "ChEMBL" in err
    assert G.check_smiles("C[Pt](Cl)(Cl)N", affinity=False) is None


def _fake(tmp_path, body):
    f = tmp_path / "fake_worker.py"
    f.write_text("import sys, os, json, time\nfor line in sys.stdin:\n    req = json.loads(line)\n    s = req['smiles']\n" + body +
                 "    sys.stdout.write('@@' + json.dumps({'error': None}) + '\\n'); sys.stdout.flush()\n")
    return [sys.executable, str(f)]


def test_a_child_that_aborts_marks_only_its_smiles_and_the_rest_are_still_checked(tmp_path, monkeypatch, _fresh):
    monkeypatch.setattr(G, "worker_cmd", lambda: _fake(tmp_path, "    if 'Pt' in s: os.abort()\n"))
    items = [("[Fe]", False), ("C[Pt](Cl)Cl", False), ("[Zn]", False), ("C[Pt](C)C", False), ("[Cu]", False)]
    r = G.probe(items)
    assert [r[i] for i in items] == [None, r[items[1]], None, r[items[3]], None]
    assert "crashed (signal 6)" in r[items[1]] and "crashed (signal 6)" in r[items[3]]
    assert len(_fresh) == 3          # first child dies at item 2, the second at item 4, the third finishes


def test_a_child_that_hangs_is_killed(tmp_path, monkeypatch):
    monkeypatch.setattr(G, "worker_cmd", lambda: _fake(tmp_path, "    if 'Pt' in s: time.sleep(60)\n"))
    t = time.time()
    r = G.probe([("[Fe]", False), ("C[Pt]", False), ("[Zn]", False)], timeout=1.0)
    assert r[("[Fe]", False)] is None and r[("[Zn]", False)] is None and "did not finish" in r[("C[Pt]", False)]
    assert time.time() - t < 20


def test_a_child_that_cannot_start_is_an_error_not_a_verdict(tmp_path, monkeypatch):
    monkeypatch.setattr(G, "worker_cmd", lambda: [sys.executable, "-c", "import sys; sys.stderr.write('no rdkit'); sys.exit(3)"])
    with pytest.raises(RuntimeError, match="failed to start"):
        G.probe([("[Fe]", False)])


def test_probe_cost():
    """Measured, not just asserted: classification of ordinary input and one child for a batch of unusual input."""
    ordinary = [f"{'C' * k}{r}" for k in range(1, 41) for r in ("N(CC)c1ccccc1Cl", "C(=O)OC", "c1ccc2ccccc2c1", "C[C@H](N)C(=O)O", "S(=O)(=O)N")]
    t = time.perf_counter()
    G.check_many(ordinary)
    per_smiles_ms = (time.perf_counter() - t) / len(ordinary) * 1e3
    from boltz_mlx.screen.inputs import Ligand, validate_ligands

    t = time.perf_counter()
    validate_ligands([Ligand(f'l{i}', s, i) for i, s in enumerate(ordinary)])
    whole_ms = (time.perf_counter() - t) / len(ordinary) * 1e3
    unusual = [f"C[Hg]{'C' * k}" for k in range(1, 61)]
    t = time.perf_counter()
    r = G.check_many(unusual)
    batch_s = time.perf_counter() - t
    print(f"\nsmiles_guard cost: {per_smiles_ms:.3f} ms/SMILES ordinary (whole validate_ligands: {whole_ms:.2f} ms/ligand); {len(unusual)} unusual SMILES in one child: {batch_s:.2f} s")
    assert all(v is None for v in r.values())
    assert per_smiles_ms < 5 and batch_s < 10


# ------------------------------------------------------------------------------------------------ screen
def test_screen_marks_bad_ligands_invalid_and_keeps_the_rest(_fresh):
    from boltz_mlx.screen.inputs import Ligand, validate_ligands

    smis = [ASPIRIN, "C(C", "CS(=O)(=O)(=O)(=O)C", "[Fe]", "C[Pt](Cl)(Cl)N", SALT, BR, "CCO.O"]
    ligs = validate_ligands([Ligand(f"l{i}", s, i, f"l{i}") for i, s in enumerate(smis)])
    st = {l.ligand_id: (l.status, l.error) for l in ligs}
    assert st["l0"][0] == st["l5"][0] == st["l6"][0] == st["l7"][0] == "pending"
    assert st["l1"][0] == st["l2"][0] == "invalid" and "RDKit" in st["l1"][1]
    assert all(st[k][0] == "invalid" and "ChEMBL" in st[k][1] for k in ("l3", "l4"))   # metals: upstream's standardize refuses them
    assert ligs[5].smiles_used and "." not in ligs[5].smiles_used and ligs[5].n_atoms
    assert len(_fresh) == 1          # [Fe], the platinum compound and nothing else shared one child


def test_validate_ligand_single(_fresh):
    from boltz_mlx.screen.inputs import Ligand, validate_ligand

    assert validate_ligand(Ligand("a", ASPIRIN, 0)).status == "pending" and _fresh == []
    assert validate_ligand(Ligand("b", "CS(=O)(=O)(=O)(=O)C", 0)).status == "invalid"
    assert validate_ligand(Ligand("c", "C[Pt](Cl)(Cl)N", 0, ), affinity=False).status == "pending"


# ------------------------------------------------------------------------------------------------ GUI
def test_gui_form_error_for_a_bad_ligand():
    from boltz_mlx.gui.predict import check_chains

    prot = {"type": "protein", "id": "A", "sequence": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEV", "msa": {"mode": "empty"}}
    for bad in ("C(C", "CS(=O)(=O)(=O)(=O)C"):
        with pytest.raises(ApiError) as ex:
            check_chains({"chains": [prot, {"type": "ligand", "id": "B", "smiles": bad}]})
        assert ex.value.status == 400 and "chain B" in ex.value.message and "RDKit" in ex.value.message
    with pytest.raises(ApiError, match="ChEMBL"):   # the affinity binder is standardized by upstream, a co-factor is not
        check_chains({"chains": [prot, {"type": "ligand", "id": "B", "smiles": "C[Pt](Cl)(Cl)N"}], "affinity": {"binder": "B"}})
    ok = check_chains({"chains": [prot, {"type": "ligand", "id": "B", "smiles": "C[Pt](Cl)(Cl)N"}]})
    assert ok["chains"][1]["smiles"]


def test_gui_depict(_fresh):
    from boltz_mlx.gui import depict

    depict._CACHE.clear()
    with pytest.raises(ApiError) as ex:
        depict.depict_svg("CS(=O)(=O)(=O)(=O)C")
    assert ex.value.status == 400
    assert depict.depict_svg(ASPIRIN).startswith("<?xml") or "<svg" in depict.depict_svg(ASPIRIN)
    assert _fresh == []
    assert "<svg" in depict.depict_svg("C[Hg]C") and len(_fresh) == 1


# ------------------------------------------------------------------------------------------------ predict
def _predict_rc(tmp_path, capfd, yaml_text, name="in.yaml"):
    from boltz_mlx import cli

    f = tmp_path / name
    f.write_text(yaml_text)
    rc = 0
    try:
        cli.validate_predict_input(f, False)
    except cli.CliError as ex:   # what cli.main turns into exit code ex.code and one stderr line
        rc = ex.code
        return rc, ex.line()
    return rc, ""


def _yaml(smiles, binder=True):
    prot = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQ"
    return ("version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: " + prot + "\n      msa: empty\n"
            f"  - ligand:\n      id: B\n      smiles: '{smiles}'\n" + ("properties:\n  - affinity:\n      binder: B\n" if binder else ""))


@pytest.mark.parametrize("bad", ["C(C", "CS(=O)(=O)(=O)(=O)C"])
def test_predict_exit_2_names_the_ligand(tmp_path, capfd, bad):
    rc, line = _predict_rc(tmp_path, capfd, _yaml(bad))
    assert rc == 2 and "ligand B" in line and "invalid SMILES" in line and "Fix:" in line


def test_predict_exit_2_for_a_metal_affinity_binder_but_not_a_cofactor(tmp_path, capfd):
    rc, line = _predict_rc(tmp_path, capfd, _yaml("C[Pt](Cl)(Cl)N"))
    assert rc == 2 and "ChEMBL" in line
    assert _predict_rc(tmp_path, capfd, _yaml("C[Pt](Cl)(Cl)N", binder=False))[0] == 0


def test_predict_accepts_the_fork_salt_and_overflow_smiles(tmp_path, capfd):
    assert _predict_rc(tmp_path, capfd, _yaml(SALT))[0] == 0
    assert _predict_rc(tmp_path, capfd, _yaml(BR))[0] == 0


def test_predict_checks_fasta_smiles_and_non_string(tmp_path, capfd):
    rc, line = _predict_rc(tmp_path, capfd, ">A|protein|empty\nMKTAYIAKQRQISFVKSHFSRQLEERLGLIEV\n>B|smiles\nCS(=O)(=O)(=O)(=O)C\n", "in.fasta")
    assert rc == 2 and "ligand B" in line
    prot = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEV"
    rc, line = _predict_rc(tmp_path, capfd, f"version: 1\nsequences:\n  - protein: {{id: A, sequence: {prot}, msa: empty}}\n  - ligand: {{id: B, smiles: 12}}\n")
    assert rc == 2 and "must be a string" in line


def test_predict_main_exit_code_2_end_to_end(tmp_path, capfd, monkeypatch):
    from boltz_mlx import check as CK
    from boltz_mlx import cli

    monkeypatch.setattr(CK, "preflight", lambda *a, **k: None)
    f = tmp_path / "in.yaml"
    f.write_text(_yaml("CS(=O)(=O)(=O)(=O)C"))
    with pytest.raises(SystemExit) as ex:
        cli.main(["predict", str(f), "--out_dir", str(tmp_path)])
    assert ex.value.code == 2
    assert "invalid SMILES" in capfd.readouterr().err
