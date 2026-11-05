"""GUI: type-ahead protein search (network stubbed: lookup._fetch is replaced, no live calls), the Predict form's constraints,
modifications and templates (written into the Boltz-2 YAML and checked first), and the decluttered About / preset texts.

CPU only; the predict subprocess is the fake one (tests/gui_fake_predict.py)."""

from __future__ import annotations

import json
import urllib.parse
from pathlib import Path

import pytest
import yaml
from test_gui_targets import PROT, STRUCT, app, call, env, http_gui, wait_state  # noqa: F401  (fixtures)

from boltz_mlx.gui import constraints as CX
from boltz_mlx.gui import lookup as LK
from boltz_mlx.gui import predict as PR
from boltz_mlx.gui import presets as GP
from boltz_mlx.gui.common import PRIVACY_GUI, ApiError


def up(acc, name, gene, org, tax, seq, reviewed=True):
    return {"primaryAccession": acc, "entryType": "UniProtKB reviewed (Swiss-Prot)" if reviewed else "UniProtKB unreviewed (TrEMBL)",
            "proteinDescription": {"recommendedName": {"fullName": {"value": name}}}, "genes": [{"geneName": {"value": gene}}],
            "organism": {"scientificName": org, "taxonId": tax}, "sequence": {"value": seq, "length": len(seq)}}


@pytest.fixture
def net(monkeypatch):
    """The one network function, answering by URL: a callable(url) -> text|dict, or raising."""
    calls = []
    box = {"answer": None}

    def fake(url):
        calls.append(url)
        r = box["answer"](url)
        if isinstance(r, Exception):
            raise r
        return r if isinstance(r, str) else json.dumps(r)

    monkeypatch.setattr(LK, "_fetch", fake)
    monkeypatch.setattr(LK, "_cache", {})
    return calls, box


def q_of(url):
    return urllib.parse.parse_qs(urllib.parse.urlparse(url).query)


# ------------------------------------------------------------------------------------------------------------ search
def test_search_puts_human_swissprot_first_and_carries_the_sequence(net):
    calls, box = net

    def answer(url):
        q = q_of(url)["query"][0]
        if "organism_id:9606" in q:
            return {"results": [up("Q86T13", "C-type lectin 14A", "CLEC14A", "Homo sapiens", 9606, "MAAA"),
                                up("P00533", "Epidermal growth factor receptor", "EGFR", "Homo sapiens", 9606, "MRPSG")]}
        return {"results": [up("Q01279", "Epidermal growth factor receptor", "Egfr", "Mus musculus", 10090, "MRPSA"),
                            up("A0A000", "EGFR-like", "egfr", "Danio rerio", 7955, "MKK", reviewed=False),
                            up("P00533", "Epidermal growth factor receptor", "EGFR", "Homo sapiens", 9606, "MRPSG")]}

    box["answer"] = answer
    r = LK.search({"query": "  egfr ", "type": "protein"})
    assert [x["id"] for x in r["results"]] == ["P00533", "Q86T13", "Q01279", "A0A000"]   # exact gene first among human Swiss-Prot
    top = r["results"][0]
    assert top["sequence"] == "MRPSG" and top["gene"] == "EGFR" and top["organism"] == "Homo sapiens" and top["length"] == 5 and top["reviewed"]
    assert not r["results"][-1]["reviewed"] and not r["offline"]
    assert len(calls) == 2 and all(u.startswith("https://rest.uniprot.org/uniprotkb/search?") for u in calls)
    qs = sorted(q_of(u)["query"][0] for u in calls)
    assert qs == ['"egfr"', '("egfr") AND (organism_id:9606) AND (reviewed:true)']   # words quoted: never query syntax
    assert q_of(calls[0])["fields"][0].startswith("accession,protein_name,gene_names,organism_name") and q_of(calls[0])["format"] == ["json"]
    # the same search again is answered from memory
    assert LK.search({"query": "EGFR", "type": "protein"})["cached"] and len(calls) == 2


def test_search_by_accession_and_by_pdb_id(net):
    calls, box = net
    fasta = (">8CPK_1|Chains A, B|Glycoprotein|Escherichia coli (562)\nMKTAYIAKQR\n>8CPK_2|Chain C|DNA strand|\nACGTACGT\n")
    box["answer"] = lambda url: ({"results": [up("P00533", "Epidermal growth factor receptor", "EGFR", "Homo sapiens", 9606, "MRPSG")]}
                                 if "uniprot" in url else fasta)
    r = LK.search({"query": "p00533"})
    assert [x["id"] for x in r["results"]] == ["P00533"] and q_of(calls[-1])["query"] == ["accession:P00533"]
    iso = LK.search({"query": "P00533-2"})["results"][0]
    assert iso["id"] == "P00533-2" and iso["sequence"] == ""          # an isoform: the page fetches its own sequence by id
    n = len(calls)
    r = LK.search({"query": "8cpk", "type": "protein"})
    assert calls[n:] == ["https://www.rcsb.org/fasta/entry/8CPK"]     # a PDB id: the entry's chains only, no UniProt text search
    assert [(x["id"], x["chains"], x["organism"], x["length"]) for x in r["results"]] == [("8CPK_A", ["A", "B"], "Escherichia coli", 10)]
    assert [x["id"] for x in LK.search({"query": "8CPK_C", "type": "dna"})["results"]] == ["8CPK_C"]
    assert LK.search({"query": "8CPK_B", "type": "protein"})["results"][0]["id"] == "8CPK_B"


@pytest.mark.parametrize("q,t,msg", [("E", "protein", "at least 2"), ("x" * 61, "protein", "at most 60"), ("EGFR; DROP", "protein", "use letters"),
                                     ("<script>", "protein", "use letters"), ("EGFR", "ligand", "protein, DNA or RNA")])
def test_search_refuses_bad_text_before_any_request(net, q, t, msg):
    with pytest.raises(ApiError, match=msg):
        LK.search({"query": q, "type": t})
    assert net[0] == []


def test_search_offline_is_an_answer_not_an_error_and_is_not_cached(net):
    calls, box = net
    box["answer"] = lambda url: ApiError(502, "could not reach the server (offline)")
    r = LK.search({"query": "thrombin"})
    assert r["offline"] and r["results"] == [] and r["message"] == LK.OFFLINE and "offline" in LK.OFFLINE
    box["answer"] = lambda url: {"results": [up("P00734", "Prothrombin", "F2", "Homo sapiens", 9606, "MAHVR")]}
    assert LK.search({"query": "thrombin"})["results"][0]["id"] == "P00734"   # back online: asked again
    # DNA / RNA: only a PDB id can be searched
    n = len(calls)
    assert LK.search({"query": "thrombin", "type": "dna"})["message"].startswith("for DNA / RNA") and len(calls) == n


def test_search_route(http_gui, net):
    _, box = net
    box["answer"] = lambda url: {"results": [up("O60885", "Bromodomain-containing protein 4", "BRD4", "Homo sapiens", 9606, "MSAES")]}
    s, _, b = call(http_gui, "POST", "/api/search", {"query": "BRD4", "type": "protein"})
    assert s == 200 and json.loads(b)["results"][0]["gene"] == "BRD4"
    s, _, b = call(http_gui, "POST", "/api/search", {"query": "a"})
    assert s == 400 and "at least 2" in json.loads(b)["error"]


# ------------------------------------------------------------------------------------------------------------ constraints
LIG = "CC(=O)Nc1ccc(O)cc1"   # paracetamol


def form(**kw):
    p = {"kind": "predict", "chains": [{"type": "protein", "id": "A", "sequence": PROT, "modifications": [{"position": 3, "ccd": "tpo"}]},
                                       {"type": "protein", "id": "B", "sequence": PROT[:60], "copies": 2},
                                       {"type": "ligand", "id": "L", "smiles": LIG}],
         "affinity": {"binder": "L"}, "options": {"preset": "standard"}}
    p.update(kw)
    return p


def test_smiles_atom_names_follow_upstream():
    names = CX.smiles_atom_names(LIG)
    assert len(names) == 11 and len(set(names)) == 11 and {n[0] for n in names} == {"C", "N", "O"}
    assert all(n[1:].isdigit() for n in names)


def test_form_constraints_modifications_and_templates_go_into_the_yaml(app):
    lig_atom = CX.smiles_atom_names(LIG, binder=True)[0]
    p = form(constraints=[{"pocket": {"binder": "L", "contacts": [["A", 10], ["A", 12], ["B", 5]], "max_distance": 8}},
                          {"contact": {"token1": ["A", 5], "token2": ["B", 7], "max_distance": 6}},
                          {"bond": {"atom1": ["A", 1, "sd"], "atom2": ["L", 1, lig_atom]}}],
             templates=[{"filename": "t.cif", "content": STRUCT.read_text(), "chain_id": "A", "template_id": None}])
    v = app.validate_predict(p)
    assert v["ok"], v
    doc = yaml.safe_load(v["yaml"])
    assert doc["sequences"][0]["protein"]["modifications"] == [{"position": 3, "ccd": "TPO"}]
    assert doc["constraints"][0] == {"pocket": {"binder": "L", "contacts": [["A", 10], ["A", 12], ["B", 5]], "max_distance": 8}}
    assert doc["constraints"][1] == {"contact": {"token1": ["A", 5], "token2": ["B", 7]}}   # 6 A is the default: left out
    assert doc["constraints"][2] == {"bond": {"atom1": ["A", 1, "SD"], "atom2": ["L", 1, lig_atom]}}
    assert doc["templates"] == [{"cif": "inputs/template_1.cif", "chain_id": "A"}]
    assert v["templates"][0]["template_chain"] and v["constraints"] == doc["constraints"]
    # a run stages the one-chain template and points the YAML at it; Use inputs gives everything back
    rid = app.create_predict_run(p)["id"]
    rd = app.runs.run_dir(rid)
    staged = yaml.safe_load((rd / "inputs" / "job.yaml").read_text())
    tpath = Path(staged["templates"][0]["cif"])
    assert tpath.is_absolute() and tpath.is_file() and tpath.parent == (rd / "inputs").resolve() and tpath.read_text().startswith("data_")
    wait_state(app, rid, {"complete", "failed"})
    back = app.run_inputs(rid)
    assert back["constraints"] == doc["constraints"] and back["templates"][0]["chain_id"] == "A" and back["templates"][0]["content"].startswith("data_")
    assert back["chains"][0]["modifications"] == [{"position": 3, "ccd": "TPO"}]


@pytest.mark.parametrize("extra,msg", [
    ({"constraints": [{"pocket": {"binder": "L", "contacts": [["A", 999]]}}]}, "pocket 1: residue A:999 is outside chain A"),
    ({"constraints": [{"pocket": {"binder": "L", "contacts": [["L", "C1"]]}}]}, "not the binder L"),
    ({"constraints": [{"pocket": {"binder": "Q", "contacts": [["A", 3]]}}]}, "there is no chain Q"),
    ({"constraints": [{"pocket": {"binder": "L", "contacts": [["A", 3]], "max_distance": 2}}]}, "4-20"),
    ({"constraints": [{"contact": {"token1": ["A", 5], "token2": ["A", 7]}}]}, "different chains"),
    ({"constraints": [{"contact": {"token1": ["A", 5], "token2": ["L", 3]}}]}, "ligand L has no atom 3"),
    ({"constraints": [{"bond": {"atom1": ["A", 1, "SG"], "atom2": ["L", 1, "C1"]}}]}, r"bond 1: A:1 \(MET\) has no atom SG"),
    ({"constraints": [{"bond": {"atom1": ["A", 2, "NZ"], "atom2": ["L", 1, "ZZ9"]}}]}, "ligand L has no atom ZZ9"),
    ({"constraints": [{"bond": {"atom1": ["A", 2, "NZ"], "atom2": ["L", 2, "C1"]}}]}, "a ligand is residue 1"),
    ({"constraints": [{"glue": {}}]}, "pocket, contact or bond"),
    ({"templates": [{"filename": "t.cif", "content": "x", "chain_id": "L"}]}, "templates are for protein chains"),
    ({"templates": [{"filename": "t.cif", "content": STRUCT.read_text(), "chain_id": "A", "template_id": "Z"}]}, "has no protein chain Z"),
    ({"templates": [{"filename": "t.exe", "content": "x", "chain_id": "A"}]}, r"use a \.pdb or \.cif"),
])
def test_form_constraint_errors_name_the_constraint(app, extra, msg):
    v = app.validate_predict(form(**extra))
    assert not v["ok"] and __import__("re").search(msg, v["error"]), v


@pytest.mark.parametrize("mods,msg", [([{"position": 999, "ccd": "SEP"}], "residue 999 is outside the chain"),
                                      ([{"position": 3, "ccd": "SEP"}, {"position": 3, "ccd": "TPO"}], "modified twice"),
                                      ([{"position": 3, "ccd": "XYZ12"}], "not in the molecule dictionary"),
                                      ([{"position": 3, "ccd": "S-P"}], "1-5 letters or digits")])
def test_modification_errors(app, mods, msg):
    p = form()
    p["chains"][0]["modifications"] = mods
    v = app.validate_predict(p)
    assert not v["ok"] and msg in v["error"] and v["error"].startswith("chain A:")


def test_copies_answer_to_their_own_ids_in_constraints(app):
    v = app.validate_predict(form())
    extra = v["chains"][1]["ids"][1]                 # B's second copy
    w = app.validate_predict(form(constraints=[{"contact": {"token1": ["A", 5], "token2": [extra, 7]}}]))
    assert w["ok"], w


def test_yaml_import_carries_constraints_and_force_is_listed(app):
    text = ("version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: " + PROT + "\n      modifications:\n        - position: 3\n          ccd: SEP\n"
            "  - ligand:\n      id: L\n      smiles: CCO\nconstraints:\n  - pocket: {binder: L, contacts: [[A, 3], [A, 9]], force: true}\n"
            "templates:\n  - cif: dir/t.cif\n    chain_id: A\n    template_id: B\n")
    r = PR.import_yaml(text)
    assert r["constraints"] == [{"pocket": {"binder": "L", "contacts": [["A", 3], ["A", 9]]}}]
    assert r["templates"] == [{"file": "t.cif", "chain_id": "A", "template_id": "B"}]
    assert r["chains"][0]["modifications"] == [{"position": 3, "ccd": "SEP"}] and r["skipped"] == ["force of a pocket constraint"]
    # the YAML path checks residues too (a job would fail on it later)
    bad = text.replace("[A, 9]", "[A, 900]").replace("templates:\n  - cif: dir/t.cif\n    chain_id: A\n    template_id: B\n", "")
    v = app.validate_predict({"kind": "predict", "yaml": {"content": bad, "files": []}, "options": {"preset": "standard"}})
    assert not v["ok"] and "A:900 is outside chain A" in v["error"]


# ------------------------------------------------------------------------------------------------------------ texts
def test_about_reports_the_molecule_folder_not_ccd_pkl(app, env):
    (Path(env["cache"]) / "ccd.pkl").unlink()
    info = app.info()
    assert info["checks"]["cache"] == {"dir": info["checks"]["cache"]["dir"], "mols": True} and not info["problems"]


def test_privacy_and_preset_texts_are_plain():
    assert "UniProt" in PRIVACY_GUI and "ColabFold" in PRIVACY_GUI and "stay on this Mac" in PRIVACY_GUI and len(PRIVACY_GUI) < 320
    for p in GP.payload("Apple M5 Max", "nax") + GP.payload("Apple M2", "mps"):
        assert p["line"] and len(p["line"]) < 90 and p["line"].startswith(p["label"] + ":")
        for word in ("PREREG", "pre-registered", "OD-50", "Kendall", "gate"):
            assert word not in p["note"] and word not in p["line"]
