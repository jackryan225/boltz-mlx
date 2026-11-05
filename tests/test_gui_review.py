"""GUI review round: sequence lookup (network mocked: no live calls), chain copies, YAML pass-through of what the form does not
hold, the Predict estimate (time + memory) on the M5 Max, and the per-Mac default preset.

CPU only. Nothing here touches the network: lookup._fetch is replaced in every test that reaches it."""

from __future__ import annotations

import json

import pytest
import yaml
from test_gui_targets import PROT, STRUCT, app, call, env, http_gui, wait_state  # noqa: F401  (fixtures)

from boltz_mlx.gui import lookup as LK
from boltz_mlx.gui import presets as GP
from boltz_mlx.gui.common import ApiError

FASTA_UNIPROT = ">sp|P00533|EGFR_HUMAN Epidermal growth factor receptor OS=Homo sapiens OX=9606 GN=EGFR PE=1 SV=2\nMRPSGTAGAA\nLLALLAAL\n"
FASTA_PDB = (">1A02_2|Chain B|DNA (5'-D(*DAP)-3')|\nACGTTGCA\n>1A02_1|Chain A|DNA one|\nTTGGAAAA\n"
             ">1A02_3|Chain C[auth N]|NFAT|Homo sapiens (9606)\nMKTAYIAKQRQ\n")


@pytest.fixture
def net(monkeypatch):
    """Replace the one network function; records the URLs asked for and answers from a dict (a prefix match)."""
    calls, answers = [], {}

    def fake(url):
        calls.append(url)
        for k, v in answers.items():
            if url.startswith(k):
                if isinstance(v, Exception):
                    raise v
                return v if isinstance(v, str) else json.dumps(v)
        raise AssertionError(f"unexpected request {url}")

    monkeypatch.setattr(LK, "_fetch", fake)
    return calls, answers


def test_uniprot_lookup_fills_a_sequence(net):
    calls, ans = net
    ans["https://rest.uniprot.org/uniprotkb/P00533.fasta"] = FASTA_UNIPROT
    r = LK.lookup({"query": " p00533 ", "type": "protein"})
    assert r["sequence"] == "MRPSGTAGAALLALLAAL" and r["length"] == 18 and r["source"] == "UniProt" and r["title"] == "Epidermal growth factor receptor"
    assert calls == ["https://rest.uniprot.org/uniprotkb/P00533.fasta"] and r["ligands"] == []


def test_pdb_lookup_by_chain_label_or_author_id_and_type(net):
    calls, ans = net
    ans["https://www.rcsb.org/fasta/entry/1A02"] = FASTA_PDB
    ans["https://data.rcsb.org/rest/v1/core/entry/1A02"] = {"rcsb_entry_container_identifiers": {"non_polymer_entity_ids": []}}
    assert LK.lookup({"query": "1a02_A", "type": "dna"})["sequence"] == "TTGGAAAA"
    assert LK.lookup({"query": "1A02_N", "type": "protein"})["sequence"] == "MKTAYIAKQRQ"   # author chain id
    assert LK.lookup({"query": "1A02", "type": "dna"})["note"].startswith("entity 1 of 3")    # first matching entity, entity order
    with pytest.raises(ApiError, match="no chain Z"):
        LK.lookup({"query": "1A02_Z", "type": "dna"})
    with pytest.raises(ApiError, match="not protein"):
        LK.lookup({"query": "1A02_A", "type": "protein"})


def test_pdb_entry_offers_bound_small_molecules_without_additives(net):
    calls, ans = net
    ans["https://www.rcsb.org/fasta/entry/4HHB"] = ">4HHB_1|Chains A, C|Hemoglobin alpha|Homo sapiens (9606)\nVLSPADKTNV\n"
    ans["https://data.rcsb.org/rest/v1/core/entry/4HHB"] = {"rcsb_entry_container_identifiers": {"non_polymer_entity_ids": ["3", "4", "5"]}}
    ans["https://data.rcsb.org/rest/v1/core/nonpolymer_entity/4HHB/3"] = {"pdbx_entity_nonpoly": {"comp_id": "HEM"}}
    ans["https://data.rcsb.org/rest/v1/core/nonpolymer_entity/4HHB/4"] = {"pdbx_entity_nonpoly": {"comp_id": "GOL"}}   # additive: skipped
    ans["https://data.rcsb.org/rest/v1/core/nonpolymer_entity/4HHB/5"] = {"pdbx_entity_nonpoly": {"comp_id": "ATP"}}
    ans["https://data.rcsb.org/rest/v1/core/chemcomp/HEM"] = {"chem_comp": {"name": "HEME"}, "rcsb_chem_comp_descriptor": {"SMILES": "C1", "SMILES_stereo": "C[C@H]1"}}
    ans["https://data.rcsb.org/rest/v1/core/chemcomp/ATP"] = {"chem_comp": {"name": "ATP"}, "rcsb_chem_comp_descriptor": {"SMILES": "CCO"}}
    r = LK.lookup({"query": "4HHB", "type": "protein"})
    assert [(x["id"], x["smiles"]) for x in r["ligands"]] == [("HEM", "C[C@H]1"), ("ATP", "CCO")]
    assert not any("chemcomp/GOL" in u for u in calls)
    # a failure of the suggestions never loses the sequence
    ans["https://data.rcsb.org/rest/v1/core/entry/4HHB"] = ApiError(502, "server answered 500")
    assert LK.lookup({"query": "4HHB", "type": "protein"})["ligands"] == []


@pytest.mark.parametrize("q,t,msg", [
    ("", "protein", "enter a UniProt"), ("hello world", "protein", "not a UniProt accession"), ("P00533", "dna", "use a PDB id"),
    ("8CPK", "ligand", "protein, DNA or RNA"),
])
def test_lookup_rejects_bad_input_before_any_request(net, q, t, msg):
    with pytest.raises(ApiError, match=msg):
        LK.lookup({"query": q, "type": t})
    assert net[0] == []


def test_lookup_network_errors_are_one_line(net):
    _, ans = net
    ans["https://rest.uniprot.org"] = ApiError(404, "not found")
    with pytest.raises(ApiError, match="not found"):
        LK.lookup({"query": "P99999"})
    ans["https://rest.uniprot.org"] = ">empty\n"
    with pytest.raises(ApiError, match="no sequence"):
        LK.lookup({"query": "P99999"})


def test_fetch_maps_urlerrors_without_a_real_request(monkeypatch):
    import urllib.error

    def boom(req, timeout=0):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(LK.urllib.request, "urlopen", boom)
    with pytest.raises(ApiError, match="could not reach the server"):
        LK._fetch("https://rest.uniprot.org/uniprotkb/P00533.fasta")


def test_lookup_route_needs_the_token_and_returns_json(http_gui, net):
    _, ans = net
    ans["https://rest.uniprot.org/uniprotkb/P00533.fasta"] = FASTA_UNIPROT
    s, _, b = call(http_gui, "POST", "/api/lookup", {"query": "P00533", "type": "protein"})
    assert s == 200 and json.loads(b)["id"] == "P00533"
    s, _, b = call(http_gui, "POST", "/api/lookup", {"query": "nope"})
    assert s == 400 and "not a UniProt accession" in json.loads(b)["error"]


# ------------------------------------------------------------------------------------------------------------ chains
def test_copies_become_one_yaml_entity_with_several_ids(app):
    chains = [{"type": "protein", "id": "A", "sequence": PROT, "copies": 2}, {"type": "protein", "id": "B", "sequence": PROT},
              {"type": "ligand", "id": "C", "smiles": "CCO", "copies": 2}]
    v = app.validate_predict({"kind": "predict", "chains": chains, "options": {"preset": "standard"}})
    assert v["ok"] and v["n_tokens"] == 3 * len(PROT) + 2 * 3 and v["chains"][0]["ids"] == ["A", "D"] and v["chains"][2]["ids"] == ["C", "E"]
    doc = yaml.safe_load(v["yaml"])
    ids = [next(iter(e.values()))["id"] for e in doc["sequences"]]
    assert ids[0] == v["chains"][0]["ids"] and ids[1] == "B"
    all_ids = [i for x in ids for i in (x if isinstance(x, list) else [x])]
    assert len(all_ids) == len(set(all_ids)) == 5   # extra ids never collide with the explicit ones


def test_copies_are_checked(app):
    base = {"kind": "predict", "options": {"preset": "standard"}}
    for bad, msg in ((0, "between 1 and"), ("x", "whole number"), (99, "between 1 and")):
        out = app.validate_predict({**base, "chains": [{"type": "protein", "id": "A", "sequence": PROT, "copies": bad}]})
        assert out["ok"] is False and msg in out["error"]
    two = {"type": "ligand", "id": "B", "smiles": "CCO", "copies": 2}
    out = app.validate_predict({**base, "chains": [{"type": "protein", "id": "A", "sequence": PROT}, two], "affinity": {"binder": "B"}})
    assert out["ok"] is False and "single copy" in out["error"]
    dup = app.validate_predict({**base, "chains": [{"type": "protein", "id": "A", "sequence": PROT}, {"type": "dna", "id": "a", "sequence": "ACGT"}]})
    assert dup["ok"] is False and "used twice" in dup["error"]
    bad = app.validate_predict({**base, "chains": [{"type": "protein", "id": "AB-1", "sequence": PROT}]})
    assert bad["ok"] is False and "1-4 letters or digits" in bad["error"]


def test_a_homo_oligomer_runs_and_comes_back_as_a_form(app):
    r = app.create_run({"kind": "predict", "chains": [{"type": "protein", "id": "A", "sequence": PROT, "copies": 2}], "options": {"preset": "standard"}})
    assert wait_state(app, r["id"], ("complete", "failed")) in ("complete", "failed")   # the fake job only reads ids as text
    assert app.run_inputs(r["id"])["chains"][0]["copies"] == 2


# ------------------------------------------------------------------------------------------- YAML pass-through
RICH = f"""version: 1
sequences:
  - protein:
      id: A
      sequence: {PROT}
      msa: empty
      modifications:
        - position: 3
          ccd: SEP
  - protein:
      id: [B, C]
      sequence: {PROT}
      msa: empty
  - ligand:
      id: D
      ccd: SAH
constraints:
  - bond:
      atom1: [A, 1, SD]
      atom2: [D, 1, C1]
  - pocket:
      binder: D
      contacts: [[A, 10], [B, 12]]
      max_distance: 6
  - contact:
      token1: [A, 5]
      token2: [B, 7]
      max_distance: 8
templates:
  - cif: t.cif
    chain_id: A
properties:
  - affinity:
      binder: D
"""


def test_the_yaml_path_keeps_constraints_templates_and_modifications(app):
    p = {"kind": "predict", "yaml": {"content": RICH, "files": [{"filename": "t.cif", "content": STRUCT.read_text()}]}, "options": {"preset": "standard"}}
    v = app.validate_predict(p)
    assert v["ok"], v
    doc = yaml.safe_load(v["yaml"])
    assert {k: len(doc[k]) for k in ("constraints", "templates")} == {"constraints": 3, "templates": 1}
    assert doc["sequences"][0]["protein"]["modifications"] == [{"position": 3, "ccd": "SEP"}] and doc["sequences"][1]["protein"]["id"] == ["B", "C"]
    assert [next(iter(c)) for c in doc["constraints"]] == ["bond", "pocket", "contact"]
    # a constraint naming a chain the YAML does not define is refused
    bad = RICH.replace("[A, 5]", "[Q, 5]")
    out = app.validate_predict({**p, "yaml": {**p["yaml"], "content": bad}})
    assert out["ok"] is False and "Q" in out["error"]


# --------------------------------------------------------------------------------------------- estimate and defaults
def test_predict_estimate_has_time_and_memory_on_the_m5_max(app, monkeypatch):
    monkeypatch.setenv("BOLTZ_MLX_GUI_CHIP", "Apple M5 Max")
    chains = [{"type": "protein", "id": "A", "sequence": PROT}, {"type": "ligand", "id": "B", "smiles": "CCO"}]
    fast = app.validate_predict({"kind": "predict", "chains": chains, "affinity": {"binder": "B"}, "options": {"preset": "standard"}})["estimate"]
    full = app.validate_predict({"kind": "predict", "chains": chains, "affinity": {"binder": "B"}, "options": {"preset": "reference"}})["estimate"]
    assert fast["available"] and full["available"] and 0 < fast["total_s"] < full["total_s"]
    assert "ESTIMATE" in fast["caveat"] and fast["memory"]["peak_gb"] > 2.6 and "basis" in fast["memory"]
    bigger = app.validate_predict({"kind": "predict", "chains": [{"type": "protein", "id": "A", "sequence": PROT * 4}], "options": {"preset": "standard"}})["estimate"]
    assert bigger["memory"]["peak_gb"] > fast["memory"]["peak_gb"]
    # an unknown chip: no time, but the memory guide is still shown
    monkeypatch.setenv("BOLTZ_MLX_GUI_CHIP", "Apple M9")
    none = app.validate_predict({"kind": "predict", "chains": chains[:1], "options": {"preset": "standard"}})["estimate"]
    assert none["available"] is False and none["memory"]["peak_gb"] > 0


def test_default_preset_is_fast_only_on_nax():
    assert GP.default_key("nax") == "standard" and GP.default_key("tensorops") == GP.default_key("portable") == "reference"
    assert [p["label"] for p in GP.GUI_PRESETS] == ["Fast", "Full"]
