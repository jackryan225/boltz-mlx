"""GUI fixes from an end-to-end pass with real runs (a first-time researcher's path through Predict and Screen): Fast's tiny-ligand
rule reaching the job from the forms, copies in the scores table, Use inputs keeping copies and an Online MSA, plain MSA choices,
"name SMILES" paste lines, modification / DNA-as-protein warnings, PDB entry copies, the screen target's MSA advice, Mol*'s volseg.

CPU only; the predict subprocess is the fake one (tests/gui_fake_predict.py); the page code is checked by its source."""

from __future__ import annotations

import re

from test_gui_search_constraints import LIG, form, net  # noqa: F401  (fixtures)
from test_gui_targets import PROT, app, env, wait_state  # noqa: F401  (fixtures)

from boltz_mlx.gui import constraints as CX
from boltz_mlx.gui import lookup as LK
from boltz_mlx.gui.inputs import plain_warning
from boltz_mlx.gui.server import STATIC
from boltz_mlx.screen.inputs import _read_smi

JS = STATIC / "js"


def test_forms_send_preset_numbers_as_the_presets_so_the_tiny_ligand_rule_applies(app):
    # the server: no step count -> Fast's rule (200 for < 10 heavy atoms); an explicit one always wins
    tiny = {"kind": "predict", "chains": [{"type": "protein", "id": "A", "sequence": PROT}, {"type": "ligand", "id": "B", "smiles": "OC(=O)CCC(O)=O"}],
            "options": {"preset": "standard", "steps": None}}
    assert app.validate_predict(tiny)["options"]["steps"] == 200
    tiny["options"]["steps"] = 20
    assert app.validate_predict(tiny)["options"]["steps"] == 20
    # ...so both forms send a number still at the preset's value as null (the fields always hold the preset's numbers)
    for page in ("predict.js", "new.js"):
        src = (JS / page).read_text()
        assert 'String(v) === presetNumbers(curCli(), [k])[k] ? null' in src or 'String(x) === presetNumbers(curCli(), [k])[k] ? null' in src, page
        assert 'steps: own("steps")' in src and 'recycling: own("recycling")' in src, page
    # Use inputs of a run whose 200 steps were Fast's own choice leaves the steps to the preset again
    assert 'steps: o.steps_auto ? "" : String(o.steps ?? "")' in (JS / "job.js").read_text()


def test_use_inputs_keeps_copies_and_an_online_msa(app):
    p = {"kind": "predict", "chains": [{"type": "protein", "id": "A", "sequence": PROT, "copies": 2, "msa": {"mode": "server"}},
                                       {"type": "ligand", "id": "B", "smiles": LIG}], "options": {"preset": "standard"}}
    rid = app.create_predict_run(p)["id"]
    wait_state(app, rid, {"complete", "failed"})
    back = app.run_inputs(rid)["chains"][0]
    assert back["copies"] == 2 and back["msa"] == {"mode": "server"}
    job = (JS / "job.js").read_text()
    assert 'copies: String(c.copies ?? 1)' in job and 'c.msa?.mode === "server" ? "server"' in job
    # a reloaded Predict draft keeps the Online choice too
    assert '["file", "server"].includes(c.msa?.mode) ? c.msa.mode : "empty"' in (JS / "predict.js").read_text()


def test_scores_table_counts_the_copies_of_a_chain():
    # chains_ptm / pair_chains_iptm are per chain of the structure (A, C for a 2-copy A, then B): the result page expands copies
    src = (JS / "job.js").read_text()
    assert "chains = chains.flatMap((c) => (c.ids?.length ? c.ids.map((id) => ({ ...c, id })) : [c]));" in src
    assert src.index("chains = chains.flatMap") < src.index("replace(elChains, chainsCard(chains, conf, d.pocket));")


def test_msa_choices_say_what_they_do_on_both_pages():
    pred = (JS / "predict.js").read_text()
    assert '["empty", "None (faster, less accurate)"], ["server", "Online: ColabFold (sends the sequence)"], ["file", "File (.a3m / .csv)"]' in pred
    assert '"Server (sends sequence)"' not in pred
    assert '["none", "None (faster, less accurate)"], ["file", "File (.a3m / .csv)"]' in (JS / "targets.js").read_text()
    assert '["empty", "None (faster, less accurate)"], ["file", "File (.a3m / .csv)"]' in (JS / "new.js").read_text()
    # the screen target's advice names the page's MSA choice, not a command-line flag
    w = "protein chain(s) ['A', 'C'] have no MSA: running single-sequence (msa: empty), which costs accuracy; pass --msa <a3m> for a precomputed MSA"
    assert plain_warning(w) == "chain A, C: no MSA (single sequence): less accurate; an MSA file (.a3m / .csv) improves it"
    assert plain_warning("something else") == "something else"


def test_smi_lines_may_put_the_name_first(tmp_path):
    f = tmp_path / "l.smi"
    f.write_text("acetazolamide CC(=O)Nc1nnc(s1)S(N)(=O)=O\nOc1ccccc1 phenol\nCCO\nmy compound CCN\nnot_a_smiles\n")
    assert _read_smi(f) == [("acetazolamide", "CC(=O)Nc1nnc(s1)S(N)(=O)=O"), ("phenol", "Oc1ccccc1"), ("lig_00003", "CCO"),
                            ("compound CCN", "my"), ("lig_00005", "not_a_smiles")]


def test_modification_on_another_residue_and_dna_as_protein_warn(app):
    p = form()
    p["chains"][0]["modifications"] = [{"position": 1, "ccd": "SEP"}]     # PROT[0] is M
    v = app.validate_predict(p)
    assert v["ok"] and "chain A: residue 1 is M, but SEP (phospho-Ser) replaces S" in v["warnings"]
    assert not CX.modification_warnings({"sequence": "MKS", "modifications": [{"position": 3, "ccd": "SEP"}]}, "chain A")
    d = app.validate_predict({"kind": "predict", "chains": [{"type": "protein", "id": "A", "sequence": "ACGTACGTACGTAC"}], "options": {"preset": "standard"}})
    assert d["ok"] and any("looks like a DNA / RNA sequence" in w for w in d["warnings"])
    assert not any("DNA" in w for w in app.validate_predict(form())["warnings"])


def test_pdb_lookup_counts_the_copies_of_the_entity(net):
    calls, box = net
    fasta = ">1HSG_1|Chains A, B[auth C]|HIV-1 PROTEASE|Human immunodeficiency virus 1 (11676)\nPQITLWQRPL\n"
    box["answer"] = lambda url: fasta if "fasta" in url else {}
    r = LK.lookup({"query": "1HSG", "type": "protein"})
    assert r["copies"] == 2 and r["ligands"] == []
    assert LK.lookup({"query": "1HSG_A", "type": "protein"})["copies"] == 2   # a picked chain still says how many the entry has
    src = (JS / "predict.js").read_text()
    assert "Use ${l.copies} copies" in src and "no bound ligands" in src


def test_viewer_frame_leaves_out_the_volseg_extension():
    # volseg lists entries from http://localhost:9000 at start-up when the frame's host is localhost: blocked, an uncaught error
    assert 'disabledExtensions: ["volseg"]' in (JS / "viewer-frame.js").read_text()


def test_predict_run_name_follows_what_was_looked_up():
    src = (JS / "predict.js").read_text()
    assert 'name: st.name.trim() || autoName() || undefined' in src
    assert re.search(r"named\(c, `\$\{r\.gene \|\| r\.name\} \(\$\{r\.id\}\)`\)", src)   # the same "CA2 (P00918)" as a saved Screen target
