"""boltz_mlx.upfix.parse_guards / names: upstream #494 (atom names over 4 characters), the valence check, and standardize().

Fork SMILES (boltz-community tests/test_bug_fixes.py, commit 9cca9d6): "Br" + "C" * 50 and "Cl" + "C" * 60 + "Cl" overflow
upstream's element + canonical-rank names; aspirin is the normal ligand. Normal inputs must featurize byte-identically
with and without the patches (tests/upfix_feats.py, run in two fresh interpreters)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
BR = "Br" + "C" * 50
CLCL = "Cl" + "C" * 60 + "Cl"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
SALT = "C#CCN(C)CC(=C)c1ccc(Cl)c(Cl)c1.O=C(O)C(=O)O"   # fork 7407b86 (terminal alkyne + oxalic acid)

from boltz_mlx.upfix import install_upstream_fixes, names  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _installed():
    install_upstream_fixes()


def _parse(smiles):
    from boltz.data.parse.schema import parse_boltz_schema

    schema = {"version": 1, "sequences": [{"ligand": {"id": "L1", "smiles": smiles}}]}
    return parse_boltz_schema("t", schema, ccd={}, boltz_2=True)


def _py(code, *args):
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    return subprocess.run([sys.executable, "-c", code, *args], capture_output=True, text=True, timeout=300, env=env)


def test_stock_upstream_rejects_the_overflow_smiles():
    r = _py("import sys; from boltz.data.parse.schema import parse_boltz_schema as p\n"
            "p('t', {'version': 1, 'sequences': [{'ligand': {'id': 'L1', 'smiles': sys.argv[1]}}]}, ccd={}, boltz_2=True)", BR)
    assert r.returncode != 0 and "longer than 4 characters" in r.stderr


@pytest.mark.parametrize("smiles", [BR, CLCL])
def test_overflow_smiles_parses_with_unique_short_per_element_names(smiles, capfd):
    st = _parse(smiles).structure
    nm = [str(n) for n in st.atoms["name"]]
    assert len(nm) == len(set(nm)) and all(1 < len(n) <= 4 for n in nm)
    from collections import Counter

    for el, cnt in Counter(n.rstrip("0123456789") for n in nm).items():   # per element, numbered 1..n without gaps
        assert sorted(int(n[len(el):]) for n in nm if n.rstrip("0123456789") == el) == list(range(1, cnt + 1))
    assert nm.count("BR1") == smiles.startswith("Br") and ("CL1" in nm) == smiles.startswith("Cl")
    assert "would exceed 4 characters" in capfd.readouterr().err   # it said so


def test_numbering_follows_canonical_rank():
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = AllChem.AddHs(Chem.MolFromSmiles(BR))
    ranks = list(AllChem.CanonicalRankAtoms(mol))
    nums = names.per_element_numbers([a.GetSymbol() for a in mol.GetAtoms()], ranks)
    carbons = sorted((r, n) for a, r, n in zip(mol.GetAtoms(), ranks, nums) if a.GetSymbol() == "C")
    assert [n for _, n in carbons] == list(range(1, 51))


def test_fitting_ligand_keeps_upstream_ranks_object_and_prints_nothing(capfd):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = AllChem.AddHs(Chem.MolFromSmiles(ASPIRIN))
    sentinel = []
    out = names.canonical_rank_atoms(lambda m, *a, **k: sentinel or AllChem.CanonicalRankAtoms(m), mol)
    assert list(out) == list(AllChem.CanonicalRankAtoms(mol))
    st = _parse(ASPIRIN).structure
    upstream = [a.GetSymbol().upper() + str(r + 1) for a, r in zip(mol.GetAtoms(), AllChem.CanonicalRankAtoms(mol)) if a.GetAtomicNum() > 1]
    assert [str(n) for n in st.atoms["name"]] == upstream   # the structure keeps the heavy atoms, with upstream's names
    assert capfd.readouterr().err == ""


def test_element_limits():
    assert names.element_names_fit(["C"] * 999, list(range(999)))
    assert not names.element_names_fit(["C"] * 1000, list(range(1000)))
    assert names.element_names_fit(["CL"] * 99, list(range(99))) and not names.element_names_fit(["CL"] * 100, list(range(100)))
    assert names.upstream_names_fit(["C"] * 999, list(range(999))) and not names.upstream_names_fit(["CL"] * 99, [99] * 99)


def test_install_is_idempotent_and_composes_with_the_conformer_seed():
    import boltz.data.parse.schema as schema

    from boltz_mlx.seeding import install_conformer_seed

    install_conformer_seed()
    a, b = schema.AllChem, schema.rdMolStandardize
    install_upstream_fixes()
    install_upstream_fixes()
    assert schema.AllChem is a and schema.rdMolStandardize is b
    assert hasattr(schema.compute_3d_conformer, "_boltz_mlx_conformer_seed")
    st = _parse(BR).structure   # seeded conformer + per-element names together
    assert len(st.atoms) == 51


def test_validate_ligand_accepts_what_now_parses():
    from boltz_mlx.screen.inputs import Ligand, validate_ligand

    lig = validate_ligand(Ligand("big", BR, 0))
    assert lig.status == "pending" and lig.error is None


def test_unreadable_and_bad_valence_smiles_raise_value_error_with_a_notice(capfd):
    from boltz.data.parse.schema import parse_boltz_schema

    for bad in ("C(C", "not a smiles", "CS(=O)(=O)(=O)(=O)C"):
        with pytest.raises(ValueError, match="SMILES"):
            parse_boltz_schema("t", {"version": 1, "sequences": [{"ligand": {"id": "L1", "smiles": bad}}]}, ccd={}, boltz_2=True)
    assert "RDKit" in capfd.readouterr().err


def test_standardize_of_the_fork_salt():
    from boltz.data.parse.schema import standardize

    out = standardize(SALT)
    assert out and "." not in out and "C#C" in out   # the largest fragment (the amine), terminal alkyne intact
    assert "calcImplicitValence" not in out


def test_standardize_is_unchanged_for_molecules_that_worked():
    """Same SMILES as stock standardize for every input the stock function handled (fresh interpreter = stock)."""
    smis = [ASPIRIN, "CCO", "C[N+](C)(C)C.[Cl-]", "c1ccccc1.O", "CC(C)[C@H](N)C(=O)O", "OC(=O)C(=O)O.CCN"]
    code = ("import json,sys\nfrom boltz.data.parse.schema import standardize\n"
            "r = {}\nfor s in json.loads(sys.argv[1]):\n    try: r[s] = standardize(s)\n    except Exception as e: r[s] = 'ERR'\nprint('@@' + json.dumps(r))")
    stock = json.loads(_py(code, json.dumps(smis)).stdout.rsplit("@@", 1)[1])
    from boltz.data.parse.schema import standardize

    for s in smis:
        if stock[s] != "ERR":
            assert standardize(s) == stock[s], s


@pytest.mark.skipif(not (Path(os.environ.get("BOLTZ_MLX_CACHE", Path.home() / ".boltz")).expanduser() / "mols").is_dir(), reason="upstream mols/ dictionary not installed")
def test_normal_inputs_featurize_byte_identically_with_and_without_the_patches(tmp_path):
    cache = Path(os.environ.get("BOLTZ_MLX_CACHE", Path.home() / ".boltz")).expanduser()
    res = {}
    for mode in ("stock", "patched"):
        r = subprocess.run([sys.executable, str(ROOT / "tests" / "upfix_feats.py"), mode, str(cache), str(tmp_path / mode)],
                           capture_output=True, text=True, timeout=900, env={**os.environ, "PYTHONPATH": str(ROOT)})
        assert r.returncode == 0, r.stderr[-1500:]
        res[mode] = json.loads(r.stdout.split("RESULT", 1)[1])
    assert set(res["stock"]) == {"prot_smiles", "prot_ccd", "multi_chain"}
    for case in res["stock"]:
        a, b = res["stock"][case], res["patched"][case]
        assert a["n_tensors"] > 50 and a["features"] == b["features"], case          # every feature tensor, bit for bit
        assert a["atom_names"] == b["atom_names"] and a["extra_mol_names"] == b["extra_mol_names"], case
    assert res["stock"]["multi_chain"]["extra_mol_names"]   # the SMILES ligands' names were compared
