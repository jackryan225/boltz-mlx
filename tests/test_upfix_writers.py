"""Upstream PDB / mmCIF writer bugs and boltz_mlx.upfix.writers (see its docstring for the fork commits).

For each bug the first test shows stock boltz 2.2.1's writer misbehaving on a structure parsed by upstream (so the bug is
in our output path: BoltzWriter is what boltz-mlx predict and screen write poses with), the next that the patched writer fixes
it on the CIF / PDB text, and a last one that everything else stays byte-identical. CPU only: the structure comes from
upstream's parser (no model, coordinates are random), and BoltzWriter is fed a real featurized batch with fake coordinates."""

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from boltz_mlx.upfix import install_upstream_fixes, writers as W

CACHE = Path(os.environ.get("BOLTZ_MLX_CACHE", Path.home() / ".boltz")).expanduser()
pytestmark = pytest.mark.skipif(not (CACHE / "mols").is_dir(), reason="upstream mols/ dictionary not installed")

DOC = {"version": 1, "sequences": [
    {"protein": {"id": "A", "sequence": "CYIQNCPLG", "msa": "empty", "cyclic": True}},      # cyclic peptide
    {"protein": {"id": "L1", "sequence": "GSHMLEDP", "msa": "empty"}},                       # two-character chain id
    {"ligand": {"id": "B", "smiles": "Brc1ccc(Cl)cc1C=O"}},                                  # double + aromatic bonds
    {"ligand": {"id": "C", "ccd": "SAH"}},
    {"ligand": {"id": "E", "smiles": "[Na+]"}}, {"ligand": {"id": "F", "smiles": "[Cu+2]"}},  # upstream writes N and C
    {"dna": {"id": "G", "sequence": "ACGTAC", "cyclic": True}},
]}
ELEMENT_OF = {"E": "NA", "F": "CU"}   # chain -> true element of its only atom


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    from boltz.data.types import StructureV2
    from boltz.main import process_inputs

    from boltz_mlx.seeding import install_conformer_seed

    install_conformer_seed()
    install_upstream_fixes()
    d = tmp_path_factory.mktemp("writers")
    y = d / "w.yaml"
    y.write_text(yaml.safe_dump(DOC))
    process_inputs(data=[y], out_dir=d, ccd_path=CACHE / "ccd.pkl", mol_dir=CACHE / "mols", msa_server_url="",
                   msa_pairing_strategy="greedy", use_msa_server=False, boltz2=True)
    st = StructureV2.load(d / "processed" / "structures" / "w.npz")
    st.atoms["coords"][:] = np.random.RandomState(0).randn(len(st.atoms), 3).astype("f4") * 5
    st.atoms["is_present"][:] = True
    return d, st


@pytest.fixture(scope="module")
def case1(case):
    """The same structure with chain L1 renamed to Z: stock and patched PDB text then differ only by the other fixes."""
    from dataclasses import replace

    d, st = case
    chains = st.chains.copy()
    chains["name"] = [("Z" if n == "L1" else n) for n in chains["name"]]
    return d, replace(st, chains=chains)


def _true_elements(case_dir):
    """The model's own per-atom elements (ref_element of the featurized batch), as the patched BoltzWriter reads them."""
    batch, pred = _batch(case_dir)
    return W.elements_from_batch(pred, batch)


def _batch(case_dir):
    import torch
    from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
    from boltz.data.types import Manifest

    proc = case_dir / "processed"
    dm = Boltz2InferenceDataModule(manifest=Manifest.load(proc / "manifest.json"), target_dir=proc / "structures", msa_dir=proc / "msa",
                                   mol_dir=CACHE / "mols", num_workers=0, constraints_dir=None, template_dir=None,
                                   extra_mols_dir=proc / "mols", override_method=None, affinity=False)
    torch.manual_seed(0)
    batch = next(iter(dm.predict_dataloader()))
    return batch, {"exception": False, "masks": batch["atom_pad_mask"]}


def _stock():
    from boltz.data.write.mmcif import to_mmcif
    from boltz.data.write.pdb import to_pdb

    return to_pdb, to_mmcif


def _atom_lines(pdb):
    return [ln for ln in pdb.splitlines() if ln.startswith(("ATOM", "HETATM"))]


# ============================================================================================ element (upstream #458)
def test_stock_writes_the_wrong_element_for_some_smiles_ligand_atoms(case1):
    d, st = case1
    to_pdb, to_cif = _stock()
    # chains E ([Na+]) and F ([Cu+2]): upstream takes the first letter of the atom name
    lines = [ln for ln in _atom_lines(to_pdb(st, boltz2=True)) if ln[12:16].strip() in ("NA1", "CU1")]
    assert [ln[76:78].strip() for ln in lines] == ["N", "C"], lines
    rows = [r.split() for r in to_cif(st, boltz2=True).splitlines() if r.startswith("HETATM") and r.split()[3] in ("NA1", "CU1")]
    assert [r[2] for r in rows] == ["N", "C"]


def test_audit_how_many_smiles_ligand_elements_stock_gets_wrong():
    from rdkit import Chem

    from boltz.data import const
    pt = Chem.GetPeriodicTable()

    def stock_element(atom_name, res_name):
        key = re.sub(r"\d", "", atom_name)
        amb = const.ambiguous_atoms
        if key in amb:
            return amb[key] if isinstance(amb[key], str) else amb[key].get(res_name, amb[key]["*"])
        return key[0]

    bad = sorted(pt.GetElementSymbol(z).upper() for z in range(1, 119)
                 if stock_element(pt.GetElementSymbol(z).upper() + "1", "LIG") != pt.GetElementSymbol(z).upper())
    assert {"NA", "CA", "CU", "CO", "NI", "CD", "PD", "PB", "SR", "BA"} <= set(bad) and len(bad) == 58
    assert not {"FE", "ZN", "MG", "MN", "BR", "CL", "SE", "SI", "PT"} & set(bad)    # these were right


def test_patched_writer_fixes_the_element_and_changes_nothing_else(case1):
    d, st = case1
    to_pdb, to_cif = _stock()
    el = _true_elements(d)
    assert len(el) == len(st.atoms) and "NA" in el and "CU" in el
    old, new = to_pdb(st, boltz2=True), W.to_pdb(st, boltz2=True, elements=el)
    ol, nl = _atom_lines(old), _atom_lines(new)
    changed = [(a, b) for a, b in zip(ol, nl) if a != b]
    assert sorted(b[12:16].strip() for _, b in changed) == ["CU1", "NA1"]
    assert [b[76:78] for _, b in changed] == ["NA", "CU"]
    assert [b[:76] for _, b in changed] == [a[:76] for a, _ in changed]       # only the element columns differ
    # all atom lines except those two are byte-identical, and the elements now match the model's input everywhere
    assert [ln[76:78].strip() for ln in nl] == el
    oc, nc = to_cif(st, boltz2=True), W.to_mmcif(st, boltz2=True, elements=el)
    orow = [r for r in oc.splitlines() if r.startswith(("ATOM ", "HETATM "))]
    nrow = [r for r in nc.splitlines() if r.startswith(("ATOM ", "HETATM "))]
    assert [r.split()[2] for r in nrow] == el
    assert [(a, b) for a, b in zip(orow, nrow) if a != b] and sum(a != b for a, b in zip(orow, nrow)) == 2
    assert [a.split()[2] for a, b in zip(orow, nrow) if a != b] == ["N", "C"] and [b.split()[2] for a, b in zip(orow, nrow) if a != b] == ["NA", "CU"]


def test_cif_atom_names_with_primes_survive_the_element_rewrite(case1):
    d, st = case1
    el = _true_elements(d)
    cif = W.to_mmcif(st, boltz2=True, elements=el)
    assert any("\"O3'\"" in r or "O3'" in r for r in cif.splitlines() if r.startswith("ATOM "))    # the DNA chain
    gemmi = pytest.importorskip("gemmi")
    block = gemmi.cif.read_string(cif)[0]
    assert list(block.find_values("_atom_site.type_symbol")) == [f"{e}" for e in el]


# ========================================================================================== chain ids (fork PR #23)
def test_stock_pdb_shifts_the_columns_of_a_two_character_chain_id(case):
    d, st = case
    to_pdb, _ = _stock()
    pdb = to_pdb(st, boltz2=True)
    bad = [ln for ln in _atom_lines(pdb) if ln[21:23] == "L1"]
    assert len(bad) == 60                                           # every atom of chain L1 ...
    for ln in bad:                                                  # ... has its fields one column late: the occupancy column holds "N  1.0"
        assert ln[22] == "1" and ln[54:60].endswith("  1.0") and ln[54:60].strip() != "1.00"
        with pytest.raises(ValueError):
            float(ln[54:60])
    assert all(float(ln[54:60]) == 1.0 for ln in _atom_lines(pdb) if ln[21:23] != "L1")


def test_patched_pdb_has_one_chain_character_and_valid_columns(case, capfd):
    d, st = case
    pdb = W.to_pdb(st, boltz2=True)
    err = capfd.readouterr().err
    assert "L1 -> " in err and "mmCIF output keeps the original names" in err
    names = [str(c["name"]) for c in st.chains]
    renamed = dict(zip(names, W.pdb_chain_ids(names)))
    assert renamed["L1"] not in {"A", "B", "C", "E", "F", "G"} and renamed["A"] == "A"
    assert {ln[21:22] for ln in _atom_lines(pdb)} == set(renamed.values())
    for ln in _atom_lines(pdb):
        assert len(ln) == 80
        float(ln[30:38]); float(ln[38:46]); float(ln[46:54]); int(ln[22:26])      # fixed-width columns parse
    gemmi = pytest.importorskip("gemmi")
    assert sorted(ch.name for ch in gemmi.read_pdb_string(pdb)[0]) == sorted(renamed.values())
    assert 'L1' in W.to_mmcif(st, boltz2=True)                                       # mmCIF keeps the original name


def test_pdb_chain_id_assignment():
    assert W.pdb_chain_ids(["A", "B"]) == ["A", "B"]
    assert W.pdb_chain_ids(["A", "AA", "B", "A"]) == ["A", "C", "B", "D"]      # the duplicated "A" is not unique: both get pool characters, B is kept
    assert W.pdb_chain_ids(["L1", "L2", "A"]) == ["B", "C", "A"]
    with pytest.raises(ValueError, match="62"):
        W.pdb_chain_ids([f"X{i}" for i in range(63)])


# ================================================================================ bond order (fork PR #22) and ring closure
def _conect(pdb):
    return [ln for ln in pdb.splitlines() if ln.startswith("CONECT")]


def test_stock_pdb_writes_every_bond_once_and_no_ring_closure(case1):
    d, st = case1
    to_pdb, to_cif = _stock()
    pdb = to_pdb(st, boltz2=True)
    assert len(_conect(pdb)) == len(st.bonds)
    assert all(len(ln.rstrip()) == 16 for ln in _conect(pdb))                     # CONECT + two serials, never a repeat
    assert any(int(b["type"]) == 2 for b in st.bonds)                              # there are double bonds to lose
    assert "_struct_conn" not in to_cif(st, boltz2=True)                           # stock mmCIF has no bond records
    assert len(W.closure_bonds(st)) == 2                                           # but two chains are cyclic


def test_patched_conect_has_bond_orders_and_the_ring_closing_bonds(case1):
    d, st = case1
    pdb = W.to_pdb(st, boltz2=True)
    serial = [int(ln[6:11]) for ln in _atom_lines(pdb)]
    con = _conect(pdb)
    closures = W.closure_bonds(st)
    assert len(closures) == 2                                                      # A (protein) and G (DNA)
    assert len(con) == len(st.bonds) + 2
    order = {2: 2, 3: 3}
    for ln, b in zip(con, st.bonds):                                               # same bonds, same order, orders as repeats
        n = order.get(int(b["type"]), 1)
        assert ln.rstrip() == f"CONECT{serial[b['atom_1']]:>5}" + f"{serial[b['atom_2']]:>5}" * n
    assert sum(len(ln.rstrip()) == 21 for ln in con) == sum(int(b["type"]) == 2 for b in st.bonds)
    extra = [ln.rstrip() for ln in con[-2:]]
    names = lambda i: str(st.atoms[i]["name"])  # noqa: E731
    assert [(names(c["atom_1"]), names(c["atom_2"])) for c in closures] == [("C", "N"), ("O3'", "P")]
    assert extra == [f"CONECT{serial[c['atom_1']]:>5}{serial[c['atom_2']]:>5}" for c in closures]
    assert all(len(ln) == 80 for ln in con)
    assert pdb.splitlines()[-2].startswith("END")
    gemmi = pytest.importorskip("gemmi")
    gemmi.read_pdb_string(pdb)                                                     # parsers accept the repeats


def test_closure_is_not_duplicated_when_the_bond_is_already_declared(case1):
    from dataclasses import replace

    d, st = case1
    c = W.closure_bonds(st)[0]
    row = np.zeros(1, dtype=st.bonds.dtype)
    row["chain_1"] = row["chain_2"] = c["chain"]
    row["res_1"], row["res_2"], row["atom_1"], row["atom_2"], row["type"] = c["res_1"], c["res_2"], c["atom_1"], c["atom_2"], 5
    st2 = replace(st, bonds=np.concatenate([st.bonds, row]))
    assert len(_conect(W.to_pdb(st2, boltz2=True))) == len(st2.bonds) + 1          # only the DNA one is added


def test_patched_mmcif_has_the_closing_struct_conn_rows(case1):
    d, st = case1
    cif = W.to_mmcif(st, boltz2=True)
    gemmi = pytest.importorskip("gemmi")
    block = gemmi.cif.read_string(cif)[0]
    t = block.find(["_struct_conn.id", "_struct_conn.conn_type_id", "_struct_conn.ptnr1_label_asym_id", "_struct_conn.ptnr1_label_comp_id",
                    "_struct_conn.ptnr1_label_seq_id", "_struct_conn.ptnr1_label_atom_id", "_struct_conn.ptnr2_label_seq_id",
                    "_struct_conn.ptnr2_label_atom_id", "_struct_conn.pdbx_dist_value"])
    rows = [[gemmi.cif.as_string(v) for v in r] for r in t]
    assert len(rows) == 2
    assert rows[0][1:8] == ["covale", "A", "GLY", "9", "C", "1", "N"] and rows[0][0] == "covale1"
    assert rows[1][1:3] == ["covale", "G"] and rows[1][5] == "O3'" and rows[1][7] == "P"
    assert all(float(r[8]) > 0 for r in rows)
    structure = gemmi.make_structure_from_block(block)
    assert len(structure.connections) == 2
    # everything before the new category is upstream's file
    _, to_cif = _stock()
    assert cif.startswith(to_cif(st, boltz2=True).rstrip("\n"))


# ================================================================================================ normal output unchanged
def test_a_structure_without_the_bugs_is_written_byte_identically(case, tmp_path):
    d, st = case
    to_pdb, to_cif = _stock()
    sub = _single_chain(st, 0, cyclic=False)                                        # chain A as a plain linear chain
    assert W.to_pdb(sub, boltz2=True) == to_pdb(sub, boltz2=True)
    assert W.to_mmcif(sub, boltz2=True) == to_cif(sub, boltz2=True)
    el = [re.sub(r"\d", "", str(a["name"]))[0] for a in sub.atoms]
    assert W.to_pdb(sub, boltz2=True, elements=el) == to_pdb(sub, boltz2=True)      # correct elements: nothing rewritten
    assert W.to_mmcif(sub, boltz2=True, elements=el) == to_cif(sub, boltz2=True)


def _single_chain(st, k, cyclic):
    """The structure of chain k alone (atoms, residues, no bonds), as a plain linear polymer."""
    from dataclasses import replace

    ch = st.chains[k:k + 1].copy()
    a0, an, r0, rn = int(ch["atom_idx"][0]), int(ch["atom_num"][0]), int(ch["res_idx"][0]), int(ch["res_num"][0])
    atoms = st.atoms[a0:a0 + an].copy()
    res = st.residues[r0:r0 + rn].copy()
    res["atom_idx"] -= a0
    res["res_idx"] = np.arange(rn)
    ch["atom_idx"], ch["res_idx"], ch["asym_id"], ch["sym_id"], ch["entity_id"] = 0, 0, 0, 0, 0
    ch["cyclic_period"] = rn if cyclic else 0
    coords = np.array([(c,) for c in atoms["coords"]], dtype=st.coords.dtype)
    ens = np.array([(0, an)], dtype=st.ensemble.dtype)
    return replace(st, atoms=atoms, residues=res, chains=ch, bonds=np.array([], dtype=st.bonds.dtype), coords=coords,
                   ensemble=ens, interfaces=np.array([], dtype=st.interfaces.dtype))


# ================================================================================ through BoltzWriter (the real output path)
def test_boltz_writer_output_uses_the_model_elements_and_all_fixes(case, tmp_path, capfd):
    import torch
    from boltz.data.write.writer import BoltzWriter

    d, st = case
    batch, pred = _batch(d)
    n = int(batch["atom_pad_mask"].shape[-1])
    pred["coords"] = torch.from_numpy(np.random.RandomState(1).randn(1, n, 3).astype("f4") * 5)
    for fmt, ext in (("pdb", "pdb"), ("mmcif", "cif")):
        w = BoltzWriter(data_dir=d / "processed" / "structures", output_dir=tmp_path / fmt, output_format=fmt, boltz2=True)
        w.write_on_batch_end(None, None, pred, None, batch, 0, 0)
        out = next((tmp_path / fmt).rglob(f"*.{ext}")).read_text()
        if fmt == "pdb":
            assert {ln[12:16].strip(): ln[76:78].strip() for ln in _atom_lines(out) if ln[12:16].strip() in ("NA1", "CU1")} == {"NA1": "NA", "CU1": "CU"}
            assert len(_conect(out)) == len(st.bonds) + 2 and len({ln[21] for ln in _atom_lines(out)}) == 7
        else:
            assert "covale1" in out and "covale2" in out
            assert [r.split()[3] for r in out.splitlines() if r.startswith("HETATM") and r.split()[3] in ("NA1", "CU1")] == ["NA1", "CU1"]
            assert [r.split()[2] for r in out.splitlines() if r.startswith("HETATM") and r.split()[3] in ("NA1", "CU1")] == ["NA", "CU"]
    capfd.readouterr()


def test_install_is_idempotent_and_leaves_upstream_alone(case):
    import boltz.data.write.pdb as P
    import boltz.data.write.writer as WR

    f = WR.BoltzWriter.write_on_batch_end
    W.install_writers()
    assert WR.BoltzWriter.write_on_batch_end is f and WR.to_pdb is W.to_pdb and WR.to_mmcif is W.to_mmcif
    assert P.to_pdb is not W.to_pdb and f.__wrapped__.__module__ == "boltz.data.write.writer"
    r = subprocess.run([sys.executable, "-c", "import boltz.data.write.writer as w, boltz.data.write.pdb as p; print(w.to_pdb is p.to_pdb)"],
                       capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])})
    assert r.stdout.strip() == "True"        # a fresh interpreter is stock until install_upstream_fixes()
