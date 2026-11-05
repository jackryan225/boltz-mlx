"""Unit tests for the round-2 additions to tools/validation/score.py: ligand pose RMSD in the pocket frame
(symmetry-aware), chirality check, and Pearson / Kendall / ligand-weighted pooled statistics with bootstrap CIs."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats as sps
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "validation"))
import score  # noqa: E402

pytest.importorskip("rdkit")
from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

IBU = "CC(C)Cc1ccc(cc1)[C@H](C)C(=O)O"  # (S)-ibuprofen-like: isopropyl and phenyl symmetry, one stereocentre
BENZENE = "c1ccccc1"
DATA = Path(os.environ.get("BOLTZ_MLX_DATA", Path(__file__).resolve().parents[1] / "data")).expanduser()


def embed(smiles, seed=7):
    """Heavy-atom 3D coordinates in SMILES atom order (what a Boltz mmCIF ligand looks like) + element symbols."""
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(m, randomSeed=seed) == 0
    AllChem.MMFFOptimizeMolecule(m)
    m = Chem.RemoveHs(m)
    return [a.GetSymbol() for a in m.GetAtoms()], m.GetConformer().GetPositions()


def tmpl_mol(smiles, xyz):
    return score._with_conformer(score._template(smiles), xyz)


# ------------------------------------------------------------------ symmetry-aware RMSD


def test_benzene_ring_rotation_is_zero_after_symmetry():
    sym, x = embed(BENZENE)
    c = x.mean(0)
    # rotate the ring about its normal by 60 degrees (atoms permute onto each other): a symmetry operation
    n = np.cross(x[1] - x[0], x[2] - x[0])
    n /= np.linalg.norm(n)
    R = Rotation.from_rotvec(n * np.deg2rad(60)).as_matrix()
    y = (x - c) @ R.T + c
    assert score.naive_rmsd(x, y) > 1.0  # identity mapping sees a big difference
    assert score.symmetric_rmsd_in_place(score._template(BENZENE), y, x) == pytest.approx(0.0, abs=1e-5)


def test_benzene_flip_and_real_displacement():
    sym, x = embed(BENZENE)
    t = score._template(BENZENE)
    # in-plane shift of 0.7 A: no mapping can fix a rigid translation
    y = x + np.array([0.7, 0, 0]) * 0
    n = np.cross(x[1] - x[0], x[2] - x[0])
    n /= np.linalg.norm(n)
    y = x + 0.7 * n
    assert score.symmetric_rmsd_in_place(t, y, x) == pytest.approx(0.7, abs=1e-6)


def test_ibuprofen_swapped_methyls_and_ring_flip_give_zero():
    sym, x = embed(IBU)
    t = score._template(IBU)
    # atoms: 0 C(H3), 1 CH, 2 CH3 (isopropyl methyls 0 and 2); ring atoms 5..10 with para substituents at 4 and 7
    mapping = Chem.MolFromSmiles(IBU).GetSubstructMatches(Chem.MolFromSmiles(IBU), uniquify=False)
    assert len(mapping) >= 4  # 2 methyl swaps x 2 ring flips (plus carboxyl handled below)
    y = x.copy()
    y[[0, 2]] = y[[2, 0]]  # swap the isopropyl methyls
    assert score.naive_rmsd(x, y) > 0.5
    assert score.symmetric_rmsd_in_place(t, y, x) == pytest.approx(0.0, abs=1e-5)
    # phenyl flip: swap ring atoms ortho/meta pairs across the para axis
    mol = Chem.MolFromSmiles(IBU)
    ring = [a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic()]
    assert len(ring) == 6
    y2 = x.copy()
    ipso, o1, m1, para, m2, o2 = ring  # SMILES order: c1 c c c(para-subst) c c1
    y2[[o1, o2]] = y2[[o2, o1]]
    y2[[m1, m2]] = y2[[m2, m1]]
    assert score.naive_rmsd(x, y2) > 0.3
    assert score.symmetric_rmsd_in_place(t, y2, x) == pytest.approx(0.0, abs=1e-5)


def test_carboxylate_oxygen_swap_symmetrised_but_not_a_different_pose():
    sym, x = embed("CC(=O)[O-]")
    t = score._template("CC(=O)[O-]")
    y = x.copy()
    y[[2, 3]] = y[[3, 2]]
    assert score.symmetric_rmsd_in_place(t, y, x, symmetrize_terminal_groups=True) == pytest.approx(0.0, abs=1e-5)
    assert score.symmetric_rmsd_in_place(t, y, x, symmetrize_terminal_groups=False) > 0.1  # bond-order-strict graph
    # a genuinely moved methyl is not fixed by any mapping
    z = x.copy()
    z[0] += 1.0
    assert score.symmetric_rmsd_in_place(t, z, x) > 0.2


def test_symmetric_rmsd_not_larger_than_identity_and_equals_calcrms():
    from rdkit.Chem import rdMolAlign

    sym, x = embed(IBU)
    rng = np.random.default_rng(0)
    y = x + rng.normal(0, 0.3, x.shape)
    t = score._template(IBU)
    s = score.symmetric_rmsd_in_place(t, y, x)
    assert s <= score.naive_rmsd(y, x) + 1e-9
    assert s == pytest.approx(rdMolAlign.CalcRMS(tmpl_mol(IBU, y), tmpl_mol(IBU, x)), abs=1e-9)


# ------------------------------------------------------------------ bond-order reconstruction


def test_mol_from_coords_shuffled_atoms_uses_template_and_recovers_order():
    sym, x = embed(IBU)
    rng = np.random.default_rng(1)
    perm = rng.permutation(len(sym))
    sym_p = [sym[i] for i in perm]
    mol, info = score.ligand_mol_from_coords(IBU, sym_p, x[perm])
    assert info["method"] == "bond_order_template"
    # the molecule carries the SMILES bond orders (aromatic ring + C=O) and the coordinates are a permutation of x
    assert sum(1 for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE) == 1
    assert sum(1 for a in mol.GetAtoms() if a.GetIsAromatic()) == 6
    got = mol.GetConformer().GetPositions()
    assert sorted(map(tuple, np.round(got, 6))) == sorted(map(tuple, np.round(x, 6)))
    # and the RMSD to the unshuffled pose is 0 up to symmetry
    assert score.symmetric_rmsd_in_place(score._template(IBU), got, x) == pytest.approx(0.0, abs=1e-5)


def test_mol_from_coords_smiles_order_fast_path_and_errors():
    sym, x = embed(IBU)
    mol, info = score.ligand_mol_from_coords(IBU, sym, x)
    assert info["method"] == "smiles_order"
    np.testing.assert_allclose(mol.GetConformer().GetPositions(), x)
    with pytest.raises(ValueError):
        score.ligand_mol_from_coords(IBU, sym[:-1], x[:-1])  # atom count mismatch
    with pytest.raises(ValueError):
        score.ligand_mol_from_coords("CCO", ["C", "C", "O"], np.array([[0, 0, 0], [10, 0, 0], [20, 0, 0]], float))


# ------------------------------------------------------------------ chirality


def test_chirality_ok_and_mirror_image_flips_every_centre():
    smi = "N[C@@H](C)C(=O)O"  # one centre
    sym, x = embed(smi)
    r = score.chirality_flips(smi, sym, x)
    assert (r["n_centres"], r["n_flips"]) == (1, 0)
    mirror = x * np.array([1, 1, -1])
    r = score.chirality_flips(smi, sym, mirror)
    assert (r["n_centres"], r["n_flips"]) == (1, 1)


def test_chirality_two_centres_single_inversion_and_shuffled_atoms():
    smi = "C[C@H](N)[C@@H](O)C(=O)O"
    sym, x = embed(smi)
    assert score.chirality_flips(smi, sym, x)["n_flips"] == 0
    assert score.chirality_flips(smi, sym, x * np.array([-1, 1, 1]))["n_flips"] == 2
    # invert only the first centre: swap the positions of its N-less neighbours' substituents (C and H-side): reflect the
    # methyl across the plane of the other three neighbours of atom 1
    mol = Chem.MolFromSmiles(smi)  # keep a reference: neighbour atoms dangle if the molecule is collected
    nb = [a.GetIdx() for a in mol.GetAtomWithIdx(1).GetNeighbors()]  # C0, N2, C3
    y = x.copy()
    y[[nb[0], nb[1]]] = y[[nb[1], nb[0]]]  # swapping two of three heavy neighbours inverts the centre
    r = score.chirality_flips(smi, sym, y)
    assert r["n_flips"] == 1
    # same answer through the bond-order template path (atoms shuffled)
    perm = np.random.default_rng(3).permutation(len(sym))
    r2 = score.chirality_flips(smi, [sym[i] for i in perm], y[perm])
    assert r2["n_flips"] == 1


def test_chirality_unspecified_centre_is_not_counted_and_double_bond_flip():
    sym, x = embed("NC(C)C(=O)O")  # no stereo in SMILES
    assert score.chirality_flips("NC(C)C(=O)O", sym, x * np.array([1, 1, -1]))["n_centres"] == 0
    smi = "C/C=C/C(=O)O"  # (E)-crotonic acid
    sym, x = embed(smi)
    r = score.chirality_flips(smi, sym, x)
    assert (r["n_double_bonds"], r["n_double_bond_flips"]) == (1, 0)
    smi_z = "C/C=C\\C(=O)O"
    r = score.chirality_flips(smi_z, sym, x)  # E geometry scored against a Z SMILES
    assert r["n_double_bond_flips"] == 1


# ------------------------------------------------------------------ full pipeline on synthetic mmCIFs


def _write_complex(path, ca, lig_sym, lig_xyz, lig_chain="B", names=None):
    import gemmi

    st = gemmi.Structure()
    st.cell = gemmi.UnitCell(1, 1, 1, 90, 90, 90)
    m = gemmi.Model("1")
    ch = gemmi.Chain("A")
    for i, xyz in enumerate(ca):
        r = gemmi.Residue()
        r.name = "ALA"
        r.seqid = gemmi.SeqId(1000 + i, " ")
        r.label_seq = i + 1
        r.entity_type = gemmi.EntityType.Polymer
        a = gemmi.Atom()
        a.name = "CA"
        a.element = gemmi.Element("C")
        a.pos = gemmi.Position(*xyz)
        r.add_atom(a)
        ch.add_residue(r)
    m.add_chain(ch)
    lc = gemmi.Chain(lig_chain)
    r = gemmi.Residue()
    r.name = "LIG1"
    r.seqid = gemmi.SeqId(1, " ")
    r.entity_type = gemmi.EntityType.NonPolymer
    r.het_flag = "H"
    for k, (s, p) in enumerate(zip(lig_sym, lig_xyz)):
        a = gemmi.Atom()
        a.name = f"{s}{k}" if names is None else names[k]
        a.element = gemmi.Element(s)
        a.pos = gemmi.Position(*[float(v) for v in p])
        r.add_atom(a)
    lc.add_residue(r)
    m.add_chain(lc)
    st.add_model(m)
    ent = gemmi.Entity("1")
    ent.entity_type = gemmi.EntityType.Polymer
    ent.polymer_type = gemmi.PolymerType.PeptideL
    ent.subchains = ["A"]
    ent.full_sequence = ["ALA"] * len(ca)
    st.entities.append(ent)
    st.make_mmcif_document().write_file(str(path))


def _receptor(seed=0):
    """Two domains: a pocket domain around the origin and a far domain 60 A away."""
    rng = np.random.default_rng(seed)
    pocket = rng.normal(0, 7, (30, 3))
    far = rng.normal(0, 5, (30, 3)) + np.array([60.0, 0, 0])
    return np.vstack([pocket, far])


def test_pipeline_rigid_motion_zero_and_far_domain_change_ignored(tmp_path):
    sym, lig = embed(IBU)
    lig = lig - lig.mean(0)
    rec = _receptor()
    ref = tmp_path / "ref.cif"
    _write_complex(ref, rec, sym, lig)
    # (a) whole complex moved rigidly -> 0
    R = Rotation.random(random_state=5).as_matrix()
    t = np.array([5.0, -7.0, 11.0])
    pred = tmp_path / "pred_rigid.cif"
    _write_complex(pred, rec @ R.T + t, sym, lig @ R.T + t)
    r = score.ligand_pose_rmsd(pred, ref, "B", IBU)
    assert r["ligand_rmsd"] == pytest.approx(0.0, abs=1e-4)
    assert r["n_pocket_ca_used"] >= 20 and r["pocket_ca_rmsd"] < 1e-4
    assert r["pred_atom_method"] == "smiles_order"
    # (b) the far domain rotated 40 degrees: the pocket frame is unaffected, ligand RMSD still 0
    rec2 = rec.copy()
    far = rec2[30:]
    c = far.mean(0)
    rec2[30:] = (far - c) @ Rotation.from_euler("z", 40, degrees=True).as_matrix().T + c + np.array([8.0, 3, 0])
    pred2 = tmp_path / "pred_far.cif"
    _write_complex(pred2, rec2, sym, lig)
    assert score.ligand_pose_rmsd(pred2, ref, "B", IBU)["ligand_rmsd"] == pytest.approx(0.0, abs=1e-4)
    # (c) ligand displaced 1.5 A in the pocket frame -> 1.5 A
    pred3 = tmp_path / "pred_shift.cif"
    _write_complex(pred3, rec, sym, lig + np.array([1.5, 0, 0]))
    assert score.ligand_pose_rmsd(pred3, ref, "B", IBU)["ligand_rmsd"] == pytest.approx(1.5, abs=1e-4)


def test_pipeline_symmetry_and_no_realignment_of_ligand(tmp_path):
    sym, lig = embed(IBU)
    lig = lig - lig.mean(0)
    rec = _receptor(1)
    ref = tmp_path / "ref.cif"
    _write_complex(ref, rec, sym, lig)
    # swapped isopropyl methyls -> 0 with symmetry (identity-map RMSD > 0)
    sw = lig.copy()
    sw[[0, 2]] = sw[[2, 0]]
    p = tmp_path / "sw.cif"
    _write_complex(p, rec, sym, sw)
    r = score.ligand_pose_rmsd(p, ref, "B", IBU)
    assert r["ligand_rmsd"] == pytest.approx(0.0, abs=1e-4)
    assert r["ligand_rmsd_identity_map"] > 0.3
    # a ligand rotated about its own centroid by 30 deg: a "best-fit" RMSD would be ~0, the pocket-frame RMSD must not be
    rot = Rotation.from_euler("x", 30, degrees=True).as_matrix()
    q = tmp_path / "rot.cif"
    _write_complex(q, rec, sym, lig @ rot.T)
    r = score.ligand_pose_rmsd(q, ref, "B", IBU)
    expected = score.naive_rmsd(lig @ rot.T, lig)
    assert r["ligand_rmsd"] > 0.5
    assert r["ligand_rmsd"] <= expected + 1e-6
    # shuffled ligand atom order in the prediction (bond-order template path) gives the same number
    perm = np.random.default_rng(2).permutation(len(sym))
    q2 = tmp_path / "rot_shuf.cif"
    _write_complex(q2, rec, [sym[i] for i in perm], (lig @ rot.T)[perm])
    r2 = score.ligand_pose_rmsd(q2, ref, "B", IBU)
    assert r2["pred_atom_method"] == "bond_order_template"
    assert r2["ligand_rmsd"] == pytest.approx(r["ligand_rmsd"], abs=1e-6)
    # chirality of the prediction: mirrored ligand flips the one centre
    mir = tmp_path / "mir.cif"
    _write_complex(mir, rec, sym, lig * np.array([1, 1, -1]))
    assert score.ligand_pose_rmsd(mir, ref, "B", IBU)["chirality_flips"] == 1
    assert r["chirality_flips"] == 0


def test_pipeline_reference_sdf_pose_and_cli(tmp_path, capsys):
    sym, lig = embed(IBU)
    lig = lig - lig.mean(0)
    rec = _receptor(2)
    ref_cif = tmp_path / "ref_recep.cif"
    _write_complex(ref_cif, rec, sym, lig + 50.0)  # the cif's own ligand is a decoy far away
    m = Chem.MolFromSmiles(IBU)
    m = Chem.AddHs(m)
    AllChem.EmbedMolecule(m, randomSeed=7)
    AllChem.MMFFOptimizeMolecule(m)
    # place the embedded conformer at `lig` (heavy atoms) by rigid fit, with hydrogens, as an SDF
    from rdkit.Chem import rdMolAlign

    heavy = [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() > 1]
    ref_conf = Chem.RemoveHs(Chem.Mol(m))
    tm = score._with_conformer(score._template(IBU), lig)
    rdMolAlign.AlignMol(m, tm, atomMap=list(zip(heavy, range(len(heavy)))))
    sdf = tmp_path / "ref.sdf"
    w = Chem.SDWriter(str(sdf))
    w.write(m)
    w.close()
    pred = tmp_path / "pred.cif"
    _write_complex(pred, rec, sym, lig + np.array([0, 2.0, 0]))
    r = score.ligand_pose_rmsd(pred, ref_cif, "B", IBU, ref_ligand_file=sdf)
    assert r["ligand_rmsd"] == pytest.approx(2.0, abs=0.05)
    assert r["n_pocket_residues"] > 5  # pocket defined by the SDF pose, not the decoy
    # CLI
    rc = score.main([str(pred), str(ref_cif), "--ligand-chain", "B", "--smiles", IBU, "--ref-ligand", str(sdf),
                     "--ligand-only", "--json"])
    assert rc == 0
    import json

    out = json.loads(capsys.readouterr().out)
    assert out["ligand"]["ligand_rmsd"] == pytest.approx(2.0, abs=0.05)


@pytest.mark.skipif(
    not (DATA / "lig_aff" / "ref" / "predictions" / "lig_aff" / "lig_aff_model_0.cif").exists(), reason="no Boltz output"
)
def test_real_boltz_ligand_self_rmsd_and_template_path_agree():
    smi = "N[C@@H](Cc1ccc(O)cc1)C(=O)O"
    f = DATA / "lig_aff" / "ref" / "predictions" / "lig_aff" / "lig_aff_model_0.cif"
    r = score.ligand_pose_rmsd(f, f, "B", smi)
    assert r["ligand_rmsd"] == pytest.approx(0.0, abs=1e-6)
    assert r["chirality_centres"] == 1 and r["chirality_flips"] in (0, 1)  # Boltz without potentials can flip a centre
    c = score.load_complex(f, "B")
    m1, i1 = score.ligand_mol_from_coords(smi, c.lig_symbols, c.lig_xyz)
    m2, i2 = score.ligand_mol_from_coords(smi, c.lig_symbols, c.lig_xyz, assume_smiles_order=False)
    assert (i1["method"], i2["method"]) == ("smiles_order", "bond_order_template")
    assert score.symmetric_rmsd_in_place(
        score._template(smi), m1.GetConformer().GetPositions(), m2.GetConformer().GetPositions()
    ) == pytest.approx(0.0, abs=1e-6)
    assert (
        score.chirality_flips(smi, c.lig_symbols, c.lig_xyz)["n_flips"]
        == score.chirality_flips(smi, c.lig_symbols, c.lig_xyz, assume_smiles_order=False)["n_flips"]
    )


# ------------------------------------------------------------------ correlation statistics


def test_pearson_kendall_match_scipy_including_ties():
    rng = np.random.default_rng(0)
    x = rng.normal(size=40)
    y = 0.6 * x + rng.normal(size=40)
    assert score.pearson(x, y) == pytest.approx(sps.pearsonr(x, y)[0], abs=1e-12)
    assert score.kendall_tau(x, y) == pytest.approx(sps.kendalltau(x, y)[0], abs=1e-12)
    xt, yt = np.round(x, 0), np.round(y, 0)  # heavy ties -> tau-b
    assert score.kendall_tau(xt, yt) == pytest.approx(sps.kendalltau(xt, yt)[0], abs=1e-12)
    assert np.isnan(score.pearson([1, 2], [1, 2])) and np.isnan(score.kendall_tau([1, 1, 1], [1, 2, 3]))


def test_known_correlations():
    x = np.arange(10.0)
    assert score.pearson(x, 2 * x + 1) == pytest.approx(1.0)
    assert score.pearson(x, -x) == pytest.approx(-1.0)
    assert score.kendall_tau(x, x**3) == pytest.approx(1.0)
    assert score.kendall_tau(x, x[::-1]) == pytest.approx(-1.0)
    # hand-computed: one discordant pair out of 6 -> tau = (5-1)/6
    assert score.kendall_tau([1, 2, 3, 4], [1, 3, 2, 4]) == pytest.approx(4 / 6)


def _toy_targets(seed=0):
    rng = np.random.default_rng(seed)
    sizes = {"t1": 40, "t2": 25, "t3": 15}
    groups, exp, pred = [], [], []
    for g, n in sizes.items():
        e = rng.normal(size=n)
        groups += [g] * n
        exp += list(e)
        pred += list(0.7 * e + 0.7 * rng.normal(size=n))
    return np.array(groups), np.array(exp), np.array(pred)


def test_weighted_vs_unweighted_pooling():
    groups, exp, pred = _toy_targets()
    per = score.per_target_stat(groups, pred, exp, score.pearson)
    n = {"t1": 40, "t2": 25, "t3": 15}
    w = sum(per[g] * n[g] for g in per) / 80
    u = np.mean(list(per.values()))
    assert score.weighted_pearson(pred, exp, groups) == pytest.approx(w)
    assert score.unweighted_pearson(pred, exp, groups) == pytest.approx(u)
    assert w != pytest.approx(u)
    tau = score.per_target_stat(groups, pred, exp, score.kendall_tau)
    assert score.pooled_kendall(pred, exp, groups) == pytest.approx(sum(tau[g] * n[g] for g in tau) / 80)


def test_bootstrap_ci_covers_estimate_deterministic_and_fisher_width():
    groups, exp, pred = _toy_targets(1)
    r1 = score.ranking_statistics(exp, {"a": pred}, groups, n_boot=2000, seed=11)
    r2 = score.ranking_statistics(exp, {"a": pred}, groups, n_boot=2000, seed=11)
    r3 = score.ranking_statistics(exp, {"a": pred}, groups, n_boot=2000, seed=12)
    a1, a2, a3 = r1["arms"]["a"], r2["arms"]["a"], r3["arms"]["a"]
    assert a1["weighted_pearson_ci"] == a2["weighted_pearson_ci"]  # seedable
    assert a1["weighted_pearson_ci"] != a3["weighted_pearson_ci"]
    lo, hi = a1["weighted_pearson_ci"]
    assert lo < a1["weighted_pearson"] < hi
    lo, hi = a1["pooled_tau_vs_exp_ci"]
    assert lo < a1["pooled_tau_vs_exp"] < hi
    # width vs the single-target Fisher-z interval (n=200, r~0.7): bootstrap half-width within 25%
    rng = np.random.default_rng(5)
    e = rng.normal(size=200)
    p = e + rng.normal(size=200)
    r = score.pearson(p, e)
    fz = np.arctanh(r)
    half = (np.tanh(fz + 1.96 / np.sqrt(197)) - np.tanh(fz - 1.96 / np.sqrt(197))) / 2
    b = score.ranking_statistics(e, {"a": p}, np.array(["x"] * 200), n_boot=4000, seed=0)["arms"]["a"]["weighted_pearson_ci"]
    assert (b[1] - b[0]) / 2 == pytest.approx(half, rel=0.25)


def test_paired_bootstrap_delta_identical_arms_and_degraded_arm():
    groups, exp, pred = _toy_targets(2)
    rng = np.random.default_rng(9)
    same = score.ranking_statistics(exp, {"strict": pred, "cand": pred.copy()}, groups, ref_arm="strict", n_boot=1000)
    c = same["arms"]["cand"]
    assert c["delta_weighted_pearson"] == 0 and c["delta_weighted_pearson_ci"] == (0.0, 0.0)
    assert c["tau_vs_ref"] == pytest.approx(1.0) and c["tau_vs_ref_ci"] == (1.0, 1.0)
    bad = pred + rng.normal(0, 2.0, len(pred))  # noisy candidate: Pearson drops, tau(cand, strict) drops
    r = score.ranking_statistics(exp, {"strict": pred, "cand": bad}, groups, ref_arm="strict", n_boot=1000)["arms"]["cand"]
    assert r["delta_weighted_pearson"] < -0.1 and r["delta_weighted_pearson_ci"][1] < 0
    assert r["tau_vs_ref"] < 0.6
    # a pure offset/monotone transform of the strict predictions changes neither Pearson sign nor tau
    mono = score.ranking_statistics(exp, {"strict": pred, "cand": pred + 3.0}, groups, ref_arm="strict", n_boot=500)
    assert mono["arms"]["cand"]["delta_weighted_pearson"] == pytest.approx(0, abs=1e-12)
    assert mono["arms"]["cand"]["tau_vs_ref"] == pytest.approx(1.0)


def test_bootstrap_resamples_within_target():
    groups, exp, pred = _toy_targets(3)
    boots = score.grouped_bootstrap_indices(groups, 50, np.random.default_rng(0))
    for members, bi in boots:
        assert bi.shape == (50, len(members))
        assert set(np.unique(bi)) <= set(members.tolist())  # never borrows a ligand from another target


def test_reference_ligand_from_pdb_file(tmp_path):
    import gemmi

    sym, lig = embed(IBU)
    lig = lig - lig.mean(0)
    rec = _receptor(4)
    ref = tmp_path / "ref.cif"
    _write_complex(ref, rec, sym, lig)
    st = gemmi.read_structure(str(ref))
    pdb = tmp_path / "ref.pdb"
    st.write_pdb(str(pdb))
    pred = tmp_path / "pred.cif"
    _write_complex(pred, rec, sym, lig + np.array([0, 0, 1.0]))
    r = score.ligand_pose_rmsd(pred, ref, "B", IBU, ref_ligand_file=pdb, ref_ligand_chain="B")
    assert r["ligand_rmsd"] == pytest.approx(1.0, abs=0.02)  # PDB coordinates are rounded to 3 decimals
