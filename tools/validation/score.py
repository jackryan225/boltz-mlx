"""Score a predicted structure against an experimental one: CA-lDDT, TM-score, CA RMSD.

Residues are matched by SEQRES index (mmCIF ``label_seq_id``), so predicted and experimental
structures must be built on the same sequence (the Boltz YAML uses the SEQRES sequence).
Only residues that are modeled (have a CA) in the *experimental* structure are scored.

  CA-lDDT : global, inclusion radius 15 A, thresholds 0.5/1/2/4 A, over experimental-modeled residues.
            Pairs (i, j), i != j, with d_ref(i, j) < 15 A; a pair scores the fraction of thresholds t with
            |d_pred - d_ref| < t; residue score = mean over its pairs; global = mean over residues having
            at least one pair. A residue missing in the prediction scores 0 for all of its pairs.
  TM-score: CA, normalized by the number of experimental-modeled residues, d0 = 1.24 (L-15)^(1/3) - 1.8
            (0.5 for L <= 21, floor 0.5). Superposition search follows Zhang & Skolnick's TM-score program:
            fragment-seeded Kabsch (fragment lengths L, L/2, L/4, ... down to 4, every start offset), then
            iterative re-superposition on the residues within a distance cutoff that grows if too few remain.
  RMSD    : CA RMSD of the common residues after a least-squares (Kabsch) superposition.

Round 2 additions (see the "ligand pose" and "correlation statistics" sections below):
  ligand_pose_rmsd     : ligand heavy-atom RMSD after superposing pocket CA atoms (pocket = reference residues within
                         10 A of the ligand), symmetry-aware (RDKit), no re-alignment of the ligand.
  chirality_flips      : stereocentres whose CIP label in the 3D pose differs from the input SMILES.
  pearson / kendall_tau / weighted_pearson / pooled_kendall / ranking_statistics : correlation statistics with paired
                         (within-target) bootstrap CIs.

Usage: python tools/validation/score.py PRED.cif EXP.cif [--pred-chain A] [--exp-chain A] [--json]
       python tools/validation/score.py PRED.cif REF.cif --ligand-chain B --smiles SMILES
              [--ref-ligand-chain B] [--ref-ligand POSE.sdf|.pdb] [--ligand-only] [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

LDDT_RADIUS = 15.0
LDDT_THRESHOLDS = (0.5, 1.0, 2.0, 4.0)


# --------------------------------------------------------------------------- parsing


def load_ca(path, chain: str | None = None, length: int | None = None) -> np.ndarray:
    """Return CA coordinates indexed by SEQRES position, shape (n, 3) float64, NaN where not modeled.

    ``n`` = entity SEQRES length (or max label_seq_id seen). Uses model 0; first polymer chain unless ``chain``.
    """
    import gemmi

    st = gemmi.read_structure(str(path))
    st.setup_entities()
    model = st[0]
    ch = None
    for c in model:
        if chain is not None and c.name != chain:
            continue
        if len(c.get_polymer()) > 0:
            ch = c
            break
    if ch is None:
        raise ValueError(f"{path}: no polymer chain{'' if chain is None else ' ' + chain}")
    pol = ch.get_polymer()
    idx, xyz = [], []
    seqres_len = 0
    ent = st.get_entity(pol[0].entity_id) if pol[0].entity_id else None
    if ent is not None:
        seqres_len = len(ent.full_sequence)
    for k in range(len(pol)):
        r = pol[k]
        ls = r.label_seq
        i = (int(ls) - 1) if ls is not None and ls > 0 else k  # fall back to order of appearance
        a = r.find_atom("CA", "*")
        if a is None:
            continue
        idx.append(i)
        xyz.append([a.pos.x, a.pos.y, a.pos.z])
    n = max([length or 0, seqres_len, (max(idx) + 1) if idx else 0])
    out = np.full((n, 3), np.nan)
    if idx:
        out[np.array(idx)] = np.array(xyz)
    return out


# --------------------------------------------------------------------------- geometry


def kabsch(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares rigid transform: returns (R, t) with mobile @ R.T + t ~= target (R proper rotation)."""
    mc, tc = mobile.mean(0), target.mean(0)
    H = (mobile - mc).T @ (target - tc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d if d != 0 else 1.0])
    R = Vt.T @ D @ U.T
    return R, tc - R @ mc


def apply(R, t, x):
    return x @ R.T + t


def rmsd_after_kabsch(mobile: np.ndarray, target: np.ndarray) -> float:
    R, t = kabsch(mobile, target)
    return float(np.sqrt(((apply(R, t, mobile) - target) ** 2).sum(1).mean()))


# --------------------------------------------------------------------------- lDDT


def lddt_ca(
    pred: np.ndarray,
    ref: np.ndarray,
    radius: float = LDDT_RADIUS,
    thresholds=LDDT_THRESHOLDS,
    per_residue: bool = False,
):
    """CA-lDDT over residues modeled in ``ref`` (NaN rows = not modeled). pred may have NaN rows too (score 0)."""
    n = min(len(pred), len(ref))
    ref, pred = ref[:n], pred[:n]
    ok = ~np.isnan(ref).any(1)
    idx = np.where(ok)[0]
    r = ref[idx]
    p = pred[idx]
    dref = np.linalg.norm(r[:, None] - r[None], axis=-1)
    dprd = np.linalg.norm(p[:, None] - p[None], axis=-1)  # NaN where pred residue missing
    m = (dref < radius) & ~np.eye(len(idx), dtype=bool)
    diff = np.abs(dprd - dref)
    thr = np.asarray(thresholds)
    # NaN diff compares False -> counts as failure (score 0) for pairs involving a missing predicted residue
    with np.errstate(invalid="ignore"):
        score = (diff[..., None] < thr).mean(-1)
    npairs = m.sum(1)
    res = np.where(npairs > 0, (score * m).sum(1) / np.maximum(npairs, 1), np.nan)
    valid = npairs > 0
    glob = float(res[valid].mean()) if valid.any() else float("nan")
    if per_residue:
        full = np.full(n, np.nan)
        full[idx] = res
        return glob, full
    return glob


# --------------------------------------------------------------------------- TM-score


def tm_d0(l_norm: int) -> float:
    if l_norm <= 21:
        return 0.5
    return max(0.5, 1.24 * (l_norm - 15) ** (1.0 / 3.0) - 1.8)


def _tm_sum(d2: np.ndarray, d0: float) -> float:
    return float((1.0 / (1.0 + d2 / (d0 * d0))).sum())


def tm_score(
    pred: np.ndarray,
    ref: np.ndarray,
    l_norm: int | None = None,
    return_transform: bool = False,
):
    """TM-score of ``pred`` against ``ref`` (aligned residue-by-residue; rows with NaN in either are dropped).

    Normalized by ``l_norm`` (default: number of residues modeled in ``ref``). Returns the best score found by the
    fragment-seeded iterative search (and optionally (R, t) mapping pred onto ref).
    """
    n = min(len(pred), len(ref))
    ok = ~(np.isnan(ref[:n]).any(1))
    l_ref = int(ok.sum())
    L = l_norm or l_ref
    ok &= ~np.isnan(pred[:n]).any(1)
    P = pred[:n][ok].astype(np.float64)
    Q = ref[:n][ok].astype(np.float64)
    Lali = len(P)
    if Lali < 3:
        return (0.0, (np.eye(3), np.zeros(3))) if return_transform else 0.0

    d0 = tm_d0(L)
    d0_search = min(8.0, max(4.5, d0))
    n_it = 20
    L_ini_min = min(4, Lali)

    # seed fragment lengths: Lali, Lali/2, Lali/4, ... until <= L_ini_min (as in TM-score)
    frag_lens = []
    for i in range(6 - 1):
        li = int(Lali / (2.0**i))
        if li <= L_ini_min:
            li = L_ini_min
            frag_lens.append(li)
            break
        frag_lens.append(li)
    else:
        frag_lens.append(L_ini_min)

    def select(d2, cutoff):
        sel = d2 < cutoff * cutoff
        c = cutoff
        while sel.sum() < min(3, Lali):
            c += 0.5
            sel = d2 < c * c
        return sel

    best, best_T = -1.0, (np.eye(3), np.zeros(3))
    for Lf in frag_lens:
        starts = [0] if Lf >= Lali else range(0, Lali - Lf + 1)
        for s in starts:
            sl = slice(s, s + Lf)
            if Lf < 3:
                continue
            R, t = kabsch(P[sl], Q[sl])
            d2 = ((apply(R, t, P) - Q) ** 2).sum(1)
            sc = _tm_sum(d2, d0)
            if sc > best:
                best, best_T = sc, (R, t)
            sel = select(d2, d0_search - 1.0)
            prev = None
            for _ in range(n_it):
                if prev is not None and np.array_equal(sel, prev):
                    break
                prev = sel
                R, t = kabsch(P[sel], Q[sel])
                d2 = ((apply(R, t, P) - Q) ** 2).sum(1)
                sc = _tm_sum(d2, d0)
                if sc > best:
                    best, best_T = sc, (R, t)
                sel = select(d2, d0_search + 1.0)
    tm = best / L
    return (tm, best_T) if return_transform else tm


# --------------------------------------------------------------------------- ligand pose (round 2, L3-L)
#
# Ligand heavy-atom RMSD in the POCKET frame:
#   1. pocket = receptor residues (polymer chains of the REFERENCE) having any heavy atom within ``radius`` (10 A)
#      of any ligand heavy atom of the reference;
#   2. superpose the prediction onto the reference by Kabsch on the CA atoms of those residues (matched by SEQRES
#      index = label_seq_id, same convention as ``load_ca``);
#   3. apply that transform to the predicted ligand and compute the heavy-atom RMSD to the reference ligand WITHOUT any
#      further alignment (the ligand is never moved to improve the fit), minimised only over symmetry-equivalent atom
#      mappings (RDKit ``rdMolAlign.CalcRMS``: all graph automorphisms, conjugated terminal groups such as
#      carboxylate/nitro oxygens symmetrised).
# Boltz mmCIF ligands carry no bond orders: they are rebuilt from the input SMILES (``AssignBondOrdersFromTemplate``),
# or, when the atom order and elements equal the SMILES heavy-atom order and every SMILES bond has a plausible length
# (Boltz writes ligand atoms in SMILES order), the coordinates are assigned to the SMILES atoms directly.

POCKET_RADIUS = 10.0


def _rdkit():
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdDetermineBonds, rdMolAlign

    return Chem, AllChem, rdDetermineBonds, rdMolAlign


def _template(smiles: str):
    Chem = _rdkit()[0]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit cannot parse SMILES {smiles!r}")
    mol = Chem.RemoveHs(mol)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return mol


def _with_conformer(template, xyz: np.ndarray):
    """Copy of ``template`` carrying a single conformer with coordinates ``xyz`` (template atom order)."""
    Chem = _rdkit()[0]
    from rdkit.Geometry import Point3D

    m = Chem.Mol(template)
    m.RemoveAllConformers()
    conf = Chem.Conformer(m.GetNumAtoms())
    for i, p in enumerate(np.asarray(xyz, float)):
        conf.SetAtomPosition(i, Point3D(float(p[0]), float(p[1]), float(p[2])))
    m.AddConformer(conf, assignId=True)
    return m


def ligand_mol_from_coords(smiles: str, symbols, xyz, assume_smiles_order: bool = True, max_matches: int = 1000):
    """Place heavy-atom coordinates (``symbols``, ``xyz``: no bond information) on the molecule described by ``smiles``.

    Returns ``(mol, info)``: ``mol`` is the SMILES molecule (bond orders, stereo from SMILES) with one conformer in the
    SMILES atom order; ``info`` has ``method`` ("smiles_order" or "bond_order_template"), ``atom_map`` (template atom
    index -> input atom index) and ``alt_maps`` (all graph-equivalent maps, capped; only for "bond_order_template").
    """
    Chem, AllChem, rdDetermineBonds, _ = _rdkit()
    tmpl = _template(smiles)
    xyz = np.asarray(xyz, float).reshape(-1, 3)
    symbols = [str(s).strip().capitalize() for s in symbols]
    keep = [i for i, s in enumerate(symbols) if s not in ("H", "D")]
    symbols = [symbols[i] for i in keep]
    xyz = xyz[keep]
    n = tmpl.GetNumAtoms()
    if len(symbols) != n:
        raise ValueError(f"ligand has {len(symbols)} heavy atoms but SMILES has {n}")
    if assume_smiles_order and all(a.GetSymbol() == s for a, s in zip(tmpl.GetAtoms(), symbols)):
        d = np.array([np.linalg.norm(xyz[b.GetBeginAtomIdx()] - xyz[b.GetEndAtomIdx()]) for b in tmpl.GetBonds()])
        if len(d) == 0 or (d.min() > 0.8 and d.max() < 2.4):
            return _with_conformer(tmpl, xyz), {
                "method": "smiles_order",
                "atom_map": list(range(n)),
                "alt_maps": [list(range(n))],
            }
    # connectivity from distances, bond orders from the SMILES template
    rw = Chem.RWMol()
    for s in symbols:
        rw.AddAtom(Chem.Atom(s))
    conf = Chem.Conformer(len(symbols))
    for i, p in enumerate(xyz):
        conf.SetAtomPosition(i, tuple(float(v) for v in p))
    rw.AddConformer(conf)
    rdDetermineBonds.DetermineConnectivity(rw)
    pdbmol = rw.GetMol()
    maps = []
    try:
        pdbmol.UpdatePropertyCache(strict=False)
        withbo = AllChem.AssignBondOrdersFromTemplate(tmpl, pdbmol)
        withbo.UpdatePropertyCache(strict=False)
        maps = [list(m) for m in withbo.GetSubstructMatches(tmpl, uniquify=False, maxMatches=max_matches)]
    except Exception:  # noqa: BLE001  (RDKit raises ValueError/RuntimeError on bond-order assignment failure)
        maps = []
    if not maps:  # graph-only fallback (bond orders ignored)
        sk_t, sk_p = Chem.RWMol(tmpl), Chem.RWMol(pdbmol)
        for sk in (sk_t, sk_p):
            for a in sk.GetAtoms():
                a.SetFormalCharge(0)
                a.SetNoImplicit(True)
                a.SetNumExplicitHs(0)
                a.SetIsAromatic(False)
            for b in sk.GetBonds():
                b.SetBondType(Chem.BondType.SINGLE)
                b.SetIsAromatic(False)
        maps = [list(m) for m in sk_p.GetMol().GetSubstructMatches(sk_t.GetMol(), uniquify=False, maxMatches=max_matches)]
    if not maps:
        raise ValueError(
            "ligand connectivity inferred from the coordinates does not match the SMILES graph "
            "(distorted geometry, wrong SMILES, or different protonation/tautomer graph)"
        )
    m0 = maps[0]
    return _with_conformer(tmpl, xyz[m0]), {
        "method": "bond_order_template",
        "atom_map": m0,
        "alt_maps": maps,
        "_xyz_input": xyz,
    }


def chirality_flips(smiles: str, symbols, xyz, assume_smiles_order: bool = True) -> dict:
    """Stereocentres whose CIP label in the 3D pose differs from the input SMILES (RDKit AssignStereochemistryFrom3D).

    Only centres that are *specified* in the SMILES are checked (an unspecified centre has no label to flip).
    A specified centre that is planar in the pose (no 3D label) counts as a flip. Double-bond (E/Z) flips are
    returned separately. When the atom map is ambiguous (graph-equivalent maps) the map with the fewest flips is used.
    """
    Chem = _rdkit()[0]
    mol0, info = ligand_mol_from_coords(smiles, symbols, xyz, assume_smiles_order)
    tmpl = _template(smiles)
    want = {a.GetIdx(): a.GetProp("_CIPCode") for a in tmpl.GetAtoms() if a.HasProp("_CIPCode")}
    want_db = {
        b.GetIdx(): b.GetStereo()
        for b in tmpl.GetBonds()
        if b.GetStereo() in (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ)
    }
    if info["method"] == "smiles_order":
        cands = [mol0.GetConformer().GetPositions()]
    else:
        xin = info["_xyz_input"]
        cands = [xin[m] for m in info["alt_maps"]]
    best = None
    for pos in cands:
        m = _with_conformer(tmpl, pos)
        Chem.AssignStereochemistryFrom3D(m)
        got = {a.GetIdx(): (a.GetProp("_CIPCode") if a.HasProp("_CIPCode") else None) for a in m.GetAtoms()}
        flipped = [i for i, lab in want.items() if got.get(i) != lab]
        db_flipped = [i for i, st in want_db.items() if m.GetBondWithIdx(i).GetStereo() != st]
        key = (len(flipped), len(db_flipped))
        if best is None or key < best[0]:
            best = (key, flipped, db_flipped)
        if key == (0, 0):
            break
    (n_f, n_db), flipped, db_flipped = best
    return {
        "n_centres": len(want),
        "n_flips": n_f,
        "flipped_atoms": flipped,
        "n_double_bonds": len(want_db),
        "n_double_bond_flips": n_db,
        "n_maps_tried": len(cands),
    }


def symmetric_rmsd_in_place(template, xyz_probe, xyz_ref, symmetrize_terminal_groups: bool = True) -> float:
    """Heavy-atom RMSD between two poses of the same molecule, NO alignment, min over symmetry-equivalent mappings."""
    rdMolAlign = _rdkit()[3]
    p = _with_conformer(template, xyz_probe)
    r = _with_conformer(template, xyz_ref)
    return float(
        rdMolAlign.CalcRMS(p, r, maxMatches=100000, symmetrizeConjugatedTerminalGroups=symmetrize_terminal_groups)
    )


def naive_rmsd(xyz_a, xyz_b) -> float:
    d = np.asarray(xyz_a, float) - np.asarray(xyz_b, float)
    return float(np.sqrt((d * d).sum(1).mean()))


# ---- structure loading for ligand scoring


class ComplexData:
    """Receptor CA atoms (keyed (chain, SEQRES index)), all receptor heavy atoms (per residue key) and the ligand atoms."""

    def __init__(self, ca, prot_xyz, prot_keys, lig_names, lig_symbols, lig_xyz):
        self.ca, self.prot_xyz, self.prot_keys = ca, prot_xyz, prot_keys
        self.lig_names, self.lig_symbols, self.lig_xyz = lig_names, lig_symbols, lig_xyz


def load_complex(path, ligand_chain: str | None = None, receptor_chains=None) -> ComplexData:
    """Read an mmCIF/PDB: polymer residues (key (chain, label_seq_id-1)) and, if ``ligand_chain`` is given, the heavy
    atoms of every residue in that chain (matched on chain name, falling back to label_asym_id)."""
    import gemmi

    st = gemmi.read_structure(str(path))
    st.setup_entities()
    model = st[0]
    ca, pxyz, pkeys = {}, [], []
    lig_n, lig_s, lig_x = [], [], []
    for ch in model:
        pol = ch.get_polymer()
        if len(pol) > 0 and (receptor_chains is None or ch.name in receptor_chains):
            for k in range(len(pol)):
                r = pol[k]
                ls = r.label_seq
                i = (int(ls) - 1) if ls is not None and ls > 0 else k
                for a in r:
                    if a.element.name in ("H", "D"):
                        continue
                    pxyz.append([a.pos.x, a.pos.y, a.pos.z])
                    pkeys.append((ch.name, i))
                    if a.name == "CA":
                        ca[(ch.name, i)] = np.array([a.pos.x, a.pos.y, a.pos.z])
        if ligand_chain is not None and len(pol) == 0 and (ch.name == ligand_chain):
            for r in ch:
                for a in r:
                    if a.element.name in ("H", "D"):
                        continue
                    lig_n.append(a.name)
                    lig_s.append(a.element.name)
                    lig_x.append([a.pos.x, a.pos.y, a.pos.z])
    if ligand_chain is not None and not lig_x:  # fall back to label_asym_id (subchain)
        for ch in model:
            for sub in ch.subchains():
                if sub.subchain_id() == ligand_chain and len(ch.get_polymer()) == 0:
                    for r in sub:
                        for a in r:
                            if a.element.name in ("H", "D"):
                                continue
                            lig_n.append(a.name)
                            lig_s.append(a.element.name)
                            lig_x.append([a.pos.x, a.pos.y, a.pos.z])
    if ligand_chain is not None and not lig_x:
        raise ValueError(f"{path}: no ligand atoms found in chain {ligand_chain!r}")
    return ComplexData(
        ca,
        np.array(pxyz).reshape(-1, 3),
        pkeys,
        lig_n,
        lig_s,
        np.array(lig_x).reshape(-1, 3),
    )


def load_ligand_file(path, chain: str | None = None):
    """Heavy-atom ``(symbols, xyz)`` of a reference ligand pose from an SDF/MOL (first molecule) or PDB/mmCIF file
    (``chain`` selects the ligand chain; without it, all non-water non-polymer residues)."""
    p = str(path)
    low = p.lower()
    if low.endswith((".sdf", ".mol")):
        Chem = _rdkit()[0]
        sup = Chem.SDMolSupplier(p, removeHs=True, sanitize=False)
        mol = next(iter(sup))
        if mol is None:
            raise ValueError(f"{p}: cannot read the first SDF record")
        mol = Chem.RemoveHs(mol, sanitize=False)
        return [a.GetSymbol() for a in mol.GetAtoms()], mol.GetConformer().GetPositions()
    import gemmi

    st = gemmi.read_structure(p)
    st.setup_entities()
    sym, xyz = [], []
    for ch in st[0]:
        if chain is not None and ch.name != chain:
            continue
        if chain is None and len(ch.get_polymer()) > 0:
            continue
        for r in ch:
            if chain is None and r.name in ("HOH", "WAT", "DOD"):
                continue
            if chain is None and r.entity_type == gemmi.EntityType.Polymer:
                continue
            for a in r:
                if a.element.name in ("H", "D"):
                    continue
                sym.append(a.element.name)
                xyz.append([a.pos.x, a.pos.y, a.pos.z])
    if not xyz:
        raise ValueError(f"{p}: no ligand atoms found")
    return sym, np.array(xyz)


def pocket_keys(ref: ComplexData, lig_xyz: np.ndarray, radius: float = POCKET_RADIUS):
    """Residue keys of the reference with any heavy atom within ``radius`` of any ligand heavy atom."""
    from scipy.spatial import cKDTree

    if len(ref.prot_xyz) == 0:
        return set()
    d, _ = cKDTree(lig_xyz).query(ref.prot_xyz, distance_upper_bound=radius)
    near = np.where(np.isfinite(d))[0]
    return {ref.prot_keys[i] for i in near}


def ligand_pose_rmsd(
    pred_path,
    ref_path,
    ligand_chain: str,
    smiles: str,
    *,
    ref_ligand_chain: str | None = None,
    ref_ligand_file=None,
    chain_map: dict | None = None,
    radius: float = POCKET_RADIUS,
    symmetrize_terminal_groups: bool = True,
    min_pocket_ca: int = 3,
) -> dict:
    """Ligand heavy-atom RMSD of ``pred_path`` against ``ref_path`` after superposing pocket CA atoms (see header).

    ``ligand_chain``: ligand chain in the prediction (and, unless ``ref_ligand_chain`` is given, in the reference).
    ``ref_ligand_file``: optional SDF/PDB pose giving the reference ligand coordinates instead of ``ref_path``'s ligand
    (must be in the frame of ``ref_path``'s receptor, which still defines the pocket).
    ``chain_map``: reference polymer chain -> predicted chain (default identity, for all reference polymer chains).
    """
    ref_lc = ref_ligand_chain or ligand_chain
    ref = load_complex(ref_path, None if ref_ligand_file else ref_lc)
    pred = load_complex(pred_path, ligand_chain)
    if ref_ligand_file is not None:
        rs, rx = load_ligand_file(ref_ligand_file, ref_ligand_chain)
    else:
        rs, rx = ref.lig_symbols, ref.lig_xyz
    ref_mol, ref_info = ligand_mol_from_coords(smiles, rs, rx)
    pred_mol, pred_info = ligand_mol_from_coords(smiles, pred.lig_symbols, pred.lig_xyz)
    ref_x = ref_mol.GetConformer().GetPositions()
    pred_x = pred_mol.GetConformer().GetPositions()
    keys = sorted(pocket_keys(ref, ref_x, radius))
    ref_chain_names = {k[0] for k in ref.ca}
    cmap = {c: c for c in ref_chain_names}
    if chain_map:
        cmap.update(chain_map)
    A, B = [], []  # pred, ref pocket CA
    n_missing = 0
    for k in keys:
        if k not in ref.ca:
            n_missing += 1
            continue
        pk = (cmap.get(k[0], k[0]), k[1])
        if pk not in pred.ca:
            n_missing += 1
            continue
        A.append(pred.ca[pk])
        B.append(ref.ca[k])
    if len(A) < min_pocket_ca:
        raise ValueError(f"only {len(A)} pocket CA atoms matched between prediction and reference (need {min_pocket_ca})")
    A, B = np.array(A), np.array(B)
    R, t = kabsch(A, B)
    pred_in_ref = apply(R, t, pred_x)
    sym = symmetric_rmsd_in_place(_template(smiles), pred_in_ref, ref_x, symmetrize_terminal_groups)
    chir = chirality_flips(smiles, pred.lig_symbols, pred.lig_xyz)
    return {
        "ligand_rmsd": sym,
        "ligand_rmsd_identity_map": naive_rmsd(pred_in_ref, ref_x),
        "pocket_ca_rmsd": float(np.sqrt(((apply(R, t, A) - B) ** 2).sum(1).mean())),
        "n_pocket_residues": len(keys),
        "n_pocket_ca_used": int(len(A)),
        "n_pocket_missing": int(n_missing),
        "n_ligand_atoms": int(len(ref_x)),
        "pred_atom_method": pred_info["method"],
        "ref_atom_method": ref_info["method"],
        "chirality_flips": chir["n_flips"],
        "chirality_centres": chir["n_centres"],
        "double_bond_flips": chir["n_double_bond_flips"],
    }


# --------------------------------------------------------------------------- correlation statistics + bootstrap
#
# All bootstraps resample LIGANDS WITHIN TARGET (stratified); the same resampled indices are applied to every arm
# (paired). Pooled statistics are LIGAND-COUNT-WEIGHTED means of within-target statistics (Pearson r and Kendall
# tau-b). Kendall tau is tau-b (ties handled), identical to scipy.stats.kendalltau.


def pearson(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return float("nan")
    xm, ym = x - x.mean(), y - y.mean()
    den = np.sqrt((xm * xm).sum() * (ym * ym).sum())
    return float((xm * ym).sum() / den) if den > 0 else float("nan")


def _sign_matrix(x):
    x = np.asarray(x, float)
    return np.sign(x[:, None] - x[None, :]).astype(np.int8)


def kendall_tau(x, y) -> float:
    """Kendall tau-b (O(n^2)); nan if n < 3 or a variable is constant."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return float("nan")
    sx, sy = _sign_matrix(x).astype(np.int64), _sign_matrix(y).astype(np.int64)
    den = np.sqrt((sx * sx).sum() * (sy * sy).sum())
    return float((sx * sy).sum() / den) if den > 0 else float("nan")


def _pearson_rows(x, y):
    """Row-wise Pearson for (B, n) arrays; nan where degenerate."""
    xm = x - x.mean(1, keepdims=True)
    ym = y - y.mean(1, keepdims=True)
    num = (xm * ym).sum(1)
    den = np.sqrt((xm * xm).sum(1) * (ym * ym).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def _kendall_rows(sx, sy, idx, chunk: int = 256):
    """Row-wise tau-b of resamples idx (B, n) of two sign matrices sx, sy (n, n)."""
    out = np.empty(len(idx))
    for s in range(0, len(idx), chunk):
        ii = idx[s : s + chunk]
        gx = sx[ii[:, :, None], ii[:, None, :]].astype(np.int32)
        gy = sy[ii[:, :, None], ii[:, None, :]].astype(np.int32)
        num = (gx * gy).sum((1, 2)).astype(float)
        den = np.sqrt((gx * gx).sum((1, 2)).astype(float) * (gy * gy).sum((1, 2)).astype(float))
        with np.errstate(invalid="ignore", divide="ignore"):
            out[s : s + chunk] = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    return out


def _group_index(groups):
    groups = np.asarray(groups)
    names = sorted(set(groups.tolist()))
    return names, [np.where(groups == g)[0] for g in names]


def wmean(values, weights) -> float:
    """Weighted mean ignoring NaN values (their weight is dropped too); nan if nothing finite."""
    v, w = np.asarray(values, float), np.asarray(weights, float)
    ok = np.isfinite(v)
    return float((v[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() and w[ok].sum() > 0 else float("nan")


def per_target_stat(groups, a, b, fn) -> dict:
    names, idxs = _group_index(groups)
    a, b = np.asarray(a, float), np.asarray(b, float)
    return {g: fn(a[i], b[i]) for g, i in zip(names, idxs)}


def weighted_pearson(pred, exp, groups) -> float:
    """Ligand-count-weighted mean over targets of the within-target Pearson r(pred, exp)."""
    names, idxs = _group_index(groups)
    pred, exp = np.asarray(pred, float), np.asarray(exp, float)
    return wmean([pearson(pred[i], exp[i]) for i in idxs], [len(i) for i in idxs])


def unweighted_pearson(pred, exp, groups) -> float:
    """Unweighted mean over targets of the within-target Pearson r (A4 primary aggregation)."""
    names, idxs = _group_index(groups)
    pred, exp = np.asarray(pred, float), np.asarray(exp, float)
    return wmean([pearson(pred[i], exp[i]) for i in idxs], np.ones(len(idxs)))


def pooled_kendall(x, y, groups) -> float:
    """Ligand-count-weighted mean over targets of the within-target Kendall tau-b(x, y)."""
    names, idxs = _group_index(groups)
    x, y = np.asarray(x, float), np.asarray(y, float)
    return wmean([kendall_tau(x[i], y[i]) for i in idxs], [len(i) for i in idxs])


def grouped_bootstrap_indices(groups, n_boot: int, rng) -> list:
    """Stratified (within-target) bootstrap: list of ``(member_indices, resampled (n_boot, n_g) indices into the
    full arrays)`` per target, in sorted target-name order. Drawn once and reused for all arms (paired)."""
    names, idxs = _group_index(groups)
    out = []
    for i in idxs:
        draw = rng.integers(0, len(i), size=(n_boot, len(i)))
        out.append((i, i[draw]))
    return out


def boot_weighted_pearson(pred_arms: dict, exp, boots, weights="ligands") -> dict:
    """Bootstrap distribution (n_boot,) of the weighted (or unweighted) mean per-target Pearson for each arm."""
    exp = np.asarray(exp, float)
    res = {}
    n_boot = boots[0][1].shape[0]
    w = np.array([len(i) for i, _ in boots], float) if weights == "ligands" else np.ones(len(boots))
    for name, p in pred_arms.items():
        p = np.asarray(p, float)
        r = np.stack([_pearson_rows(p[bi], exp[bi]) for _, bi in boots], 1)  # (B, T)
        ok = np.isfinite(r)
        ww = np.where(ok, w[None], 0.0)
        res[name] = np.where(ww.sum(1) > 0, (np.where(ok, r, 0) * ww).sum(1) / np.maximum(ww.sum(1), 1e-300), np.nan)
    return res


def boot_pooled_kendall(x, y, boots) -> np.ndarray:
    """Bootstrap distribution of the ligand-weighted pooled Kendall tau-b(x, y)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    taus = []
    for i, bi in boots:
        # sign matrices on the full arrays restricted to the group members, resample indices local to the group
        sx, sy = _sign_matrix(x[i]), _sign_matrix(y[i])
        local = np.searchsorted(i, bi)  # members ``i`` are sorted
        taus.append(_kendall_rows(sx, sy, local))
    taus = np.stack(taus, 1)
    w = np.array([len(i) for i, _ in boots], float)
    ok = np.isfinite(taus)
    ww = np.where(ok, w[None], 0.0)
    return np.where(ww.sum(1) > 0, (np.where(ok, taus, 0) * ww).sum(1) / np.maximum(ww.sum(1), 1e-300), np.nan)


def percentile_ci(samples, level: float = 0.95):
    s = np.asarray(samples, float)
    s = s[np.isfinite(s)]
    if len(s) == 0:
        return (float("nan"), float("nan"))
    a = (1 - level) / 2
    lo, hi = np.quantile(s, [a, 1 - a])
    return float(lo), float(hi)


def ranking_statistics(exp, arms: dict, groups, ref_arm: str | None = None, n_boot: int = 10000, seed: int = 0) -> dict:
    """Pearson / Kendall statistics with paired bootstrap CIs (ligands resampled within target, ``n_boot`` resamples).

    ``arms``: name -> per-ligand predictions (same ligand order as ``exp`` and ``groups``).
    Per arm: ligand-weighted and unweighted mean per-target Pearson vs ``exp``, weighted pooled Kendall tau vs ``exp``,
    each with a percentile 95% CI and the per-target values. If ``ref_arm`` is given, for every other arm also:
    ``delta_weighted_pearson`` (arm - ref; CI) and ``tau_vs_ref`` (weighted pooled Kendall tau(arm, ref); CI).
    """
    rng = np.random.default_rng(seed)
    exp = np.asarray(exp, float)
    boots = grouped_bootstrap_indices(groups, n_boot, rng)
    wp = boot_weighted_pearson(arms, exp, boots, "ligands")
    up = boot_weighted_pearson(arms, exp, boots, "targets")
    out = {"n_ligands": int(len(exp)), "n_boot": int(n_boot), "seed": int(seed), "arms": {}}
    for name, p in arms.items():
        p = np.asarray(p, float)
        tau_b = boot_pooled_kendall(p, exp, boots)
        out["arms"][name] = {
            "weighted_pearson": weighted_pearson(p, exp, groups),
            "weighted_pearson_ci": percentile_ci(wp[name]),
            "unweighted_pearson": unweighted_pearson(p, exp, groups),
            "unweighted_pearson_ci": percentile_ci(up[name]),
            "pooled_tau_vs_exp": pooled_kendall(p, exp, groups),
            "pooled_tau_vs_exp_ci": percentile_ci(tau_b),
            "per_target_pearson": per_target_stat(groups, p, exp, pearson),
            "per_target_tau": per_target_stat(groups, p, exp, kendall_tau),
        }
    if ref_arm is not None:
        ref = np.asarray(arms[ref_arm], float)
        for name, p in arms.items():
            if name == ref_arm:
                continue
            p = np.asarray(p, float)
            d = wp[name] - wp[ref_arm]
            tv = boot_pooled_kendall(p, ref, boots)
            out["arms"][name].update(
                {
                    "delta_weighted_pearson": weighted_pearson(p, exp, groups) - weighted_pearson(ref, exp, groups),
                    "delta_weighted_pearson_ci": percentile_ci(d),
                    "delta_weighted_pearson_boot": d,
                    "tau_vs_ref": pooled_kendall(p, ref, groups),
                    "tau_vs_ref_ci": percentile_ci(tv),
                    "tau_vs_ref_boot": tv,
                    "per_target_tau_vs_ref": per_target_stat(groups, p, ref, kendall_tau),
                }
            )
    return out


# --------------------------------------------------------------------------- driver


def score_arrays(pred: np.ndarray, ref: np.ndarray) -> dict:
    n = max(len(pred), len(ref))
    pred = np.vstack([pred, np.full((n - len(pred), 3), np.nan)])
    ref = np.vstack([ref, np.full((n - len(ref), 3), np.nan)])
    ref_ok = ~np.isnan(ref).any(1)
    both = ref_ok & ~np.isnan(pred).any(1)
    tm, (R, t) = tm_score(pred, ref, return_transform=True)
    rmsd = rmsd_after_kabsch(pred[both], ref[both]) if both.sum() >= 3 else float("nan")
    tm_rmsd = float(np.sqrt(((apply(R, t, pred[both]) - ref[both]) ** 2).sum(1).mean())) if both.sum() else float("nan")
    return {
        "lddt_ca": lddt_ca(pred, ref),
        "tm_score": float(tm),
        "rmsd_ca": rmsd,
        "rmsd_ca_tm_superposition": tm_rmsd,
        "n_ref_modeled": int(ref_ok.sum()),
        "n_scored": int(both.sum()),
        "n_seqres": int(n),
    }


def score_files(pred_cif, exp_cif, pred_chain=None, exp_chain=None) -> dict:
    ref = load_ca(exp_cif, exp_chain)
    pred = load_ca(pred_cif, pred_chain, length=len(ref))
    if len(pred) != len(ref):
        raise ValueError(f"SEQRES length mismatch: pred {len(pred)} vs experimental {len(ref)}")
    return score_arrays(pred, ref)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pred", type=Path)
    ap.add_argument("exp", type=Path)
    ap.add_argument("--pred-chain", default=None)
    ap.add_argument("--exp-chain", default=None)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--ligand-chain", default=None, help="ligand chain id (prediction); enables ligand RMSD + chirality")
    ap.add_argument("--smiles", default=None, help="input SMILES of the ligand (bond orders + stereo)")
    ap.add_argument("--ref-ligand-chain", default=None, help="ligand chain in the reference (default: --ligand-chain)")
    ap.add_argument("--ref-ligand", type=Path, default=None, help="reference ligand pose (SDF/PDB) in the reference frame")
    ap.add_argument("--ligand-only", action="store_true", help="skip CA-lDDT/TM scoring")
    a = ap.parse_args(argv)
    res = {}
    if not a.ligand_only:
        res = score_files(a.pred, a.exp, a.pred_chain, a.exp_chain)
    if a.ligand_chain:
        if not a.smiles:
            ap.error("--ligand-chain needs --smiles")
        res["ligand"] = ligand_pose_rmsd(
            a.pred,
            a.exp,
            a.ligand_chain,
            a.smiles,
            ref_ligand_chain=a.ref_ligand_chain,
            ref_ligand_file=a.ref_ligand,
        )
    if a.json:
        print(json.dumps(res))
    else:
        if "lddt_ca" in res:
            print(
                f"CA-lDDT {res['lddt_ca']:.4f}  TM-score {res['tm_score']:.4f}  CA-RMSD {res['rmsd_ca']:.3f} A  "
                f"({res['n_scored']}/{res['n_ref_modeled']} modeled residues scored, SEQRES {res['n_seqres']})"
            )
        if "ligand" in res:
            L = res["ligand"]
            print(
                f"ligand RMSD (pocket frame, symmetry-aware) {L['ligand_rmsd']:.3f} A  "
                f"[identity map {L['ligand_rmsd_identity_map']:.3f}]  pocket {L['n_pocket_ca_used']}/{L['n_pocket_residues']} CA "
                f"(fit {L['pocket_ca_rmsd']:.3f} A)  chirality flips {L['chirality_flips']}/{L['chirality_centres']}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
