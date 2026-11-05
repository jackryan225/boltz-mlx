"""Lead optimisation helpers: poses superposed onto a reference pose (so analogues sit in the same frame side by side) and
2D depictions drawn in the reference compound's orientation with the atoms that differ from it highlighted.

Both run on the CPU (gemmi, RDKit) on files a screen already wrote; nothing here runs the model.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path

from boltz_mlx.gui.common import ApiError

MIN_CA = 3          # fewer matched C-alpha atoms than this: no superposition
MCS_TIMEOUT_S = 2   # RDKit FindMCS time budget per pair
_CACHE: "OrderedDict[tuple, object]" = OrderedDict()
_LOCK = threading.Lock()
_MAX = 256


def _cached(key, make):
    with _LOCK:
        if key in _CACHE:
            _CACHE.move_to_end(key)
            return _CACHE[key]
    val = make()
    with _LOCK:
        _CACHE[key] = val
        while len(_CACHE) > _MAX:
            _CACHE.popitem(last=False)
    return val


def _ca_positions(path: Path) -> dict:
    import gemmi

    st = gemmi.read_structure(str(path))
    st.setup_entities()
    out = {}
    for ch in st[0]:
        for r in ch:
            if r.het_flag != "A":
                continue
            a = r.find_atom("CA", "*")
            if a is not None:
                out[(ch.name, r.seqid.num, r.seqid.icode)] = a.pos
    return out


def superpose(path: Path, ref_path: Path):
    """(gemmi.Transform moving `path` onto `ref_path` by the C-alpha atoms both have, rmsd A, n atoms)."""
    import gemmi

    a, b = _ca_positions(ref_path), _ca_positions(path)
    keys = [k for k in a if k in b]
    if len(keys) < MIN_CA:
        raise ApiError(400, f"cannot superpose: only {len(keys)} matching C-alpha atoms")
    sup = gemmi.superpose_positions([a[k] for k in keys], [b[k] for k in keys])
    return sup.transform, float(sup.rmsd), len(keys)


def aligned_pose(path: Path, ref_path: Path) -> tuple[str, str, float, int]:
    """The pose at `path` with its coordinates moved onto `ref_path`'s frame: (text, format, rmsd, n_ca). mmCIF files keep
    every category (pLDDT, chem_comp, ...): only the Cartn_x/y/z columns of atom_site are rewritten."""
    key = ("pose", str(path), path.stat().st_mtime, str(ref_path), ref_path.stat().st_mtime)

    def make():
        import gemmi

        tr, rmsd, n = superpose(path, ref_path)
        if path.suffix == ".pdb":
            st = gemmi.read_structure(str(path))
            st[0].transform_pos_and_adp(tr)
            return st.make_pdb_string(), "pdb", rmsd, n
        doc = gemmi.cif.read(str(path))
        block = doc.sole_block()
        tab = block.find("_atom_site.", ["Cartn_x", "Cartn_y", "Cartn_z"])
        if not len(tab):
            raise ApiError(500, f"{path.name}: no atom_site coordinates")
        for row in tab:
            p = tr.apply(gemmi.Position(float(row[0]), float(row[1]), float(row[2])))
            row[0], row[1], row[2] = f"{p.x:.3f}", f"{p.y:.3f}", f"{p.z:.3f}"
        return doc.as_string(), "mmcif", rmsd, n

    return _cached(key, make)


def _changed_sites(mol, ref, pairs) -> dict:
    """Atoms and bonds inside the common part (pairs: [(mol idx, ref idx)]) that still differ: formal charge, isotope, aromaticity,
    R/S and E/Z labels (or one side unspecified), and the order of a mapped bond. The MCS compares elements and bond orders only."""
    from rdkit import Chem

    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    Chem.AssignStereochemistry(ref, cleanIt=True, force=True)
    to_ref = dict(pairs)
    atoms: set[int] = set()
    for mi, ri in pairs:
        a, b = mol.GetAtomWithIdx(mi), ref.GetAtomWithIdx(ri)
        cip = lambda x: x.GetProp("_CIPCode") if x.HasProp("_CIPCode") else ""  # noqa: E731
        if (a.GetFormalCharge() != b.GetFormalCharge() or a.GetIsotope() != b.GetIsotope() or a.GetIsAromatic() != b.GetIsAromatic()
                or cip(a) != cip(b)):
            atoms.add(mi)
    bonds: set[int] = set()
    for bd in mol.GetBonds():
        i, j = bd.GetBeginAtomIdx(), bd.GetEndAtomIdx()
        if i in to_ref and j in to_ref:
            rb = ref.GetBondBetweenAtoms(to_ref[i], to_ref[j])
            if rb is None or bd.GetBondType() != rb.GetBondType() or bd.GetStereo() != rb.GetStereo():
                bonds.add(bd.GetIdx())
    return {"atoms": atoms, "bonds": bonds}


def depict_vs(smiles: str, ref_smiles: str, width: int = 320, height: int = 240, dark: bool = False) -> tuple[str, dict]:
    """SVG of `smiles` laid out like `ref_smiles` (by their maximum common substructure), with the atoms outside that common
    part highlighted: what this analogue changes relative to the reference compound. (svg, {"mcs_atoms", "changed_atoms"})."""
    width, height = max(80, min(int(width), 900)), max(80, min(int(height), 900))
    key = ("2d", smiles, ref_smiles, width, height, bool(dark))

    def make():
        from rdkit import Chem, RDLogger
        from rdkit.Chem import rdDepictor, rdFMCS
        from rdkit.Chem.Draw import rdMolDraw2D

        RDLogger.DisableLog("rdApp.*")

        def big(s):
            from boltz_mlx import smiles_guard

            err = smiles_guard.check_smiles(s or "")
            if err:
                raise ApiError(400, err)
            m = Chem.MolFromSmiles(s or "")
            if m is None:
                raise ApiError(400, "RDKit could not parse the SMILES")
            frags = Chem.GetMolFrags(m, asMols=True)
            return max(frags, key=lambda x: (x.GetNumHeavyAtoms(), Chem.MolToSmiles(x))) if len(frags) > 1 else m

        mol, ref = big(smiles), big(ref_smiles)
        rdDepictor.SetPreferCoordGen(True)
        rdDepictor.Compute2DCoords(ref)
        res = rdFMCS.FindMCS([ref, mol], timeout=MCS_TIMEOUT_S, ringMatchesRingOnly=True, completeRingsOnly=True)
        core = Chem.MolFromSmarts(res.smartsString) if res.numAtoms else None
        match = mol.GetSubstructMatch(core) if core is not None else ()
        if match:
            try:
                rdDepictor.GenerateDepictionMatching2DStructure(mol, ref, refPatt=core)
            except ValueError:
                rdDepictor.Compute2DCoords(mol)
        else:
            rdDepictor.Compute2DCoords(mol)
        rmatch = ref.GetSubstructMatch(core) if match and core is not None else ()
        pairs = list(zip(match, rmatch)) if len(match) == len(rmatch) else []
        changed = {a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() not in set(match)}
        sites = _changed_sites(mol, ref, pairs)    # mapped atoms and bonds that differ: charge, isotope, stereo, bond type
        changed |= sites["atoms"]
        bonds = {b.GetIdx() for b in mol.GetBonds() if b.GetBeginAtomIdx() in changed or b.GetEndAtomIdx() in changed} | sites["bonds"]
        changed, bonds = sorted(changed), sorted(bonds)
        d = rdMolDraw2D.MolDraw2DSVG(width, height)
        if dark:
            rdMolDraw2D.SetDarkMode(d)
        opts = d.drawOptions()
        opts.clearBackground = False
        opts.bondLineWidth = 2
        opts.padding = 0.08
        opts.addStereoAnnotation = True
        colour = (0.96, 0.62, 0.27) if dark else (0.98, 0.72, 0.42)
        d.DrawMolecule(mol, highlightAtoms=changed, highlightBonds=bonds,
                       highlightAtomColors={i: colour for i in changed}, highlightBondColors={i: colour for i in bonds})
        d.FinishDrawing()
        return d.GetDrawingText(), {"mcs_atoms": int(res.numAtoms), "changed_atoms": len(changed), "n_atoms": mol.GetNumAtoms(),
                                    "mcs_timed_out": bool(res.canceled), "mapped_changes": len(sites["atoms"]) + len(sites["bonds"])}

    return _cached(key, make)
