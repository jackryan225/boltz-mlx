"""2D depiction of a ligand as SVG, rendered server-side with RDKit (a dependency of the screen); nothing is fetched."""

from __future__ import annotations

import threading
from collections import OrderedDict

from boltz_mlx.gui.common import ApiError

_CACHE: "OrderedDict[tuple, str]" = OrderedDict()
_LOCK = threading.Lock()
_MAX = 512
MAX_SMILES = 5000


def depict_svg(smiles: str, width: int = 320, height: int = 240, dark: bool = False) -> str:
    """SVG text for a SMILES (transparent background so the page theme shows through). Raises ApiError(400) if RDKit cannot
    parse it."""
    smiles = (smiles or "").strip()
    if not smiles or len(smiles) > MAX_SMILES:
        raise ApiError(400, "empty or too long SMILES")
    width = max(80, min(int(width), 900))
    height = max(80, min(int(height), 900))
    key = (smiles, width, height, bool(dark))
    with _LOCK:
        hit = _CACHE.get(key)
        if hit is not None:
            _CACHE.move_to_end(key)
            return hit
    from boltz_mlx import smiles_guard

    err = smiles_guard.check_smiles(smiles)   # this server process must not meet an RDKit abort: unusual molecules are tried in a child
    if err:
        raise ApiError(400, err)
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdDepictor
    from rdkit.Chem.Draw import rdMolDraw2D

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ApiError(400, "RDKit could not parse the SMILES")
    # the largest fragment only (the screen keeps it too), so a salt does not shrink the drawing
    frags = Chem.GetMolFrags(mol, asMols=True)
    if len(frags) > 1:
        mol = max(frags, key=lambda m: (m.GetNumHeavyAtoms(), Chem.MolToSmiles(m)))
    rdDepictor.SetPreferCoordGen(True)
    rdDepictor.Compute2DCoords(mol)
    d = rdMolDraw2D.MolDraw2DSVG(width, height)
    if dark:
        rdMolDraw2D.SetDarkMode(d)
    opts = d.drawOptions()
    opts.clearBackground = False
    opts.bondLineWidth = 2
    opts.padding = 0.08
    opts.addStereoAnnotation = True
    d.DrawMolecule(mol)
    d.FinishDrawing()
    svg = d.GetDrawingText()
    with _LOCK:
        _CACHE[key] = svg
        while len(_CACHE) > _MAX:
            _CACHE.popitem(last=False)
    return svg
