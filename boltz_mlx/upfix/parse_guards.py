"""Patches of boltz.data.parse.schema (parse_boltz_schema, standardize) for SMILES ligands. Upstream is not edited.

Three module-level names of the upstream module are replaced by thin wrappers around upstream's own objects:

  schema.AllChem          -> _PatchedAllChem, a proxy that forwards everything to rdkit.Chem.AllChem except
        MolFromSmiles       - fork b555625 (upstream #649): a SMILES that RDKit cannot read raises ValueError("Failed to parse
                              SMILES ...") and the molecule gets UpdatePropertyCache(strict=True), so a bad valence is a Python
                              exception here instead of a SIGSEGV / SIGILL inside AddHs (the fork's report; on this RDKit
                              build the failure is a RuntimeError, not an abort, but the check costs nothing). For every
                              valid molecule the call returns upstream's molecule unchanged (the cache is already computed).
        CanonicalRankAtoms  - upstream #494, names.canonical_rank_atoms: per-element atom names, only for a ligand whose
                              upstream names would exceed 4 characters.
  schema.rdMolStandardize -> _PatchedStandardize: LargestFragmentChooser().choose(mol) first calls
                              mol.UpdatePropertyCache(strict=False) (fork 7407b86). standardize() parses with sanitize=False,
                              so the implicit-valence cache is unset, and for some salts (the fork's example: a terminal alkyne
                              with oxalic acid) the chooser's getNumImplicitHs() raises a pre-condition violation (an uncatchable
                              abort in the fork's environment; a RuntimeError that failed the ligand in this one).

Notices (one stderr line) are printed only when a patch changed something: an unparseable SMILES, an overflowing name,
a molecule whose valence cache was unset.

Fork commits: github.com/boltz-community/boltz (MIT, Wohlwend, Corso, Passaro), b555625, 7407b86, 9cca9d6.
"""

from __future__ import annotations

import threading

from boltz_mlx.upfix import names, say

_lock = threading.Lock()


class _PatchedAllChem:
    _boltz_mlx_parse_guards = True

    def __init__(self, real):
        self._real = real

    def MolFromSmiles(self, smiles, *args, **kwargs):  # noqa: N802 (RDKit's name)
        mol = self._real.MolFromSmiles(smiles, *args, **kwargs)
        if mol is None:
            say(f"RDKit cannot read the ligand SMILES {str(smiles)[:60]!r}")
            raise ValueError(f"Failed to parse SMILES: {smiles}")
        if kwargs.get("sanitize", args[0] if args else True):
            try:
                mol.UpdatePropertyCache(strict=True)
            except Exception as ex:  # noqa: BLE001  (RDKit raises RuntimeError / KekulizeException subclasses)
                say(f"RDKit rejects the valence of the ligand SMILES {str(smiles)[:60]!r}")
                raise ValueError(f"Invalid valence in SMILES {smiles}: {str(ex).splitlines()[0]}") from None
        return mol

    def CanonicalRankAtoms(self, mol, *args, **kwargs):  # noqa: N802
        return names.canonical_rank_atoms(self._real.CanonicalRankAtoms, mol, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._real, name)


class _FragmentChooser:
    def __init__(self, real):
        self._real = real

    def choose(self, mol, *args, **kwargs):
        try:
            unset = any(a.NeedsUpdateProperty("implicitValence") for a in mol.GetAtoms())
        except Exception:  # noqa: BLE001
            unset = False
        mol.UpdatePropertyCache(strict=False)   # strict=False keeps the unsanitized salt intact
        if unset:
            say("standardize: computed the implicit valences of an unsanitized molecule before choosing its largest fragment")
        return self._real.choose(mol, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._real, name)


class _PatchedStandardize:
    _boltz_mlx_parse_guards = True

    def __init__(self, real):
        self._real = real

    def LargestFragmentChooser(self, *args, **kwargs):  # noqa: N802
        return _FragmentChooser(self._real.LargestFragmentChooser(*args, **kwargs))

    def __getattr__(self, name):
        return getattr(self._real, name)


def _installed(obj) -> bool:
    """True if obj is, or wraps (seeding's temporary AllChem proxy), one of our proxies."""
    while obj is not None:
        if "_boltz_mlx_parse_guards" in type(obj).__dict__:
            return True
        obj = obj.__dict__.get("_real")
    return False


def install_parse_guards() -> None:
    """Idempotent."""
    import boltz.data.parse.schema as schema

    with _lock:
        if not _installed(schema.AllChem):
            schema.AllChem = _PatchedAllChem(schema.AllChem)
        if not _installed(schema.rdMolStandardize):
            schema.rdMolStandardize = _PatchedStandardize(schema.rdMolStandardize)
