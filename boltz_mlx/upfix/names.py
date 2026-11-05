"""Ligand atom names for SMILES ligands that overflow upstream's 4-character limit (upstream issue #494).

Upstream (boltz/data/parse/schema.py parse_boltz_schema) names every atom of a SMILES ligand <ELEMENT><canonical rank + 1>
and raises when that is longer than 4 characters: a Cl or Br whose rank is >= 100 ("CL100"), a C of rank >= 10000. Atom
names are a model input (ref_atom_name_chars), so they must not change for ligands that work today. This patch changes
the names only of a ligand where upstream's naming would overflow: such a ligand gets <ELEMENT><n>, n counting the atoms of
that element in canonical-rank order (C1, C2, ..., BR1, CL1). Every other ligand keeps upstream's names exactly.

Mechanism: upstream computes the names from AllChem.CanonicalRankAtoms(mol) in one place, as symbol + str(rank + 1).
parse_guards.py replaces schema.AllChem by a proxy whose CanonicalRankAtoms is canonical_rank_atoms below: upstream's ranks
untouched when every name fits, per-element counts (minus 1, so that upstream's own "+ 1" gives n) when one would not.

Follows the boltz-community fork commit 9cca9d6 (MIT, Wohlwend, Corso, Passaro), which numbers every ligand per element;
here only the ligands that overflow are renumbered.
"""

from __future__ import annotations

MAX_NAME = 4  # PDB atom-name columns; upstream's ValueError threshold


def _name(symbol: str, number: int) -> str:
    return symbol.upper() + str(number)


def upstream_names_fit(symbols, ranks) -> bool:
    """True when <SYMBOL><rank + 1> is at most 4 characters for every atom (upstream's naming works)."""
    return all(len(_name(s, int(r) + 1)) <= MAX_NAME for s, r in zip(symbols, ranks))


def per_element_numbers(symbols, ranks) -> list[int]:
    """For each atom, its 1-based count among the atoms of its element, taken in canonical-rank order."""
    symbols = [s.upper() for s in symbols]
    order = sorted(range(len(symbols)), key=lambda i: (int(ranks[i]), i))
    seen: dict[str, int] = {}
    out = [0] * len(symbols)
    for i in order:
        seen[symbols[i]] = seen.get(symbols[i], 0) + 1
        out[i] = seen[symbols[i]]
    return out


def element_names_fit(symbols, ranks) -> bool:
    """True when the per-element names fit in 4 characters (False: more than 999 C, or 99 Cl, ... in one ligand)."""
    return all(len(_name(s, n)) <= MAX_NAME for s, n in zip(symbols, per_element_numbers(symbols, ranks)))


def atom_name_problem(mol_with_hs) -> str | None:
    """None when the ligand can be named (by upstream's scheme or, if that overflows, per element), else why not."""
    from rdkit.Chem import AllChem

    ranks = list(AllChem.CanonicalRankAtoms(mol_with_hs))
    symbols = [a.GetSymbol() for a in mol_with_hs.GetAtoms()]
    if upstream_names_fit(symbols, ranks) or element_names_fit(symbols, ranks):
        return None
    return "too many atoms of one element for 4-character atom names (more than 999 of a one-letter element, or 99 of a two-letter one)"


def canonical_rank_atoms(real_canonical_rank_atoms, mol, *args, **kwargs):
    """Upstream's ranks when every upstream name fits in 4 characters, else per-element numbers minus 1."""
    ranks = real_canonical_rank_atoms(mol, *args, **kwargs)
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    if upstream_names_fit(symbols, ranks):
        return ranks   # untouched: the names of every normal ligand are upstream's
    numbers = per_element_numbers(symbols, ranks)
    from boltz_mlx.upfix import say

    say(f"ligand with {len(symbols)} atoms: upstream atom names (element + canonical rank) would exceed 4 characters; "
        "named per element instead (C1, C2, ..., CL1). Ligands whose names fit are unchanged.")
    return [n - 1 for n in numbers]   # upstream adds 1
