"""Fast preset rule for tiny ligands (owner decision): Fast runs 20 structure steps, but at 20 steps a 4-atom ligand (BCT, PDB
8gpp) collapsed in every run; at 50 and 200 steps it did not. So when the preset is Fast (CLI "screen") and the user has not
set the structure steps, any ligand with fewer than TINY_HEAVY_ATOMS heavy atoms makes the structure pass run TINY_STEPS (200)
steps. The affinity pass stays at the preset's steps. An explicit step count always wins; Full is unchanged.

Heavy atoms = atoms with atomic number > 1. SMILES: RDKit. CCD codes: the upstream Boltz-2 `mols/` dictionary (<code>.pkl,
RDKit mols), the same files `predict` featurizes from. This module has no model code: it runs on the CPU, before any GPU work.

  predict / GUI predict  -> one decision for the whole call (the sampler is built once): any tiny ligand in the input(s) -> 200
  screen / GUI screen    -> per ligand (ScreenConfig.steps_for): only the tiny ligands run 200, the rest keep Fast's steps
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

TINY_HEAVY_ATOMS = 10      # a ligand with fewer heavy atoms than this is tiny
TINY_STEPS = 200           # structure steps for a run with a tiny ligand (affinity steps are not changed)
RULE_PRESET = "screen"     # the CLI / internal name of Fast (GUI key "standard", public name "fast")


@dataclass
class LigandSize:
    label: str               # "ligand B" (+ file name for a directory of inputs)
    kind: str                # "smiles" | "ccd"
    value: str
    heavy: int | None        # None: could not be counted (invalid SMILES, CCD code not in the dictionary)

    @property
    def tiny(self) -> bool:
        return self.heavy is not None and self.heavy < TINY_HEAVY_ATOMS


def heavy_atoms_smiles(smiles: str) -> int | None:
    """Atoms with atomic number > 1 in the SMILES (RDKit), or None when RDKit cannot read it."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        mol = Chem.MolFromSmiles(str(smiles).strip(), sanitize=False)
    return None if mol is None else sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)


def heavy_atoms_ccd(code: str, mols_dir: str | Path | None) -> int | None:
    """Atoms with atomic number > 1 of the CCD component `code` in the upstream mols/ dictionary, or None when it is not there."""
    if mols_dir is None:
        return None
    p = Path(mols_dir) / f"{str(code).strip()}.pkl"
    if not p.is_file():
        return None
    try:
        with p.open("rb") as f:
            mol = pickle.load(f)  # noqa: S301  (the upstream dictionary, the same files predict loads)
        return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    except Exception:  # noqa: BLE001
        return None


def size_of(label: str, kind: str, value: str, mols_dir=None) -> LigandSize:
    n = heavy_atoms_smiles(value) if kind == "smiles" else heavy_atoms_ccd(value, mols_dir)
    return LigandSize(label, kind, str(value), n)


def _first_id(ids) -> str:
    return str((ids[0] if isinstance(ids, list) and ids else ids) or "?")


def sizes_from_doc(doc: dict, mols_dir=None, prefix: str = "") -> list[LigandSize]:
    """Every `ligand:` entry of a Boltz YAML document (one entry with several ids counts once)."""
    out: list[LigandSize] = []
    for ent in (doc.get("sequences") or []) if isinstance(doc, dict) else []:
        body = ent.get("ligand") if isinstance(ent, dict) else None
        if not isinstance(body, dict):
            continue
        label = f"{prefix}ligand {_first_id(body.get('id'))}"
        if body.get("smiles"):
            out.append(size_of(label, "smiles", body["smiles"], mols_dir))
        elif body.get("ccd"):
            out.append(size_of(label, "ccd", str(body["ccd"]).strip().upper(), mols_dir))
    return out


def sizes_from_fasta(text: str, mols_dir=None, prefix: str = "") -> list[LigandSize]:
    """Ligands of an upstream-style FASTA (`>A|smiles|` or `>A|ccd|`, the ligand on the next line)."""
    out: list[LigandSize] = []
    head = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(">"):
            head = line[1:].split("|")
        elif line and head is not None:
            if len(head) >= 2 and head[1].strip().lower() in ("smiles", "ccd"):
                kind = head[1].strip().lower()
                out.append(size_of(f"{prefix}ligand {head[0].strip() or '?'}", kind, line.upper() if kind == "ccd" else line, mols_dir))
            head = None
    return out


def sizes_from_path(path: str | Path, mols_dir=None) -> list[LigandSize]:
    """The ligands of an input file, or of every .yaml / .yml / .fasta file in a directory (file names prefix the labels then)."""
    import yaml

    p = Path(path).expanduser()
    files = sorted(f for f in p.iterdir() if f.suffix.lower() in (".yaml", ".yml", ".fasta", ".fa", ".fas")) if p.is_dir() else [p]
    out: list[LigandSize] = []
    for f in files:
        prefix = f"{f.name}: " if p.is_dir() else ""
        try:
            text = f.read_text()
            if f.suffix.lower() in (".yaml", ".yml"):
                out += sizes_from_doc(yaml.safe_load(text), mols_dir, prefix)
            else:
                out += sizes_from_fasta(text, mols_dir, prefix)
        except (OSError, ValueError, yaml.YAMLError):
            continue   # unreadable input: predict reports it
    return out


def reason(tiny: list[LigandSize], limit: int = 3) -> str:
    """'ligand B has 4 heavy atoms (< 10)'; several: 'ligand B has 4 heavy atoms, ligand C has 6 heavy atoms (< 10)'."""
    parts = [f"{t.label} has {t.heavy} heavy atom{'s' if t.heavy != 1 else ''}" for t in tiny[:limit]]
    if len(tiny) > limit:
        parts.append(f"{len(tiny) - limit} more")
    return ", ".join(parts) + f" (< {TINY_HEAVY_ATOMS})"


def decide(sizes: list[LigandSize], steps: int) -> tuple[int, str | None]:
    """(structure steps, reason) for a call whose Fast step count is `steps`: TINY_STEPS and the reason when any ligand is tiny."""
    tiny = [s for s in sizes if s.tiny]
    if not tiny:
        return steps, None
    return max(steps, TINY_STEPS), reason(tiny)


def uncounted(sizes: list[LigandSize]) -> list[LigandSize]:
    return [s for s in sizes if s.heavy is None]
