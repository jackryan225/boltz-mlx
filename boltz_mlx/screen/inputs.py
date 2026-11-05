"""Screen inputs: ligand lists (.csv / .tsv / .smi / .sdf), RDKit validation and size limits, the target (YAML / FASTA / sequence).

Ligand size follows upstream (boltz/data/parse/schema.py, docs/prediction.md "Properties (affinity)"): atoms of
RemoveHs(AddHs(mol), sanitize=False), i.e. heavy atoms plus the hydrogens RDKit RemoveHs keeps. > 128 is rejected
(upstream raises), > 56 is a warning (training maximum; affinity may be inaccurate).
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

MAX_ATOMS = 128   # upstream hard limit for the affinity binder
WARN_ATOMS = 56   # upstream training maximum
ID_COLS = ("id", "ligand_id", "compound_id", "name", "title", "mol_id", "cid")
SMILES_COLS = ("smiles", "smi", "canonical_smiles", "isomeric_smiles")
AA = set("ACDEFGHIKLMNPQRSTVWYXBZUO")


class InputError(ValueError):
    """Bad screen input (whole run): unreadable ligand file, bad target, ..."""


@dataclass
class Ligand:
    ligand_id: str                 # as given by the user
    smiles: str                    # as given (or written from the SDF record)
    index: int                     # position in the input file (0-based)
    record_id: str = ""            # filesystem/record-safe id (seeds use it)
    smiles_used: str | None = None  # what goes into the YAML (largest fragment); None if invalid
    n_atoms: int | None = None     # upstream size definition
    n_heavy: int | None = None
    mw: float | None = None
    status: str = "pending"        # pending | invalid
    error: str | None = None
    warnings: list[str] = field(default_factory=list)


# ================================================================================================ ligand files


def sanitize_id(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s).strip()).strip("_.")
    return s


def _read_csv(path: Path, id_col: str | None, smiles_col: str | None) -> list[tuple[str, str]]:
    delim = "\t" if path.suffix.lower() in (".tsv", ".tab") else ","
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f, delimiter=delim))
    rows = [r for r in rows if r and any(c.strip() for c in r)]
    if not rows:
        return []
    header = [c.strip() for c in rows[0]]
    low = [c.lower() for c in header]

    def find(explicit, names):
        if explicit:
            if explicit.lower() not in low:
                raise InputError(f"{path}: column {explicit!r} not in header {header}")
            return low.index(explicit.lower())
        return next((low.index(n) for n in names if n in low), None)

    si = find(smiles_col, SMILES_COLS)
    if si is None:
        raise InputError(f"{path}: no SMILES column in header {header} (expected one of {SMILES_COLS}, or pass --smiles-col)")
    ii = find(id_col, ID_COLS)
    out = []
    for k, r in enumerate(rows[1:]):
        smi = r[si].strip() if si < len(r) else ""
        lid = r[ii].strip() if (ii is not None and ii < len(r)) else ""
        out.append((lid or f"lig_{k + 1:05d}", smi))
    return out


def _read_smi(path: Path) -> list[tuple[str, str]]:
    out = []
    n = 0
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if n == 0 and parts[0].lower() in SMILES_COLS:
            continue  # header line
        n += 1
        if len(parts) > 1 and " " not in parts[1].strip() and not _parses(parts[0]) and _parses(parts[1].strip()):
            parts = [parts[1].strip(), parts[0]]   # "name SMILES" (a spreadsheet's column order): the SMILES is the second word
        lid = parts[1].strip() if len(parts) > 1 else ""
        out.append((lid or f"lig_{n:05d}", parts[0]))
    return out


def _parses(smiles: str) -> bool:
    """RDKit reads it as a molecule (no sanitising: this only tells a SMILES from a name)."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    return Chem.MolFromSmiles(smiles, sanitize=False) is not None


def _read_sdf(path: Path, id_col: str | None) -> list[tuple[str, str]]:
    from rdkit import Chem

    out = []
    suppl = Chem.SDMolSupplier(str(path), sanitize=True, removeHs=True)
    for k, mol in enumerate(suppl):
        if mol is None:
            out.append((f"sdf_{k + 1:05d}", ""))  # unparseable record -> invalid row (empty SMILES)
            continue
        lid = ""
        if id_col and mol.HasProp(id_col):
            lid = mol.GetProp(id_col).strip()
        elif mol.HasProp("_Name"):
            lid = mol.GetProp("_Name").strip()
        if mol.GetNumConformers() and mol.GetConformer().Is3D():
            Chem.AssignStereochemistryFrom3D(mol)
        out.append((lid or f"sdf_{k + 1:05d}", Chem.MolToSmiles(mol, isomericSmiles=True)))
    return out


def read_ligands(path: str | Path, id_col: str | None = None, smiles_col: str | None = None) -> list[Ligand]:
    """Ligands in file order; ids made record-safe and unique (duplicates become invalid rows)."""
    path = Path(path).expanduser()
    if not path.exists():
        raise InputError(f"ligand file not found: {path}")
    suf = path.suffix.lower()
    if suf in (".csv", ".tsv", ".tab"):
        pairs = _read_csv(path, id_col, smiles_col)
    elif suf in (".smi", ".smiles", ".txt"):
        pairs = _read_smi(path)
    elif suf in (".sdf", ".sd", ".mol"):
        pairs = _read_sdf(path, id_col)
    else:
        raise InputError(f"unsupported ligand file {path.name}: use .csv, .tsv, .smi or .sdf")
    ligs: list[Ligand] = []
    seen: dict[str, str] = {}
    for k, (lid, smi) in enumerate(pairs):
        lig = Ligand(ligand_id=lid, smiles=smi, index=k, record_id=sanitize_id(lid) or f"lig_{k + 1:05d}")
        key = lig.record_id.lower()  # case-insensitive filesystems
        if key in seen:
            lig.status, lig.error = "invalid", f"duplicate ligand id (same record id as {seen[key]!r})"
            lig.record_id = f"{lig.record_id}__dup{k + 1}"
        else:
            seen[key] = lid
        ligs.append(lig)
    return ligs


def _prepare(lig: Ligand):
    """Parse and salt-strip (safe RDKit calls only). Returns (mol, smiles to use) or None when lig is now invalid."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    if not lig.smiles:
        lig.status, lig.error = "invalid", "empty SMILES (or unreadable SDF record)"
        return None
    try:
        mol = Chem.MolFromSmiles(lig.smiles)
        if mol is None:
            lig.status, lig.error = "invalid", "RDKit could not parse the SMILES"
            return None
        frags = Chem.GetMolFrags(mol, asMols=True)
        if len(frags) > 1:
            big = max(frags, key=lambda m: (m.GetNumHeavyAtoms(), Chem.MolToSmiles(m)))
            lig.warnings.append(f"{len(frags)} fragments: kept the largest ({big.GetNumHeavyAtoms()} heavy atoms); one binder per complex")
            mol = big
            smi = Chem.MolToSmiles(mol, isomericSmiles=True)
        else:
            smi = lig.smiles.strip()  # passed through unchanged: upstream parses it exactly as `boltz predict` would
    except Exception as ex:  # noqa: BLE001
        lig.status, lig.error = "invalid", f"RDKit error: {ex}"
        return None
    return mol, smi


def _finish(lig: Ligand, mol, smi: str, error: str | None, max_atoms: int, warn_atoms: int) -> Ligand:
    """Size limits, atom names, stereo warnings. Only after smiles_guard cleared `smi` (error None): AddHs is not safe before."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, Descriptors

    RDLogger.DisableLog("rdApp.*")
    if error:
        lig.status, lig.error = "invalid", error
        return lig
    try:
        mh = AllChem.AddHs(Chem.MolFromSmiles(smi))
        no_h = AllChem.RemoveHs(mh, sanitize=False)
        lig.n_atoms = int(no_h.GetNumAtoms())
        lig.n_heavy = int(mol.GetNumHeavyAtoms())
        # true molecular weight (with implicit H). NB: upstream's affinity_mw (used by the MW correction, reported as
        # affinity_mw) is MolWt of RemoveHs(..., sanitize=False), which leaves out the hydrogens (OCC: 40.02 vs 46.07).
        lig.mw = float(Descriptors.MolWt(Chem.MolFromSmiles(smi)))
        lig.smiles_used = smi
        if lig.n_atoms > max_atoms:
            lig.status, lig.error = "invalid", f"ligand too large: {lig.n_atoms} atoms > {max_atoms} (upstream affinity limit)"
            return lig
        if lig.n_atoms > warn_atoms:
            lig.warnings.append(f"{lig.n_atoms} atoms > {warn_atoms} (upstream training maximum): affinity may be inaccurate")
        # upstream atom names are <element><canonical rank> and must fit in 4 characters; a ligand where they do not gets
        # per-element names (boltz_mlx/upfix/names.py), so only more than 999 (99) atoms of one element are refused
        from boltz_mlx.upfix import names

        problem = names.atom_name_problem(mh)
        if problem:
            lig.status, lig.error = "invalid", problem
            return lig
        unassigned = [c for c in Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False) if c[1] == "?"]
        if unassigned:
            lig.warnings.append(f"{len(unassigned)} unassigned stereocentre(s): RDKit picks one stereoisomer for the conformer")
    except Exception as ex:  # noqa: BLE001
        lig.status, lig.error = "invalid", f"RDKit error: {ex}"
    return lig


def validate_ligand(lig: Ligand, max_atoms: int = MAX_ATOMS, warn_atoms: int = WARN_ATOMS, affinity: bool = True) -> Ligand:
    """RDKit parse, salt stripping (largest fragment), upstream size limits, stereo warnings. Sets status invalid + error.
    Before anything that can abort RDKit (AddHs), boltz_mlx.smiles_guard checks the SMILES (a child process only for
    unusual molecules). One ligand at a time; validate_ligands shares one child between a whole list."""
    from boltz_mlx import smiles_guard

    if lig.status == "invalid":
        return lig
    prep = _prepare(lig)
    if prep is None:
        return lig
    mol, smi = prep
    return _finish(lig, mol, smi, smiles_guard.check_smiles(smi, affinity), max_atoms, warn_atoms)


def validate_ligands(ligs: list[Ligand], max_atoms: int = MAX_ATOMS, warn_atoms: int = WARN_ATOMS, affinity: bool = True) -> list[Ligand]:
    """validate_ligand for a list: the unusual molecules of the whole list are checked in one child process."""
    from boltz_mlx import smiles_guard

    prepared = [(lig, None if lig.status == "invalid" else _prepare(lig)) for lig in ligs]
    todo = [p[1][1] for p in prepared if p[1] is not None]
    verdicts = smiles_guard.check_many(todo, affinity) if todo else {}
    for lig, prep in prepared:
        if prep is not None:
            _finish(lig, prep[0], prep[1], verdicts[prep[1]], max_atoms, warn_atoms)
    return ligs


# ======================================================================================================== target


@dataclass
class Target:
    doc: dict                      # Boltz YAML document without the screened ligand / affinity property
    ligand_chain: str              # chain id given to the screened ligand
    n_residues: int                # polymer residues (tokens); ligand atoms add one token each
    msa_source: str                # "empty" | "a3m:<name>" | "csv:<name>" | "mixed:..."
    msa_paths: list[str]           # absolute MSA paths used (sorted, as upstream orders them)
    source: str                    # where the target came from
    warnings: list[str] = field(default_factory=list)
    msa_sha: list[str] = field(default_factory=list)  # content hashes of msa_paths, taken when the target is loaded

    def key(self) -> str:
        """Hash of the target definition (+ MSA file contents): part of the resume config key."""
        h = hashlib.sha256(json.dumps(self.doc, sort_keys=True, default=str).encode())
        for sha in self.msa_sha:
            h.update(sha.encode())
        return h.hexdigest()[:16]

    def ligand_yaml(self, smiles: str) -> dict:
        doc = json.loads(json.dumps(self.doc))
        doc.setdefault("sequences", []).append({"ligand": {"id": self.ligand_chain, "smiles": smiles}})
        doc["properties"] = [{"affinity": {"binder": self.ligand_chain}}]
        return doc


def _file_sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _chain_ids(entry: dict) -> list[str]:
    ids = entry.get("id")
    return [ids] if isinstance(ids, str) else list(ids or [])


def _parse_fasta_text(text: str) -> list[tuple[str, str]]:
    out, name, seq = [], None, []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith(">"):
            if name is not None:
                out.append((name, "".join(seq)))
            name, seq = line[1:].strip(), []
        else:
            seq.append(line.replace(" ", ""))
    if name is not None:
        out.append((name, "".join(seq)))
    return out


def _letters():
    import string

    yield from string.ascii_uppercase
    for a in string.ascii_uppercase:
        for b in string.ascii_uppercase:
            yield a + b


def load_target(target: str, msa: str | None = None, pocket: list[str] | None = None, ligand_chain: str | None = None) -> Target:
    """--target: a Boltz YAML, a FASTA file (protein chains only), or a raw amino-acid sequence.

    msa: None (keep the YAML's per-chain MSAs; chains without one run single-sequence with a warning),
         "empty", or a path to an .a3m/.csv applied to every protein chain. The MSA server is never used (offline).
    pocket: ["A:45", "A:67", ...] residue contacts for a pocket constraint on the screened ligand.
    """
    import yaml

    warnings: list[str] = []
    p = Path(target).expanduser()
    base = Path.cwd()
    if p.suffix.lower() in (".yaml", ".yml"):
        if not p.exists():
            raise InputError(f"target YAML not found: {p}")
        doc = yaml.safe_load(p.read_text()) or {}
        base = p.resolve().parent
        source = str(p)
    elif p.suffix.lower() in (".fa", ".fasta", ".fas", ".faa"):
        if not p.exists():
            raise InputError(f"target FASTA not found: {p}")
        recs = _parse_fasta_text(p.read_text())
        if not recs:
            raise InputError(f"no sequences in {p}")
        letters = _letters()
        doc = {"version": 1, "sequences": [{"protein": {"id": next(letters), "sequence": s.upper()}} for _, s in recs]}
        source = str(p)
    else:
        seq = re.sub(r"\s+", "", target).upper()
        if not seq or not set(seq) <= AA:
            raise InputError(f"--target {target!r}: not a .yaml/.fasta path or an amino-acid sequence")
        doc = {"version": 1, "sequences": [{"protein": {"id": "A", "sequence": seq}}]}
        source = "sequence"
    seqs = doc.get("sequences") or []
    if not any("protein" in s for s in seqs):
        raise InputError("target has no protein chain")
    # a complex YAML (receptor + one ligand + affinity property) works as a target: its binder ligand is the slot the
    # screened ligands take (same chain id, same chain order as the original file)
    binder = None
    for prop in doc.pop("properties", None) or []:
        if isinstance(prop, dict) and "affinity" in prop:
            binder = (prop["affinity"] or {}).get("binder")
    if binder is not None:
        keep = [s for s in seqs if not ("ligand" in s and binder in _chain_ids(s["ligand"]))]
        if len(keep) != len(seqs):
            warnings.append(f"target YAML's affinity binder ligand {binder!r} was removed: each screened ligand takes its place")
            seqs[:] = keep
            if ligand_chain is None:
                ligand_chain = binder
    used = {c for s in seqs for v in s.values() for c in _chain_ids(v)}
    for s in seqs:
        if "ligand" in s:
            warnings.append(f"target contains ligand chain(s) {_chain_ids(s['ligand'])}: kept as co-factor(s); affinity is for the screened ligand only")

    # MSA handling (no server: missing MSAs become single-sequence with a warning)
    if msa not in (None, "empty") and len({s["protein"].get("sequence") for s in seqs if "protein" in s}) > 1:
        raise InputError("--msa <file> applies one MSA to every protein chain, but the target has several different protein "
                         "sequences: put per-chain `msa:` paths in the target YAML instead")
    msa_paths: set[str] = set()
    for s in seqs:
        if "protein" not in s:
            continue
        pr = s["protein"]
        if msa is not None:
            val = msa
        else:
            val = pr.get("msa")
            if val in (None, "", 0):
                warnings.append(f"protein chain(s) {_chain_ids(pr)} have no MSA: running single-sequence (msa: empty), which costs "
                                f"accuracy; pass --msa <a3m> for a precomputed MSA (the MSA server is never used)")
                val = "empty"
        if val != "empty":
            mp = Path(str(val)).expanduser()
            if not mp.is_absolute():
                mp = (Path.cwd() if msa is not None else base) / mp
            mp = mp.resolve()
            if not mp.exists():
                raise InputError(f"MSA file not found: {mp}")
            if mp.suffix not in (".a3m", ".csv"):
                raise InputError(f"MSA {mp.name}: only .a3m or .csv are supported")
            val = str(mp)
            msa_paths.add(val)
        pr["msa"] = val
    msas = sorted(msa_paths)
    kinds = {Path(m).suffix.lstrip(".") + ":" + Path(m).name for m in msas}
    has_empty = any("protein" in s and s["protein"].get("msa") == "empty" for s in seqs)
    if not msas:
        msa_source = "empty"
    elif has_empty:
        msa_source = "mixed:" + ",".join(sorted(kinds)) + "+empty"
    else:
        msa_source = ",".join(sorted(kinds))
    # templates: resolve relative paths
    for t in doc.get("templates") or []:
        for k in ("cif", "pdb"):
            if k in t and not Path(t[k]).is_absolute():
                t[k] = str((base / t[k]).resolve())

    # ligand chain id: an explicit choice, else the binder named by a pocket constraint in the YAML, else L, else a free id
    lig_chain = ligand_chain
    if lig_chain is None:
        for c in doc.get("constraints") or []:
            b = (c.get("pocket") or {}).get("binder")
            if b and b not in used:
                lig_chain = b
                break
    if lig_chain is None:
        lig_chain = "L" if "L" not in used else next(x for x in _letters() if x not in used)
    if lig_chain in used:
        raise InputError(f"ligand chain id {lig_chain!r} is already used by the target")
    if pocket:
        contacts = []
        for tok in pocket:
            for part in str(tok).split(","):
                part = part.strip()
                if not part:
                    continue
                m = re.fullmatch(r"([A-Za-z0-9]+):(\d+)", part)
                if not m or m.group(1) not in used:
                    raise InputError(f"--pocket {part!r}: expected CHAIN:RESIDUE_INDEX with a target chain id")
                contacts.append([m.group(1), int(m.group(2))])
        doc.setdefault("constraints", []).append({"pocket": {"binder": lig_chain, "contacts": contacts}})
    n_res = 0
    for s in seqs:
        for kind in ("protein", "dna", "rna"):
            if kind in s:
                n_res += len(re.sub(r"\(.*?\)", "X", s[kind].get("sequence", ""))) * max(1, len(_chain_ids(s[kind])))
        if "ligand" in s:
            n_res += 30 * max(1, len(_chain_ids(s["ligand"])))  # rough: co-factor atoms (estimate only)
    return Target(doc=doc, ligand_chain=lig_chain, n_residues=n_res, msa_source=msa_source, msa_paths=msas, source=source,
                  warnings=warnings, msa_sha=[_file_sha(Path(m)) for m in msas])
