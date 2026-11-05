"""Constraints, residue modifications and templates for the Predict form, as the Boltz-2 YAML schema takes them.

What upstream boltz/data/parse/schema.py (parse_boltz_schema) accepts, and what is checked here first so a job never fails on it:
  constraints:
    - pocket:  {binder: L, contacts: [[A, 45], [A, 67], [L2, C1]], max_distance: 6}   polymer contact = [chain, residue (1-based)],
                                                                                   ligand contact = [chain, atom name]
    - contact: {token1: [A, 45], token2: [B, 12], max_distance: 6}                 same token forms
    - bond:    {atom1: [A, 45, SG], atom2: [L, 1, C1]}                            [chain, residue, atom name]; a ligand is residue 1
  protein / dna / rna entry: modifications: [{position: 3, ccd: SEP}]           1-based, replaces that residue with the CCD code
  templates: [{cif: path, chain_id: A}]                                          (pdb: path also works; template_id names the
                                                                                   file's chain; force + threshold need potentials)
max_distance: Boltz-2 was trained with contact cutoffs of 4-20 A (conditioning_cutoff_min / _max), so the form takes 4-20; 6 is the
default and is left out of the YAML. `force` (steering potentials) is not offered: boltz-mlx does not port the potentials
(boltz_mlx/cli.py), so constraints and templates inform the trunk but are not enforced during sampling.

Atom names follow upstream: standard residues use boltz.data.const.ref_atoms, CCD codes the mols/<CODE>.pkl molecule, SMILES ligands
<ELEMENT><canonical rank + 1> on the molecule with hydrogens (standardized first for the affinity binder). A template upload is
reduced to one protein chain of model 1 (ligands and waters removed) and written as mmCIF, the same as the Screen target library."""

from __future__ import annotations

import re
from pathlib import Path

from boltz_mlx.gui.common import ApiError, safe_filename

MAX_CONSTRAINTS = 40
MAX_CONTACTS = 200
MAX_MODS = 50
MAX_TEMPLATES = 4
TEMPLATE_LIMIT = 32 * 1024 * 1024
DIST = (4.0, 20.0)
DEFAULT_DIST = 6.0
CCD_RE = re.compile(r"^[A-Za-z0-9]{1,5}$")
ATOM_RE = re.compile(r"^[A-Za-z0-9']{1,4}$")
TEMPLATE_EXT = (".pdb", ".ent", ".cif", ".mmcif")


# ------------------------------------------------------------------------------------------------------ modifications
# Common modified residues and the standard residue each one replaces: a code put on another residue is almost always a slip
# (SEP on a Glu), so the form warns; upstream would model it as asked.
MOD_PARENT = {"SEP": ("S", "phospho-Ser"), "TPO": ("T", "phospho-Thr"), "PTR": ("Y", "phospho-Tyr"), "MSE": ("M", "seleno-Met"),
              "HYP": ("P", "hydroxy-Pro"), "ALY": ("K", "acetyl-Lys"), "MLY": ("K", "dimethyl-Lys"), "M3L": ("K", "trimethyl-Lys"),
              "MLZ": ("K", "methyl-Lys"), "CSO": ("C", "S-hydroxy-Cys"), "CME": ("C", "S,S-(2-hydroxyethyl)thiocysteine"),
              "NEP": ("H", "phospho-His"), "PCA": ("Q", "pyroglutamate"), "CGU": ("E", "gamma-carboxy-Glu")}


def modification_warnings(chain: dict, where: str) -> list[str]:
    """One line per modification whose code replaces a different residue than the one at its position (protein chains)."""
    seq = chain.get("sequence") or ""
    out = []
    for m in chain.get("modifications") or []:
        parent = MOD_PARENT.get(m["ccd"])
        aa = seq[m["position"] - 1] if 0 < m["position"] <= len(seq) else ""
        if parent and aa and aa != parent[0]:
            out.append(f"{where}: residue {m['position']} is {aa}, but {m['ccd']} ({parent[1]}) replaces {parent[0]}")
    return out


def check_modifications(raw, chain: dict, where: str, mols_dir: Path | None) -> list[dict]:
    """[{"position": int, "ccd": "SEP"}] of one polymer chain (chain["sequence"] already clean), positions unique and in range."""
    if raw in (None, "", []):
        return []
    if not isinstance(raw, list):
        raise ApiError(400, f"{where}: modifications must be a list")
    if len(raw) > MAX_MODS:
        raise ApiError(400, f"{where}: at most {MAX_MODS} modifications")
    n, out, seen = len(chain["sequence"]), [], set()
    for k, m in enumerate(raw, 1):
        m = m if isinstance(m, dict) else {}
        try:
            pos = int(m.get("position"))
        except (TypeError, ValueError):
            raise ApiError(400, f"{where}: modification {k}: the position must be a residue number") from None
        if not 1 <= pos <= n:
            raise ApiError(400, f"{where}: modification {k}: residue {pos} is outside the chain (1-{n})")
        if pos in seen:
            raise ApiError(400, f"{where}: residue {pos} is modified twice")
        seen.add(pos)
        ccd = str(m.get("ccd") or "").strip().upper()
        if not CCD_RE.match(ccd):
            raise ApiError(400, f"{where}: modification {k}: a CCD code is 1-5 letters or digits (SEP, TPO, MSE ...)")
        if mols_dir is not None and mols_dir.is_dir() and not (mols_dir / f"{ccd}.pkl").exists():
            raise ApiError(400, f"{where}: modification {k}: {ccd} is not in the molecule dictionary")
        out.append({"position": pos, "ccd": ccd})
    return sorted(out, key=lambda m: m["position"])


# ------------------------------------------------------------------------------------------------------ atom names
def _mol_atoms(ccd: str, mols_dir: Path | None) -> set[str] | None:
    """Heavy-atom names of a CCD molecule from mols/<CODE>.pkl; None when the dictionary is not there to check against."""
    p = (mols_dir / f"{ccd}.pkl") if mols_dir is not None else None
    if p is None or not p.is_file():
        return None
    try:
        import pickle

        with open(p, "rb") as f:
            mol = pickle.load(f)  # noqa: S301 (the user's own Boltz molecule dictionary)
        return {a.GetProp("name") for a in mol.GetAtoms() if a.GetAtomicNum() > 1 and a.HasProp("name")}
    except Exception:  # noqa: BLE001
        return None


def smiles_atom_names(smiles: str, binder: bool = False) -> list[str]:
    """Heavy-atom names upstream gives a SMILES ligand: <ELEMENT><canonical rank + 1> on the molecule with hydrogens."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    if binder:
        from boltz.data.parse.schema import standardize

        smiles = standardize(smiles) or smiles
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    mol = AllChem.AddHs(mol)
    ranks = AllChem.CanonicalRankAtoms(mol)
    return [a.GetSymbol().upper() + str(r + 1) for a, r in zip(mol.GetAtoms(), ranks) if a.GetAtomicNum() > 1]


def _residue_token(c: dict, pos: int) -> str | None:
    from boltz.data import const

    mods = {m["position"]: m["ccd"] for m in c.get("modifications") or []}
    if pos in mods:
        return mods[pos]
    letter = c["sequence"][pos - 1]
    table = {"protein": const.prot_letter_to_token, "dna": const.dna_letter_to_token, "rna": const.rna_letter_to_token}[c["type"]]
    return table.get(letter, "UNK" if c["type"] == "protein" else "N" if c["type"] == "rna" else "DN")


class Chains:
    """The checked chains by every id they answer to (a chain with copies has several), with lazily computed atom names."""

    def __init__(self, checked: dict, mols_dir: Path | None):
        self.by: dict[str, dict] = {}
        for c in checked["chains"]:
            for i in c.get("ids") or [c["id"]]:
                self.by[str(i)] = c
        self.binder = checked.get("binder")
        self.mols_dir = mols_dir
        self._lig: dict[str, set[str] | None] = {}

    def get(self, cid, where: str) -> dict:
        c = self.by.get(str(cid))
        if c is None:
            raise ApiError(400, f"{where}: there is no chain {cid} (chains: {', '.join(self.by)})")
        return c

    def ligand_atoms(self, c: dict) -> set[str] | None:
        if c["id"] not in self._lig:
            if c.get("ccd"):
                self._lig[c["id"]] = _mol_atoms(c["ccd"], self.mols_dir)
            else:
                try:
                    self._lig[c["id"]] = set(smiles_atom_names(c["smiles"], binder=c["id"] == self.binder)) or None
                except Exception:  # noqa: BLE001
                    self._lig[c["id"]] = None
        return self._lig[c["id"]]

    def residue(self, c: dict, res, cid: str, where: str) -> int:
        try:
            r = int(res)
        except (TypeError, ValueError):
            raise ApiError(400, f"{where}: {cid}:{res} is not a residue number") from None
        if isinstance(res, float) and res != r:
            raise ApiError(400, f"{where}: {cid}:{res} is not a residue number")
        n = len(c["sequence"])
        if not 1 <= r <= n:
            raise ApiError(400, f"{where}: residue {cid}:{r} is outside chain {cid} (1-{n})")
        return r

    def atom(self, name, known: set[str] | None, label: str, where: str) -> str:
        a = str(name or "").strip().upper()
        if not ATOM_RE.match(a):
            raise ApiError(400, f"{where}: {label}: an atom name is 1-4 letters or digits (CA, SG, NZ, C1 ...)")
        if known is not None and a not in known:
            show = ", ".join(sorted(known, key=lambda x: (len(x), x))[:24])
            raise ApiError(400, f"{where}: {label} has no atom {a} (atoms: {show}{' ...' if len(known) > 24 else ''})")
        return a

    def token(self, spec, where: str) -> list:
        """[chain, residue] (polymer) or [chain, atom name] (ligand), checked."""
        if not isinstance(spec, (list, tuple)) or len(spec) != 2:
            raise ApiError(400, f"{where}: expected chain:residue (A:45) or, on a ligand, chain:atom (L:C1)")
        cid, x = str(spec[0]).strip(), spec[1]
        c = self.get(cid, where)
        if c["type"] == "ligand":
            return [cid, self.atom(x, self.ligand_atoms(c), f"ligand {cid}", where)]
        return [cid, self.residue(c, x, cid, where)]

    def atom_spec(self, spec, where: str) -> list:
        """[chain, residue, atom name], checked (a ligand is residue 1)."""
        if not isinstance(spec, (list, tuple)) or len(spec) != 3:
            raise ApiError(400, f"{where}: expected chain:residue:atom (A:45:SG, L:1:C1)")
        cid = str(spec[0]).strip()
        c = self.get(cid, where)
        if c["type"] == "ligand":
            if str(spec[1]).strip() != "1":
                raise ApiError(400, f"{where}: a ligand is residue 1 ({cid}:1:atom)")
            return [cid, 1, self.atom(spec[2], self.ligand_atoms(c), f"ligand {cid}", where)]
        r = self.residue(c, spec[1], cid, where)
        tok = _residue_token(c, r)
        from boltz.data import const

        known = set(const.ref_atoms[tok]) if tok in const.ref_atoms else _mol_atoms(tok, self.mols_dir)
        return [cid, r, self.atom(spec[2], known, f"{cid}:{r} ({tok})", where)]


def _dist(v, where: str) -> float | None:
    if v in (None, ""):
        return None
    try:
        d = float(v)
    except (TypeError, ValueError):
        raise ApiError(400, f"{where}: the distance must be a number of Å") from None
    if not DIST[0] <= d <= DIST[1]:
        raise ApiError(400, f"{where}: the distance must be {DIST[0]:g}-{DIST[1]:g} Å (what Boltz-2 was trained on)")
    return None if d == DEFAULT_DIST else (int(d) if d == int(d) else d)


def check_constraints(raw, checked: dict, mols_dir: Path | None) -> list[dict]:
    """Boltz `constraints` entries (pocket / contact / bond), every chain, residue and atom checked against the chains.
    Returns the list as it goes into the YAML. Raises ApiError(400) naming the constraint ("pocket 1: ...")."""
    if raw in (None, []):
        return []
    if not isinstance(raw, list):
        raise ApiError(400, "constraints must be a list")
    if len(raw) > MAX_CONSTRAINTS:
        raise ApiError(400, f"at most {MAX_CONSTRAINTS} constraints")
    ch = Chains(checked, mols_dir)
    out: list[dict] = []
    n = {"pocket": 0, "contact": 0, "bond": 0}
    for c in raw:
        if not isinstance(c, dict) or len(c) != 1 or next(iter(c)) not in n or not isinstance(next(iter(c.values())), dict):
            raise ApiError(400, "each constraint is one of pocket, contact or bond")
        (kind, body), = c.items()
        n[kind] += 1
        where = f"{kind} {n[kind]}"
        force = bool(body.get("force")) if "force" in body else None
        if kind == "pocket":
            binder = str(body.get("binder") or "").strip()
            if not binder:
                raise ApiError(400, f"{where}: choose the binder (the chain that sits in the pocket)")
            ch.get(binder, where)
            contacts = body.get("contacts")
            if not isinstance(contacts, list) or not contacts:
                raise ApiError(400, f"{where}: list the pocket residues (A:45, A:67)")
            if len(contacts) > MAX_CONTACTS:
                raise ApiError(400, f"{where}: at most {MAX_CONTACTS} pocket residues")
            toks = []
            for t in contacts:
                if isinstance(t, (list, tuple)) and t and str(t[0]).strip() == binder:
                    raise ApiError(400, f"{where}: pocket residues belong to the other chains, not the binder {binder}")
                tok = ch.token(t, where)
                if tok not in toks:
                    toks.append(tok)
            v: dict = {"binder": binder, "contacts": toks}
            d = _dist(body.get("max_distance"), where)
        elif kind == "contact":
            t1, t2 = ch.token(body.get("token1"), where), ch.token(body.get("token2"), where)
            if t1[0] == t2[0]:
                raise ApiError(400, f"{where}: the two residues must be on different chains")
            v = {"token1": t1, "token2": t2}
            d = _dist(body.get("max_distance"), where)
        else:
            a1, a2 = ch.atom_spec(body.get("atom1"), where), ch.atom_spec(body.get("atom2"), where)
            if a1 == a2:
                raise ApiError(400, f"{where}: the two atoms are the same")
            v, d = {"atom1": a1, "atom2": a2}, None
        if d is not None:
            v["max_distance"] = d
        if force is not None:
            v["force"] = force
        out.append({kind: v})
    return out


# ------------------------------------------------------------------------------------------------------ templates
def single_chain_cif(st, chain: str) -> str:
    """Model 1 of a (ligand- and water-free) gemmi Structure with only `chain` kept, as mmCIF text."""
    while len(st) > 1:
        del st[len(st) - 1]
    model = st[0]
    for c in list(model):
        if c.name != chain:
            model.remove_chain(c.name)
    st.setup_entities()
    return st.make_mmcif_document().as_string()


def check_templates(raw, checked: dict) -> list[dict]:
    """[{"filename", "content", "chain_id": form chain, "template_id": chain of the file or None}] -> [{"name" (staged file),
    "from" (uploaded name), "content" (one-chain mmCIF), "chain_id": [ids], "template_chain", "length"}]."""
    if raw in (None, []):
        return []
    if not isinstance(raw, list):
        raise ApiError(400, "templates must be a list")
    if len(raw) > MAX_TEMPLATES:
        raise ApiError(400, f"at most {MAX_TEMPLATES} templates")
    from boltz_mlx.gui.targets import read_structure, structure_chains

    ch = Chains(checked, None)
    out = []
    for k, t in enumerate(raw, 1):
        where = f"template {k}"
        t = t if isinstance(t, dict) else {}
        cid = str(t.get("chain_id") or "").strip()
        if not cid:
            raise ApiError(400, f"{where}: choose the protein chain it is a template for")
        c = ch.get(cid, where)
        if c["type"] != "protein":
            raise ApiError(400, f"{where}: chain {cid} is a {c['type']} chain; templates are for protein chains")
        text = t.get("content")
        name, ext = safe_filename(t.get("filename"))
        if not isinstance(text, str) or not text.strip():
            raise ApiError(400, f"{where}: choose a PDB or mmCIF file")
        if len(text) > TEMPLATE_LIMIT:
            raise ApiError(413, f"{where}: the file is too large (limit {TEMPLATE_LIMIT // (1024 * 1024)} MB)")
        if ext not in TEMPLATE_EXT:
            raise ApiError(400, f"{where}: use a .pdb or .cif file")
        try:
            st = read_structure(text, name or f"template{ext}")
        except ApiError as ex:
            raise ApiError(400, f"{where}: {ex.message.replace('structure: ', '')}") from None
        found = structure_chains(st)
        if not found:
            raise ApiError(400, f"{where}: {name} has no protein chain")
        want = str(t.get("template_id") or "").strip()
        pick = next((x for x in found if x["chain"] == want), None) if want else found[0]
        if pick is None:
            raise ApiError(400, f"{where}: {name} has no protein chain {want} (it has {', '.join(x['chain'] for x in found)})")
        out.append({"name": f"template_{k}.cif", "from": name, "content": single_chain_cif(st, pick["chain"]),
                    "chain_id": c.get("ids") or [c["id"]], "template_chain": pick["chain"], "length": len(pick["sequence"])})
    return out


def check_extras(payload: dict, checked: dict, mols_dir: Path | None) -> dict:
    """The form's constraints and templates, checked against the checked chains: sets checked["constraints"] / ["templates"]."""
    checked["constraints"] = check_constraints(payload.get("constraints"), checked, mols_dir)
    checked["templates"] = check_templates(payload.get("templates"), checked)
    return checked
