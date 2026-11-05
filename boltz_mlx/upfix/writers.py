"""PDB / mmCIF writer fixes for stock boltz 2.2.1 (boltz/data/write/{pdb,mmcif}.py, called by BoltzWriter). Upstream is not edited.

BoltzWriter looks up `to_pdb` and `to_mmcif` as globals of boltz.data.write.writer; install_writers replaces those two names
by the functions below, which call upstream's own writers and then correct their text:

  elements   (upstream #458; fork 9cca9d6) Upstream infers the element of an atom from its name, which is wrong for SMILES
             ligands of some metals (a [Na+] is written N, [Ca+2] C, [Cu+2] C, Pd P, Cd C, ...: 58 elements, see
             tests/test_upfix_writers.py). The model's own input is right (ref_element, from RDKit / the CCD), so the patched
             BoltzWriter.write_on_batch_end hands the element of every atom to the two writers; only a column whose element
             differs is rewritten, so every correct line stays byte-identical. The fork stores an element field in the atom
             table instead, which would break stock 2.2.1's cached structures.
  chain ids  (fork 321d13f, PR #23) PDB has one chain-id column; a name like "L1" shifts every later column. Chains whose name
             is not a unique single character get the first unused character (A-Z a-z 0-9, in chain order) for the PDB text
             (a notice lists the renaming; mmCIF keeps the names). More than 62 chains: ValueError, as in the fork.
  bond order (fork 772a2a6, PR #22, with the final convention of 321d13f's parent: f91495c) CONECT records of double and triple
             bonds list the partner twice / three times (the RDKit / Open Babel / PyMOL convention); single, aromatic and other
             bonds are written once, as upstream does. Without this the ligand bond orders of structure.bonds are lost.
  cyclic     (fork 8450e66, PR #27) `cyclic: true` is positional information only: no bond closes the ring. The head-to-tail bond
             (C of the last residue to N of the first for a protein; O3' to P for DNA / RNA) is written as a CONECT record in
             PDB and as a _struct_conn row (covale) in mmCIF, for chains whose cyclic_period is the whole chain. Stock 2.2.1
             writes no _struct_conn category at all, so this is the only row.

Fork code is MIT (Wohlwend, Corso, Passaro); this module follows its behaviour and test expectations, written against the
stock data types (see NOTICE). A notice (stderr) is printed only for a chain renaming.
"""

from __future__ import annotations

import contextvars
import re
import threading
from dataclasses import replace

import numpy as np

_CHAIN_ID_POOL = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
_ELEMENTS: contextvars.ContextVar = contextvars.ContextVar("boltz_mlx_upfix_elements", default=None)
_lock = threading.Lock()


# ------------------------------------------------------------------------------------------------------- elements
def elements_from_batch(prediction: dict, batch: dict) -> list[str] | None:
    """Upper-case element symbols of the unpadded atoms of the (single) record of a BoltzWriter batch, from the model's
    ref_element input; None when the batch does not hold exactly that."""
    try:
        records = batch["record"]
        if len(records) != 1:
            return None
        mask = np.asarray(prediction["masks"][0].cpu() if hasattr(prediction["masks"][0], "cpu") else prediction["masks"][0]).astype(bool)
        oh = batch["ref_element"][0]
        oh = np.asarray(oh.cpu() if hasattr(oh, "cpu") else oh)
        z = oh.argmax(-1)[mask]
    except Exception:  # noqa: BLE001  (a batch without the keys: the writers keep their own elements)
        return None
    from rdkit import Chem

    pt = Chem.GetPeriodicTable()
    return [pt.GetElementSymbol(int(v)).upper() for v in z]


def fix_pdb_elements(text: str, elements: list[str]) -> str:
    """Rewrite the element columns (77-78) of the ATOM / HETATM lines where they differ from `elements` (in atom order)."""
    out, k = [], 0
    for line in text.split("\n"):
        if line.startswith(("ATOM  ", "HETATM")):
            if k < len(elements) and line[76:78] != f"{elements[k]:>2}":
                line = line[:76] + f"{elements[k]:>2}" + line[78:]
            k += 1
        out.append(line)
    if k != len(elements):
        return text
    return "\n".join(out)


_TOKEN = re.compile(r"""'(?:[^']|'(?=\S))*'(?=\s|$)|"(?:[^"]|"(?=\S))*"(?=\s|$)|\S+""")


def fix_cif_elements(text: str, elements: list[str]) -> str:
    """Rewrite the _atom_site.type_symbol of every row where it differs from `elements` (in atom order)."""
    lines = text.split("\n")
    tags = [i for i, ln in enumerate(lines) if ln.startswith("_atom_site.")]
    if not tags:
        return text
    names = [lines[i].strip() for i in range(tags[0], tags[-1] + 1)]
    if "_atom_site.type_symbol" not in names:
        return text
    col = names.index("_atom_site.type_symbol")
    rows = [i for i in range(tags[-1] + 1, len(lines)) if lines[i].startswith(("ATOM ", "HETATM "))]
    if len(rows) != len(elements):
        return text
    for i, el in zip(rows, elements):
        toks = list(_TOKEN.finditer(lines[i]))
        if len(toks) <= col:
            return text
        t = toks[col]
        if t.group(0) != el:
            lines[i] = lines[i][: t.start()] + el + lines[i][t.end():]
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------- chain ids
def pdb_chain_ids(names: list[str]) -> list[str]:
    """One unique character per chain: single-character unique names are kept, every other chain gets the first unused
    character of A-Z a-z 0-9 in chain order (fork 321d13f). ValueError when 62 are not enough."""
    single = [n for n in names if len(n) == 1]
    keep = {n for n in single if single.count(n) == 1}
    pool = (c for c in _CHAIN_ID_POOL if c not in keep)
    ids = []
    for n in names:
        if n in keep:
            ids.append(n)
            continue
        try:
            ids.append(next(pool))
        except StopIteration:
            raise ValueError(f"Cannot write {len(names)} chains to PDB, which allows at most {len(_CHAIN_ID_POOL)} one-character "
                             "chain IDs. Use --output_format mmcif instead.") from None
    return ids


# ------------------------------------------------------------------------------------------ bonds / ring closure
def _atom_name(atom) -> str:
    return str(atom["name"])


def closure_bonds(structure) -> list[dict]:
    """Head-to-tail bonds of the cyclic polymer chains: dicts with chain, res_1/res_2 (residue indices) and atom_1/atom_2
    (atom indices). Only chains whose cyclic_period is the whole chain (fork 8450e66)."""
    from boltz.data import const

    mol = {const.chain_type_ids["PROTEIN"]: ("C", "N"), const.chain_type_ids["DNA"]: ("O3'", "P"),
           const.chain_type_ids["RNA"]: ("O3'", "P")}
    out = []
    for chain in structure.chains:
        names = mol.get(int(chain["mol_type"]))
        n_res = int(chain["res_num"])
        if names is None or n_res < 2 or int(chain["cyclic_period"]) != n_res:
            continue
        first = int(chain["res_idx"])
        last = first + n_res - 1
        ends = []
        for res_idx, name in ((last, names[0]), (first, names[1])):
            res = structure.residues[res_idx]
            start = int(res["atom_idx"])
            hit = [start + i for i, a in enumerate(structure.atoms[start: start + int(res["atom_num"])]) if _atom_name(a) == name]
            ends.append(hit[0] if hit else None)
        if None in ends:
            continue
        out.append({"chain": int(chain["asym_id"]), "res_1": last, "res_2": first, "atom_1": ends[0], "atom_2": ends[1]})
    return out


def _rewrite_conect(text: str, structure) -> str:
    """CONECT records with bond orders (partner repeated for double / triple) and the ring-closing bonds."""
    from boltz.data import const

    lines = text.split("\n")
    serials = [int(ln[6:11]) for ln in lines if ln.startswith(("ATOM  ", "HETATM"))]
    idx = [i for i, ln in enumerate(lines) if ln.startswith("CONECT")]
    closures = closure_bonds(structure)
    order_of = {const.bond_type_ids["DOUBLE"]: 2, const.bond_type_ids["TRIPLE"]: 3}
    names = structure.bonds.dtype.names or ()
    has_order = "type" in names and any(int(b["type"]) in order_of for b in structure.bonds)
    if (not has_order and not closures) or len(serials) != len(structure.atoms):
        return text
    present = structure.atoms["is_present"]
    new, seen = [], set()
    for b in structure.bonds:
        a1, a2 = int(b["atom_1"]), int(b["atom_2"])
        if not present[a1] or not present[a2]:
            continue
        seen.add(frozenset((a1, a2)))
        new.append(f"CONECT{serials[a1]:>5}" + f"{serials[a2]:>5}" * order_of.get(int(b["type"]), 1))
    for c in closures:
        if frozenset((c["atom_1"], c["atom_2"])) not in seen and present[c["atom_1"]] and present[c["atom_2"]]:
            new.append(f"CONECT{serials[c['atom_1']]:>5}{serials[c['atom_2']]:>5}")
    if len(new) < len(idx) or (idx and idx != list(range(idx[0], idx[0] + len(idx)))):
        return text   # not the layout upstream writes (all bonds, contiguous): leave it alone
    new = [ln.ljust(80) for ln in new]
    at = idx[0] if idx else lines.index("END")
    return "\n".join(lines[:at] + new + lines[at + len(idx):])


def _cif_quote(s: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_.\-+*]+", s):
        return s
    return f'"{s}"' if "'" in s else f"'{s}'"


_STRUCT_CONN = ["id", "conn_type_id", "ptnr1_label_asym_id", "ptnr1_label_comp_id", "ptnr1_label_seq_id", "ptnr1_label_atom_id",
                "ptnr1_auth_asym_id", "ptnr1_auth_comp_id", "ptnr1_auth_seq_id", "ptnr1_auth_atom_id",
                "ptnr2_label_asym_id", "ptnr2_label_comp_id", "ptnr2_label_seq_id", "ptnr2_label_atom_id",
                "ptnr2_auth_asym_id", "ptnr2_auth_comp_id", "ptnr2_auth_seq_id", "ptnr2_auth_atom_id", "pdbx_dist_value"]


def add_struct_conn(text: str, structure) -> str:
    """Append a _struct_conn loop with the ring-closing bonds (covale) of cyclic chains; unchanged without any."""
    closures = closure_bonds(structure)
    if not closures or "_struct_conn." in text:
        return text
    by_asym = {int(c["asym_id"]): str(c["name"]) for c in structure.chains}
    rows = []
    for k, c in enumerate(closures, 1):
        chain, r1, r2 = by_asym[c["chain"]], structure.residues[c["res_1"]], structure.residues[c["res_2"]]
        a1, a2 = structure.atoms[c["atom_1"]], structure.atoms[c["atom_2"]]
        dist = float(np.linalg.norm(np.asarray(a1["coords"], dtype=float) - np.asarray(a2["coords"], dtype=float)))
        vals = [f"covale{k}", "covale"]
        for r, a in ((r1, a1), (r2, a2)):
            comp, seq, name = str(r["name"]), str(int(r["res_idx"]) + 1), _atom_name(a)
            vals += [chain, comp, seq, name, chain, comp, seq, name]
        rows.append(" ".join(_cif_quote(v) for v in vals + [f"{dist:.3f}"]))
    block = ["#", "loop_"] + [f"_struct_conn.{t}" for t in _STRUCT_CONN] + rows + ["#", ""]
    return text.rstrip("\n") + "\n" + "\n".join(block)


# --------------------------------------------------------------------------------------------------- the writers
def to_pdb(structure, plddts=None, boltz2: bool = False, elements: list[str] | None = None) -> str:
    """boltz.data.write.pdb.to_pdb plus the fixes of the module docstring (boltz2 structures; v1 is passed through)."""
    from boltz.data.write.pdb import to_pdb as upstream

    if not boltz2:
        return upstream(structure, plddts=plddts, boltz2=boltz2)
    elements = _ELEMENTS.get() if elements is None else elements
    names = [str(c["name"]) for c in structure.chains]
    ids = pdb_chain_ids(names)
    st = structure
    if ids != names:
        from boltz_mlx.upfix import say

        say("PDB chain IDs are one character, so these chains are renamed in the PDB output: "
            + ", ".join(f"{n} -> {c}" for n, c in zip(names, ids) if n != c) + ". mmCIF output keeps the original names.")
        chains = structure.chains.copy()
        chains["name"] = ids
        st = replace(structure, chains=chains)
    text = upstream(st, plddts=plddts, boltz2=boltz2)
    if elements is not None and len(elements) == len(structure.atoms):
        text = fix_pdb_elements(text, elements)
    return _rewrite_conect(text, structure)


def to_mmcif(structure, plddts=None, boltz2: bool = False, elements: list[str] | None = None) -> str:
    """boltz.data.write.mmcif.to_mmcif plus the element and ring-closure fixes (boltz2 structures; v1 is passed through)."""
    from boltz.data.write.mmcif import to_mmcif as upstream

    text = upstream(structure, plddts=plddts, boltz2=boltz2)
    if not boltz2:
        return text
    elements = _ELEMENTS.get() if elements is None else elements
    if elements is not None and len(elements) == len(structure.atoms):
        text = fix_cif_elements(text, elements)
    return add_struct_conn(text, structure)


def install_writers() -> None:
    """Idempotent: boltz.data.write.writer.{to_pdb, to_mmcif} -> the functions above; BoltzWriter.write_on_batch_end hands them
    the batch's elements."""
    import boltz.data.write.writer as W

    with _lock:
        if getattr(W.BoltzWriter.write_on_batch_end, "_boltz_mlx_elements", False):
            return
        upstream = W.BoltzWriter.write_on_batch_end

        def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
            tok = _ELEMENTS.set(elements_from_batch(prediction, batch) if self.boltz2 and not prediction.get("exception") else None)
            try:
                return upstream(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx)
            finally:
                _ELEMENTS.reset(tok)

        write_on_batch_end._boltz_mlx_elements = True
        write_on_batch_end.__wrapped__ = upstream
        write_on_batch_end.__doc__ = upstream.__doc__
        W.BoltzWriter.write_on_batch_end = write_on_batch_end
        W.to_pdb = to_pdb
        W.to_mmcif = to_mmcif
