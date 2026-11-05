"""Keep user SMILES from killing a process: validate before RDKit does anything dangerous with them.

Why. Two upstream failures of a valid-looking SMILES are reported by the boltz-community fork as C-level aborts (no Python
exception to catch): AllChem.AddHs on a molecule whose valence cache is unusable (upstream #649: SIGSEGV / SIGILL, fork
b555625) and standardize() of a salt with a terminal alkyne (fork 7407b86; a RuntimeError on the RDKit build used here, an
abort in the fork's). In the GUI server (which runs RDKit in its own process for ligand checks and drawings), in a screen
and in predict an abort would take the whole process or batch down. This module is the check that runs first.

What it does, per SMILES, cheaply and in this order:
  1. in-process, safe operations only: MolFromSmiles (sanitizing), UpdatePropertyCache(strict=True) -> a Python error with a
     clear message for anything RDKit refuses;
  2. classification: a molecule is "suspicious" when it has an element outside the organic subset (H B C N O F P S Cl Br I),
     an atom beyond its element's usual valence, a charge of 2 or more, a radical, a dative bond, or (for the affinity
     ligand) several fragments, which standardize() reduces to the largest. Ordinary drug-like input is never suspicious;
  3. only suspicious SMILES run the dangerous steps (AddHs, canonical ranks, stereo assignment, 2D depiction, and for the
     affinity ligand upstream's standardize sequence) in a throw-away child interpreter, one child for the whole batch.
     If the child dies (signal, abort) the SMILES it was working on is the culprit: it is marked invalid with a clear
     message and a new child takes the rest. A child that does not answer within PROBE_TIMEOUT_S gets the same verdict.

Cost (CPU, M-series Mac, tests/test_smiles_guard.py::test_probe_cost): ~0.4 ms per SMILES for the in-process checks (parse,
strict valence, classification; validate_ligands as a whole is ~1.0 ms per ligand); one child costs ~0.2 s (interpreter +
RDKit import) plus ~1 ms per SMILES; a screen of drug-like ligands starts none.
"""

from __future__ import annotations

import json
import queue
import subprocess
import sys
import tempfile
import threading
from collections import OrderedDict
from pathlib import Path

ORGANIC_Z = frozenset({1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53})
PROBE_TIMEOUT_S = 30.0          # per SMILES: a child that stays silent this long is killed and the SMILES refused
MAX_SMILES_CHARS = 4000
_ROOT = str(Path(__file__).resolve().parents[1])
_WORKER_CODE = "import sys; sys.path.insert(0, sys.argv[1]); from boltz_mlx.smiles_guard import worker_main; worker_main()"


def worker_cmd() -> list[str]:
    """The child interpreter command (isolated mode: no PYTHONPATH / cwd / user-site imports). Tests replace it."""
    return [sys.executable, "-I", "-c", _WORKER_CODE, _ROOT]


class SmilesError(ValueError):
    """A SMILES that must not be processed; str() is the message shown to the user."""


# ------------------------------------------------------------------------------------------------ in-process (safe)
def _first_line(ex: Exception) -> str:
    return (str(ex).strip().splitlines() or [type(ex).__name__])[0].strip()


def parse_checked(smiles: str):
    """The sanitized RDKit molecule, or SmilesError. Only operations that raise Python exceptions on bad input."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    smiles = str(smiles or "").strip()
    if not smiles:
        raise SmilesError("empty SMILES")
    if len(smiles) > MAX_SMILES_CHARS:
        raise SmilesError(f"SMILES longer than {MAX_SMILES_CHARS} characters")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise SmilesError("RDKit could not parse the SMILES")
    try:
        mol.UpdatePropertyCache(strict=True)
    except Exception as ex:  # noqa: BLE001
        raise SmilesError(f"RDKit rejects the valence of this SMILES ({_first_line(ex)})") from None
    return mol


def suspicion(mol, affinity: bool = False) -> str | None:
    """Why this molecule is sent to the child process (None: ordinary, safe in-process)."""
    from rdkit import Chem

    pt = Chem.GetPeriodicTable()
    for a in mol.GetAtoms():
        z = a.GetAtomicNum()
        if z not in ORGANIC_Z:
            return f"element {a.GetSymbol()}"
        if abs(a.GetFormalCharge()) >= 2:
            return "charge of 2 or more"
        if a.GetNumRadicalElectrons():
            return "radical"
        vals = [v for v in pt.GetValenceList(z) if v > 0]
        if vals and a.GetTotalValence() > max(vals) + (1 if a.GetFormalCharge() else 0):
            return f"hypervalent {a.GetSymbol()}"
    if any(b.GetBondType() == Chem.BondType.DATIVE for b in mol.GetBonds()):
        return "dative bond"
    if affinity and len(Chem.GetMolFrags(mol)) > 1:
        return "salt / several fragments (affinity ligand)"
    return None


# ------------------------------------------------------------------------------------------------ child process
def dangerous_steps(smiles: str, affinity: bool) -> str | None:
    """What upstream and the GUI do with the SMILES after the checks above, run where an abort is survivable. Returns an
    error message or None. Used by the child (worker_main); also callable in-process by tests."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, rdDepictor

    RDLogger.DisableLog("rdApp.*")
    try:
        mol = parse_checked(smiles)
        if affinity:   # upstream's standardize() sequence (boltz/data/parse/schema.py), with the valence cache filled in
            from chembl_structure_pipeline.exclude_flag import exclude_flag
            from chembl_structure_pipeline.standardizer import standardize_mol
            from rdkit.Chem.MolStandardize import rdMolStandardize

            raw = Chem.MolFromSmiles(smiles, sanitize=False)
            if exclude_flag(raw, includeRDKitSanitization=False):
                return "rejected by the ChEMBL structure pipeline that upstream applies to the affinity ligand (metals, unusual elements, ...)"
            raw.UpdatePropertyCache(strict=False)
            raw = rdMolStandardize.LargestFragmentChooser().choose(raw)
            raw = standardize_mol(raw)
            if Chem.MolFromSmiles(Chem.MolToSmiles(raw)) is None:
                return "the molecule is broken by standardization"
        mh = AllChem.AddHs(mol)
        AllChem.CanonicalRankAtoms(mh)
        Chem.AssignStereochemistry(mh, force=True, cleanIt=True)
        draw = Chem.Mol(mol)
        rdDepictor.Compute2DCoords(draw)
    except SmilesError as ex:
        return str(ex)
    except Exception as ex:  # noqa: BLE001
        return f"RDKit error: {_first_line(ex)}"
    return None


def worker_main() -> None:
    """Child process: JSON lines {"smiles", "affinity"} on stdin, one '@@{"error": ...}' line per answer on stdout."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        sys.stdout.write("@@" + json.dumps({"error": dangerous_steps(req["smiles"], bool(req["affinity"]))}) + "\n")
        sys.stdout.flush()


# ------------------------------------------------------------------------------------------------ parent side
def _crash_message(returncode: int | None) -> str:
    if returncode is not None and returncode < 0:
        return f"RDKit crashed (signal {-returncode}) on this SMILES; it was refused to protect the process"
    return "RDKit stopped on this SMILES; it was refused to protect the process"


def probe(items: list[tuple[str, bool]], timeout: float | None = None) -> dict[tuple[str, bool], str | None]:
    """Run dangerous_steps for each (smiles, affinity) in one child interpreter (a new child after each death).
    Returns {(smiles, affinity): error message or None}."""
    timeout = PROBE_TIMEOUT_S if timeout is None else timeout
    pending = list(dict.fromkeys(items))
    out: dict[tuple[str, bool], str | None] = {}
    while pending:
        with tempfile.TemporaryFile("w+") as errf:
            proc = subprocess.Popen(worker_cmd(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errf, text=True)
            lines: queue.Queue = queue.Queue()

            def feed(batch=tuple(pending), p=proc):
                try:
                    for smi, aff in batch:
                        p.stdin.write(json.dumps({"smiles": smi, "affinity": aff}) + "\n")
                    p.stdin.close()
                except (BrokenPipeError, OSError, ValueError):
                    pass

            def read(p=proc):
                for ln in p.stdout:
                    if ln.startswith("@@"):
                        lines.put(ln[2:])
                lines.put(None)

            threading.Thread(target=feed, daemon=True).start()
            threading.Thread(target=read, daemon=True).start()
            done, culprit_msg = 0, None
            while done < len(pending):
                try:
                    ln = lines.get(timeout=timeout)
                except queue.Empty:
                    proc.kill()
                    culprit_msg = f"RDKit did not finish checking this SMILES within {timeout:g} s; it was refused"
                    break
                if ln is None:
                    try:
                        rc = proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        rc = None
                    culprit_msg = _crash_message(rc)
                    errf.seek(0)
                    if rc is not None and rc > 0 and done == 0:   # the interpreter itself failed (not a signal): not the SMILES' fault
                        raise RuntimeError(f"SMILES check process failed to start: {errf.read()[-300:].strip()}")
                    break
                out[pending[done]] = json.loads(ln)["error"]
                done += 1
            if proc.poll() is None:
                proc.kill()
            proc.wait()
        if done < len(pending):
            out[pending[done]] = culprit_msg
            done += 1
        pending = pending[done:]
    return out


_VERDICTS: OrderedDict = OrderedDict()   # (smiles, affinity) -> error | None, for repeated GUI requests (depictions)
_VERDICTS_MAX = 512
_vlock = threading.Lock()


def check_many(smiles_list, affinity=False) -> dict[str, str | None]:
    """{smiles: error message or None} for every SMILES: in-process checks first, a child only for the suspicious ones
    (one child for all of them). `affinity` is a bool for all, or a list of bools aligned with smiles_list; a SMILES
    listed with both flags gets the error of either."""
    smiles_list = list(smiles_list)
    aff = list(affinity) if isinstance(affinity, (list, tuple)) else [bool(affinity)] * len(smiles_list)
    keys = list(dict.fromkeys((str(s or "").strip(), bool(a)) for s, a in zip(smiles_list, aff)))
    verdict: dict[tuple[str, bool], str | None] = {}
    to_probe: list[tuple[str, bool]] = []
    for key in keys:
        with _vlock:
            if key in _VERDICTS:
                _VERDICTS.move_to_end(key)
                verdict[key] = _VERDICTS[key]
                continue
        try:
            why = suspicion(parse_checked(key[0]), key[1])
        except SmilesError as ex:
            verdict[key] = str(ex)
            continue
        if why is None:
            verdict[key] = None
        else:
            to_probe.append(key)
    if to_probe:
        probed = probe(to_probe)
        verdict.update(probed)
        with _vlock:
            _VERDICTS.update(probed)
            while len(_VERDICTS) > _VERDICTS_MAX:
                _VERDICTS.popitem(last=False)
    result: dict[str, str | None] = {}
    for smi, af in zip(smiles_list, aff):
        err = verdict[(str(smi or "").strip(), bool(af))]
        result[smi] = result.get(smi) or err
    return result


def check_smiles(smiles: str, affinity: bool = False) -> str | None:
    """Error message for one SMILES, or None when it is safe to hand to RDKit / upstream."""
    return check_many([smiles], affinity)[smiles]
