"""Shared constants and small helpers of the GUI: limits, id / file-name sanitising, path containment, API errors."""

from __future__ import annotations

import os
import re
import secrets
import time
from pathlib import Path

# ----------------------------------------------------------------------------------------------------- limits
KB = 1024
MB = 1024 * KB
LIMITS = {
    "target": 2 * MB,        # YAML / FASTA / sequence
    "msa": 64 * MB,          # .a3m / .csv (the screen reads at most --max-msa-seqs rows of it)
    "ligands": 64 * MB,      # .csv / .tsv / .smi / .sdf
    "body": 140 * MB,        # whole JSON request
    "name": 80,              # run name, characters
    "preview_rows": 20000,   # rows of the ligand preview sent back to the browser
    "pocket": 200,           # pocket contacts
}
ALLOWED_EXT = {
    "target": (".yaml", ".yml", ".fasta", ".fa", ".fas", ".faa"),
    "msa": (".a3m", ".csv"),
    "ligands": (".csv", ".tsv", ".smi", ".smiles", ".txt", ".sdf"),
}
DEFAULT_PORT = 8765
PORT_TRIES = 20
DEFAULT_RUNS_DIR = "~/boltz-mlx-runs"
LOCAL_HOSTS = ("127.0.0.1", "localhost")

# verbatim from docs/researcher.md (the owner asked for it in the UI footer and About box)
PRIVACY = ("Ligand structures never leave the Mac. The target MSA is a one-time, per-target step that sends only the protein "
           "sequence (often public; public for every benchmark target here). Fully private targets without local MSA "
           "databases (hundreds of GB) run `msa: empty`, at an accuracy cost that this round measures only if pre-flight MSAs "
           "exist (strict FEP+ seed 0, MSA vs empty).")
# The About box: plain words for an ordinary user, true to what the code does (lookup.py is the only search / lookup traffic; the
# MSA server is used only by a Predict protein chain whose MSA is set to "Server": predict.py adds --use_msa_server for it).
PRIVACY_GUI = ("Your structures, ligands and results stay on this Mac. The protein search sends only the name or ID you type or "
               "pick to UniProt and the PDB (RCSB). The MSA server option sends that protein's sequence to ColabFold's server, "
               "and only for a chain where you choose it.")

RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
RECORD_ID_RE = re.compile(r"^(?!\.)[A-Za-z0-9_.-]{1,200}$")


class ApiError(Exception):
    """An error with an HTTP status; the message is shown to the user."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def new_token() -> str:
    return secrets.token_urlsafe(24)


# ------------------------------------------------------------------------------------------------ file names
def safe_filename(name: object) -> tuple[str, str]:
    """(display_name, extension) of a user-supplied file name: directory parts dropped, odd characters replaced, at most 80
    characters. Uploaded content is always stored under a fixed name (inputs/target.<ext>, ...), so a file name can never
    choose where anything is written; the name is only shown back to the user and its extension picks the parser."""
    base = re.split(r"[\\/]", str(name or ""))[-1]
    base = re.sub(r"[^A-Za-z0-9._ +()-]+", "_", base).strip(" .")
    base = re.sub(r"\.{2,}", ".", base)[:80] or "upload"
    ext = os.path.splitext(base)[1].lower()
    return base, ext


def slug(s: object, default: str = "run", n: int = 40) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", str(s or "")).strip("-")[:n].strip("-")
    return s or default


def is_within(path: str | os.PathLike, root: str | os.PathLike) -> bool:
    """True when `path` (symlinks resolved) lies inside `root` (symlinks resolved)."""
    try:
        Path(os.path.realpath(path)).relative_to(os.path.realpath(root))
        return True
    except ValueError:
        return False


def valid_run_id(run_id: object) -> bool:
    return isinstance(run_id, str) and bool(RUN_ID_RE.match(run_id)) and ".." not in run_id


def valid_record_id(rid: object) -> bool:
    return isinstance(rid, str) and bool(RECORD_ID_RE.match(rid))


def iso(ts: float | None) -> str | None:
    return None if ts is None else time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts))
