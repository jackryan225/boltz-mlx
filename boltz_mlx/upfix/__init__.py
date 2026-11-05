"""Narrow patches for upstream boltz 2.2.1 bugs that boltz-mlx inherits (it uses upstream's parsing, featurisation and writers).

boltz-mlx has to keep working with stock `boltz==2.2.1`, so nothing upstream is edited: each fix replaces one module-level
name of an upstream module with a small wrapper that calls upstream's own code (the pattern of boltz_mlx/seeding.py).
Every patch is idempotent, leaves the output for normal inputs byte-identical, and prints nothing unless it fires.

  names.py    ligand atom names that overflow 4 characters (upstream #494), applied by parse_guards
  parse_guards.py  upstream's SMILES parsing: atom names, valence check (#649), salts in standardize -> install_parse_guards
  writers.py  PDB / mmCIF writer bugs (#458 element, chain-id columns, cyclic closure, bond order) -> install_writers

install_upstream_fixes() installs all of them; call it where install_conformer_seed() is called (before upstream
preprocessing and before any BoltzWriter output). The SMILES validation that keeps bad input away from RDKit in the GUI,
the screen and predict lives in boltz_mlx/smiles_guard.py.

Credits: the ideas for the atom-name, valence, standardize and writer fixes come from the boltz-community fork of Boltz
(MIT, Wohlwend, Corso, Passaro; https://github.com/boltz-community/boltz); each module names the fork commit it follows.
The code here is written against stock boltz 2.2.1 and does not copy the fork's changes to its data types.
"""

from __future__ import annotations

import sys


def say(msg: str) -> None:
    """The one-line notice of a patch that fired (stderr; stdout carries --json output)."""
    print(f"boltz-mlx: {msg}", file=sys.stderr, flush=True)


def install_upstream_fixes() -> None:
    from boltz_mlx.upfix import parse_guards, writers

    parse_guards.install_parse_guards()
    writers.install_writers()
