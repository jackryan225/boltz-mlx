#!/bin/zsh
# Public RCSB entries for the macbench cases (FASTA + mmCIF), fetched 2026-10-31 and committed in tools/macbench/ref/ with
# sha256 in ref/MANIFEST.json. Re-run only to verify: files are compared, never overwritten.
#   8RF4 (nsp1 + EQT)  9MOJ (GINS tetramer)  9B3H (PmSLP-1 + bovine factor I)  9GQU / 9GUI (NTCA apo / with DNA)
set -eu
D=${1:?usage: fetch_ref.sh OUTDIR}
mkdir -p $D
for id in 8RF4 9MOJ 9B3H 9GQU 9GUI; do
  curl -fsSL -m 120 -o $D/$id.fasta https://www.rcsb.org/fasta/entry/$id
  curl -fsSL -m 120 -o $D/$id.cif https://files.rcsb.org/download/$id.cif
done
shasum -a 256 $D/*
