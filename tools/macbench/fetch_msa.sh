#!/bin/zsh
# Fetch the ColabFold-server MSAs for the public PDB sequences of the macbench cases, once, and cache them as csv files in the
# layout the case YAMLs expect (tools/macbench/cases/*_msa.yaml read macbench-env/msa/<stem>_<i>.csv, relative to the checkout).
# Only these public sequences are sent (8RF4, 9MOJ, 9B3H, 9GQU entries; 8EIL, 7QSJ, 8CPK, 8GPP for the NIM-size cases), to
# https://api.colabfold.com (the default server), pairing strategy greedy (the default). MSA time per case is written to
# MSA_DIR/fetch_times.jsonl (seconds, wall). The MSAs are not redistributed with this repository.
#
#   zsh tools/macbench/fetch_msa.sh [MSA_DIR]       default <checkout>/macbench-env/msa ($MACBENCH_ENV/msa); never overwrites
#   MSA_CASES="nim_8eil nim_7qsj nim_8cpk nim_8gpp" zsh tools/macbench/fetch_msa.sh      the NIM-size cases (report section 10)
#
# Environment: BOLTZ = a `boltz` executable with --use_msa_server (default: `boltz` on PATH; the published MSAs were fetched with
# the RBVI fork's `--msa_only` option, docs/macbench/ENV.md, which stops after the MSA step; without it a short CPU prediction runs
# after the MSA is written). BOLTZ_MLX_CACHE = Boltz cache (default ~/.boltz).
# The pinned hashes are in tools/macbench/msa_manifest.json (check: python tools/macbench/check_env.py --need msa). The ColabFold
# databases change over time, so a fresh fetch can differ from the pinned files.
set -u
zmodload zsh/datetime
HERE=${0:A:h}
ROOT=${HERE:h:h}
ENV_DIR=${MACBENCH_ENV:-$ROOT/macbench-env}
MSA_DIR=${1:-$ENV_DIR/msa}
WORK=$MSA_DIR/work
BOLTZ=${BOLTZ:-boltz}
CACHE=${BOLTZ_MLX_CACHE:-$HOME/.boltz}
MSA_ONLY=()
$BOLTZ predict --help 2>/dev/null | grep -q -- '--msa_only' && MSA_ONLY=(--msa_only --accelerator cpu)
[[ ${#MSA_ONLY} -eq 0 ]] && MSA_ONLY=(--accelerator cpu --sampling_steps 1 --recycling_steps 0 --diffusion_samples 1)
mkdir -p $MSA_DIR $WORK
for c in ${=MSA_CASES:-8rf4 9moj 9b3h ntca_aff}; do
  if ls $MSA_DIR/${c%_aff}_0.csv >/dev/null 2>&1 || ls $MSA_DIR/${c}_0.csv >/dev/null 2>&1; then echo "$c: MSA already cached, skipping"; continue; fi
  t0=$EPOCHREALTIME
  nice $BOLTZ predict $HERE/cases/$c.yaml --use_msa_server $MSA_ONLY --out_dir $WORK/$c --cache $CACHE \
      --preprocessing-threads 1 > $WORK/$c.log 2>&1
  rc=$?
  t1=$EPOCHREALTIME
  printf '{"case": "%s", "rc": %d, "wall_s": %.1f, "start_epoch": %.1f}\n' $c $rc $(( t1 - t0 )) $t0 >> $MSA_DIR/fetch_times.jsonl
  tail -n 3 $WORK/$c.log
  stem=${c%_aff}; [ $c = ntca_aff ] && stem=ntca
  for f in $WORK/$c/boltz_results_$c/msa/*.csv; do
    i=${f:t:r}; i=${i##*_}
    cp -n $f $MSA_DIR/${stem}_${i}.csv
  done
done
shasum -a 256 $MSA_DIR/*.csv
