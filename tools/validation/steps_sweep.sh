#!/bin/sh
# Fast-mode accuracy sweep: sampling steps x diffusion precision x 3 seeds on the validation set.
# Usage: sh tools/validation/steps_sweep.sh
set -e
cd "$(dirname "$0")/../.."
for cfg in "200 fp32" "100 fp16" "50 fp16" "50 fp32"; do
  set -- $cfg; steps=$1; dt=$2
  for seed in 0 1 2; do
    run="sweep_s${steps}_${dt}_seed${seed}"
    BOLTZ_MLX_DIFF_DTYPE=$dt python tools/validation/mlx_feats_predict.py --run "$run" --steps "$steps" --seed "$seed" > /dev/null 2>&1
    python tools/validation/run_all.py --name "$run" --score-only --pred-dir "data/validation/runs/$run/{id}/predictions" --out "validation/sweep/$run.json" 2>&1 | grep "mean over"
  done
done
