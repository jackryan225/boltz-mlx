#!/bin/sh
# Turbo (TF32 + fp16 diffusion + 50 steps) with fewer recycles, 3 seeds, validation set.
set -e
cd "$(dirname "$0")/../.."
for rc in 0 1; do
  for seed in 0 1 2; do
    run="sweep_turbo_r${rc}_seed${seed}"
    BOLTZ_MLX_TF32=1 BOLTZ_MLX_DIFF_DTYPE=fp16 python tools/validation/mlx_feats_predict.py --run "$run" --steps 50 --recycling "$rc" --seed "$seed" > /dev/null 2>&1
    python tools/validation/run_all.py --name "$run" --score-only --pred-dir "data/validation/runs/$run/{id}/predictions" --out "validation/sweep/$run.json" 2>&1 | grep "mean over" | sed "s/^/$run /"
  done
done
