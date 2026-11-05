#!/bin/sh
# strict vs turbo on all 30 validation cases, 3 seeds each (MLX RNG, same features per case).
# Usage: sh tools/validation/modes_sweep30.sh
set -e
cd "$(dirname "$0")/../.."
M=validation/manifest_all.json
for seed in 0 1 2; do
  run="v30_strict_seed${seed}"
  python tools/validation/mlx_feats_predict.py --manifest $M --run "$run" --seed "$seed" > /dev/null 2>&1
  python tools/validation/run_all.py --manifest $M --name "$run" --score-only --pred-dir "data/validation/runs/$run/{id}/predictions" --out "validation/sweep/$run.json" 2>&1 | grep "mean over" | sed "s/^/$run /"
  run="v30_turbo_seed${seed}"
  BOLTZ_MLX_TF32=1 BOLTZ_MLX_DIFF_DTYPE=fp16 python tools/validation/mlx_feats_predict.py --manifest $M --run "$run" --seed "$seed" --steps 50 > /dev/null 2>&1
  python tools/validation/run_all.py --manifest $M --name "$run" --score-only --pred-dir "data/validation/runs/$run/{id}/predictions" --out "validation/sweep/$run.json" 2>&1 | grep "mean over" | sed "s/^/$run /"
done
