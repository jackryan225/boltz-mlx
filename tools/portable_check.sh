#!/usr/bin/env bash
# Portability test kit: run on any Apple-silicon Mac, copy the output folder back.
#
#   tools/portable_check.sh [OUT_DIR] [extra options of `tools/portable_check.py run`]
#
# Environment:
#   BOLTZ_MLX_PYTHON   python with mlx + numpy + pytest (default: <checkout>/.venv/bin/python, else python3)
#   BOLTZ_MLX_DATA     data dir (default <checkout>/data; files: `tools/portable_check.py files`)
#   BOLTZ_MLX_WEIGHTS  converted weights incl. the *_mixed files (default: <checkout>/weights, then the dev location)
#   BOLTZ_MLX_KERNEL_PATH / BOLTZ_MLX_NAX   force one f16 kernel path (default: measure every available one)
# Fully offline. Nothing is written outside OUT_DIR except ~/.cache/boltz_mlx/tensorops_check.json (the per-device
# tensor-op check cache; BOLTZ_MLX_CAP_CACHE=0 disables it) and, for the install check, OUT_DIR/install.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="${BOLTZ_MLX_PYTHON:-}"
if [[ -z "$PY" ]]; then
  if [[ -x "$REPO/.venv/bin/python" ]]; then PY="$REPO/.venv/bin/python"; else PY="$(command -v python3)"; fi
fi
"$PY" -c 'import mlx.core, numpy, pytest' 2>/dev/null || {
  echo "portable_check: $PY cannot import mlx / numpy / pytest (set BOLTZ_MLX_PYTHON to one that can)" >&2; exit 1; }
echo "== python $PY ($("$PY" -c 'import mlx.core as mx, platform; print(platform.python_version(), "mlx", mx.__version__)'))"
echo "== files (MISSING ones are skipped where possible; prot400 is optional)"
"$PY" "$REPO/tools/portable_check.py" files | sed -n '1,13p'
OUT="${1:-}"
if [[ -n "$OUT" && "$OUT" != --* ]]; then shift; set -- --out "$OUT" "$@"; fi
cd "$REPO"
exec "$PY" "$REPO/tools/portable_check.py" run "$@"
