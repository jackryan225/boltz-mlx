#!/usr/bin/env bash
# Install boltz-mlx from this checkout into a FRESH venv and check that `boltz-mlx screen` works offline.
#
#   tools/install_check.sh [DIR]          DIR default: <checkout>/tmp_install (venv at DIR/venv, scratch at DIR/run)
#
# Environment:
#   BOLTZ_UPSTREAM   local upstream boltz checkout to install, e.g. ~/src/boltz; default: `boltz` from the package
#                    index (via the uv cache when --offline works). It is COPIED to DIR/src first (the checkout is never touched).
#   BOLTZ_MLX_CACHE  dir with the upstream mols/ dictionary (+ boltz2_*.ckpt if weights must be converted); default ~/.boltz
#   BOLTZ_MLX_WEIGHTS  dir with converted boltz2_{conf,aff}.safetensors (+ .json); converted into it from the cache if missing
#   ALLOW_NETWORK=1  let uv fall back to the package index when the offline install fails (pypi.org only)
#   PYTHON           interpreter version for the venv (default 3.12)
#
# Checks (exit non-zero on the first failure): fresh venv + install, console entry points, imports + versions, weights and
# mols present (no hard-coded paths), `screen --dry-run` on a tiny target, tests/test_offline.py (socket connect disabled).
# It never runs the model on the GPU; the GPU smoke test is a separate, queued command. It refuses macOS < 26 (Metal 4
# is required) and prints the detected GPU capability and the kernel path --precision f16 takes on this Mac (nax on
# M5-class GPUs, tensorops on M1-M4, the portable fallback only if the device check fails): detection runs the
# tensor-op check once per device + OS (a few one-simdgroup kernels; cached in ~/.cache/boltz_mlx).
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
DIR="${1:-$REPO/tmp_install}"
PY="${PYTHON:-3.12}"
CACHE="${BOLTZ_MLX_CACHE:-${BOLTZ_CACHE:-$HOME/.boltz}}"
step() { printf '\n== %s\n' "$*"; }
fail() { printf 'INSTALL CHECK FAILED: %s\n' "$*" >&2; exit 1; }

MACOS="$(sw_vers -productVersion 2>/dev/null || echo 0)"
[[ "${MACOS%%.*}" -ge 26 ]] || fail "macOS $MACOS: boltz-mlx requires macOS 26 or later (Metal 4; every Apple-silicon Mac runs it)"
command -v uv >/dev/null || fail "uv not found (https://docs.astral.sh/uv/); install it first"
step "fresh venv in $DIR/venv (python $PY)"
rm -rf "$DIR/venv" "$DIR/run"
mkdir -p "$DIR/run"
uv venv --python "$PY" "$DIR/venv"
VPY="$DIR/venv/bin/python"

step "install boltz-mlx (+ upstream boltz, pytest) from the committed tree of $REPO (git archive HEAD, non-editable)"
# a copy, so the build never writes into the checkout (egg-info) and the check sees exactly what is committed
SRC="$DIR/src/boltz-mlx"
rm -rf "$DIR/src"; mkdir -p "$SRC"
git -C "$REPO" archive --format=tar HEAD | tar -x -C "$SRC"
[[ -z "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]] || echo "WARNING: $REPO has uncommitted changes; checking HEAD only"
PKGS=("$SRC[screen,test]")
if [[ -n "${BOLTZ_UPSTREAM:-}" ]]; then
  mkdir -p "$DIR/src/boltz"
  rsync -a --exclude .git --exclude '*.egg-info' --exclude '__pycache__' "$BOLTZ_UPSTREAM/" "$DIR/src/boltz/"
  PKGS+=("$DIR/src/boltz")
fi
if ! uv pip install --python "$VPY" --offline "${PKGS[@]}"; then
  if [[ "${ALLOW_NETWORK:-0}" == "1" ]]; then
    echo "offline install failed; retrying with the package index (ALLOW_NETWORK=1)"
    uv pip install --python "$VPY" "${PKGS[@]}"
  else
    fail "offline install failed (uv cache incomplete); rerun with ALLOW_NETWORK=1 to allow pypi.org"
  fi
fi

step "entry points"
"$DIR/venv/bin/boltz-mlx" --help >/dev/null || fail "boltz-mlx --help"
"$DIR/venv/bin/boltz-mlx" screen --help >/dev/null || fail "boltz-mlx screen --help"

step "imports / versions"
cd "$DIR/run"  # not the source tree: import the installed package
"$VPY" - <<'EOF' || fail "imports"
import importlib.metadata as md, boltz_mlx, boltz, rdkit, mlx.core as mx
print("boltz_mlx", md.version("boltz-mlx"), boltz_mlx.__file__)
print("boltz", md.version("boltz"), "| rdkit", rdkit.__version__, "| mlx", mx.__version__)
EOF

step "GPU capability and code path (python -m boltz_mlx.capability)"
"$VPY" -m boltz_mlx.capability | tee "$DIR/run/capability.txt" || fail "capability detection"
grep -q "run as:" "$DIR/run/capability.txt" || fail "capability detection printed no code path"

step "cache and weights (cache: $CACHE)"
[[ -d "$CACHE/mols" ]] || fail "$CACHE/mols missing: copy the upstream molecule dictionary there (upstream \`boltz predict\` downloads it to ~/.boltz once), or set BOLTZ_MLX_CACHE"
WDIR="$(BOLTZ_MLX_CACHE="$CACHE" "$VPY" -c 'from boltz_mlx.paths import weights_dir; print(weights_dir())')"
if [[ ! -f "$WDIR/boltz2_conf.safetensors" || ! -f "$WDIR/boltz2_aff.safetensors" ]]; then
  [[ -f "$CACHE/boltz2_conf.ckpt" && -f "$CACHE/boltz2_aff.ckpt" ]] || fail "no converted weights in $WDIR and no checkpoints in $CACHE"
  WDIR="${BOLTZ_MLX_WEIGHTS:-$CACHE/mlx_weights}"
  echo "converting checkpoints from $CACHE into $WDIR"
  "$VPY" "$REPO/tools/convert_weights.py" --cache "$CACHE" --out "$WDIR" || fail "weight conversion"
fi
if [[ ! -f "$WDIR/boltz2_conf_mixed.safetensors" || ! -f "$WDIR/boltz2_aff_mixed.safetensors" ]]; then
  echo "writing the mixed fp16 checkpoints (--precision f16) into $WDIR"
  "$VPY" "$REPO/tools/make_mixed_checkpoint.py" --src "$WDIR" --report '' || fail "mixed checkpoint conversion"
fi
echo "weights: $WDIR"; ls -la "$WDIR" | sed -n '1,10p'

step "screen --dry-run (no model)"
cat > "$DIR/run/ligands.csv" <<'EOF'
id,smiles
phenol,Oc1ccccc1
acetaminophen,CC(=O)Nc1ccc(O)cc1
not_a_molecule,C1CC
EOF
(cd "$DIR/run" && BOLTZ_MLX_CACHE="$CACHE" BOLTZ_MLX_WEIGHTS="$WDIR" "$DIR/venv/bin/boltz-mlx" screen \
  --target MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQ --msa empty \
  --ligands ligands.csv --out screen_dry --dry-run) | tee "$DIR/run/dry_run.txt"
grep -q "ESTIMATE" "$DIR/run/dry_run.txt" || fail "dry run printed no estimate"
grep -q "1 invalid" "$DIR/run/dry_run.txt" || fail "dry run did not flag the invalid ligand"

step "offline test (socket connect disabled; real featurization, mocked model)"
# run from a scratch dir against the INSTALLED package (not a source tree)
mkdir -p "$DIR/run/tests"; cp "$SRC/tests/test_offline.py" "$SRC/tests/screen_fakes.py" "$DIR/run/tests/"
(cd "$DIR/run" && BOLTZ_MLX_CACHE="$CACHE" "$VPY" -m pytest -q -p no:cacheprovider --rootdir "$DIR/run" \
   -o pythonpath=tests tests/test_offline.py) || fail "tests/test_offline.py"
(cd "$DIR/run" && "$VPY" -c 'import boltz_mlx, sys; p = boltz_mlx.__file__; print("tested package:", p); sys.exit("site-packages" not in p)') \
  || fail "the offline test did not use the installed package"

step "OK: install check passed ($DIR)"
