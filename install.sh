#!/usr/bin/env bash
# boltz-mlx installer for Apple-silicon Macs (macOS 26+). Run from the cloned repository:
#
#   ./install.sh               install `boltz-mlx`, download + convert the model (~6.2 GB, once), open the interface
#   ./install.sh --no-gui      the same without opening the interface at the end
#   ./install.sh --no-setup    install the command only (the first `boltz-mlx gui` / `boltz-mlx setup` downloads later)
#   ./install.sh --editable    developer install: the command runs this checkout's code
#
# What it does:
#   1. checks the Mac: Apple silicon, macOS 26 or later;
#   2. needs uv (https://docs.astral.sh/uv/, the Python package manager). If uv is missing it asks before installing it
#      with uv's official installer (or install it yourself: `brew install uv`). BOLTZ_MLX_INSTALL_UV=1 answers yes;
#   3. `uv tool install`s this checkout with Python 3.12 into its own environment (~/.local/share/uv/tools/boltz-mlx)
#      with the tested versions pinned (constraints.txt: boltz 2.2.1, mlx 0.32.4, torch, rdkit, ...), and links the
#      `boltz-mlx` command into uv's bin directory (~/.local/bin);
#   4. runs `boltz-mlx setup`: downloads the Boltz-2 weights and molecule dictionary from the official Boltz / Hugging Face
#      sources into ~/.boltz (resumable, checksum verified) and converts them to MLX weights in ~/.boltz/mlx_weights;
#   5. starts `boltz-mlx gui` (http://127.0.0.1:8765, this Mac only) when run in a terminal.
# Re-running is safe: finished steps are skipped. After `git pull`, run it again to update.
# Locations: BOLTZ_MLX_CACHE (default ~/.boltz), BOLTZ_MLX_WEIGHTS; uv's own UV_TOOL_DIR / UV_TOOL_BIN_DIR are respected.
set -euo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
PY_VERSION="3.12"
SETUP=1 GUI=1 EDITABLE=0
for a in "$@"; do
  case "$a" in
    --no-setup) SETUP=0; GUI=0 ;;
    --no-gui) GUI=0 ;;
    --editable) EDITABLE=1 ;;
    -h|--help) sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "install.sh: unknown option $a (see ./install.sh --help)" >&2; exit 2 ;;
  esac
done
say() { printf '\033[1m==> %s\033[0m\n' "$*"; }
die() { printf 'install.sh: %s\n' "$*" >&2; exit 1; }

# ------------------------------------------------------------------ 1. this Mac
[[ "$(uname -s)" == "Darwin" ]] || die "boltz-mlx runs on Apple-silicon Macs only (this is $(uname -s))."
if [[ "$(uname -m)" != "arm64" ]]; then
  if [[ "$(sysctl -n hw.optional.arm64 2>/dev/null || echo 0)" == "1" ]]; then
    die "this terminal runs under Rosetta (x86_64). Open a native Terminal (Get Info > uncheck 'Open using Rosetta') and run ./install.sh again."
  fi
  die "boltz-mlx needs an Apple-silicon Mac (M1 or later); this Mac is $(uname -m)."
fi
MACOS="$(sysctl -n kern.osproductversion 2>/dev/null || sw_vers -productVersion)"
MAJOR="${MACOS%%.*}"
[[ "$MAJOR" == "16" ]] && MAJOR=26   # compatibility numbering of macOS 26 seen by some tools
[[ "$MAJOR" -ge 26 ]] || die "macOS $MACOS: boltz-mlx needs macOS 26 or later (Metal 4). Update in System Settings > General > Software Update."
CHIP="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'Apple silicon')"
say "$CHIP, macOS $MACOS"

# ------------------------------------------------------------------ 2. uv
find_uv() { command -v uv 2>/dev/null || { [[ -x "$HOME/.local/bin/uv" ]] && echo "$HOME/.local/bin/uv"; } || true; }
UV="$(find_uv)"
if [[ -z "$UV" ]]; then
  echo "boltz-mlx is installed with uv, a Python package manager (https://docs.astral.sh/uv/), which is not on this Mac."
  echo "It can be installed now with uv's official installer:"
  echo "    curl -LsSf https://astral.sh/uv/install.sh | sh"
  echo "(it puts uv in ~/.local/bin and adds that directory to your shell's PATH). Or install it yourself: brew install uv"
  ANSWER="${BOLTZ_MLX_INSTALL_UV:-}"
  if [[ -z "$ANSWER" && -t 0 ]]; then
    read -r -p "Install uv now? [y/N] " ANSWER
  fi
  case "$ANSWER" in
    y|Y|yes|1) curl -LsSf https://astral.sh/uv/install.sh | sh || die "the uv installer failed; install uv yourself (brew install uv) and run ./install.sh again." ;;
    *) die "uv is required. Install it (see above) and run ./install.sh again." ;;
  esac
  UV="$(find_uv)"
  [[ -n "$UV" ]] || die "uv was installed but is not found; open a new terminal and run ./install.sh again."
fi
say "uv: $UV ($("$UV" --version))"

# ------------------------------------------------------------------ 3. the boltz-mlx command
SRC="$REPO[screen]"
INSTALL=("$UV" tool install --force --reinstall-package boltz-mlx --python "$PY_VERSION" --constraints "$REPO/constraints.txt")
if [[ "$EDITABLE" == "1" ]]; then INSTALL+=(--editable); fi
say "installing boltz-mlx with Python $PY_VERSION and the tested package versions (first time: ~1 GB of packages, a few minutes)"
"${INSTALL[@]}" "$SRC"
BIN="$("$UV" tool dir --bin)"
CMD="$BIN/boltz-mlx"
[[ -x "$CMD" ]] || die "boltz-mlx was not installed into $BIN"
ON_PATH=1
case ":$PATH:" in *":$BIN:"*) ;; *) ON_PATH=0 ;; esac

# ------------------------------------------------------------------ 4. model weights + molecule dictionary
if [[ "$SETUP" == "1" ]]; then
  say "downloading and converting the Boltz-2 model (skipped if already done)"
  "$CMD" setup || die "setup did not finish (the reason is above). Run \`$CMD setup\` again to continue where it stopped."
fi

echo
say "boltz-mlx is installed."
if [[ "$ON_PATH" == "0" ]]; then
  echo "    $BIN is not on your PATH yet. Run this once (then open a new terminal):"
  echo "        $UV tool update-shell"
  echo "    Until then start it as: $CMD gui"
fi
echo "    Start the interface:   boltz-mlx gui      (opens http://127.0.0.1:8765 in your browser; Ctrl-C quits)"
echo "    Check the setup:       boltz-mlx check"
echo "    Update later:          git pull && ./install.sh"

# ------------------------------------------------------------------ 5. open it
if [[ "$GUI" == "1" && -t 0 && -t 1 ]]; then
  echo
  say "starting the interface now (Ctrl-C to quit; next time just run: boltz-mlx gui)"
  exec "$CMD" gui
fi
