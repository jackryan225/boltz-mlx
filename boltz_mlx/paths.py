"""Portable locations: no hard-coded per-user paths.

Cache (upstream molecule dictionary `mols/`, Boltz checkpoints):
  --cache option > $BOLTZ_MLX_CACHE > $BOLTZ_CACHE (upstream's variable) > ~/.boltz (upstream default).
Converted MLX weights (boltz2_{conf,aff}.safetensors + .json):
  $BOLTZ_MLX_WEIGHTS > <source checkout>/weights > <cache>/mlx_weights > legacy development location (rounds 1-2 dev box),
  the first that holds boltz2_conf.json; when none does: <cache>/mlx_weights (where `boltz-mlx setup` converts them).
"""

from __future__ import annotations

import os
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parents[1]


def default_cache(explicit: str | os.PathLike | None = None) -> Path:
    for v in (explicit, os.environ.get("BOLTZ_MLX_CACHE"), os.environ.get("BOLTZ_CACHE")):
        if v:
            return Path(v).expanduser()
    return Path("~/.boltz").expanduser()


def weights_dir(cache: str | os.PathLike | None = None) -> Path:
    env = os.environ.get("BOLTZ_MLX_WEIGHTS")
    if env:
        return Path(env).expanduser()
    cands = [_PKG_ROOT / "weights", default_cache(cache) / "mlx_weights"]
    for c in cands:
        if (c / "boltz2_conf.json").exists():
            return c
    return cands[1]
