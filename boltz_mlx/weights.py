"""Weight loading: converted safetensors keep PyTorch qualified names."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from boltz_mlx.paths import weights_dir  # noqa: E402

WEIGHTS = weights_dir()  # $BOLTZ_MLX_WEIGHTS > checkout weights/ > <cache>/mlx_weights > legacy dev path (boltz_mlx/paths.py)


@lru_cache(maxsize=2)
def load_all(name: str = "conf") -> dict:
    return mx.load(str(WEIGHTS / f"boltz2_{name}.safetensors"))


def subtree(prefix: str, name: str = "conf") -> dict:
    p = prefix.rstrip(".") + "."
    return {k[len(p):]: v for k, v in load_all(name).items() if k.startswith(p)}


def load_module(module: nn.Module, prefix: str, name: str = "conf", strict: bool = True) -> nn.Module:
    """Load weights under `prefix` into `module` (names relative to prefix)."""
    w = subtree(prefix, name)
    if strict:
        ours = {k for k, _ in tree_flatten(module.parameters())}
        missing, extra = ours - set(w), set(w) - ours
        if missing or extra:
            raise ValueError(f"{prefix}: missing={sorted(missing)[:10]} unexpected={sorted(extra)[:10]}")
    module.load_weights(list(w.items()), strict=strict)
    return module
