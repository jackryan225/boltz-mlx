"""Small synthetic Boltz-2 models for the mixed-checkpoint tests (tests/test_mixed_checkpoint.py).

The real hyperparameters (weights/boltz2_{conf,aff}.json) with every depth cut to 1-2 blocks (the hard-coded 64
pairformer / 4 MSA blocks are truncated after construction) and MLX's random initialization, so a CPU f16 forward on
the prot117 / lig_aff features takes seconds and a few hundred MB. Every module class of the full model is present,
so every fp16 weight site of the f16 code paths is exercised.
"""

from __future__ import annotations

import copy
import json

import mlx.core as mx
from mlx.utils import tree_flatten

from boltz_mlx.model import Boltz2
from boltz_mlx.weights import WEIGHTS


def small_hp(name: str) -> dict:
    hp = copy.deepcopy(json.loads((WEIGHTS / f"boltz2_{name}.json").read_text()))
    sm = hp["score_model_args"]
    sm.update(token_transformer_depth=2, atom_encoder_depth=1, atom_decoder_depth=1)
    hp["confidence_model_args"]["pairformer_args"]["num_blocks"] = 1
    for k in ("affinity_model_args1", "affinity_model_args2"):
        if k in hp:
            hp[k]["pairformer_args"]["num_blocks"] = 1
            hp[k]["transformer_args"]["num_blocks"] = 1
    return hp


def small_model(name: str, seed: int = 0) -> Boltz2:
    mx.random.seed(seed)
    model = Boltz2(small_hp(name), affinity=(name == "aff"), compile_diffusion=False)
    model.pairformer_module.layers = model.pairformer_module.layers[:2]
    model.msa_module.layers = model.msa_module.layers[:1]
    # random init has unit-scale LayerNorms and uniform(+-1/sqrt(fan_in)) linears: fine for fp16 ranges at these depths
    mx.eval(model.parameters())
    return model


def checkpoint(model) -> dict:
    """The model's parameters as an fp32 'checkpoint' dict (aliases included, as in the converted files)."""
    return dict(tree_flatten(model.parameters()))


def load_into(model, weights: dict) -> None:
    tops = set(model.children().keys())
    model.load_weights([(k, v) for k, v in weights.items() if k.split(".")[0] in tops], strict=True)
    mx.eval(model.parameters())
