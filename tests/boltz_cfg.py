"""Boltz-2 (conf checkpoint) hyperparameters + dump helpers shared by encoder/diffusion tests."""

import mlx.core as mx
import numpy as np

from conftest import mxa

TOKEN_S, TOKEN_Z, ATOM_S, ATOM_Z = 384, 128, 128, 16
W, H = 32, 128
ATOM_FEATURE_DIM = 388

SCORE_MODEL_ARGS = dict(
    token_s=TOKEN_S,
    atom_s=ATOM_S,
    atoms_per_window_queries=W,
    atoms_per_window_keys=H,
    sigma_data=16,
    dim_fourier=256,
    atom_encoder_depth=3,
    atom_encoder_heads=4,
    token_transformer_depth=24,
    token_transformer_heads=16,
    atom_decoder_depth=3,
    atom_decoder_heads=4,
    conditioning_transition_layers=2,
    transformer_post_ln=False,
)

DIFFUSION_COND_ARGS = dict(
    token_s=TOKEN_S,
    token_z=TOKEN_Z,
    atom_s=ATOM_S,
    atom_z=ATOM_Z,
    atoms_per_window_queries=W,
    atoms_per_window_keys=H,
    atom_encoder_depth=3,
    atom_encoder_heads=4,
    token_transformer_depth=24,
    token_transformer_heads=16,
    atom_decoder_depth=3,
    atom_decoder_heads=4,
    atom_feature_dim=ATOM_FEATURE_DIM,
    conditioning_transition_layers=2,
)

INPUT_EMBEDDER_ARGS = dict(
    atom_s=ATOM_S,
    atom_z=ATOM_Z,
    token_s=TOKEN_S,
    token_z=TOKEN_Z,
    atoms_per_window_queries=W,
    atoms_per_window_keys=H,
    atom_feature_dim=ATOM_FEATURE_DIM,
    atom_encoder_depth=3,
    atom_encoder_heads=4,
    add_mol_type_feat=True,
    add_method_conditioning=True,
    add_modified_flag=True,
    add_cyclic_flag=True,
)


def is_feats(v) -> bool:
    return isinstance(v, np.ndarray) and v.dtype.kind == "U" and str(v) == "FEATS"


def arg(d: dict, key: str, feats: dict):
    """Fetch a dump entry as an MLX value (FEATS -> feats dict; 0-d ints -> python int)."""
    v = d[key]
    if is_feats(v):
        return feats
    if v.ndim == 0 and v.dtype.kind in "iu":
        return int(v)
    return mxa(v)
