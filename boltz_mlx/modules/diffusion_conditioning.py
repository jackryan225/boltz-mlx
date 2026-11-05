"""MLX port of boltz/model/modules/diffusion_conditioning.py."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from boltz_mlx import kernels as _K

from boltz_mlx.modules.encodersv2 import AtomEncoder, PairwiseConditioning
from boltz_mlx.layers.norm import LayerNorm as _FastLN


def _proj_list(dim: int, heads: int, depth: int):
    # torch: ModuleList of Sequential(LayerNorm(dim), Linear(dim, heads, bias=False))
    return [[_FastLN(dim), nn.Linear(dim, heads, bias=False)] for _ in range(depth)]


def _apply(layers, x):
    for f in layers:
        x = f(x)
    return x


class DiffusionConditioning(nn.Module):
    def __init__(
        self,
        token_s: int,
        token_z: int,
        atom_s: int,
        atom_z: int,
        atoms_per_window_queries: int = 32,
        atoms_per_window_keys: int = 128,
        atom_encoder_depth: int = 3,
        atom_encoder_heads: int = 4,
        token_transformer_depth: int = 24,
        token_transformer_heads: int = 8,
        atom_decoder_depth: int = 3,
        atom_decoder_heads: int = 4,
        atom_feature_dim: int = 128,
        conditioning_transition_layers: int = 2,
        use_no_atom_char: bool = False,
        use_atom_backbone_feat: bool = False,
        use_residue_feats_atoms: bool = False,
    ) -> None:
        super().__init__()
        self.pairwise_conditioner = PairwiseConditioning(
            token_z=token_z,
            dim_token_rel_pos_feats=token_z,
            num_transitions=conditioning_transition_layers,
        )
        self.atom_encoder = AtomEncoder(
            atom_s=atom_s,
            atom_z=atom_z,
            token_s=token_s,
            token_z=token_z,
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys,
            atom_feature_dim=atom_feature_dim,
            structure_prediction=True,
            use_no_atom_char=use_no_atom_char,
            use_atom_backbone_feat=use_atom_backbone_feat,
            use_residue_feats_atoms=use_residue_feats_atoms,
        )
        self.atom_enc_proj_z = _proj_list(atom_z, atom_encoder_heads, atom_encoder_depth)
        self.atom_dec_proj_z = _proj_list(atom_z, atom_decoder_heads, atom_decoder_depth)
        self.token_trans_proj_z = _proj_list(token_z, token_transformer_heads, token_transformer_depth)

    def __call__(self, s_trunk, z_trunk, relative_position_encoding, feats):
        N = z_trunk.shape[1]
        if _K.chunking(N):
            # pair rows are independent: build z and the token-transformer bias in row chunks
            zs, tbs = [], []
            for i0 in range(0, N, _K.CHUNK_ROWS):
                zc = self.pairwise_conditioner(z_trunk[:, i0 : i0 + _K.CHUNK_ROWS], relative_position_encoding[:, i0 : i0 + _K.CHUNK_ROWS])
                tbc = mx.concatenate([_apply(layer, zc) for layer in self.token_trans_proj_z], axis=-1)
                mx.eval(zc, tbc)
                zs.append(zc)
                tbs.append(tbc)
            z = mx.concatenate(zs, axis=1)
            token_trans_bias = mx.concatenate(tbs, axis=1)
            del zs, tbs
            q, c, p, to_keys = self.atom_encoder(feats=feats, s_trunk=s_trunk, z=z)
            atom_enc_bias = mx.concatenate([_apply(layer, p) for layer in self.atom_enc_proj_z], axis=-1)
            atom_dec_bias = mx.concatenate([_apply(layer, p) for layer in self.atom_dec_proj_z], axis=-1)
            return q, c, to_keys, atom_enc_bias, atom_dec_bias, token_trans_bias

        z = self.pairwise_conditioner(z_trunk, relative_position_encoding)

        q, c, p, to_keys = self.atom_encoder(feats=feats, s_trunk=s_trunk, z=z)

        atom_enc_bias = mx.concatenate([_apply(layer, p) for layer in self.atom_enc_proj_z], axis=-1)
        atom_dec_bias = mx.concatenate([_apply(layer, p) for layer in self.atom_dec_proj_z], axis=-1)
        token_trans_bias = mx.concatenate([_apply(layer, z) for layer in self.token_trans_proj_z], axis=-1)

        return q, c, to_keys, atom_enc_bias, atom_dec_bias, token_trans_bias
