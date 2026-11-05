"""MLX port of InputEmbedder and ContactConditioning from boltz/model/modules/trunkv2.py."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from boltz_mlx.modules.encodersv2 import AtomAttentionEncoder, AtomEncoder, FourierEmbedding
from boltz_mlx.layers.norm import LayerNorm as _FastLN

# boltz.data.const values
NUM_TOKENS = 33
NUM_METHOD_TYPES = 12
NUM_CHAIN_TYPES = 4
NUM_CONTACT_CONDITIONING_INFO = 5  # len(const.contact_conditioning_info)


class ContactConditioning(nn.Module):
    def __init__(self, token_z: int, cutoff_min: float, cutoff_max: float) -> None:
        super().__init__()
        self.fourier_embedding = FourierEmbedding(token_z)
        self.encoder = nn.Linear(token_z + NUM_CONTACT_CONDITIONING_INFO - 1, token_z)
        self.encoding_unspecified = mx.zeros((token_z,))
        self.encoding_unselected = mx.zeros((token_z,))
        self.cutoff_min = cutoff_min
        self.cutoff_max = cutoff_max

    def __call__(self, feats):
        cc_full = feats["contact_conditioning"].astype(mx.float32)
        contact_conditioning = cc_full[:, :, :, 2:]
        contact_threshold = feats["contact_threshold"]
        contact_threshold_normalized = (contact_threshold - self.cutoff_min) / (self.cutoff_max - self.cutoff_min)
        contact_threshold_fourier = self.fourier_embedding(contact_threshold_normalized.reshape(-1)).reshape(
            contact_threshold_normalized.shape + (-1,)
        )

        contact_conditioning = mx.concatenate(
            [contact_conditioning, contact_threshold_normalized[..., None], contact_threshold_fourier],
            axis=-1,
        )
        contact_conditioning = self.encoder(contact_conditioning)

        return (
            contact_conditioning * (1 - cc_full[:, :, :, 0:2].sum(axis=-1, keepdims=True))
            + self.encoding_unspecified * cc_full[:, :, :, 0:1]
            + self.encoding_unselected * cc_full[:, :, :, 1:2]
        )


class InputEmbedder(nn.Module):
    def __init__(
        self,
        atom_s: int,
        atom_z: int,
        token_s: int,
        token_z: int,
        atoms_per_window_queries: int,
        atoms_per_window_keys: int,
        atom_feature_dim: int,
        atom_encoder_depth: int,
        atom_encoder_heads: int,
        activation_checkpointing: bool = False,  # ignored
        add_method_conditioning: bool = False,
        add_modified_flag: bool = False,
        add_cyclic_flag: bool = False,
        add_mol_type_feat: bool = False,
        use_no_atom_char: bool = False,
        use_atom_backbone_feat: bool = False,
        use_residue_feats_atoms: bool = False,
    ) -> None:
        super().__init__()
        self.token_s = token_s
        self.add_method_conditioning = add_method_conditioning
        self.add_modified_flag = add_modified_flag
        self.add_cyclic_flag = add_cyclic_flag
        self.add_mol_type_feat = add_mol_type_feat

        self.atom_encoder = AtomEncoder(
            atom_s=atom_s,
            atom_z=atom_z,
            token_s=token_s,
            token_z=token_z,
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys,
            atom_feature_dim=atom_feature_dim,
            structure_prediction=False,
            use_no_atom_char=use_no_atom_char,
            use_atom_backbone_feat=use_atom_backbone_feat,
            use_residue_feats_atoms=use_residue_feats_atoms,
        )

        self.atom_enc_proj_z = [
            _FastLN(atom_z),
            nn.Linear(atom_z, atom_encoder_depth * atom_encoder_heads, bias=False),
        ]

        self.atom_attention_encoder = AtomAttentionEncoder(
            atom_s=atom_s,
            token_s=token_s,
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys,
            atom_encoder_depth=atom_encoder_depth,
            atom_encoder_heads=atom_encoder_heads,
            structure_prediction=False,
        )

        self.res_type_encoding = nn.Linear(NUM_TOKENS, token_s, bias=False)
        self.msa_profile_encoding = nn.Linear(NUM_TOKENS + 1, token_s, bias=False)

        if add_method_conditioning:
            self.method_conditioning_init = nn.Embedding(NUM_METHOD_TYPES, token_s)
        if add_modified_flag:
            self.modified_conditioning_init = nn.Embedding(2, token_s)
        if add_cyclic_flag:
            self.cyclic_conditioning_init = nn.Linear(1, token_s, bias=False)
        if add_mol_type_feat:
            self.mol_type_conditioning_init = nn.Embedding(NUM_CHAIN_TYPES, token_s)

    def __call__(self, feats, affinity: bool = False):
        res_type = feats["res_type"].astype(mx.float32)
        if affinity:
            profile = feats["profile_affinity"]
            deletion_mean = feats["deletion_mean_affinity"][..., None]
        else:
            profile = feats["profile"]
            deletion_mean = feats["deletion_mean"][..., None]

        q, c, p, to_keys = self.atom_encoder(feats)
        atom_enc_bias = p
        for f in self.atom_enc_proj_z:
            atom_enc_bias = f(atom_enc_bias)
        a, _, _, _ = self.atom_attention_encoder(
            feats=feats, q=q, c=c, atom_enc_bias=atom_enc_bias, to_keys=to_keys
        )

        s = (
            a
            + self.res_type_encoding(res_type)
            + self.msa_profile_encoding(mx.concatenate([profile, deletion_mean], axis=-1))
        )

        if self.add_method_conditioning:
            s = s + self.method_conditioning_init(feats["method_feature"])
        if self.add_modified_flag:
            s = s + self.modified_conditioning_init(feats["modified"])
        if self.add_cyclic_flag:
            cyclic = mx.minimum(feats["cyclic_period"], 1.0)[..., None]
            s = s + self.cyclic_conditioning_init(cyclic)
        if self.add_mol_type_feat:
            s = s + self.mol_type_conditioning_init(feats["mol_type"])

        return s
