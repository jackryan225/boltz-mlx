"""MLX port of boltz.model.modules.affinity (AffinityModule, AffinityHeadsTransformer); inference only.

Plus `affinity_ensemble`, the combination of the two checkpoint heads done inline in
`Boltz2.forward` upstream (sigmoid on the binary logits, mean over the ensemble, optional MW correction).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from boltz_mlx.layers.pairformer import PairformerNoSeqModule
from boltz_mlx.modules.confidence_utils import cdist, repeat_interleave
from boltz_mlx.modules.encodersv2 import PairwiseConditioning
from boltz_mlx.modules.trunkv2 import _torch_linspace_f32


class AffinityModule(nn.Module):
    """Algorithm 31"""

    def __init__(
        self,
        token_s,
        token_z,
        pairformer_args: dict,
        transformer_args: dict,
        num_dist_bins=64,
        max_dist=22,
        use_cross_transformer: bool = False,
        groups: dict = None,
    ):
        super().__init__()
        if use_cross_transformer:
            raise NotImplementedError("use_cross_transformer is not used by the released Boltz-2 affinity heads")
        # registered buffer upstream (persistent -> present in the checkpoint)
        self.boundaries = mx.array(_torch_linspace_f32(2.0, float(max_dist), num_dist_bins - 1))
        self.dist_bin_pairwise_embed = nn.Embedding(num_dist_bins, token_z)

        self.s_to_z_prod_in1 = nn.Linear(token_s, token_z, bias=False)
        self.s_to_z_prod_in2 = nn.Linear(token_s, token_z, bias=False)

        self.z_norm = nn.LayerNorm(token_z)
        self.z_linear = nn.Linear(token_z, token_z, bias=False)

        self.pairwise_conditioner = PairwiseConditioning(
            token_z=token_z, dim_token_rel_pos_feats=token_z, num_transitions=2
        )

        self.pairformer_stack = PairformerNoSeqModule(token_z, **pairformer_args)
        self.affinity_heads = AffinityHeadsTransformer(
            token_z,
            transformer_args["token_s"],
            transformer_args["num_blocks"],
            transformer_args["num_heads"],
            transformer_args["activation_checkpointing"],
            False,
            groups=groups,
        )

    def __call__(self, s_inputs, z, x_pred, feats, multiplicity=1, use_kernels=False):
        z = self.z_linear(self.z_norm(z))
        z = repeat_interleave(z, multiplicity, 0)

        z = z + self.s_to_z_prod_in1(s_inputs)[:, :, None, :] + self.s_to_z_prod_in2(s_inputs)[:, None, :, :]

        token_to_rep_atom = repeat_interleave(feats["token_to_rep_atom"], multiplicity, 0)
        if x_pred.ndim == 4:
            B, mult, N, _ = x_pred.shape
            x_pred = x_pred.reshape(B * mult, N, -1)
        x_pred_repr = mx.matmul(token_to_rep_atom.astype(mx.float32), x_pred)
        d = cdist(x_pred_repr, x_pred_repr)

        distogram = (d[..., None] > self.boundaries).sum(axis=-1).astype(mx.int64)
        distogram = self.dist_bin_pairwise_embed(distogram)

        z = z + self.pairwise_conditioner(z_trunk=z, token_rel_pos_feats=distogram)

        pad_token_mask = repeat_interleave(feats["token_pad_mask"], multiplicity, 0)
        rec_mask = repeat_interleave((feats["mol_type"] == 0).astype(mx.float32), multiplicity, 0) * pad_token_mask
        lig_mask = repeat_interleave((feats["affinity_token_mask"] != 0).astype(mx.float32), multiplicity, 0)
        lig_mask = lig_mask * pad_token_mask
        cross_pair_mask = (
            lig_mask[:, :, None] * rec_mask[:, None, :]
            + rec_mask[:, :, None] * lig_mask[:, None, :]
            + lig_mask[:, :, None] * lig_mask[:, None, :]
        )
        z = self.pairformer_stack(z, pair_mask=cross_pair_mask, use_kernels=use_kernels)

        out_dict = {}
        out_dict.update(self.affinity_heads(z=z, feats=feats, multiplicity=multiplicity))
        return out_dict


class AffinityHeadsTransformer(nn.Module):
    def __init__(
        self,
        token_z,
        input_token_s,
        num_blocks,
        num_heads,
        activation_checkpointing,
        use_cross_transformer,
        groups=None,
    ):
        super().__init__()
        # lists instead of nn.Sequential: weight names `x.<idx>.weight` match torch (ReLUs keep their index)
        self.affinity_out_mlp = [
            nn.Linear(token_z, token_z),
            nn.ReLU(),
            nn.Linear(token_z, input_token_s),
            nn.ReLU(),
        ]
        self.to_affinity_pred_value = [
            nn.Linear(input_token_s, input_token_s),
            nn.ReLU(),
            nn.Linear(input_token_s, input_token_s),
            nn.ReLU(),
            nn.Linear(input_token_s, 1),
        ]
        self.to_affinity_pred_score = [
            nn.Linear(input_token_s, input_token_s),
            nn.ReLU(),
            nn.Linear(input_token_s, input_token_s),
            nn.ReLU(),
            nn.Linear(input_token_s, 1),
        ]
        self.to_affinity_logits_binary = nn.Linear(1, 1)

    @staticmethod
    def _seq(layers, x):
        for f in layers:
            x = f(x)
        return x

    def __call__(self, z, feats, multiplicity=1):
        pad_token_mask = repeat_interleave(feats["token_pad_mask"], multiplicity, 0)[..., None]
        rec_mask = repeat_interleave((feats["mol_type"] == 0).astype(mx.float32), multiplicity, 0)[..., None]
        rec_mask = rec_mask * pad_token_mask
        lig_mask = repeat_interleave((feats["affinity_token_mask"] != 0).astype(mx.float32), multiplicity, 0)[..., None]
        lig_mask = lig_mask * pad_token_mask
        cross_pair_mask = (
            lig_mask[:, :, None] * rec_mask[:, None, :]
            + rec_mask[:, :, None] * lig_mask[:, None, :]
            + (lig_mask[:, :, None] * lig_mask[:, None, :])
        ) * (1 - mx.eye(lig_mask.shape[1]))[None, :, :, None]

        g = mx.sum(z * cross_pair_mask, axis=(1, 2)) / (mx.sum(cross_pair_mask, axis=(1, 2)) + 1e-7)

        g = self._seq(self.affinity_out_mlp, g)

        affinity_pred_value = self._seq(self.to_affinity_pred_value, g).reshape(-1, 1)
        affinity_pred_score = self._seq(self.to_affinity_pred_score, g).reshape(-1, 1)
        affinity_logits_binary = self.to_affinity_logits_binary(affinity_pred_score).reshape(-1, 1)
        return {
            "affinity_pred_value": affinity_pred_value,
            "affinity_logits_binary": affinity_logits_binary,
        }


# ---------------------------------------------------------------------------------------
# glue from Boltz2.forward (affinity branch)
# ---------------------------------------------------------------------------------------
def affinity_cross_pair_mask(feats: dict) -> mx.array:
    """(N, N) bool mask used upstream to zero z before the affinity modules (batch element 0)."""
    pad = feats["token_pad_mask"][0]
    rec = (feats["mol_type"][0] == 0).astype(mx.float32) * pad
    lig = (feats["affinity_token_mask"][0] != 0).astype(mx.float32) * pad
    return (lig[:, None] * rec[None, :] + rec[:, None] * lig[None, :] + lig[:, None] * lig[None, :]) != 0


def mask_z_for_affinity(z: mx.array, feats: dict) -> mx.array:
    """`z_affinity = z * cross_pair_mask[None, :, :, None]`."""
    return z * affinity_cross_pair_mask(feats).astype(z.dtype)[None, :, :, None]


def best_sample_index(iptm: mx.array) -> int:
    """`argsort(iptm, descending=True)[0]`: the diffusion sample whose coordinates feed the affinity module."""
    return int(mx.argmax(iptm).item())


# Boltz2 affinity MW correction constants (affinity_mw_correction=True; off in the released boltz2_aff ckpt)
MW_MODEL_COEF = 1.03525938
MW_MW_COEF = -0.59992683
MW_BIAS = 2.83288489


def affinity_ensemble(out1: dict, out2: dict, feats: dict | None = None, mw_correction: bool = False) -> dict:
    """Combine the two ensemble members exactly as `Boltz2.forward` does.

    `out1`/`out2` are AffinityModule outputs (`affinity_pred_value`, `affinity_logits_binary`).  Returns
    affinity_pred_value / affinity_probability_binary (ensemble means) plus the per-member `..1` / `..2` values.
    With `mw_correction=True` the mean value is replaced by `1.03526*value - 0.59993*MW**0.3 + 2.83288`
    (needs `feats["affinity_mw"]`).
    """
    p1 = mx.sigmoid(out1["affinity_logits_binary"])
    p2 = mx.sigmoid(out2["affinity_logits_binary"])
    ens = {
        "affinity_pred_value": (out1["affinity_pred_value"] + out2["affinity_pred_value"]) / 2,
        "affinity_probability_binary": (p1 + p2) / 2,
    }
    if mw_correction:
        mw = feats["affinity_mw"][0] ** 0.3
        ens["affinity_pred_value"] = MW_MODEL_COEF * ens["affinity_pred_value"] + MW_MW_COEF * mw + MW_BIAS
    ens["affinity_pred_value1"] = out1["affinity_pred_value"]
    ens["affinity_probability_binary1"] = p1
    ens["affinity_pred_value2"] = out2["affinity_pred_value"]
    ens["affinity_probability_binary2"] = p2
    return ens
