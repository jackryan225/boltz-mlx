"""MLX port of boltz.model.modules.confidencev2 (ConfidenceModule, ConfidenceHeads); inference only."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from boltz_mlx.layers.pairformer import PairformerModule
from boltz_mlx.modules.confidence_utils import (
    NONPOLYMER,
    cdist,
    compute_aggregated_metric,
    compute_ptms,
    repeat_interleave,
)
from boltz_mlx.modules.encodersv2 import RelativePositionEncoder
from boltz_mlx.modules.trunkv2 import _torch_linspace_f32
from boltz_mlx.modules.trunkv2_embed import ContactConditioning

NUM_BOND_TYPES = 6  # len(const.bond_types)


class ConfidenceModule(nn.Module):
    """Algorithm 31"""

    def __init__(
        self,
        token_s,
        token_z,
        pairformer_args: dict,
        num_dist_bins=64,
        token_level_confidence=True,
        max_dist=22,
        add_s_to_z_prod=False,
        add_s_input_to_s=False,
        add_z_input_to_z=False,
        maximum_bond_distance=0,
        bond_type_feature=False,
        confidence_args: dict = None,
        compile_pairformer=False,
        fix_sym_check=False,
        cyclic_pos_enc=False,
        return_latent_feats=False,
        conditioning_cutoff_min=None,
        conditioning_cutoff_max=None,
        **kwargs,
    ):
        super().__init__()
        if not token_level_confidence:
            raise NotImplementedError("only token_level_confidence=True is ported")
        self.max_num_atoms_per_token = 23
        self.no_update_s = pairformer_args.get("no_update_s", False)
        # registered buffer upstream (persistent -> present in the checkpoint)
        self.boundaries = mx.array(_torch_linspace_f32(2.0, float(max_dist), num_dist_bins - 1))
        self.dist_bin_pairwise_embed = nn.Embedding(num_dist_bins, token_z)
        self.token_level_confidence = token_level_confidence

        self.s_to_z = nn.Linear(token_s, token_z, bias=False)
        self.s_to_z_transpose = nn.Linear(token_s, token_z, bias=False)

        self.add_s_to_z_prod = add_s_to_z_prod
        if add_s_to_z_prod:
            self.s_to_z_prod_in1 = nn.Linear(token_s, token_z, bias=False)
            self.s_to_z_prod_in2 = nn.Linear(token_s, token_z, bias=False)
            self.s_to_z_prod_out = nn.Linear(token_z, token_z, bias=False)

        self.s_inputs_norm = nn.LayerNorm(token_s)
        if not self.no_update_s:
            self.s_norm = nn.LayerNorm(token_s)
        self.z_norm = nn.LayerNorm(token_z)

        self.add_s_input_to_s = add_s_input_to_s
        if add_s_input_to_s:
            self.s_input_to_s = nn.Linear(token_s, token_s, bias=False)

        self.add_z_input_to_z = add_z_input_to_z
        if add_z_input_to_z:
            self.rel_pos = RelativePositionEncoder(token_z, fix_sym_check=fix_sym_check, cyclic_pos_enc=cyclic_pos_enc)
            self.token_bonds = nn.Linear(
                1 if maximum_bond_distance == 0 else maximum_bond_distance + 2, token_z, bias=False
            )
            self.bond_type_feature = bond_type_feature
            if bond_type_feature:
                self.token_bonds_type = nn.Embedding(NUM_BOND_TYPES + 1, token_z)

            self.contact_conditioning = ContactConditioning(
                token_z=token_z, cutoff_min=conditioning_cutoff_min, cutoff_max=conditioning_cutoff_max
            )
        pairformer_args = dict(pairformer_args)
        pairformer_args["v2"] = True
        self.pairformer_stack = PairformerModule(token_s, token_z, **pairformer_args)
        self.return_latent_feats = return_latent_feats

        self.confidence_heads = ConfidenceHeads(
            token_s, token_z, token_level_confidence=token_level_confidence, **(confidence_args or {})
        )

    def __call__(
        self,
        s_inputs,  # (b, n, ts)
        s,  # (b, n, ts)
        z,  # (b, n, n, tz)
        x_pred,  # (bm, m, 3)
        feats,
        pred_distogram_logits,
        multiplicity=1,
        run_sequentially=False,
        use_kernels: bool = False,
    ):
        if run_sequentially and multiplicity > 1:
            assert z.shape[0] == 1, "Not supported with batch size > 1"
            out_dicts = []
            for sample_idx in range(multiplicity):
                out = self(
                    s_inputs,
                    s,
                    z,
                    x_pred[sample_idx : sample_idx + 1],
                    feats,
                    pred_distogram_logits,
                    multiplicity=1,
                    run_sequentially=False,
                    use_kernels=use_kernels,
                )
                # materialise per-sample results so the graph of one sample is freed before the next
                mx.eval([v for k, v in out.items() if k != "pair_chains_iptm"])
                out_dicts.append(out)

            out_dict = {}
            for key in out_dicts[0]:
                if key != "pair_chains_iptm":
                    out_dict[key] = mx.concatenate([out[key] for out in out_dicts], axis=0)
                else:
                    pair_chains_iptm = {}
                    for chain_idx1 in out_dicts[0][key]:
                        chains_iptm = {}
                        for chain_idx2 in out_dicts[0][key][chain_idx1]:
                            chains_iptm[chain_idx2] = mx.concatenate(
                                [out[key][chain_idx1][chain_idx2] for out in out_dicts], axis=0
                            )
                        pair_chains_iptm[chain_idx1] = chains_iptm
                    out_dict[key] = pair_chains_iptm
            return out_dict

        s_inputs = self.s_inputs_norm(s_inputs)
        if not self.no_update_s:
            s = self.s_norm(s)

        if self.add_s_input_to_s:
            s = s + self.s_input_to_s(s_inputs)

        z = self.z_norm(z)

        if self.add_z_input_to_z:
            relative_position_encoding = self.rel_pos(feats)
            z = z + relative_position_encoding
            z = z + self.token_bonds(feats["token_bonds"].astype(mx.float32))
            if self.bond_type_feature:
                z = z + self.token_bonds_type(feats["type_bonds"].astype(mx.int64))
            z = z + self.contact_conditioning(feats)

        s = repeat_interleave(s, multiplicity, 0)

        z = z + self.s_to_z(s_inputs)[:, :, None, :] + self.s_to_z_transpose(s_inputs)[:, None, :, :]
        if self.add_s_to_z_prod:
            z = z + self.s_to_z_prod_out(
                self.s_to_z_prod_in1(s_inputs)[:, :, None, :] * self.s_to_z_prod_in2(s_inputs)[:, None, :, :]
            )

        z = repeat_interleave(z, multiplicity, 0)
        s_inputs = repeat_interleave(s_inputs, multiplicity, 0)

        token_to_rep_atom = repeat_interleave(feats["token_to_rep_atom"], multiplicity, 0)
        if x_pred.ndim == 4:
            B, mult, N, _ = x_pred.shape
            x_pred = x_pred.reshape(B * mult, N, -1)
        x_pred_repr = mx.matmul(token_to_rep_atom.astype(mx.float32), x_pred)
        d = cdist(x_pred_repr, x_pred_repr)
        distogram = (d[..., None] > self.boundaries).sum(axis=-1).astype(mx.int64)
        distogram = self.dist_bin_pairwise_embed(distogram)
        z = z + distogram

        mask = repeat_interleave(feats["token_pad_mask"], multiplicity, 0)
        pair_mask = mask[:, :, None] * mask[:, None, :]

        s_t, z_t = self.pairformer_stack(s, z, mask=mask, pair_mask=pair_mask, use_kernels=use_kernels)

        # AF3 has residual connections, we remove them
        s = s_t
        z = z_t

        out_dict = {}
        if self.return_latent_feats:
            out_dict["s_conf"] = s
            out_dict["z_conf"] = z

        out_dict.update(
            self.confidence_heads(
                s=s,
                z=z,
                x_pred=x_pred,
                d=d,
                feats=feats,
                multiplicity=multiplicity,
                pred_distogram_logits=pred_distogram_logits,
            )
        )
        return out_dict


class ConfidenceHeads(nn.Module):
    def __init__(
        self,
        token_s,
        token_z,
        num_plddt_bins=50,
        num_pde_bins=64,
        num_pae_bins=64,
        token_level_confidence=True,
        use_separate_heads: bool = False,
        **kwargs,
    ):
        super().__init__()
        if not token_level_confidence:
            raise NotImplementedError("only token_level_confidence=True is ported")
        self.max_num_atoms_per_token = 23
        self.token_level_confidence = token_level_confidence
        self.use_separate_heads = use_separate_heads

        if self.use_separate_heads:
            self.to_pae_intra_logits = nn.Linear(token_z, num_pae_bins, bias=False)
            self.to_pae_inter_logits = nn.Linear(token_z, num_pae_bins, bias=False)
            self.to_pde_intra_logits = nn.Linear(token_z, num_pde_bins, bias=False)
            self.to_pde_inter_logits = nn.Linear(token_z, num_pde_bins, bias=False)
        else:
            self.to_pae_logits = nn.Linear(token_z, num_pae_bins, bias=False)
            self.to_pde_logits = nn.Linear(token_z, num_pde_bins, bias=False)

        self.to_plddt_logits = nn.Linear(token_s, num_plddt_bins, bias=False)
        self.to_resolved_logits = nn.Linear(token_s, 2, bias=False)

    def __call__(
        self,
        s,  # (b, n, ts)
        z,  # (b, n, n, tz)
        x_pred,  # (bm, m, 3)
        d,
        feats,
        pred_distogram_logits,
        multiplicity=1,
    ):
        if self.use_separate_heads:
            asym_id_token = feats["asym_id"]
            is_same_chain = asym_id_token[:, :, None] == asym_id_token[:, None, :]
            is_different_chain = ~is_same_chain

            pae_intra_logits = self.to_pae_intra_logits(z) * is_same_chain.astype(mx.float32)[..., None]
            pae_inter_logits = self.to_pae_inter_logits(z) * is_different_chain.astype(mx.float32)[..., None]
            pae_logits = pae_inter_logits + pae_intra_logits

            z_sym = z + mx.swapaxes(z, 1, 2)
            pde_intra_logits = self.to_pde_intra_logits(z_sym) * is_same_chain.astype(mx.float32)[..., None]
            pde_inter_logits = self.to_pde_inter_logits(z_sym) * is_different_chain.astype(mx.float32)[..., None]
            pde_logits = pde_inter_logits + pde_intra_logits
        else:
            pae_logits = self.to_pae_logits(z)
            pde_logits = self.to_pde_logits(z + mx.swapaxes(z, 1, 2))
        resolved_logits = self.to_resolved_logits(s)
        plddt_logits = self.to_plddt_logits(s)

        ligand_weight = 20
        non_interface_weight = 1
        interface_weight = 10

        token_type = repeat_interleave(feats["mol_type"], multiplicity, 0)
        is_ligand_token = (token_type == NONPOLYMER).astype(mx.float32)

        plddt = compute_aggregated_metric(plddt_logits)
        token_pad_mask = repeat_interleave(feats["token_pad_mask"], multiplicity, 0)
        complex_plddt = (plddt * token_pad_mask).sum(axis=-1) / token_pad_mask.sum(axis=-1)

        is_contact = (d < 8).astype(mx.float32)
        is_different_chain = (feats["asym_id"][:, :, None] != feats["asym_id"][:, None, :]).astype(mx.float32)
        is_different_chain = repeat_interleave(is_different_chain, multiplicity, 0)
        token_interface_mask = mx.max(
            is_contact * is_different_chain * (1 - is_ligand_token)[:, :, None],
            axis=-1,
        )
        token_non_interface_mask = (1 - token_interface_mask) * (1 - is_ligand_token)
        iplddt_weight = (
            is_ligand_token * ligand_weight
            + token_interface_mask * interface_weight
            + token_non_interface_mask * non_interface_weight
        )
        complex_iplddt = (plddt * token_pad_mask * iplddt_weight).sum(axis=-1) / mx.sum(
            token_pad_mask * iplddt_weight, axis=-1
        )

        # Compute the gPDE and giPDE
        pde = compute_aggregated_metric(pde_logits, end=32)
        pred_distogram_prob = repeat_interleave(mx.softmax(pred_distogram_logits, axis=-1), multiplicity, 0)
        contacts = mx.concatenate([mx.ones((1, 1, 1, 20)), mx.zeros((1, 1, 1, 44))], axis=-1).astype(
            pred_distogram_prob.dtype
        )
        prob_contact = (pred_distogram_prob * contacts).sum(axis=-1)
        token_pad_pair_mask = (
            token_pad_mask[:, :, None]
            * token_pad_mask[:, None, :]
            * (1 - mx.eye(token_pad_mask.shape[1]))[None]
        )
        token_pair_mask = token_pad_pair_mask * prob_contact
        complex_pde = (pde * token_pair_mask).sum(axis=(1, 2)) / token_pair_mask.sum(axis=(1, 2))
        asym_id = repeat_interleave(feats["asym_id"], multiplicity, 0)
        token_interface_pair_mask = token_pair_mask * (asym_id[:, :, None] != asym_id[:, None, :])
        complex_ipde = (pde * token_interface_pair_mask).sum(axis=(1, 2)) / (
            token_interface_pair_mask.sum(axis=(1, 2)) + 1e-5
        )
        out_dict = dict(
            pde_logits=pde_logits,
            plddt_logits=plddt_logits,
            resolved_logits=resolved_logits,
            pde=pde,
            plddt=plddt,
            complex_plddt=complex_plddt,
            complex_iplddt=complex_iplddt,
            complex_pde=complex_pde,
            complex_ipde=complex_ipde,
        )
        out_dict["pae_logits"] = pae_logits
        out_dict["pae"] = compute_aggregated_metric(pae_logits, end=32)

        ptm, iptm, ligand_iptm, protein_iptm, pair_chains_iptm = compute_ptms(pae_logits, x_pred, feats, multiplicity)
        out_dict["ptm"] = ptm
        out_dict["iptm"] = iptm
        out_dict["ligand_iptm"] = ligand_iptm
        out_dict["protein_iptm"] = protein_iptm
        out_dict["pair_chains_iptm"] = pair_chains_iptm
        return out_dict
