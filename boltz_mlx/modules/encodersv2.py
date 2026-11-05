"""MLX port of boltz/model/modules/encodersv2.py (inference only)."""

from __future__ import annotations

from math import pi

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from boltz_mlx import kernels as _K
from boltz_mlx.layers.transition import Transition
from boltz_mlx.modules.transformersv2 import AtomTransformer
from boltz_mlx.layers.norm import LayerNorm as _FastLN


def one_hot(idx: mx.array, num_classes: int) -> mx.array:
    """torch.nn.functional.one_hot semantics (int -> 0/1 along a new last axis); returns float32."""
    return (idx[..., None] == mx.arange(num_classes)).astype(mx.float32)


class FourierEmbedding(nn.Module):
    """Algorithm 22."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(1, dim)

    def __call__(self, times: mx.array) -> mx.array:  # (b,) -> (b, d)
        times = times[:, None]
        rand_proj = self.proj(times)
        return mx.cos(2 * pi * rand_proj)


class RelativePositionEncoder(nn.Module):
    """Algorithm 3."""

    def __init__(self, token_z, r_max=32, s_max=2, fix_sym_check=False, cyclic_pos_enc=False) -> None:
        super().__init__()
        self.r_max = r_max
        self.s_max = s_max
        self.linear_layer = nn.Linear(4 * (r_max + 1) + 2 * (s_max + 1) + 1, token_z, bias=False)
        self.fix_sym_check = fix_sym_check
        self.cyclic_pos_enc = cyclic_pos_enc

    def __call__(self, feats):
        asym = feats["asym_id"]
        res = feats["residue_index"]
        b_same_chain = asym[:, :, None] == asym[:, None, :]
        b_same_residue = res[:, :, None] == res[:, None, :]
        ent = feats["entity_id"]
        b_same_entity = ent[:, :, None] == ent[:, None, :]

        d_residue = res[:, :, None] - res[:, None, :]

        if self.cyclic_pos_enc and bool(mx.any(feats["cyclic_period"] > 0).item()):
            cp = feats["cyclic_period"]
            period = mx.where(cp > 0, cp, mx.zeros_like(cp) + 10000)
            d_residue = (d_residue - period * mx.round(d_residue / period)).astype(mx.int64)

        d_residue = mx.clip(d_residue + self.r_max, 0, 2 * self.r_max)
        d_residue = mx.where(b_same_chain, d_residue, mx.zeros_like(d_residue) + 2 * self.r_max + 1)
        a_rel_pos = one_hot(d_residue, 2 * self.r_max + 2)

        tok = feats["token_index"]
        d_token = mx.clip(tok[:, :, None] - tok[:, None, :] + self.r_max, 0, 2 * self.r_max)
        d_token = mx.where(
            b_same_chain & b_same_residue, d_token, mx.zeros_like(d_token) + 2 * self.r_max + 1
        )
        a_rel_token = one_hot(d_token, 2 * self.r_max + 2)

        sym = feats["sym_id"]
        d_chain = mx.clip(sym[:, :, None] - sym[:, None, :] + self.s_max, 0, 2 * self.s_max)
        d_chain = mx.where(
            (~b_same_entity) if self.fix_sym_check else b_same_chain,
            mx.zeros_like(d_chain) + 2 * self.s_max + 1,
            d_chain,
        )
        a_rel_chain = one_hot(d_chain, 2 * self.s_max + 2)

        return self.linear_layer(
            mx.concatenate(
                [a_rel_pos, a_rel_token, b_same_entity[..., None].astype(mx.float32), a_rel_chain],
                axis=-1,
            )
        )


class SingleConditioning(nn.Module):
    """Algorithm 21."""

    def __init__(
        self,
        sigma_data: float,
        token_s: int = 384,
        dim_fourier: int = 256,
        num_transitions: int = 2,
        transition_expansion_factor: int = 2,
        eps: float = 1e-20,
        disable_times: bool = False,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.sigma_data = sigma_data
        self.disable_times = disable_times

        self.norm_single = _FastLN(2 * token_s)
        self.single_embed = nn.Linear(2 * token_s, 2 * token_s)
        if not self.disable_times:
            self.fourier_embed = FourierEmbedding(dim_fourier)
            self.norm_fourier = _FastLN(dim_fourier)
            self.fourier_to_single = nn.Linear(dim_fourier, 2 * token_s, bias=False)

        self.transitions = [
            Transition(dim=2 * token_s, hidden=transition_expansion_factor * 2 * token_s)
            for _ in range(num_transitions)
        ]

    def __call__(self, times, s_trunk, s_inputs):
        s = mx.concatenate([s_trunk, s_inputs], axis=-1)
        s = self.single_embed(self.norm_single(s))
        normed_fourier = None
        if not self.disable_times:
            fourier_embed = self.fourier_embed(times)  # sigma rescaling done in diffusion module
            normed_fourier = self.norm_fourier(fourier_embed)
            fourier_to_single = self.fourier_to_single(normed_fourier)
            s = fourier_to_single[:, None, :] + s

        for transition in self.transitions:
            s = _K.radd(s, transition(s))

        return s, normed_fourier


class PairwiseConditioning(nn.Module):
    """Algorithm 21."""

    def __init__(self, token_z, dim_token_rel_pos_feats, num_transitions=2, transition_expansion_factor=2) -> None:
        super().__init__()
        self.dim_pairwise_init_proj = [
            _FastLN(token_z + dim_token_rel_pos_feats),
            nn.Linear(token_z + dim_token_rel_pos_feats, token_z, bias=False),
        ]
        self.transitions = [
            Transition(dim=token_z, hidden=transition_expansion_factor * token_z) for _ in range(num_transitions)
        ]

    def __call__(self, z_trunk, token_rel_pos_feats):
        z = mx.concatenate([z_trunk, token_rel_pos_feats], axis=-1)
        for f in self.dim_pairwise_init_proj:
            z = f(z)
        for transition in self.transitions:
            z = _K.radd(z, transition(z))
        return z


def get_indexing_matrix(K: int, W: int, H: int) -> mx.array:
    """Matrix (2K, h*K) mapping 2K half-windows (W//2 atoms each) to K key windows of H atoms."""
    assert W % 2 == 0
    assert H % (W // 2) == 0

    h = H // (W // 2)
    assert h % 2 == 0

    arange = mx.arange(2 * K)
    index = mx.clip((arange[None, :] - arange[:, None]) + h // 2, 0, h + 1)
    index = index.reshape(K, 2, 2 * K)[:, 0, :]
    onehot = one_hot(index, h + 2)[..., 1:-1]  # (K, 2K, h)
    onehot = mx.swapaxes(onehot, 1, 0)  # (2K, K, h)
    return onehot.reshape(2 * K, h * K).astype(mx.float32)


def single_to_keys(single: mx.array, indexing_matrix: mx.array, W: int, H: int) -> mx.array:
    B, N, D = single.shape
    K = N // W
    single = single.reshape(B, 2 * K, W // 2, D)
    out = mx.einsum("bjid,jk->bkid", single, indexing_matrix)  # j = 2K, i = W//2, k = h * K
    return out.reshape(B, K, H, D)


class ToKeys:
    """The `to_keys` callable for N atoms (N % W == 0), windows W (queries) / H (keys): __call__ is upstream's
    one-hot contraction (single_to_keys). gather() is the same map as an index gather (every column of the indexing
    matrix holds at most one 1): identical values (x * 1 + sum of 0 * y == x) up to the sign of the zeros of key slots
    past the sequence ends, which are masked. Used for the per-step atom attention keys in --precision f16, where the
    contraction was ~1.5 GFLOP fp32 + transposed copies per atom layer and step (lig_aff x5)."""

    def __init__(self, N: int, W: int, H: int):
        K = N // W
        self.W, self.H, self.K = W, H, K
        self.indexing_matrix = get_indexing_matrix(K, W, H)
        self._idx = None

    def __call__(self, single: mx.array) -> mx.array:
        return single_to_keys(single, self.indexing_matrix, self.W, self.H)

    def gather(self, single: mx.array) -> mx.array:
        if self._idx is None:
            m = self.indexing_matrix  # (2K, h * K) one-hot columns
            valid = m.max(axis=0) > 0
            self._idx = (mx.where(valid, mx.argmax(m, axis=0), 0).astype(mx.uint32), valid)
            mx.eval(*self._idx)
        idx, valid = self._idx
        B, N, D = single.shape
        g = mx.take(single.reshape(B, 2 * self.K, self.W // 2, D), idx, axis=1)  # (B, h * K, W / 2, D)
        g = mx.where(valid[None, :, None, None], g, mx.zeros((), g.dtype))
        return g.reshape(B, self.K, self.H, D)


def make_to_keys(N: int, W: int, H: int):
    """Build the `to_keys` callable for N atoms (N % W == 0), windows W (queries) / H (keys)."""
    return ToKeys(N, W, H)


class AtomEncoder(nn.Module):
    def __init__(
        self,
        atom_s,
        atom_z,
        token_s,
        token_z,
        atoms_per_window_queries,
        atoms_per_window_keys,
        atom_feature_dim,
        structure_prediction=True,
        use_no_atom_char=False,
        use_atom_backbone_feat=False,
        use_residue_feats_atoms=False,
    ) -> None:
        super().__init__()
        self.embed_atom_features = nn.Linear(atom_feature_dim, atom_s)
        self.embed_atompair_ref_pos = nn.Linear(3, atom_z, bias=False)
        self.embed_atompair_ref_dist = nn.Linear(1, atom_z, bias=False)
        self.embed_atompair_mask = nn.Linear(1, atom_z, bias=False)
        self.atoms_per_window_queries = atoms_per_window_queries
        self.atoms_per_window_keys = atoms_per_window_keys
        self.use_no_atom_char = use_no_atom_char
        self.use_atom_backbone_feat = use_atom_backbone_feat
        self.use_residue_feats_atoms = use_residue_feats_atoms

        self.structure_prediction = structure_prediction
        if structure_prediction:
            self.s_to_c_trans = [_FastLN(token_s), nn.Linear(token_s, atom_s, bias=False)]
            self.z_to_p_trans = [_FastLN(token_z), nn.Linear(token_z, atom_z, bias=False)]

        self.c_to_p_trans_k = [nn.ReLU(), nn.Linear(atom_s, atom_z, bias=False)]
        self.c_to_p_trans_q = [nn.ReLU(), nn.Linear(atom_s, atom_z, bias=False)]
        self.p_mlp = [
            nn.ReLU(),
            nn.Linear(atom_z, atom_z, bias=False),
            nn.ReLU(),
            nn.Linear(atom_z, atom_z, bias=False),
            nn.ReLU(),
            nn.Linear(atom_z, atom_z, bias=False),
        ]

    @staticmethod
    def _seq(layers, x):
        for f in layers:
            x = f(x)
        return x

    def __call__(self, feats, s_trunk=None, z=None):
        B, N, _ = feats["ref_pos"].shape
        atom_mask = feats["atom_pad_mask"].astype(mx.bool_)

        atom_ref_pos = feats["ref_pos"].astype(mx.float32)
        atom_uid = feats["ref_space_uid"]

        atom_feats = [
            atom_ref_pos,
            feats["ref_charge"].astype(mx.float32)[..., None],
            feats["ref_element"].astype(mx.float32),
        ]
        if not self.use_no_atom_char:
            atom_feats.append(feats["ref_atom_name_chars"].reshape(B, N, 4 * 64).astype(mx.float32))
        if self.use_atom_backbone_feat:
            atom_feats.append(feats["atom_backbone_feat"].astype(mx.float32))
        if self.use_residue_feats_atoms:
            res_feats = mx.concatenate(
                [
                    feats["res_type"].astype(mx.float32),
                    feats["modified"].astype(mx.float32)[..., None],
                    one_hot(feats["mol_type"], 4),
                ],
                axis=-1,
            )
            atom_to_token = feats["atom_to_token"].astype(mx.float32)
            atom_feats.append(atom_to_token @ res_feats)

        atom_feats = mx.concatenate(atom_feats, axis=-1)
        c = self.embed_atom_features(atom_feats)

        # windows are created up front
        W, H = self.atoms_per_window_queries, self.atoms_per_window_keys
        K = N // W
        to_keys = make_to_keys(N, W, H)

        atom_ref_pos_queries = atom_ref_pos.reshape(B, K, W, 1, 3)
        atom_ref_pos_keys = to_keys(atom_ref_pos).reshape(B, K, 1, H, 3)

        d = atom_ref_pos_keys - atom_ref_pos_queries  # (b k w h 3)
        d_norm = mx.sum(d * d, axis=-1, keepdims=True)
        d_norm = 1 / (1 + d_norm)  # AF3 feeds in the reciprocal of the distance norm

        atom_mask_queries = atom_mask.reshape(B, K, W, 1)
        atom_mask_keys = to_keys(atom_mask[..., None].astype(mx.float32)).reshape(B, K, 1, H).astype(mx.bool_)
        atom_uid_queries = atom_uid.reshape(B, K, W, 1)
        atom_uid_keys = to_keys(atom_uid[..., None].astype(mx.float32)).reshape(B, K, 1, H).astype(mx.int64)
        v = (atom_mask_queries & atom_mask_keys & (atom_uid_queries == atom_uid_keys)).astype(mx.float32)[..., None]

        p = self.embed_atompair_ref_pos(d) * v
        p = p + self.embed_atompair_ref_dist(d_norm) * v
        p = p + self.embed_atompair_mask(v) * v

        q = c

        if self.structure_prediction:
            # only in the structure model, not in the initial encoding
            atom_to_token = feats["atom_to_token"].astype(mx.float32)  # (b m n)

            s_to_c = self._seq(self.s_to_c_trans, s_trunk.astype(mx.float32))
            s_to_c = atom_to_token @ s_to_c
            c = c + s_to_c

            atom_to_token_queries = atom_to_token.reshape(B, K, W, atom_to_token.shape[-1])
            atom_to_token_keys = to_keys(atom_to_token)
            z_to_p = self._seq(self.z_to_p_trans, z.astype(mx.float32))
            # "bijd,bwki,bwlj->bwkld" done as two contractions
            tmp = mx.einsum("bwki,bijd->bwkjd", atom_to_token_queries, z_to_p)
            z_to_p = mx.einsum("bwkjd,bwlj->bwkld", tmp, atom_to_token_keys)
            p = p + z_to_p

        p = p + self._seq(self.c_to_p_trans_q, c.reshape(B, K, W, 1, c.shape[-1]))
        p = p + self._seq(self.c_to_p_trans_k, to_keys(c).reshape(B, K, 1, H, c.shape[-1]))
        p = p + self._seq(self.p_mlp, p)
        return q, c, p, to_keys


class AtomAttentionEncoder(nn.Module):
    def __init__(
        self,
        atom_s,
        token_s,
        atoms_per_window_queries,
        atoms_per_window_keys,
        atom_encoder_depth=3,
        atom_encoder_heads=4,
        structure_prediction=True,
        activation_checkpointing=False,  # ignored
        transformer_post_layer_norm=False,
    ) -> None:
        super().__init__()
        self.structure_prediction = structure_prediction
        if structure_prediction:
            self.r_to_q_trans = nn.Linear(3, atom_s, bias=False)

        self.atom_encoder = AtomTransformer(
            dim=atom_s,
            dim_single_cond=atom_s,
            attn_window_queries=atoms_per_window_queries,
            attn_window_keys=atoms_per_window_keys,
            depth=atom_encoder_depth,
            heads=atom_encoder_heads,
            post_layer_norm=transformer_post_layer_norm,
        )

        self.atom_to_token_trans = [
            nn.Linear(atom_s, 2 * token_s if structure_prediction else token_s, bias=False),
            nn.ReLU(),
        ]

    def nax_ok(self, feats, c) -> bool:
        """kernels_atom path (--precision f16 + BOLTZ_MLX_NAX_ATOM) for this input: one input (samples via
        multiplicity), structure-prediction encoder, supported shapes."""
        return (self.structure_prediction and c.shape[0] == 1 and feats["atom_to_token"].shape[0] == 1
                and self.atom_encoder.nax_ok(c.shape[1]))

    def prepare(self, feats, c, atom_enc_bias, to_keys, multiplicity=1):
        if self.nax_ok(feats, c):
            from boltz_mlx import kernels_atom as _KA

            P = self.atom_encoder.prepare_nax(c, atom_enc_bias, to_keys, feats["atom_pad_mask"], multiplicity)
            # atom -> token mean as a segment sum (host-side index build, once per prediction)
            off, idx, wt = _KA.token_segments(np.asarray(feats["atom_to_token"][0].astype(mx.float32)))
            P["seg"] = (mx.array(off), mx.array(idx), mx.array(wt))
            mx.eval(P["seg"])
            return P
        atom_mask = feats["atom_pad_mask"].astype(mx.bool_)
        if multiplicity != 1:
            c = mx.repeat(c, multiplicity, axis=0)
            atom_mask = mx.repeat(atom_mask, multiplicity, axis=0)
        return self.atom_encoder.prepare((*c.shape[:-1], 1), c, atom_enc_bias, to_keys, atom_mask, multiplicity)

    def _call_nax(self, q, c, r, to_keys, multiplicity, P):
        """kernels_atom path. q, c: [1, N, D] (per input); r: [multiplicity, N, 3]. Same arithmetic as the MLX path up
        to the kernels' rounding: q + r_to_q(r) broadcast over the samples (== repeat + add), the AtomTransformer on
        kernels_atom, relu + mean over each token's atoms as a segment sum (no dense one-hot matmul)."""
        from boltz_mlx import kernels_atom as _KA

        q = q + self.r_to_q_trans(r)
        q = self.atom_encoder(q=q, mask=None, c=None, bias=None, multiplicity=multiplicity, to_keys=None, prepared=P)
        a = _KA.segment_mean_relu(self.atom_to_token_trans[0](q), P["seg"])
        if multiplicity != 1:  # lazy: only materialised if a non-NAX decoder consumed it
            c = mx.repeat(c, multiplicity, axis=0)
        return a, q, c, to_keys

    def __call__(self, feats, q, c, atom_enc_bias, to_keys, r=None, multiplicity=1, prepared=None):
        if prepared is not None and prepared.get("nax_atom"):
            return self._call_nax(q, c, r, to_keys, multiplicity, prepared)
        atom_mask = feats["atom_pad_mask"].astype(mx.bool_)

        if self.structure_prediction:
            # only here the multiplicity kicks in because we use the different positions r
            if multiplicity != 1:
                q = mx.repeat(q, multiplicity, axis=0)
            q = q + self.r_to_q_trans(r)

        if multiplicity != 1:
            c = mx.repeat(c, multiplicity, axis=0)
            atom_mask = mx.repeat(atom_mask, multiplicity, axis=0)

        q = self.atom_encoder(
            q=q, mask=atom_mask, c=c, bias=atom_enc_bias, multiplicity=multiplicity, to_keys=to_keys, prepared=prepared
        )

        q_to_a = q
        for f in self.atom_to_token_trans:
            q_to_a = f(q_to_a)
        atom_to_token = feats["atom_to_token"].astype(mx.float32)
        if multiplicity != 1:
            atom_to_token = mx.repeat(atom_to_token, multiplicity, axis=0)
        atom_to_token_mean = atom_to_token / (atom_to_token.sum(axis=1, keepdims=True) + 1e-6)
        a = mx.swapaxes(atom_to_token_mean, 1, 2) @ q_to_a

        return a, q, c, to_keys


class AtomAttentionDecoder(nn.Module):
    """Algorithm 6."""

    def __init__(
        self,
        atom_s,
        token_s,
        attn_window_queries,
        attn_window_keys,
        atom_decoder_depth=3,
        atom_decoder_heads=4,
        activation_checkpointing=False,  # ignored
        transformer_post_layer_norm=False,
    ) -> None:
        super().__init__()
        self.a_to_q_trans = nn.Linear(2 * token_s, atom_s, bias=False)

        self.atom_decoder = AtomTransformer(
            dim=atom_s,
            dim_single_cond=atom_s,
            attn_window_queries=attn_window_queries,
            attn_window_keys=attn_window_keys,
            depth=atom_decoder_depth,
            heads=atom_decoder_heads,
            post_layer_norm=transformer_post_layer_norm,
        )

        self.transformer_post_layer_norm = transformer_post_layer_norm
        if transformer_post_layer_norm:
            self.atom_feat_to_atom_pos_update = nn.Linear(atom_s, 3, bias=False)
        else:
            self.atom_feat_to_atom_pos_update = [_FastLN(atom_s), nn.Linear(atom_s, 3, bias=False)]

    def prepare(self, feats, c, atom_dec_bias, to_keys, multiplicity=1):
        """c: the atom conditioning per input (repeated over the samples here for the MLX path)."""
        if c.shape[0] == 1 and feats["atom_to_token"].shape[0] == 1 and self.atom_decoder.nax_ok(c.shape[1]):
            from boltz_mlx import kernels_atom as _KA

            P = self.atom_decoder.prepare_nax(c, atom_dec_bias, to_keys, feats["atom_pad_mask"], multiplicity)
            tok, valid = _KA.atom_tokens(np.asarray(feats["atom_to_token"][0].astype(mx.float32)))
            P["tok"] = (mx.array(tok), mx.array(valid))
            mx.eval(P["tok"])
            return P
        if multiplicity != 1:
            c = mx.repeat(c, multiplicity, axis=0)
        atom_mask = feats["atom_pad_mask"]
        if multiplicity != 1:
            atom_mask = mx.repeat(atom_mask, multiplicity, axis=0)
        return self.atom_decoder.prepare((*c.shape[:-1], 1), c, atom_dec_bias, to_keys, atom_mask, multiplicity)

    def _call_nax(self, a, q, P):
        """kernels_atom path: token -> atom broadcast as an index gather (== the one-hot matmul, which only adds
        zeros), the AtomTransformer on kernels_atom, then the position update head."""
        tok, valid = P["tok"]
        x = mx.take(self.a_to_q_trans(a.astype(mx.float32)), tok, axis=1)
        q = q + mx.where(valid[None, :, None], x, mx.zeros((), x.dtype))
        q = self.atom_decoder(q=q, mask=None, c=None, bias=None, to_keys=None, prepared=P)
        if self.transformer_post_layer_norm:
            return self.atom_feat_to_atom_pos_update(q)
        for f in self.atom_feat_to_atom_pos_update:
            q = f(q)
        return q

    def __call__(self, a, q, c, atom_dec_bias, feats, to_keys, multiplicity=1, prepared=None):
        if prepared is not None and prepared.get("nax_atom"):
            return self._call_nax(a, q, prepared)
        atom_to_token = feats["atom_to_token"].astype(mx.float32)
        atom_mask = feats["atom_pad_mask"]
        if multiplicity != 1:
            atom_to_token = mx.repeat(atom_to_token, multiplicity, axis=0)
            atom_mask = mx.repeat(atom_mask, multiplicity, axis=0)

        a_to_q = atom_to_token @ self.a_to_q_trans(a.astype(mx.float32))
        q = q + a_to_q

        q = self.atom_decoder(
            q=q, mask=atom_mask, c=c, bias=atom_dec_bias, multiplicity=multiplicity, to_keys=to_keys, prepared=prepared
        )

        if self.transformer_post_layer_norm:
            return self.atom_feat_to_atom_pos_update(q)
        for f in self.atom_feat_to_atom_pos_update:
            q = f(q)
        return q
