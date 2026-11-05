"""MLX port of boltz/model/modules/diffusionv2.py: DiffusionModule score model and the
AtomDiffusion helpers (preconditioning + sampling schedule). Steering/potentials, the sampling
loop and training losses are not ported here. Also contains `weighted_rigid_align`
(upstream: boltz/model/loss/diffusionv2.py).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from boltz_mlx.modules.encodersv2 import AtomAttentionDecoder, AtomAttentionEncoder, SingleConditioning
from boltz_mlx.modules.transformersv2 import DiffusionTransformer
from boltz_mlx.layers.norm import LayerNorm as _FastLN


def _log(t: mx.array, eps: float = 1e-20) -> mx.array:
    return mx.log(mx.maximum(t, eps))


def _det3(m: mx.array) -> mx.array:
    """Determinant of (..., 3, 3) matrices."""
    a, b, c = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    d, e, f = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    g, h, i = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def weighted_rigid_align(true_coords, pred_coords, weights, mask):
    """Algorithm 28 (aligns true_coords onto pred_coords). Returns aligned true coords (b, n, 3).

    The upstream result is detached; callers here are inference only so no gradient handling.
    """
    weights = (mask.astype(mx.float32) * weights)[..., None]

    true_centroid = (true_coords * weights).sum(axis=-2, keepdims=True) / weights.sum(axis=-2, keepdims=True)
    pred_centroid = (pred_coords * weights).sum(axis=-2, keepdims=True) / weights.sum(axis=-2, keepdims=True)

    true_centered = true_coords - true_centroid
    pred_centered = pred_coords - pred_centroid

    # cov[..., i, j] = sum_n w * pred_centered[n, i] * true_centered[n, j]
    cov = mx.einsum("...ni,...nj->...ij", weights * pred_centered, true_centered)

    # SVD (CPU only in MLX); mx returns U, S, Vh whereas upstream names V = Vh^H
    U, S, Vh = mx.linalg.svd(cov.astype(mx.float32), stream=mx.cpu)

    rot = U @ Vh  # einsum("... i j, ... k j -> ... i k", U, V)
    det = _det3(rot)
    # F = diag(1, 1, det(rot))
    diag = mx.concatenate([mx.ones(rot.shape[:-2] + (2,)), det[..., None]], axis=-1)
    F = diag[..., :, None] * mx.eye(3)
    rot = U @ F @ Vh  # einsum(U, F, V, "... i j, ... j k, ... l k -> ... i l")

    # einsum(true_centered, rot, "... n i, ... j i -> ... n j")
    return mx.einsum("...ni,...ji->...nj", true_centered, rot) + pred_centroid


class DiffusionModule(nn.Module):
    """Diffusion (score) module."""

    def __init__(
        self,
        token_s: int,
        atom_s: int,
        atoms_per_window_queries: int = 32,
        atoms_per_window_keys: int = 128,
        sigma_data: int = 16,
        dim_fourier: int = 256,
        atom_encoder_depth: int = 3,
        atom_encoder_heads: int = 4,
        token_transformer_depth: int = 24,
        token_transformer_heads: int = 8,
        atom_decoder_depth: int = 3,
        atom_decoder_heads: int = 4,
        conditioning_transition_layers: int = 2,
        activation_checkpointing: bool = False,  # ignored
        transformer_post_ln: bool = False,
    ) -> None:
        super().__init__()
        self.atoms_per_window_queries = atoms_per_window_queries
        self.atoms_per_window_keys = atoms_per_window_keys
        self.sigma_data = sigma_data

        self.single_conditioner = SingleConditioning(
            sigma_data=sigma_data,
            token_s=token_s,
            dim_fourier=dim_fourier,
            num_transitions=conditioning_transition_layers,
        )

        self.atom_attention_encoder = AtomAttentionEncoder(
            atom_s=atom_s,
            token_s=token_s,
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys,
            atom_encoder_depth=atom_encoder_depth,
            atom_encoder_heads=atom_encoder_heads,
            structure_prediction=True,
            transformer_post_layer_norm=transformer_post_ln,
        )

        self.s_to_a_linear = [_FastLN(2 * token_s), nn.Linear(2 * token_s, 2 * token_s, bias=False)]

        self.token_transformer = DiffusionTransformer(
            dim=2 * token_s,
            dim_single_cond=2 * token_s,
            depth=token_transformer_depth,
            heads=token_transformer_heads,
        )

        self.a_norm = _FastLN(2 * token_s)

        self.atom_attention_decoder = AtomAttentionDecoder(
            atom_s=atom_s,
            token_s=token_s,
            attn_window_queries=atoms_per_window_queries,
            attn_window_keys=atoms_per_window_keys,
            atom_decoder_depth=atom_decoder_depth,
            atom_decoder_heads=atom_decoder_heads,
        )

    def __call__(
        self,
        s_inputs,  # (b, n, ts)
        s_trunk,  # (b, n, ts)
        r_noisy,  # (bm, m, 3)
        times,  # (bm,)
        feats,
        diffusion_conditioning,  # dict: q, c, to_keys, atom_enc_bias, atom_dec_bias, token_trans_bias
        multiplicity: int = 1,
    ):
        prep = diffusion_conditioning.get("prepared") or {}
        tok_cond = None
        s_a_in = None
        if multiplicity != 1 and diffusion_conditioning.get("shared_times", False) and s_trunk.shape[0] == 1:
            # all samples share the noise level: time conditioning (and the token transformer's
            # s-projections, ~1/3 of its FLOPs) are identical across samples -> compute once
            s1, _ = self.single_conditioner(times[:1], s_trunk, s_inputs)
            s = mx.repeat(s1, multiplicity, axis=0)
            if (prep.get("enc") or {}).get("nax_atom"):  # kernels_atom round: s_to_a on the shared rows, broadcast
                s_a_in = s1
            if prep.get("tok") is not None:
                tok_cond = self.token_transformer.precompute_cond_shared(s1, multiplicity)
        else:
            if multiplicity != 1:
                s_trunk = mx.repeat(s_trunk, multiplicity, axis=0)
                s_inputs = mx.repeat(s_inputs, multiplicity, axis=0)
            s, _normed_fourier = self.single_conditioner(times, s_trunk, s_inputs)

        # Sequence-local atom attention and aggregation to coarse-grained tokens
        a, q_skip, c_skip, to_keys = self.atom_attention_encoder(
            feats=feats,
            q=diffusion_conditioning["q"],
            c=diffusion_conditioning["c"],
            atom_enc_bias=diffusion_conditioning["atom_enc_bias"],
            to_keys=diffusion_conditioning["to_keys"],
            r=r_noisy,
            multiplicity=multiplicity,
            prepared=prep.get("enc"),
        )

        # Full self-attention on token level
        s_a = s if s_a_in is None else s_a_in
        for f in self.s_to_a_linear:
            s_a = f(s_a)
        a = a + s_a

        mask = feats["token_pad_mask"]
        if multiplicity != 1:
            mask = mx.repeat(mask, multiplicity, axis=0)
        a = self.token_transformer(
            a,
            mask=mask.astype(mx.float32),
            s=s,
            bias=diffusion_conditioning["token_trans_bias"],  # z not expanded with multiplicity until after bias
            multiplicity=multiplicity,
            prepared=prep.get("tok") if tok_cond is None else {**prep["tok"], "cond": tok_cond},
        )
        a = self.a_norm(a)

        # Broadcast token activations to atoms and run sequence-local atom attention
        return self.atom_attention_decoder(
            a=a,
            q=q_skip,
            c=c_skip,
            atom_dec_bias=diffusion_conditioning["atom_dec_bias"],
            feats=feats,
            multiplicity=multiplicity,
            to_keys=to_keys,
            prepared=prep.get("dec"),
        )

    def prepare(self, feats, diffusion_conditioning, multiplicity=1):
        """Step-invariant tensors of the diffusion loop (attention biases, atom-level conditioning)."""
        dc = diffusion_conditioning
        c = dc["c"]
        mask = feats["token_pad_mask"]
        if multiplicity != 1:
            mask = mx.repeat(mask, multiplicity, axis=0)
        return {
            "enc": self.atom_attention_encoder.prepare(feats, c, dc["atom_enc_bias"], dc["to_keys"], multiplicity),
            "dec": self.atom_attention_decoder.prepare(feats, c, dc["atom_dec_bias"], dc["to_keys"], multiplicity),
            "tok": self.token_transformer.prepare(
                None, dc["token_trans_bias"], mask.astype(mx.float32), None, multiplicity, with_cond=False
            ),
        }


class AtomDiffusion(nn.Module):
    """Preconditioning / schedule parts of upstream AtomDiffusion (no sampling loop, no steering)."""

    def __init__(
        self,
        score_model_args,
        num_sampling_steps: int = 5,
        sigma_min: float = 0.0004,
        sigma_max: float = 160.0,
        sigma_data: float = 16.0,
        rho: float = 7,
        P_mean: float = -1.2,
        P_std: float = 1.5,
        gamma_0: float = 0.8,
        gamma_min: float = 1.0,
        noise_scale: float = 1.003,
        step_scale: float = 1.5,
        step_scale_random: list | None = None,
        coordinate_augmentation: bool = True,
        coordinate_augmentation_inference=None,
        alignment_reverse_diff: bool = False,
        synchronize_sigmas: bool = False,
        **_ignored,
    ) -> None:
        super().__init__()
        self.score_model = DiffusionModule(**score_model_args)

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.P_mean = P_mean
        self.P_std = P_std
        self.num_sampling_steps = num_sampling_steps
        self.gamma_0 = gamma_0
        self.gamma_min = gamma_min
        self.noise_scale = noise_scale
        self.step_scale = step_scale
        self.step_scale_random = step_scale_random
        self.coordinate_augmentation = coordinate_augmentation
        self.coordinate_augmentation_inference = (
            coordinate_augmentation_inference
            if coordinate_augmentation_inference is not None
            else coordinate_augmentation
        )
        self.alignment_reverse_diff = alignment_reverse_diff
        self.synchronize_sigmas = synchronize_sigmas
        self.token_s = score_model_args["token_s"]

    def c_skip(self, sigma):
        return (self.sigma_data**2) / (sigma**2 + self.sigma_data**2)

    def c_out(self, sigma):
        return sigma * self.sigma_data / mx.sqrt(self.sigma_data**2 + sigma**2)

    def c_in(self, sigma):
        return 1 / mx.sqrt(sigma**2 + self.sigma_data**2)

    def c_noise(self, sigma):
        return _log(sigma / self.sigma_data) * 0.25

    def preconditioned_network_forward(
        self,
        noised_atom_coords,  # (b, m, 3)
        sigma,  # (b,) array or float
        network_condition_kwargs: dict,
    ):
        batch = noised_atom_coords.shape[0]

        if isinstance(sigma, (float, int)):
            sigma = mx.full((batch,), float(sigma), dtype=mx.float32)

        padded_sigma = sigma[:, None, None]

        r_update = self.score_model(
            r_noisy=self.c_in(padded_sigma) * noised_atom_coords,
            times=self.c_noise(sigma),
            **network_condition_kwargs,
        )

        return self.c_skip(padded_sigma) * noised_atom_coords + self.c_out(padded_sigma) * r_update

    def sample_schedule(self, num_sampling_steps: int | None = None):
        num_sampling_steps = self.num_sampling_steps if num_sampling_steps is None else num_sampling_steps
        inv_rho = 1 / self.rho

        steps = mx.arange(num_sampling_steps, dtype=mx.float32)
        sigmas = (
            self.sigma_max**inv_rho
            + steps / (num_sampling_steps - 1) * (self.sigma_min**inv_rho - self.sigma_max**inv_rho)
        ) ** self.rho

        sigmas = sigmas * self.sigma_data

        return mx.pad(sigmas, (0, 1), constant_values=0.0)  # last step is sigma value of 0.
