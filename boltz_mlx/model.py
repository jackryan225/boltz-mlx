"""Boltz-2 top level (boltz/model/models/boltz2.py::Boltz2.forward, inference path)."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass, field

import mlx.core as mx
import mlx.nn as nn

from boltz_mlx import kernels as _K

from boltz_mlx.layers.pairformer import PairformerModule
from boltz_mlx.modules.affinity import AffinityModule, affinity_ensemble, best_sample_index, mask_z_for_affinity
from boltz_mlx.modules.confidencev2 import ConfidenceModule
from boltz_mlx.modules.diffusion_conditioning import DiffusionConditioning
from boltz_mlx.modules.diffusionv2 import AtomDiffusion
from boltz_mlx.modules.encodersv2 import RelativePositionEncoder
from boltz_mlx.modules.trunkv2 import BFactorModule, DistogramModule, MSAModule, TemplateV2Module
from boltz_mlx.modules.trunkv2_embed import ContactConditioning, InputEmbedder
from boltz_mlx.sampler import DiffusionParams, MLXNoise, NoiseSource, SamplerConfig, sample, weighted_rigid_align
from boltz_mlx.weights import WEIGHTS
from boltz_mlx.layers.norm import LayerNorm as _FastLN

NUM_BOND_TYPES = 6  # len(boltz.data.const.bond_types)


@dataclass
class PredictArgs:
    """Per-pass prediction settings. `diffusion` carries the sampler options (rho, step_scale, solver); the
    structure and affinity passes each get their own PredictArgs, so they can use different samplers."""

    recycling_steps: int = 3
    sampling_steps: int = 200
    diffusion_samples: int = 1
    diffusion: DiffusionParams = field(default_factory=DiffusionParams)

    @classmethod
    def from_sampler(cls, cfg: SamplerConfig | None, recycling_steps: int, diffusion_samples: int,
                     sampling_steps: int = 200) -> "PredictArgs":
        """PredictArgs for a sampler config (None = upstream sampler with `sampling_steps`)."""
        if cfg is None:
            return cls(recycling_steps, sampling_steps, diffusion_samples)
        return cls(recycling_steps, cfg.steps, diffusion_samples, cfg.params())


class Boltz2(nn.Module):
    def __init__(self, hp: dict, confidence: bool = True, affinity: bool = False, compile_diffusion: bool = True) -> None:
        super().__init__()
        self.compile_diffusion = compile_diffusion
        token_s, token_z = hp["token_s"], hp["token_z"]
        atom_s, atom_z = hp["atom_s"], hp["atom_z"]
        W, H = hp["atoms_per_window_queries"], hp["atoms_per_window_keys"]
        sm = hp["score_model_args"]
        self.hp = hp

        self.input_embedder = InputEmbedder(
            atom_s=atom_s,
            atom_z=atom_z,
            token_s=token_s,
            token_z=token_z,
            atoms_per_window_queries=W,
            atoms_per_window_keys=H,
            atom_feature_dim=hp["atom_feature_dim"],
            **hp["embedder_args"],
        )
        self.s_init = nn.Linear(token_s, token_s, bias=False)
        self.z_init_1 = nn.Linear(token_s, token_z, bias=False)
        self.z_init_2 = nn.Linear(token_s, token_z, bias=False)
        self.rel_pos = RelativePositionEncoder(
            token_z, fix_sym_check=hp["fix_sym_check"], cyclic_pos_enc=hp["cyclic_pos_enc"]
        )
        self.token_bonds = nn.Linear(1, token_z, bias=False)
        self.token_bonds_type = nn.Embedding(NUM_BOND_TYPES + 1, token_z)
        self.contact_conditioning = ContactConditioning(
            token_z, hp["conditioning_cutoff_min"], hp["conditioning_cutoff_max"]
        )
        self.s_norm = _FastLN(token_s)
        self.z_norm = _FastLN(token_z)
        self.s_recycle = nn.Linear(token_s, token_s, bias=False)
        self.z_recycle = nn.Linear(token_z, token_z, bias=False)

        self.use_templates = hp.get("use_templates", False)
        if self.use_templates:
            assert hp.get("use_templates_v2", False)
            ta = hp["template_args"]
            self.template_module = TemplateV2Module(
                token_z, template_dim=ta["template_dim"], template_blocks=ta["template_blocks"]
            )
        # predict-time MSAModuleArgs / PairformerArgsV2 (boltz.main), dropout off
        self.msa_module = MSAModule(msa_s=64, token_z=token_z, token_s=token_s, msa_blocks=4, msa_dropout=0.0, z_dropout=0.0)
        self.pairformer_module = PairformerModule(token_s, token_z, num_blocks=64, num_heads=16)
        self.diffusion_conditioning = DiffusionConditioning(
            token_s=token_s,
            token_z=token_z,
            atom_s=atom_s,
            atom_z=atom_z,
            atoms_per_window_queries=W,
            atoms_per_window_keys=H,
            atom_encoder_depth=sm["atom_encoder_depth"],
            atom_encoder_heads=sm["atom_encoder_heads"],
            token_transformer_depth=sm["token_transformer_depth"],
            token_transformer_heads=sm["token_transformer_heads"],
            atom_decoder_depth=sm["atom_decoder_depth"],
            atom_decoder_heads=sm["atom_decoder_heads"],
            atom_feature_dim=hp["atom_feature_dim"],
            conditioning_transition_layers=sm["conditioning_transition_layers"],
        )
        dp = DiffusionParams()
        self.structure_module = AtomDiffusion(
            score_model_args={
                "token_s": token_s,
                "atom_s": atom_s,
                "atoms_per_window_queries": W,
                "atoms_per_window_keys": H,
                **{k: v for k, v in sm.items() if k != "activation_checkpointing"},
            },
            sigma_min=dp.sigma_min,
            sigma_max=dp.sigma_max,
            sigma_data=dp.sigma_data,
            rho=dp.rho,
            gamma_0=dp.gamma_0,
            gamma_min=dp.gamma_min,
            noise_scale=dp.noise_scale,
            step_scale=dp.step_scale,
            alignment_reverse_diff=True,
        )
        self.distogram_module = DistogramModule(token_z, hp["num_bins"])
        self.predict_bfactor = hp.get("predict_bfactor", False)
        if self.predict_bfactor:
            self.bfactor_module = BFactorModule(token_s, hp["num_bins"])

        self.confidence = confidence and hp.get("confidence_prediction", False)
        if self.confidence:
            self.confidence_module = ConfidenceModule(
                token_s,
                token_z,
                token_level_confidence=hp["token_level_confidence"],
                bond_type_feature=hp["bond_type_feature"],
                fix_sym_check=hp["fix_sym_check"],
                cyclic_pos_enc=hp["cyclic_pos_enc"],
                conditioning_cutoff_min=hp["conditioning_cutoff_min"],
                conditioning_cutoff_max=hp["conditioning_cutoff_max"],
                **hp["confidence_model_args"],
            )
        self.affinity = affinity and hp.get("affinity_prediction", False)
        if self.affinity:
            assert hp.get("affinity_ensemble", False), "only the released ensemble affinity checkpoint is supported"
            self.affinity_module1 = AffinityModule(token_s, token_z, **hp["affinity_model_args1"])
            self.affinity_module2 = AffinityModule(token_s, token_z, **hp["affinity_model_args2"])

    # ------------------------------------------------------------------ trunk
    def embed(self, feats):
        _K.set_phase("embed")
        s_inputs = self.input_embedder(feats)
        s_init = self.s_init(s_inputs)
        z_init = self.z_init_1(s_inputs)[:, :, None] + self.z_init_2(s_inputs)[:, None, :]
        rpe = self.rel_pos(feats)
        z_init = z_init + rpe
        z_init = z_init + self.token_bonds(feats["token_bonds"].astype(mx.float32))
        z_init = z_init + self.token_bonds_type(feats["type_bonds"])
        z_init = z_init + self.contact_conditioning(feats)
        return s_inputs, s_init, z_init, rpe

    def trunk(self, feats, recycling_steps: int, eval_each: bool = True):
        s_inputs, s_init, z_init, rpe = self.embed(feats)
        s = mx.zeros_like(s_init)
        z = mx.zeros_like(z_init)
        mask = feats["token_pad_mask"].astype(mx.float32)
        pair_mask = mask[:, :, None] * mask[:, None, :]
        # With no templates the template module returns exactly 0 (masked sum, bias-free u_proj): skip it.
        use_templates = self.use_templates and bool(mx.any(feats["template_mask"] != 0).item())
        self.fp16_fallbacks = 0
        for _ in range(recycling_steps + 1):
            s_prev, z_prev = s, z
            s, z = self._recycle(s, z, s_init, z_init, s_inputs, feats, mask, pair_mask, use_templates)
            if _K.f16() and _K.GUARDS:
                _K.guard_residual(s, z)
                g = _K.guard_pending()
                mx.eval(*([g] if g is not None else []), s, z)
                if _K.guard_take() > 0:  # fp16 overflow somewhere in this recycle: redo it with fp32 GEMMs
                    self.fp16_fallbacks += 1
                    print(f"boltz_mlx: fp16 overflow guard hit; recycle recomputed in fp32 "
                          f"(fallback {self.fp16_fallbacks})", file=sys.stderr)
                    _K.PRECISION = "fp32"
                    try:
                        s, z = self._recycle(s_prev, z_prev, s_init, z_init, s_inputs, feats, mask, pair_mask,
                                             use_templates)
                        mx.eval(s, z)
                    finally:
                        _K.PRECISION = "f16"
            elif eval_each:
                mx.eval(s, z)
        return s_inputs, s, z, rpe

    def _recycle(self, s, z, s_init, z_init, s_inputs, feats, mask, pair_mask, use_templates):
        s = s_init + self.s_recycle(self.s_norm(s))
        z = z_init + self.z_recycle(self.z_norm(z))
        if use_templates:
            z = z + self.template_module(z, feats, pair_mask)
        _K.set_phase("trunk.msa")
        z = z + self.msa_module(z, s_inputs, feats)
        _K.set_phase("trunk.pairformer")
        s, z = self.pairformer_module(s, z, mask, pair_mask)
        _K.record("trunk.z_residual", z)
        _K.record("trunk.s_residual", s)
        return s, z

    # -------------------------------------------------------------- structure
    def structure(self, s_inputs, s, z, rpe, feats, args: PredictArgs, noise: NoiseSource):
        _K.set_phase("diffusion")
        q, c, to_keys, enc_bias, dec_bias, tt_bias = self.diffusion_conditioning(s, z, rpe, feats)
        cond = {
            "q": q,
            "c": c,
            "to_keys": to_keys,
            "atom_enc_bias": enc_bias,
            "atom_dec_bias": dec_bias,
            "token_trans_bias": tt_bias,
        }
        mult = args.diffusion_samples
        sm = self.structure_module
        cond["prepared"] = sm.score_model.prepare(feats, cond, mult)
        cond["shared_times"] = True  # the sampler uses one noise level for all samples
        mx.eval(q, c, enc_bias, dec_bias, tt_bias, cond["prepared"])

        def _denoise(x, sigma):
            _K.GUARD_SUSPEND += 1  # may be traced by mx.compile: no guard side effects in here
            try:
                return sm.preconditioned_network_forward(
                    x,
                    sigma,
                    dict(
                        s_inputs=s_inputs,
                        s_trunk=s,
                        feats=feats,
                        diffusion_conditioning=cond,
                        multiplicity=mult,
                    ),
                )
            finally:
                _K.GUARD_SUSPEND -= 1

        # One compiled graph per prediction (conditioning arrays are captured as constants).
        compiled = mx.compile(_denoise) if self.compile_diffusion else _denoise

        def denoise(x, t_hat):
            return compiled(x, mx.full((x.shape[0],), t_hat, dtype=mx.float32))

        return sample(
            denoise,
            feats["atom_pad_mask"].astype(mx.float32),
            args.sampling_steps,
            mult,
            noise,
            weighted_rigid_align,
            args.diffusion,
        )

    def __call__(
        self,
        feats,
        args: PredictArgs = PredictArgs(),
        noise: NoiseSource | None = None,
        timings: dict | None = None,
        mw_correction: bool = False,
    ):
        noise = noise or MLXNoise(0)
        if _K.f16():
            _K.nax_status()  # cached; detection before the (possibly compiled) diffusion step builds any kernel
        t0 = time.perf_counter()
        s_inputs, s, z, rpe = self.trunk(feats, args.recycling_steps)
        pdistogram = self.distogram_module(z)
        mx.eval(pdistogram)
        t1 = time.perf_counter()
        coords = self.structure(s_inputs, s, z, rpe, feats, args, noise)
        mx.eval(coords)
        t2 = time.perf_counter()
        out = {
            "s_inputs": s_inputs,
            "s": s,
            "z": z,
            "pdistogram": pdistogram,
            "sample_atom_coords": coords,
        }
        if self.predict_bfactor:
            out["pbfactor"] = self.bfactor_module(s)
        if self.confidence:
            _K.set_phase("confidence")
            conf = self.confidence_module(
                s_inputs=s_inputs,
                s=s,
                z=z,
                x_pred=coords,
                feats=feats,
                pred_distogram_logits=pdistogram[:, :, :, 0],
                multiplicity=args.diffusion_samples,
                run_sequentially=True,
            )
            mx.eval([v for k, v in conf.items() if k != "pair_chains_iptm"])
            out.update(conf)
        t3 = time.perf_counter()
        if self.affinity:
            z_aff = mask_z_for_affinity(z, feats)
            best = best_sample_index(out["iptm"])
            x_aff = coords[best][None, None]
            s_inputs_aff = self.input_embedder(feats, affinity=True)
            _K.set_phase("affinity")
            o1 = self.affinity_module1(s_inputs=s_inputs_aff, z=z_aff, x_pred=x_aff, feats=feats, multiplicity=1)
            o2 = self.affinity_module2(s_inputs=s_inputs_aff, z=z_aff, x_pred=x_aff, feats=feats, multiplicity=1)
            out.update(affinity_ensemble(o1, o2, feats, mw_correction=mw_correction))
            mx.eval(out["affinity_pred_value"])
        if _K.f16():  # trunk fallbacks + any count left from confidence / affinity (validation: must be 0)
            out["fp16_fallbacks"] = getattr(self, "fp16_fallbacks", 0)
            _K.guard_residual(*[v for k, v in out.items() if isinstance(v, mx.array) and v.dtype == mx.float32])
            out["fp16_guard_post_trunk"] = _K.guard_take()
            out["fp16_nonfinite_coords"] = int(mx.sum(~mx.isfinite(out["sample_atom_coords"])).item())
            # validation rows (run_l3): any nonzero count is a hard failure (a fallback = a trunk guard hit)
            out["guard_counts"] = {"trunk_fallbacks": out["fp16_fallbacks"], "post_trunk": out["fp16_guard_post_trunk"],
                                   "nonfinite_coords": out["fp16_nonfinite_coords"]}
        t4 = time.perf_counter()
        if timings is not None:
            timings.update(trunk_s=t1 - t0, diffusion_s=t2 - t1, confidence_s=t3 - t2, affinity_s=t4 - t3)
        return out


def load_model(name: str = "conf", weights: str = "auto", **kw) -> Boltz2:
    """weights: "fp32" (boltz2_<name>.safetensors), "mixed" (boltz2_<name>_mixed.safetensors, boltz_mlx/mixed.py) or
    "auto": mixed in f16 mode (fp16 trunk + diffusion GEMMs) when the file exists and matches this build, else fp32
    (f16 mode then casts its fp16 weights at first use; one warning line). strict / fast / turbo always load fp32.
    The checkpoint dict is not cached (an lru-cached dict kept a deleted model's weights resident)."""
    from boltz_mlx import mixed as _MX

    if _K.f16():
        _K.nax_status()  # capability detection (one tiny GPU probe on NAX candidates) here, never inside a compile trace
    hp = json.loads((WEIGHTS / f"boltz2_{name}.json").read_text())
    if name == "aff":
        kw.setdefault("affinity", True)
    model = Boltz2(hp, **kw)
    tops = set(model.children().keys())
    entry = None
    if weights == "mixed" or (weights == "auto" and _MX.want_mixed()):
        path = _MX.mixed_path(name)
        try:
            entry = _MX.load_manifest()[name]
            if not path.exists():
                raise FileNotFoundError(f"{path} not found (tools/make_mixed_checkpoint.py writes it)")
            w = _MX.load_mixed(name, path, entry, tops)
        except (FileNotFoundError, KeyError, ValueError) as e:
            if weights == "mixed":
                raise
            _MX.warn(f"f16 mode without the mixed checkpoint ({e}); using the fp32 checkpoint + runtime fp16 casts")
            entry = None
    elif weights not in ("auto", "fp32"):
        raise ValueError(f"weights must be auto | fp32 | mixed, not {weights!r}")
    if entry is None:
        w = [(k, v) for k, v in mx.load(str(WEIGHTS / f"boltz2_{name}.safetensors")).items() if k.split(".")[0] in tops]
    model.load_weights(w, strict=True)
    del w
    if entry is not None:
        _MX.prefuse(model, entry)
    mx.eval(model.parameters())
    model.weights_kind = "mixed" if entry is not None else "fp32"
    return model
