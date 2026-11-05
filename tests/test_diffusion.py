"""Parity tests: DiffusionConditioning, DiffusionModule (score model) and its children,
AtomDiffusion preconditioning/schedule, weighted_rigid_align."""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from boltz_mlx.compare import assert_close
from boltz_mlx.modules.diffusion_conditioning import DiffusionConditioning
from boltz_mlx.modules.diffusionv2 import AtomDiffusion, DiffusionModule, weighted_rigid_align
from boltz_mlx.modules.encodersv2 import make_to_keys
from boltz_mlx.modules.transformersv2 import DiffusionTransformer
from boltz_mlx.weights import load_module
import mlx.nn as nn
from boltz_cfg import DIFFUSION_COND_ARGS, H, SCORE_MODEL_ARGS, TOKEN_S, W, arg
from conftest import load_dump, load_feats, mxa

SM = "structure_module.score_model"

# Composite modules (24-layer transformer / full score model) use the standard single-module gate:
# measured max_rel is ~2e-6 .. 4e-6 (see test output), i.e. no loosening is needed.
COMPOSITE_REL = 1e-4


@pytest.fixture(scope="module")
def feats():
    return load_feats()


@pytest.fixture(scope="module")
def score_model():
    return load_module(DiffusionModule(**SCORE_MODEL_ARGS), SM)


def test_diffusion_conditioning(feats):
    d = load_dump("diffusion_conditioning")
    mod = load_module(DiffusionConditioning(**DIFFUSION_COND_ARGS), "diffusion_conditioning")
    q, c, to_keys, enc_bias, dec_bias, tok_bias = mod(
        s_trunk=mxa(d["kw_s_trunk"]),
        z_trunk=mxa(d["kw_z_trunk"]),
        relative_position_encoding=mxa(d["kw_relative_position_encoding"]),
        feats=feats,
    )
    assert_close(q, d["out_0"], name="diffusion_conditioning.q")
    assert_close(c, d["out_1"], name="diffusion_conditioning.c")
    assert_close(enc_bias, d["out_3"], name="diffusion_conditioning.atom_enc_bias")
    assert_close(dec_bias, d["out_4"], name="diffusion_conditioning.atom_dec_bias")
    assert_close(tok_bias, d["out_5"], name="diffusion_conditioning.token_trans_bias")
    assert to_keys(q).shape == (1, q.shape[1] // W, H, q.shape[-1])


def _score_kwargs(d, feats):
    n_atoms = d["kw_r_noisy"].shape[1]
    return dict(
        s_inputs=mxa(d["kw_s_inputs"]),
        s_trunk=mxa(d["kw_s_trunk"]),
        feats=feats,
        diffusion_conditioning=dict(
            q=mxa(d["kw_diffusion_conditioning_q"]),
            c=mxa(d["kw_diffusion_conditioning_c"]),
            atom_enc_bias=mxa(d["kw_diffusion_conditioning_atom_enc_bias"]),
            atom_dec_bias=mxa(d["kw_diffusion_conditioning_atom_dec_bias"]),
            token_trans_bias=mxa(d["kw_diffusion_conditioning_token_trans_bias"]),
            to_keys=make_to_keys(n_atoms, W, H),
        ),
        multiplicity=int(d["kw_multiplicity"]),
    )


def test_score_model(score_model, feats):
    d = load_dump(SM)
    out = score_model(r_noisy=mxa(d["kw_r_noisy"]), times=mxa(d["kw_times"]), **_score_kwargs(d, feats))
    assert_close(out, d["out"], rel=COMPOSITE_REL, name="score_model")


def test_score_model_children_token_transformer():
    d = load_dump(f"{SM}.token_transformer")
    mod = load_module(
        DiffusionTransformer(dim=2 * TOKEN_S, dim_single_cond=2 * TOKEN_S, depth=24, heads=16),
        f"{SM}.token_transformer",
    )
    out = mod(
        mxa(d["in_0"]),
        mask=mxa(d["kw_mask"]),
        s=mxa(d["kw_s"]),
        bias=mxa(d["kw_bias"]),
        multiplicity=int(d["kw_multiplicity"]),
    )
    assert_close(out, d["out"], rel=COMPOSITE_REL, name="score_model.token_transformer")


def test_score_model_children_s_to_a_linear_and_a_norm(score_model):
    d = load_dump(f"{SM}.s_to_a_linear")
    x = mxa(d["in_0"])
    # score_model weights were loaded strictly with torch names (s_to_a_linear.{0,1}.*)
    for f in score_model.s_to_a_linear:
        x = f(x)
    assert_close(x, d["out"], name="score_model.s_to_a_linear")

    d = load_dump(f"{SM}.a_norm")
    assert_close(score_model.a_norm(mxa(d["in_0"])), d["out"], name="score_model.a_norm")


def test_score_model_children_s_to_a_linear_standalone():
    """Same as above but loading with the exact torch prefix into a standalone list module."""
    d = load_dump(f"{SM}.s_to_a_linear")

    class Seq(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [nn.LayerNorm(2 * TOKEN_S), nn.Linear(2 * TOKEN_S, 2 * TOKEN_S, bias=False)]

    m = Seq()
    from boltz_mlx.weights import subtree

    w = subtree(f"{SM}.s_to_a_linear")
    m.load_weights([(f"layers.{k}", v) for k, v in w.items()], strict=True)
    x = mxa(d["in_0"])
    for f in m.layers:
        x = f(x)
    assert_close(x, d["out"], name="score_model.s_to_a_linear(standalone)")


def test_tied_output_projection_weights_identical():
    from boltz_mlx.weights import subtree

    w = subtree(f"{SM}.token_transformer.layers.0")
    assert np.array_equal(np.array(w["output_projection_linear.weight"]), np.array(w["output_projection.0.weight"]))
    assert np.array_equal(np.array(w["output_projection_linear.bias"]), np.array(w["output_projection.0.bias"]))


def test_atom_diffusion_loads_and_preconditioned_forward(feats):
    """AtomDiffusion(score_model.*) loads under `structure_module` and the preconditioned
    forward equals c_skip*x + c_out*score_model(c_in*x, c_noise(sigma)) built from the dump."""
    d = load_dump(SM)
    ad = load_module(AtomDiffusion(score_model_args=SCORE_MODEL_ARGS), "structure_module")
    times = mxa(d["kw_times"])
    sigma = 16.0 * mx.exp(4.0 * times)  # invert c_noise
    padded = sigma[:, None, None]
    x = mxa(d["kw_r_noisy"]) / ad.c_in(padded)  # so that c_in(sigma) * x == dumped r_noisy
    kw = _score_kwargs(d, feats)
    out = ad.preconditioned_network_forward(x, sigma, kw)
    expected = ad.c_skip(padded) * x + ad.c_out(padded) * mxa(d["out"])
    # sigma inversion costs ~1 ulp in times; amplified by the network's sensitivity
    assert_close(out, expected, rel=1e-4, name="preconditioned_network_forward")


def test_preconditioning_helpers_vs_torch():
    torch = pytest.importorskip("torch")
    from boltz.model.modules.diffusionv2 import AtomDiffusion as TAD

    fake = SimpleNamespace(sigma_data=16.0)
    sig = np.array([0.0004, 0.3, 4.0, 16.0, 160.0, 2560.0], dtype=np.float32)
    ad = AtomDiffusion.__new__(AtomDiffusion)
    ad.sigma_data = 16.0
    s = mx.array(sig)
    ts = torch.from_numpy(sig)
    for name in ("c_skip", "c_out", "c_in", "c_noise"):
        ref = getattr(TAD, name)(fake, ts).numpy()
        out = getattr(ad, name)(s)
        assert_close(out, ref, rel=1e-6, name=f"AtomDiffusion.{name}")


@pytest.mark.parametrize("steps,sigma_min", [(200, 0.0001), (200, 0.0004), (5, 0.0004), (2, 0.0004)])
def test_sample_schedule_vs_torch(steps, sigma_min):
    torch = pytest.importorskip("torch")
    from boltz.model.modules.diffusionv2 import AtomDiffusion as TAD

    fake = SimpleNamespace(
        num_sampling_steps=steps, rho=7, sigma_max=160.0, sigma_min=sigma_min, sigma_data=16.0, device="cpu"
    )
    ref = TAD.sample_schedule(fake).numpy()
    ad = AtomDiffusion.__new__(AtomDiffusion)
    for k, v in vars(fake).items():
        setattr(ad, k, v)
    out = ad.sample_schedule()
    assert out.shape == ref.shape == (steps + 1,)
    assert_close(out, ref, rel=1e-6, name=f"sample_schedule(steps={steps},smin={sigma_min})")


def test_weighted_rigid_align_vs_torch():
    torch = pytest.importorskip("torch")
    from boltz.model.loss.diffusionv2 import weighted_rigid_align as twra

    rng = np.random.default_rng(0)
    B, N = 3, 60
    true = rng.normal(size=(B, N, 3)).astype(np.float32) * 10
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    # include one batch with a reflection-prone (noisy) target
    pred = (true @ q.astype(np.float32) + rng.normal(size=(B, N, 3)).astype(np.float32) * 0.5 + 3.0)
    pred[2] = rng.normal(size=(N, 3)).astype(np.float32) * 10  # unrelated cloud
    weights = rng.uniform(0.5, 2.0, size=(B, N)).astype(np.float32)
    mask = (rng.uniform(size=(B, N)) > 0.2).astype(np.float32)
    ref = twra(torch.from_numpy(true), torch.from_numpy(pred), torch.from_numpy(weights), torch.from_numpy(mask)).numpy()
    out = weighted_rigid_align(mx.array(true), mx.array(pred), mx.array(weights), mx.array(mask))
    assert_close(out, ref, rel=1e-5, name="weighted_rigid_align")
