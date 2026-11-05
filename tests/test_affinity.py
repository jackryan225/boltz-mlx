"""Parity tests: AffinityModule (ensemble members 1 and 2), AffinityHeadsTransformer, affinity_ensemble.

Reference: data/lig_aff/ref_aff (torch, CPU, boltz2_aff ckpt; 213 tokens, 13 affinity ligand tokens).
Weights prefixes `affinity_module1.` (8 pairformer blocks) / `affinity_module2.` (4 blocks) in boltz2_aff.safetensors.
Only the aff checkpoint is loaded in this process (never together with the conf checkpoint).
"""

import ast
import json

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from boltz_mlx.compare import assert_close
from boltz_mlx.modules.affinity import (
    AffinityHeadsTransformer,
    AffinityModule,
    affinity_cross_pair_mask,
    affinity_ensemble,
    best_sample_index,
    mask_z_for_affinity,
)
from boltz_mlx.modules.encodersv2 import PairwiseConditioning
from boltz_mlx.weights import WEIGHTS, load_module, subtree
from conftest import DATA, load_dump, load_feats, mxa

REL, COS = 1e-4, 0.99999
CASE = "lig_aff"
REF = "ref_aff"
HEAD_ATOL = 1e-5


def _as_dict(v) -> dict:
    return ast.literal_eval(v) if isinstance(v, str) else v


def _hparams() -> dict:
    return json.load(open(WEIGHTS / "boltz2_aff.json"))


def build_aff(idx: int) -> AffinityModule:
    h = _hparams()
    mod = AffinityModule(h["token_s"], h["token_z"], **_as_dict(h[f"affinity_model_args{idx}"]))
    load_module(mod, f"affinity_module{idx}", name="aff")
    return mod


def _dump(name):
    return load_dump(name, CASE, REF)


@pytest.fixture(scope="module")
def feats():
    return load_feats(CASE, affinity=True)


@pytest.fixture(scope="module")
def aff1():
    return build_aff(1)


def _ev(*xs):
    mx.eval(*xs)
    return xs if len(xs) > 1 else xs[0]


def _scalar(name, got, ref, atol=HEAD_ATOL):
    m = assert_close(got, ref, rel=1.0, cos=0.0, name=name)
    assert m["max_abs"] <= atol, (name, m)
    return m


def test_boundaries_in_checkpoint_equal_torch_linspace():
    torch = pytest.importorskip("torch")
    for i in (1, 2):
        b = np.array(subtree(f"affinity_module{i}", "aff")["boundaries"])
        assert np.array_equal(b, torch.linspace(2, 22, 63).numpy())


# ---------------------------------------------------------------------------------------
# children
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "child,make",
    [
        ("z_norm", lambda: nn.LayerNorm(128)),
        ("z_linear", lambda: nn.Linear(128, 128, bias=False)),
        ("dist_bin_pairwise_embed", lambda: nn.Embedding(64, 128)),
    ],
)
def test_small_child(child, make):
    name = f"affinity_module1.{child}"
    d = _dump(name)
    mod = load_module(make(), name, name="aff")
    out = _ev(mod(mxa(d["in_0"])))
    assert_close(out, d["out"], rel=1e-5, cos=0.999999, name=name)


def test_pairwise_conditioner():
    name = "affinity_module1.pairwise_conditioner"
    d = _dump(name)
    mod = load_module(PairwiseConditioning(128, 128, num_transitions=2), name, name="aff")
    out = _ev(mod(z_trunk=mxa(d["kw_z_trunk"]), token_rel_pos_feats=mxa(d["kw_token_rel_pos_feats"])))
    assert_close(out, d["out"], rel=REL, cos=COS, name=name)


def test_pairformer_stack(aff1):
    name = "affinity_module1.pairformer_stack"
    d = _dump(name)
    out = _ev(aff1.pairformer_stack(mxa(d["in_0"]), pair_mask=mxa(d["kw_pair_mask"])))
    assert_close(out, d["out"], rel=REL, cos=COS, name=name)


def test_affinity_heads(aff1, feats):
    name = "affinity_module1.affinity_heads"
    d = _dump(name)
    out = aff1.affinity_heads(z=mxa(d["kw_z"]), feats=feats, multiplicity=int(d["kw_multiplicity"]))
    _ev(*out.values())
    _scalar(f"{name}:affinity_pred_value", out["affinity_pred_value"], d["out_affinity_pred_value"])
    _scalar(f"{name}:affinity_logits_binary", out["affinity_logits_binary"], d["out_affinity_logits_binary"])


# ---------------------------------------------------------------------------------------
# whole modules + ensemble
# ---------------------------------------------------------------------------------------
def _run_module(mod, d, feats):
    out = mod(
        s_inputs=mxa(d["kw_s_inputs"]),
        z=mxa(d["kw_z"]),
        x_pred=mxa(d["kw_x_pred"]),
        feats=feats,
        multiplicity=int(d["kw_multiplicity"]),
    )
    _ev(*out.values())
    return out


def test_affinity_module1(aff1, feats):
    name = "affinity_module1"
    d = _dump(name)
    out = _run_module(aff1, d, feats)
    _scalar(f"{name}:affinity_pred_value", out["affinity_pred_value"], d["out_affinity_pred_value"])
    _scalar(f"{name}:affinity_logits_binary", out["affinity_logits_binary"], d["out_affinity_logits_binary"])


def test_affinity_module2_and_ensemble(aff1, feats):
    """Module 2 (4 blocks) from its dumped inputs, then the ensemble vs ref_aff/outputs.npz."""
    d1, d2 = _dump("affinity_module1"), _dump("affinity_module2")
    o = np.load(DATA / CASE / REF / "outputs.npz")
    out1 = _run_module(aff1, d1, feats)
    aff2 = build_aff(2)
    out2 = _run_module(aff2, d2, feats)
    _scalar("affinity_module2:affinity_pred_value", out2["affinity_pred_value"], d2["out_affinity_pred_value"])
    _scalar("affinity_module2:affinity_logits_binary", out2["affinity_logits_binary"], d2["out_affinity_logits_binary"])

    ens = affinity_ensemble(out1, out2, feats, mw_correction=False)
    _ev(*ens.values())
    for k in [
        "affinity_pred_value", "affinity_probability_binary",
        "affinity_pred_value1", "affinity_probability_binary1",
        "affinity_pred_value2", "affinity_probability_binary2",
    ]:  # fmt: skip
        assert ens[k].shape == o[k].shape == (1, 1), k
        _scalar(f"ensemble:{k}", ens[k], o[k])


def test_ensemble_from_dumped_logits_exact_formula():
    """affinity_ensemble on the dumped member outputs reproduces outputs.npz (isolates the combination step)."""
    d1, d2 = _dump("affinity_module1"), _dump("affinity_module2")
    o = np.load(DATA / CASE / REF / "outputs.npz")
    mk = lambda d: {  # noqa: E731
        "affinity_pred_value": mxa(d["out_affinity_pred_value"]),
        "affinity_logits_binary": mxa(d["out_affinity_logits_binary"]),
    }
    ens = affinity_ensemble(mk(d1), mk(d2))
    for k in ["affinity_pred_value", "affinity_probability_binary", "affinity_pred_value1",
              "affinity_probability_binary1", "affinity_pred_value2", "affinity_probability_binary2"]:  # fmt: skip
        _scalar(f"ensemble(dumped):{k}", ens[k], o[k], atol=1e-6)


def test_ensemble_mw_correction_formula():
    """MW correction is off by default; when on it matches the upstream expression (Boltz2.forward)."""
    out1 = {"affinity_pred_value": mx.array([[1.5]]), "affinity_logits_binary": mx.array([[0.3]])}
    out2 = {"affinity_pred_value": mx.array([[2.5]]), "affinity_logits_binary": mx.array([[-0.2]])}
    feats = {"affinity_mw": mx.array([348.4])}
    off = affinity_ensemble(out1, out2, feats)
    on = affinity_ensemble(out1, out2, feats, mw_correction=True)
    assert abs(float(off["affinity_pred_value"][0, 0]) - 2.0) < 1e-7
    expected = 1.03525938 * 2.0 + (-0.59992683) * 348.4**0.3 + 2.83288489
    assert abs(float(on["affinity_pred_value"][0, 0]) - expected) < 1e-4
    assert float(on["affinity_probability_binary"][0, 0]) == float(off["affinity_probability_binary"][0, 0])


# ---------------------------------------------------------------------------------------
# glue helpers + ragged-mask check of the heads against live upstream torch
# ---------------------------------------------------------------------------------------
def test_affinity_input_helpers(feats):
    o = np.load(DATA / CASE / REF / "outputs.npz")
    d = _dump("affinity_module1")
    z_full = mxa(o["z"])
    zm = mask_z_for_affinity(z_full, feats)
    _ev(zm)
    # the dumped module input is exactly z * cross_pair_mask
    assert_close(zm, d["kw_z"], rel=0.0, cos=1.0, name="mask_z_for_affinity vs dumped kw_z")
    cm = np.array(affinity_cross_pair_mask(feats))
    assert cm.sum() > 0 and cm.shape == (213, 213)
    # best sample = argmax iptm of the 5 diffusion samples
    assert best_sample_index(mxa(o["iptm"])) == int(np.argmax(o["iptm"]))


def test_heads_ragged_mask_vs_torch(aff1):
    """AffinityHeadsTransformer with pad tokens / 2 chains / multiplicity=2 vs upstream torch (real weights)."""
    torch = pytest.importorskip("torch")
    from boltz.model.modules.affinity import AffinityHeadsTransformer as TorchHeads

    h = TorchHeads(128, 384, 12, 8, False, False)
    sd = {k[len("affinity_heads."):]: torch.from_numpy(np.array(v)) for k, v in subtree("affinity_module1", "aff").items()
          if k.startswith("affinity_heads.")}  # fmt: skip
    h.load_state_dict(sd, strict=True)
    h.eval()

    rng = np.random.default_rng(0)
    T = 30
    mol_type = np.array([0] * 14 + [3] * 6 + [1] * 4 + [0] * 6, dtype=np.int64)
    pad = np.array([1] * 24 + [0] * 6, dtype=np.float32)
    aff_tok = np.array([0] * 14 + [1] * 6 + [0] * 10, dtype=np.float32)
    f = {"token_pad_mask": pad[None], "mol_type": mol_type[None], "affinity_token_mask": aff_tok[None]}
    z = rng.standard_normal((2, T, T, 128)).astype(np.float32)
    with torch.no_grad():
        ref = h(torch.from_numpy(z), {k: torch.from_numpy(v) for k, v in f.items()}, multiplicity=2)
    out = aff1.affinity_heads(mxa(z), {k: mx.array(v) for k, v in f.items()}, multiplicity=2)
    _ev(*out.values())
    for k in ref:
        assert out[k].shape == tuple(ref[k].shape) == (2, 1), k
        m = assert_close(out[k], ref[k].numpy(), rel=1.0, cos=0.0, name=f"heads(ragged,mult=2):{k}")
        assert m["max_abs"] <= 1e-5
