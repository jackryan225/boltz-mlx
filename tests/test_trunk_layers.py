"""Parity tests for trunk layers vs. PyTorch reference dumps (prot117, first call, CPU fp32)."""

import mlx.core as mx
import pytest
from conftest import load_dump, mxa

from boltz_mlx.compare import assert_close
from boltz_mlx.layers.attentionv2 import AttentionPairBias
from boltz_mlx.layers.outer_product_mean import OuterProductMean
from boltz_mlx.layers.pair_averaging import PairWeightedAveraging
from boltz_mlx.layers.pairformer import PairformerLayer, PairformerNoSeqLayer
from boltz_mlx.layers.transition import Transition
from boltz_mlx.layers.triangular_attention.attention import (
    TriangleAttentionEndingNode,
    TriangleAttentionStartingNode,
)
from boltz_mlx.layers.triangular_mult import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)
from boltz_mlx.weights import load_module

PF = "pairformer_module.layers.0"
MSA = "msa_module.layers.0"
REL, COS = 1e-4, 0.99999


def _ev(*xs):
    mx.eval(*xs)
    return xs[0] if len(xs) == 1 else xs


def _np(x):
    import numpy as np

    return np.array(x)


# ---------------------------------------------------------------------------------------
# Transition
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name,dim,hidden",
    [
        (f"{PF}.transition_z", 128, 512),
        (f"{PF}.transition_s", 384, 1536),
        (f"{MSA}.msa_transition", 64, 256),
    ],
)
def test_transition(name, dim, hidden):
    d = load_dump(name)
    mod = load_module(Transition(dim, hidden), name)
    out = _ev(mod(mxa(d["in_0"])))
    assert_close(out, d["out"], rel=REL, cos=COS, name=name)


# ---------------------------------------------------------------------------------------
# Outer product mean / pair weighted averaging
# ---------------------------------------------------------------------------------------
def test_outer_product_mean():
    name = f"{MSA}.outer_product_mean"
    d = load_dump(name)
    mod = load_module(OuterProductMean(64, 32, 128), name)
    out = _ev(mod(mxa(d["in_0"]), mxa(d["in_1"])))
    assert_close(out, d["out"], rel=REL, cos=COS, name=name)


def test_pair_weighted_averaging():
    name = f"{MSA}.pair_weighted_averaging"
    d = load_dump(name)
    mod = load_module(PairWeightedAveraging(c_m=64, c_z=128, c_h=32, num_heads=8), name)
    out = _ev(mod(mxa(d["in_0"]), mxa(d["in_1"]), mxa(d["in_2"]), bool(d["in_3"])))
    assert_close(out, d["out"], rel=REL, cos=COS, name=name)


# ---------------------------------------------------------------------------------------
# Triangular multiplication / attention
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name,cls",
    [
        (f"{PF}.tri_mul_out", TriangleMultiplicationOutgoing),
        (f"{PF}.tri_mul_in", TriangleMultiplicationIncoming),
    ],
)
def test_triangle_multiplication(name, cls):
    d = load_dump(name)
    mod = load_module(cls(128), name)
    out = _ev(mod(mxa(d["in_0"]), mask=mxa(d["kw_mask"])))
    assert_close(out, d["out"], rel=REL, cos=COS, name=name)


@pytest.mark.parametrize(
    "name,cls",
    [
        (f"{PF}.tri_att_start", TriangleAttentionStartingNode),
        (f"{PF}.tri_att_end", TriangleAttentionEndingNode),
    ],
)
def test_triangle_attention(name, cls):
    d = load_dump(name)
    mod = load_module(cls(128, 32, 4, inf=1e9), name)
    out = _ev(mod(mxa(d["in_0"]), mask=mxa(d["kw_mask"])))
    assert_close(out, d["out"], rel=REL, cos=COS, name=name)


# ---------------------------------------------------------------------------------------
# AttentionPairBias v2
# ---------------------------------------------------------------------------------------
def test_attention_pair_bias_v2():
    name = f"{PF}.attention"
    d = load_dump(name)
    mod = load_module(AttentionPairBias(c_s=384, c_z=128, num_heads=16), name)
    out = _ev(mod(s=mxa(d["kw_s"]), z=mxa(d["kw_z"]), mask=mxa(d["kw_mask"]), k_in=mxa(d["kw_k_in"])))
    assert_close(out, d["out"], rel=REL, cos=COS, name=name)


# ---------------------------------------------------------------------------------------
# Pairformer layers
# ---------------------------------------------------------------------------------------
def test_pairformer_layer():
    name = PF
    d = load_dump(name)
    mod = load_module(PairformerLayer(384, 128, num_heads=16), name)
    s, z = _ev(*mod(mxa(d["in_0"]), mxa(d["in_1"]), mxa(d["in_2"]), mxa(d["in_3"])))
    assert_close(s, d["out_0"], rel=REL, cos=COS, name=f"{name}[s]")
    assert_close(z, d["out_1"], rel=REL, cos=COS, name=f"{name}[z]")


def test_pairformer_noseq_layer_msa():
    name = f"{MSA}.pairformer_layer"
    d = load_dump(name)
    mod = load_module(PairformerNoSeqLayer(128), name)
    out = _ev(mod(mxa(d["in_0"]), mxa(d["in_1"])))
    assert_close(out, d["out"], rel=REL, cos=COS, name=name)


def test_pairformer_noseq_layer_template():
    name = "template_module.pairformer.layers.0"
    d = load_dump(name)
    mod = load_module(PairformerNoSeqLayer(64), name)
    out = _ev(mod(mxa(d["in_0"]), mxa(d["in_1"])))
    assert_close(out, d["out"], rel=REL, cos=COS, name=name)


def test_pairformer_layer_param_names_match_torch():
    """Strict load already enforces this; also assert Identity s_post_norm adds no params."""
    from mlx.utils import tree_flatten

    names = {k for k, _ in tree_flatten(PairformerLayer(384, 128).parameters())}
    assert not any(k.startswith("s_post_norm") for k in names)
    assert "attention.proj_z.0.weight" in names and "attention.proj_z.1.weight" in names
