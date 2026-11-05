"""Parity tests for trunk modules vs. PyTorch reference dumps (prot117, first call, CPU fp32).

Measured numbers for the deep stacks (CPU, fp32) are recorded next to the thresholds below.
"""

import mlx.core as mx
import pytest
from conftest import load_dump, load_feats, mxa

from boltz_mlx.compare import assert_close
from boltz_mlx.layers.pairformer import PairformerModule
from boltz_mlx.modules.trunkv2 import (
    BFactorModule,
    DistogramModule,
    MSALayer,
    MSAModule,
    TemplateV2Module,
)
from boltz_mlx.weights import load_module

REL, COS = 1e-4, 0.99999

# Deep-stack thresholds.  Measured on MLX CPU fp32 (prot117, 117 residues):
#   pairformer_module (64 blocks): s max_rel=7.6e-7, z max_rel=1.0e-6, cos=1.0 (to 8 digits)
#   msa_module (4 blocks):         z max_rel=3.5e-7, cos=1.0
# Drift through the stacks is ~1e-6 relative (fp32 accumulation order only), 100x below the
# single-module gate, so the default 1e-4 gate is kept; it leaves headroom for GPU re-runs
# (different matmul/reduction order) without hiding real bugs (a wrong op shows up at >=1e-2).
PAIRFORMER_REL, PAIRFORMER_COS = 1e-4, 0.99999
MSA_REL, MSA_COS = 1e-4, 0.99999


def test_msa_layer():
    name = "msa_module.layers.0"
    d = load_dump(name)
    mod = load_module(MSALayer(64, 128, 0.15, 0.25), name)
    z, m = mod(mxa(d["in_0"]), mxa(d["in_1"]), mxa(d["in_2"]), mxa(d["in_3"]))
    mx.eval(z, m)
    assert_close(z, d["out_0"], rel=REL, cos=COS, name=f"{name}[z]")
    assert_close(m, d["out_1"], rel=REL, cos=COS, name=f"{name}[m]")


@pytest.mark.parametrize("case", ["prot117", "msa136"])
def test_msa_module(case):
    """Whole 4-block MSA module: prot117 (S=1 row) and msa136 (real upstream example MSA, 75 rows)."""
    name = "msa_module"
    d = load_dump(name, case)
    assert str(d["in_2"]) == "FEATS"
    mod = load_module(
        MSAModule(msa_s=64, token_z=128, token_s=384, msa_blocks=4, msa_dropout=0.15, z_dropout=0.25),
        name,
    )
    out = mod(mxa(d["in_0"]), mxa(d["in_1"]), load_feats(case))
    mx.eval(out)
    assert_close(out, d["out"], rel=MSA_REL, cos=MSA_COS, name=f"{case}:{name}")


def test_template_module():
    """prot117 has no templates (all template feats/masks are zero) -> output is exactly zero.

    The synthetic, non-trivial template test lives in test_trunk_vs_torch.py.
    """
    name = "template_module"
    d = load_dump(name)
    assert str(d["in_1"]) == "FEATS"
    mod = load_module(TemplateV2Module(128, template_dim=64, template_blocks=2), name)
    out = mod(mxa(d["in_0"]), load_feats(), mxa(d["in_2"]))
    mx.eval(out)
    assert_close(out, d["out"], rel=REL, cos=COS, name=name)


def test_distogram_module():
    name = "distogram_module"
    d = load_dump(name)
    mod = load_module(DistogramModule(128, 64), name)
    out = mod(mxa(d["in_0"]))
    mx.eval(out)
    assert out.shape == d["out"].shape
    assert_close(out, d["out"], rel=REL, cos=COS, name=name)


def test_bfactor_module():
    name = "bfactor_module"
    d = load_dump(name)
    mod = load_module(BFactorModule(384, 64), name)
    out = mod(mxa(d["in_0"]))
    mx.eval(out)
    assert_close(out, d["out"], rel=REL, cos=COS, name=name)


def test_pairformer_module():
    """Whole 64-block pairformer stack (s and z)."""
    name = "pairformer_module"
    d = load_dump(name)
    mod = load_module(PairformerModule(384, 128, num_blocks=64, num_heads=16), name)
    s, z = mod(mxa(d["in_0"]), mxa(d["in_1"]), mxa(d["kw_mask"]), mxa(d["kw_pair_mask"]))
    mx.eval(s, z)
    assert_close(s, d["out_0"], rel=PAIRFORMER_REL, cos=PAIRFORMER_COS, name=f"{name}[s]")
    assert_close(z, d["out_1"], rel=PAIRFORMER_REL, cos=PAIRFORMER_COS, name=f"{name}[z]")
