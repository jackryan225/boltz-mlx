"""Batched samples with shared noise level: computing time conditioning once must match per-sample."""

import mlx.core as mx
import numpy as np
import pytest

from boltz_mlx.model import load_model
from conftest import load_feats


@pytest.mark.skipif(mx.default_device() != mx.gpu, reason="full model; run on GPU")
def test_shared_times_matches():
    model = load_model("conf", confidence=False)
    feats = load_feats("prot117")
    s_inputs, s, z, rpe = model.trunk(feats, 0)
    q, c, tk, eb, db, tb = model.diffusion_conditioning(s, z, rpe, feats)
    sm = model.structure_module
    mult = 3
    base = dict(q=q, c=c, to_keys=tk, atom_enc_bias=eb, atom_dec_bias=db, token_trans_bias=tb)
    base["prepared"] = sm.score_model.prepare(feats, base, mult)
    x = mx.random.normal((mult, feats["atom_pad_mask"].shape[1], 3), key=mx.random.key(1)) * 10
    sig = mx.full((mult,), 3.0)
    outs = []
    for shared in (False, True):
        cond = {**base, "shared_times": shared}
        outs.append(sm.preconditioned_network_forward(
            x, sig, dict(s_inputs=s_inputs, s_trunk=s, feats=feats, diffusion_conditioning=cond, multiplicity=mult)))
    mx.eval(outs)
    d = float(mx.abs(outs[0] - outs[1]).max())
    print("max |d|", d, "scale", float(mx.abs(outs[0]).max()))
    assert d < 1e-3
