"""Extra parity tests against the live upstream PyTorch modules (real weights, synthetic inputs).

The prot117 dumps have all-ones masks, S=1 MSA and no templates, so they cannot exercise
masking, multi-row MSA, multiplicity or the template path.  These tests run the upstream
torch modules (CPU fp32, eval, use_kernels=False) on random inputs with ragged masks and
compare against the MLX port.  Skipped if torch / boltz are not importable.
"""

import mlx.core as mx
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("boltz")

from boltz_mlx.compare import assert_close  # noqa: E402
from boltz_mlx.layers.attentionv2 import AttentionPairBias  # noqa: E402
from boltz_mlx.layers.outer_product_mean import OuterProductMean  # noqa: E402
from boltz_mlx.layers.pair_averaging import PairWeightedAveraging  # noqa: E402
from boltz_mlx.layers.triangular_attention.attention import (  # noqa: E402
    TriangleAttentionEndingNode,
    TriangleAttentionStartingNode,
)
from boltz_mlx.layers.triangular_mult import (  # noqa: E402
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)
from boltz_mlx.modules.trunkv2 import MSAModule, TemplateV2Module, _torch_linspace_f32  # noqa: E402
from boltz_mlx.weights import load_module, subtree  # noqa: E402

REL, COS = 1e-4, 0.99999
N, S, B = 19, 5, 1
PF = "pairformer_module.layers.0"
MSA = "msa_module.layers.0"


def _torch_mod(mod, prefix):
    sd = {k: torch.from_numpy(np.array(v)) for k, v in subtree(prefix).items()}
    mod.load_state_dict(sd, strict=True)
    return mod.eval()


def _rng(seed=0):
    return np.random.default_rng(seed)


def _t(x):
    return torch.from_numpy(np.asarray(x))


def _pair_mask(rng, n=N):
    tok = (rng.random((B, n)) > 0.2).astype(np.float32)
    tok[:, 0] = 1.0
    return tok, tok[:, :, None] * tok[:, None, :]


def _z(rng, c=128, n=N):
    return rng.standard_normal((B, n, n, c)).astype(np.float32)


def test_linspace_matches_torch_default_bins():
    ref = torch.linspace(3.25, 50.75, 37).numpy()
    assert np.array_equal(ref, _torch_linspace_f32(3.25, 50.75, 37))


@pytest.mark.parametrize(
    "cls_mx,cls_name",
    [(TriangleMultiplicationOutgoing, "TriangleMultiplicationOutgoing"), (TriangleMultiplicationIncoming, "TriangleMultiplicationIncoming")],
)
def test_tri_mul_masked(cls_mx, cls_name):
    import boltz.model.layers.triangular_mult as T

    name = f"{PF}.tri_mul_" + ("out" if "Outgoing" in cls_name else "in")
    rng = _rng(1)
    z = _z(rng)
    _, pm = _pair_mask(rng)
    with torch.no_grad():
        ref = _torch_mod(getattr(T, cls_name)(128), name)(_t(z), _t(pm)).numpy()
    out = load_module(cls_mx(128), name)(mx.array(z), mx.array(pm))
    assert_close(out, ref, rel=REL, cos=COS, name=name + " (masked)")


@pytest.mark.parametrize("starting", [True, False])
def test_tri_att_masked(starting):
    from boltz.model.layers.triangular_attention.attention import (
        TriangleAttentionEndingNode as TE,
    )
    from boltz.model.layers.triangular_attention.attention import (
        TriangleAttentionStartingNode as TS,
    )

    name = f"{PF}.tri_att_" + ("start" if starting else "end")
    rng = _rng(2)
    z = _z(rng)
    _, pm = _pair_mask(rng)
    # also drop a few random pair entries (not only whole rows/cols)
    pm = pm * (rng.random(pm.shape) > 0.1).astype(np.float32)
    with torch.no_grad():
        ref = _torch_mod((TS if starting else TE)(128, 32, 4, inf=1e9), name)(_t(z), mask=_t(pm)).numpy()
    cls = TriangleAttentionStartingNode if starting else TriangleAttentionEndingNode
    out = load_module(cls(128, 32, 4, inf=1e9), name)(mx.array(z), mask=mx.array(pm))
    assert_close(out, ref, rel=REL, cos=COS, name=name + " (masked)")


@pytest.mark.parametrize("multiplicity", [1, 3])
def test_attention_pair_bias_masked_multiplicity(multiplicity):
    from boltz.model.layers.attentionv2 import AttentionPairBias as TA

    name = f"{PF}.attention"
    rng = _rng(3)
    tok, _ = _pair_mask(rng)
    s = rng.standard_normal((B * multiplicity, N, 384)).astype(np.float32)
    z = _z(rng)
    tokm = np.repeat(tok, multiplicity, axis=0)
    with torch.no_grad():
        ref = _torch_mod(TA(384, 128, 16), name)(
            s=_t(s), z=_t(z), mask=_t(tokm), k_in=_t(s), multiplicity=multiplicity
        ).numpy()
    out = load_module(AttentionPairBias(384, 128, 16), name)(
        s=mx.array(s), z=mx.array(z), mask=mx.array(tokm), k_in=mx.array(s), multiplicity=multiplicity
    )
    assert_close(out, ref, rel=REL, cos=COS, name=f"{name} (masked, mult={multiplicity})")


def test_pair_weighted_averaging_masked():
    from boltz.model.layers.pair_averaging import PairWeightedAveraging as TP

    name = f"{MSA}.pair_weighted_averaging"
    rng = _rng(4)
    m = rng.standard_normal((B, S, N, 64)).astype(np.float32)
    z = _z(rng)
    _, pm = _pair_mask(rng)
    with torch.no_grad():
        ref = _torch_mod(TP(64, 128, 32, 8), name)(_t(m), _t(z), _t(pm)).numpy()
    out = load_module(PairWeightedAveraging(64, 128, 32, 8), name)(mx.array(m), mx.array(z), mx.array(pm))
    assert_close(out, ref, rel=REL, cos=COS, name=name + " (masked, S=5)")


def test_outer_product_mean_masked():
    from boltz.model.layers.outer_product_mean import OuterProductMean as TO

    name = f"{MSA}.outer_product_mean"
    rng = _rng(5)
    m = rng.standard_normal((B, S, N, 64)).astype(np.float32)
    mm = (rng.random((B, S, N)) > 0.3).astype(np.int64)
    mm[:, 0] = 1
    mm[:, :, 3] = 0  # a column with zero MSA coverage -> exercises clamp(min=1)
    with torch.no_grad():
        ref = _torch_mod(TO(64, 32, 128), name)(_t(m), _t(mm)).numpy()
    out = load_module(OuterProductMean(64, 32, 128), name)(mx.array(m), mx.array(mm))
    assert_close(out, ref, rel=REL, cos=COS, name=name + " (masked, S=5)")


def _random_feats(rng, n=N, s=S):
    tok, _ = _pair_mask(rng, n)
    msa = rng.integers(0, 33, size=(B, s, n))
    f = {
        "token_pad_mask": tok,
        "msa": msa.astype(np.int64),
        "has_deletion": rng.random((B, s, n)) > 0.7,
        "deletion_value": rng.random((B, s, n)).astype(np.float32),
        "msa_paired": (rng.random((B, s, n)) > 0.5).astype(np.float32),
        "msa_mask": (rng.random((B, s, n)) > 0.2).astype(np.int64),
    }
    return f


def test_msa_module_multirow():
    from boltz.model.modules.trunkv2 import MSAModule as TM

    rng = _rng(6)
    feats = _random_feats(rng)
    z = _z(rng) * 0.5
    emb = rng.standard_normal((B, N, 384)).astype(np.float32)
    kw = dict(msa_s=64, token_z=128, token_s=384, msa_blocks=4, msa_dropout=0.15, z_dropout=0.25)
    with torch.no_grad():
        tf = {k: _t(v) for k, v in feats.items()}
        ref = _torch_mod(TM(**kw), "msa_module")(_t(z), _t(emb), tf).numpy()
    mf = {k: mx.array(v) for k, v in feats.items()}
    out = load_module(MSAModule(**kw), "msa_module")(mx.array(z), mx.array(emb), mf)
    # loose-ish: 4 blocks, random (out-of-distribution) inputs
    assert_close(out, ref, rel=2e-4, cos=COS, name="msa_module (S=5, ragged masks, random feats)")


def _random_template_feats(rng, T=2, n=N):
    def rot():
        q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        return q.astype(np.float32)

    res = np.zeros((B, T, n, 33), dtype=np.int64)
    idx = rng.integers(2, 22, size=(B, T, n))
    np.put_along_axis(res, idx[..., None], 1, axis=-1)
    tmask = np.ones((B, T, n), dtype=np.float32)
    tmask[:, T - 1] = 0.0  # last template fully masked -> exercises num_templates / template_mask
    return {
        "template_restype": res,
        "template_frame_rot": np.stack([[rot() for _ in range(n)] for _ in range(T)])[None].astype(np.float32),
        "template_frame_t": (rng.standard_normal((B, T, n, 3)) * 12).astype(np.float32),
        "template_mask_frame": (rng.random((B, T, n)) > 0.15).astype(np.float32),
        "template_cb": (rng.standard_normal((B, T, n, 3)) * 12).astype(np.float32),
        "template_ca": (rng.standard_normal((B, T, n, 3)) * 12).astype(np.float32),
        "template_mask_cb": (rng.random((B, T, n)) > 0.15).astype(np.float32),
        "visibility_ids": rng.integers(0, 2, size=(B, T, n)).astype(np.float32),
        "template_mask": tmask,
    }


@pytest.mark.parametrize("T", [1, 3])
def test_template_module_nontrivial(T):
    from boltz.model.modules.trunkv2 import TemplateV2Module as TT

    rng = _rng(7 + T)
    feats = _random_template_feats(rng, T)
    if T == 1:
        feats["template_mask"][:] = 1.0
    z = _z(rng) * 0.5
    _, pm = _pair_mask(rng)
    kw = dict(token_z=128, template_dim=64, template_blocks=2)
    with torch.no_grad():
        tf = {k: _t(v) for k, v in feats.items()}
        ref = _torch_mod(TT(**kw), "template_module")(_t(z), tf, _t(pm)).numpy()
    assert np.abs(ref).max() > 0
    mf = {k: mx.array(v) for k, v in feats.items()}
    out = load_module(TemplateV2Module(**kw), "template_module")(mx.array(z), mf, mx.array(pm))
    assert_close(out, ref, rel=2e-4, cos=COS, name=f"template_module (T={T}, random templates)")
