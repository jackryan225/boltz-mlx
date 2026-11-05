"""Parity tests: encodersv2 / transformersv2 pieces, InputEmbedder, ContactConditioning."""

import mlx.core as mx
import numpy as np
import pytest

from boltz_mlx.compare import assert_close
from boltz_mlx.modules.encodersv2 import (
    AtomAttentionDecoder,
    AtomAttentionEncoder,
    RelativePositionEncoder,
    SingleConditioning,
    get_indexing_matrix,
    make_to_keys,
)
from boltz_mlx.modules.trunkv2_embed import ContactConditioning, InputEmbedder
from boltz_mlx.weights import load_module
from boltz_cfg import (
    ATOM_S,
    H,
    INPUT_EMBEDDER_ARGS,
    SCORE_MODEL_ARGS,
    TOKEN_S,
    TOKEN_Z,
    W,
    arg,
)
from conftest import load_dump, load_feats, mxa

SM = "structure_module.score_model"


@pytest.fixture(scope="module")
def feats():
    return load_feats()


def test_rel_pos(feats):
    d = load_dump("rel_pos")
    mod = load_module(RelativePositionEncoder(TOKEN_Z, fix_sym_check=True, cyclic_pos_enc=True), "rel_pos")
    out = mod(feats)
    assert_close(out, d["out"], name="rel_pos")


def test_rel_pos_cyclic_branch_vs_torch():
    """Exercise the cyclic_period > 0 branch (inactive for prot117) against upstream torch."""
    torch = pytest.importorskip("torch")
    from boltz.model.modules.encodersv2 import RelativePositionEncoder as TorchRPE

    rng = np.random.default_rng(0)
    n = 24
    f = {
        "asym_id": np.repeat([0, 1], n // 2)[None],
        "residue_index": np.concatenate([np.arange(n // 2), np.arange(n // 2)])[None],
        "entity_id": np.repeat([0, 1], n // 2)[None],
        "token_index": np.arange(n)[None],
        "sym_id": np.zeros((1, n), dtype=np.int64),
        "cyclic_period": np.concatenate([np.full(n // 2, 7.0), np.zeros(n // 2)])[None].astype(np.float32),
    }
    t = TorchRPE(TOKEN_Z, fix_sym_check=True, cyclic_pos_enc=True).eval()
    with torch.no_grad():
        t.linear_layer.weight.copy_(torch.randn_like(t.linear_layer.weight))
        ref = t({k: torch.from_numpy(v) for k, v in f.items()}).numpy()
    m = RelativePositionEncoder(TOKEN_Z, fix_sym_check=True, cyclic_pos_enc=True)
    m.linear_layer.weight = mxa(t.linear_layer.weight.detach().numpy())
    out = m({k: mx.array(v) for k, v in f.items()})
    assert_close(out, ref, rel=1e-6, name="rel_pos(cyclic)")


def test_indexing_matrix_and_to_keys_vs_torch():
    torch = pytest.importorskip("torch")
    from boltz.model.modules.encodersv2 import get_indexing_matrix as tg, single_to_keys as ts

    K = 29
    ref_m = tg(K, W, H, "cpu").numpy()
    m = get_indexing_matrix(K, W, H)
    assert np.array_equal(np.array(m), ref_m)

    x = np.random.default_rng(1).normal(size=(2, K * W, 5)).astype(np.float32)
    ref = ts(torch.from_numpy(x), torch.from_numpy(ref_m), W, H).numpy()
    out = make_to_keys(K * W, W, H)(mx.array(x))
    assert out.shape == (2, K, H, 5)
    assert_close(out, ref, rel=1e-6, name="to_keys")


def test_contact_conditioning(feats):
    d = load_dump("contact_conditioning")
    mod = load_module(ContactConditioning(TOKEN_Z, 4.0, 20.0), "contact_conditioning")
    assert_close(mod(feats), d["out"], name="contact_conditioning")


def test_input_embedder(feats):
    d = load_dump("input_embedder")
    mod = load_module(InputEmbedder(**INPUT_EMBEDDER_ARGS), "input_embedder")
    assert_close(mod(feats), d["out"], name="input_embedder")


def test_single_conditioner():
    d = load_dump(f"{SM}.single_conditioner")
    mod = load_module(
        SingleConditioning(sigma_data=16, token_s=TOKEN_S, dim_fourier=256, num_transitions=2),
        f"{SM}.single_conditioner",
    )
    s, nf = mod(mxa(d["in_0"]), mxa(d["in_1"]), mxa(d["in_2"]))
    assert_close(s, d["out_0"], name="single_conditioner.s")
    assert_close(nf, d["out_1"], name="single_conditioner.normed_fourier")


def test_atom_attention_encoder(feats):
    d = load_dump(f"{SM}.atom_attention_encoder")
    mod = load_module(
        AtomAttentionEncoder(
            atom_s=ATOM_S,
            token_s=TOKEN_S,
            atoms_per_window_queries=W,
            atoms_per_window_keys=H,
            atom_encoder_depth=3,
            atom_encoder_heads=4,
            structure_prediction=True,
        ),
        f"{SM}.atom_attention_encoder",
    )
    n_atoms = d["kw_q"].shape[1]
    a, q, c, _ = mod(
        feats=feats,
        q=mxa(d["kw_q"]),
        c=mxa(d["kw_c"]),
        atom_enc_bias=mxa(d["kw_atom_enc_bias"]),
        to_keys=make_to_keys(n_atoms, W, H),
        r=mxa(d["kw_r"]),
        multiplicity=int(d["kw_multiplicity"]),
    )
    assert_close(a, d["out_0"], name="atom_attention_encoder.a")
    assert_close(q, d["out_1"], name="atom_attention_encoder.q")
    assert_close(c, d["out_2"], name="atom_attention_encoder.c")


def test_atom_attention_decoder(feats):
    d = load_dump(f"{SM}.atom_attention_decoder")
    mod = load_module(
        AtomAttentionDecoder(
            atom_s=ATOM_S,
            token_s=TOKEN_S,
            attn_window_queries=W,
            attn_window_keys=H,
            atom_decoder_depth=3,
            atom_decoder_heads=4,
        ),
        f"{SM}.atom_attention_decoder",
    )
    n_atoms = d["kw_q"].shape[1]
    out = mod(
        a=mxa(d["kw_a"]),
        q=mxa(d["kw_q"]),
        c=mxa(d["kw_c"]),
        atom_dec_bias=mxa(d["kw_atom_dec_bias"]),
        feats=feats,
        to_keys=make_to_keys(n_atoms, W, H),
        multiplicity=int(d["kw_multiplicity"]),
    )
    assert_close(out, d["out"], name="atom_attention_decoder")


def test_atom_encoder_multiplicity_consistency(feats):
    """multiplicity=2 with identical r must reproduce multiplicity=1 for both samples."""
    d = load_dump(f"{SM}.atom_attention_encoder")
    mod = load_module(
        AtomAttentionEncoder(
            atom_s=ATOM_S, token_s=TOKEN_S, atoms_per_window_queries=W, atoms_per_window_keys=H,
            atom_encoder_depth=3, atom_encoder_heads=4, structure_prediction=True,
        ),
        f"{SM}.atom_attention_encoder",
    )
    n_atoms = d["kw_q"].shape[1]
    tk = make_to_keys(n_atoms, W, H)
    kw = dict(feats=feats, q=mxa(d["kw_q"]), c=mxa(d["kw_c"]), atom_enc_bias=mxa(d["kw_atom_enc_bias"]), to_keys=tk)
    r = mxa(d["kw_r"])
    a2, q2, _, _ = mod(r=mx.concatenate([r, r]), multiplicity=2, **kw)
    assert_close(a2[0:1], d["out_0"], name="mult2.a[0]")
    assert_close(a2[1:2], d["out_0"], name="mult2.a[1]")
    assert_close(q2[1:2], d["out_1"], name="mult2.q[1]")


def test_contact_conditioning_all_branches_vs_torch():
    """prot117 only has UNSPECIFIED contacts; compare random one-hot contact feats to upstream torch."""
    torch = pytest.importorskip("torch")
    from boltz.model.modules.trunkv2 import ContactConditioning as TorchCC

    rng = np.random.default_rng(0)
    n = 10
    cls = rng.integers(0, 5, size=(1, n, n))
    cc = np.eye(5, dtype=np.int64)[cls]
    thr = rng.uniform(4, 20, size=(1, n, n)).astype(np.float32)
    t = TorchCC(TOKEN_Z, 4.0, 20.0).eval()
    with torch.no_grad():
        for p in t.parameters():
            p.copy_(torch.randn_like(p))
        ref = t({"contact_conditioning": torch.from_numpy(cc), "contact_threshold": torch.from_numpy(thr)}).numpy()
    m = ContactConditioning(TOKEN_Z, 4.0, 20.0)
    m.load_weights([(k, mxa(v.detach().numpy())) for k, v in t.state_dict().items()])
    out = m({"contact_conditioning": mx.array(cc), "contact_threshold": mx.array(thr)})
    assert_close(out, ref, rel=1e-6, name="contact_conditioning(random)")
