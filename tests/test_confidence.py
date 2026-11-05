"""Parity tests: ConfidenceModule / ConfidenceHeads / confidence_utils vs torch dumps (prot117, lig_aff).

prot117 is a single protein chain (iptm == 0 by construction); lig_aff is protein + 13-atom ligand
(2 chains) so the nonpolymer frame search, ligand/protein ipTM and the pair-chain ipTM are exercised.
Both dumps used diffusion_samples=1 with run_sequentially=True; multiplicity>1 (the real run_sequentially
loop) is covered against the live upstream torch module in `test_confidence_module_multiplicity_vs_torch`.
"""

import ast
import json

import mlx.core as mx
import numpy as np
import pytest

from boltz_mlx.compare import assert_close
from boltz_mlx.modules.confidence_utils import (
    cdist,
    compute_aggregated_metric,
    compute_frame_pred,
    compute_ptms,
)
from boltz_mlx.modules.confidencev2 import ConfidenceHeads, ConfidenceModule
from boltz_mlx.modules.encodersv2 import RelativePositionEncoder
from boltz_mlx.modules.trunkv2_embed import ContactConditioning
from boltz_mlx.weights import WEIGHTS, load_module, subtree
from conftest import load_dump, load_feats, mxa

import mlx.nn as nn

REL, COS = 1e-4, 0.99999
CM = "confidence_module"


def _hparams() -> dict:
    return json.load(open(WEIGHTS / "boltz2_conf.json"))


def _as_dict(v) -> dict:
    """Checkpoint hparams json stores nested args either as dict or as a python-repr string."""
    return ast.literal_eval(v) if isinstance(v, str) else v


def conf_kwargs() -> dict:
    h = _hparams()
    return dict(
        token_level_confidence=h["token_level_confidence"],
        bond_type_feature=h["bond_type_feature"],
        fix_sym_check=h["fix_sym_check"],
        cyclic_pos_enc=h["cyclic_pos_enc"],
        conditioning_cutoff_min=h["conditioning_cutoff_min"],
        conditioning_cutoff_max=h["conditioning_cutoff_max"],
        **_as_dict(h["confidence_model_args"]),
    )


def build_conf() -> ConfidenceModule:
    h = _hparams()
    mod = ConfidenceModule(h["token_s"], h["token_z"], **conf_kwargs())
    load_module(mod, CM)
    return mod


@pytest.fixture(scope="module")
def conf():
    return build_conf()


def _ev(*xs):
    mx.eval(*[x for x in xs])
    return xs if len(xs) > 1 else xs[0]


# -------------------------------------------------------------------------------------------
# small children
# -------------------------------------------------------------------------------------------
@pytest.mark.parametrize("case", ["prot117", "lig_aff"])
@pytest.mark.parametrize(
    "child,make",
    [
        ("s_inputs_norm", lambda: nn.LayerNorm(384)),
        ("s_norm", lambda: nn.LayerNorm(384)),
        ("z_norm", lambda: nn.LayerNorm(128)),
        ("s_input_to_s", lambda: nn.Linear(384, 384, bias=False)),
        ("s_to_z", lambda: nn.Linear(384, 128, bias=False)),
        ("s_to_z_transpose", lambda: nn.Linear(384, 128, bias=False)),
        ("s_to_z_prod_out", lambda: nn.Linear(128, 128, bias=False)),
        ("token_bonds", lambda: nn.Linear(1, 128, bias=False)),
        ("token_bonds_type", lambda: nn.Embedding(7, 128)),
        ("dist_bin_pairwise_embed", lambda: nn.Embedding(64, 128)),
    ],
)
def test_small_child(case, child, make):
    name = f"{CM}.{child}"
    d = load_dump(name, case)
    mod = load_module(make(), name)
    out = _ev(mod(mxa(d["in_0"])))
    assert_close(out, d["out"], rel=1e-5, cos=0.999999, name=f"{case}:{name}")


@pytest.mark.parametrize("case", ["prot117", "lig_aff"])
def test_rel_pos(case):
    name = f"{CM}.rel_pos"
    d = load_dump(name, case)
    mod = load_module(RelativePositionEncoder(128, fix_sym_check=True, cyclic_pos_enc=True), name)
    out = _ev(mod(load_feats(case)))
    assert_close(out, d["out"], rel=1e-6, name=f"{case}:{name}")


@pytest.mark.parametrize("case", ["prot117", "lig_aff"])
def test_contact_conditioning(case):
    name = f"{CM}.contact_conditioning"
    d = load_dump(name, case)
    mod = load_module(ContactConditioning(128, 4.0, 20.0), name)
    out = _ev(mod(load_feats(case)))
    assert_close(out, d["out"], rel=1e-6, name=f"{case}:{name}")


def test_boundaries_buffer_matches_linspace():
    """`boundaries` is loaded from the checkpoint (a persistent torch buffer); it must equal torch.linspace(2,22,63).
    The runtime-built fallback (trunkv2._torch_linspace_f32) is off by 1 ulp at 2 of 63 entries; only the loaded
    values are used for binning."""
    torch = pytest.importorskip("torch")
    ckpt = np.array(subtree(CM)["boundaries"])
    assert np.array_equal(ckpt, torch.linspace(2, 22, 63).numpy())
    mod = ConfidenceModule(384, 128, **conf_kwargs())
    assert np.abs(np.array(mod.boundaries) - ckpt).max() <= 1.2e-6


# -------------------------------------------------------------------------------------------
# pairformer stack (8 blocks, s and z)
# -------------------------------------------------------------------------------------------
@pytest.mark.parametrize("case", ["prot117", "lig_aff"])
def test_pairformer_stack(conf, case):
    name = f"{CM}.pairformer_stack"
    d = load_dump(name, case)
    s, z = conf.pairformer_stack(mxa(d["in_0"]), mxa(d["in_1"]), mxa(d["kw_mask"]), mxa(d["kw_pair_mask"]))
    _ev(s, z)
    assert_close(s, d["out_0"], rel=REL, cos=COS, name=f"{case}:{name}[s]")
    assert_close(z, d["out_1"], rel=REL, cos=COS, name=f"{case}:{name}[z]")


# -------------------------------------------------------------------------------------------
# confidence heads (given the dumped inputs; all outputs, incl. pair_chains_iptm)
# -------------------------------------------------------------------------------------------
HEAD_KEYS = [
    "pde_logits", "plddt_logits", "resolved_logits", "pde", "plddt", "complex_plddt", "complex_iplddt",
    "complex_pde", "complex_ipde", "pae_logits", "pae", "ptm", "iptm", "ligand_iptm", "protein_iptm",
]  # fmt: skip


def _check_outputs(out: dict, d: dict, case: str, name: str, scalar_atol=1e-5):
    for k in HEAD_KEYS:
        ref = d[f"out_{k}"]
        assert out[k].shape == ref.shape, (k, out[k].shape, ref.shape)
        if ref.size == 1:  # scalar heads: absolute 1e-5
            m = assert_close(out[k], ref, rel=scalar_atol, cos=0.0 if float(np.abs(ref).max()) == 0 else 0.99999,
                             name=f"{case}:{name}:{k}")  # fmt: skip
            assert m["max_abs"] <= scalar_atol, (k, m)
        else:
            assert_close(out[k], ref, rel=REL, cos=COS, name=f"{case}:{name}:{k}")
    n_pairs = 0
    for key in d:
        if key.startswith("out_pair_chains_iptm_"):
            i, j = key.split("_")[-2:]
            got = out["pair_chains_iptm"][int(i)][int(j)]
            m = assert_close(got, d[key], rel=scalar_atol, cos=0.0, name=f"{case}:{name}:pair_chains_iptm[{i}][{j}]")
            assert m["max_abs"] <= scalar_atol
            n_pairs += 1
    n_chains = len(out["pair_chains_iptm"])
    assert n_pairs == n_chains * n_chains


@pytest.mark.parametrize("case", ["prot117", "lig_aff"])
def test_confidence_heads(conf, case):
    name = f"{CM}.confidence_heads"
    d = load_dump(name, case)
    feats = load_feats(case)
    heads = conf.confidence_heads
    out = heads(
        s=mxa(d["kw_s"]),
        z=mxa(d["kw_z"]),
        x_pred=mxa(d["kw_x_pred"]),
        d=mxa(d["kw_d"]),
        feats=feats,
        pred_distogram_logits=mxa(d["kw_pred_distogram_logits"]),
        multiplicity=int(d["kw_multiplicity"]),
    )
    _ev(*[v for k, v in out.items() if k != "pair_chains_iptm"])
    _check_outputs(out, d, case, name)


# -------------------------------------------------------------------------------------------
# whole module (run_sequentially=True, multiplicity=1 as in the reference predict run)
# -------------------------------------------------------------------------------------------
@pytest.mark.parametrize("case", ["prot117", "lig_aff"])
def test_confidence_module(conf, case):
    d = load_dump(CM, case)
    feats = load_feats(case)
    out = conf(
        s_inputs=mxa(d["kw_s_inputs"]),
        s=mxa(d["kw_s"]),
        z=mxa(d["kw_z"]),
        x_pred=mxa(d["kw_x_pred"]),
        feats=feats,
        pred_distogram_logits=mxa(d["kw_pred_distogram_logits"]),
        multiplicity=int(d["kw_multiplicity"]),
        run_sequentially=bool(d["kw_run_sequentially"]),
    )
    _ev(*[v for k, v in out.items() if k != "pair_chains_iptm"])
    _check_outputs(out, d, case, CM)


def test_distogram_bins_exact_vs_dump():
    """dist_bin_pairwise_embed input (int bin index) recomputed from x_pred must match the dump exactly."""
    d = load_dump(f"{CM}.dist_bin_pairwise_embed", "lig_aff")
    dm = load_dump(CM, "lig_aff")
    feats = load_feats("lig_aff")
    boundaries = subtree(CM)["boundaries"]
    xr = mx.matmul(feats["token_to_rep_atom"].astype(mx.float32), mxa(dm["kw_x_pred"]))
    bins = (cdist(xr, xr)[..., None] > boundaries).sum(axis=-1)
    assert np.array_equal(np.array(bins), d["in_0"]), int((np.array(bins) != d["in_0"]).sum())


# -------------------------------------------------------------------------------------------
# vs live upstream torch (synthetic / multiplicity > 1)
# -------------------------------------------------------------------------------------------
def _synthetic_feats(rng, multi_pad=True):
    """Small ragged complex: 12-token protein, two ligands (7 and 5 single-atom tokens) + 3 pad tokens."""
    prot, l1, l2, padt = 12, 7, 5, 3
    T = prot + l1 + l2 + padt
    n_atom_prot = prot * 4
    A = n_atom_prot + l1 + l2 + 6  # 6 pad atoms
    asym = np.array([0] * prot + [1] * l1 + [2] * l2 + [0] * padt, dtype=np.int64)
    mol = np.array([0] * prot + [3] * (l1 + l2) + [0] * padt, dtype=np.int64)
    token_pad = np.array([1] * (prot + l1 + l2) + [0] * padt, dtype=np.float32)
    atom_to_token = np.zeros((A, T), dtype=np.int64)
    atom_pad = np.zeros(A, dtype=np.float32)
    a = 0
    for t in range(prot):
        for _ in range(4):
            atom_to_token[a, t] = 1
            atom_pad[a] = 1
            a += 1
    for t in range(prot, prot + l1 + l2):
        atom_to_token[a, t] = 1
        atom_pad[a] = 1
        a += 1
    for k in range(6):  # pad atoms map to pad tokens
        atom_to_token[a, prot + l1 + l2 + (k % padt)] = 1
        a += 1
    token_to_rep = np.zeros((T, A), dtype=np.int64)
    for t in range(prot):
        token_to_rep[t, t * 4 + 1] = 1
    for t in range(prot, prot + l1 + l2):
        token_to_rep[t, n_atom_prot + (t - prot)] = 1
    # polymer frames: (N, CA, C) of each residue = atoms 0,1,2; ligands/pad -> arbitrary
    frames = np.zeros((1, 1, T, 3), dtype=np.int64)
    for t in range(prot):
        frames[0, 0, t] = [t * 4, t * 4 + 1, t * 4 + 2]
    feats = {
        "asym_id": asym[None], "mol_type": mol[None], "token_pad_mask": token_pad[None],
        "atom_pad_mask": atom_pad[None], "atom_to_token": atom_to_token[None],
        "token_to_rep_atom": token_to_rep[None], "frames_idx": frames,
    }  # fmt: skip
    return feats, T, A


def _torch_feats(feats):
    import torch

    return {k: torch.from_numpy(v) for k, v in feats.items()}


def _coords(rng, A, mult):
    # random-walk "protein" + compact ligand blobs so contacts/frames are non-degenerate
    x = np.cumsum(rng.standard_normal((mult, A, 3)).astype(np.float32) * 1.5, axis=1)
    return x


def test_compute_ptms_and_frames_vs_torch():
    torch = pytest.importorskip("torch")
    from boltz.model.layers import confidence_utils as tcu

    rng = np.random.default_rng(1)
    feats, T, A = _synthetic_feats(rng)
    for mult in (1, 3):
        x = _coords(rng, A, mult)
        logits = rng.standard_normal((mult, T, T, 64)).astype(np.float32) * 2
        tf = _torch_feats(feats)
        tx = torch.from_numpy(x)
        t_idx, t_mask = tcu.compute_frame_pred(tx, tf["frames_idx"], tf, mult, inference=True)
        f = {k: mx.array(v) for k, v in feats.items()}
        m_idx, m_mask = compute_frame_pred(mx.array(x), f["frames_idx"], f, mult, inference=True)
        assert np.array_equal(np.array(m_idx), t_idx.numpy()), "frame indices differ"
        assert np.array_equal(np.array(m_mask), t_mask.numpy()), "frame masks differ"
        ref = tcu.compute_ptms(torch.from_numpy(logits), tx, tf, mult)
        got = compute_ptms(mx.array(logits), mx.array(x), f, mult)
        for nm, g, r in zip(["ptm", "iptm", "ligand_iptm", "protein_iptm"], got[:4], ref[:4]):
            m = assert_close(g, r.numpy(), rel=1e-5, cos=0.0, name=f"synthetic(mult={mult}):{nm}")
            assert m["max_abs"] <= 1e-5
        assert set(got[4]) == set(ref[4])
        for i in ref[4]:
            for j in ref[4][i]:
                m = assert_close(got[4][i][j], ref[4][i][j].numpy(), rel=1e-5, cos=0.0,
                                 name=f"synthetic(mult={mult}):pair[{i}][{j}]")  # fmt: skip
                assert m["max_abs"] <= 1e-5


def test_aggregated_metric_and_cdist_vs_torch():
    torch = pytest.importorskip("torch")
    from boltz.model.layers import confidence_utils as tcu

    rng = np.random.default_rng(2)
    for nb, end in ((50, 1.0), (64, 32)):
        lg = rng.standard_normal((2, 9, 9, nb)).astype(np.float32)
        r = tcu.compute_aggregated_metric(torch.from_numpy(lg), end=end).numpy()
        m = assert_close(compute_aggregated_metric(mx.array(lg), end=end), r, rel=1e-6, cos=0.0,
                         name=f"aggregated_metric(bins={nb},end={end})")  # fmt: skip
        assert m["max_abs"] <= 1e-6 * end
    for n in (10, 60):  # direct (<=25) and matmul (>25) torch cdist paths
        x = (rng.standard_normal((2, n, 3)) * 20).astype(np.float32)
        r = torch.cdist(torch.from_numpy(x), torch.from_numpy(x)).numpy()
        assert_close(cdist(mx.array(x), mx.array(x)), r, rel=1e-6, cos=0.0, name=f"cdist(n={n})")


def test_confidence_module_multiplicity_vs_torch(conf):
    """run_sequentially=True with multiplicity=2 (lig_aff, 2nd sample perturbed) vs upstream torch module."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("boltz")
    from boltz.model.modules.confidencev2 import ConfidenceModule as TorchConf

    torch.set_num_threads(8)
    h = _hparams()
    case = "lig_aff"
    d = load_dump(CM, case)
    feats_np = {k: np.array(v) for k, v in load_feats(case).items()}
    rng = np.random.default_rng(3)
    x0 = d["kw_x_pred"]
    x = np.concatenate([x0, x0 + rng.standard_normal(x0.shape).astype(np.float32) * 0.7], axis=0)

    t = TorchConf(h["token_s"], h["token_z"], **conf_kwargs())
    t.load_state_dict({k: torch.from_numpy(np.array(v)) for k, v in subtree(CM).items()}, strict=True)
    t.eval()
    with torch.no_grad():
        ref = t(
            s_inputs=torch.from_numpy(d["kw_s_inputs"]), s=torch.from_numpy(d["kw_s"]), z=torch.from_numpy(d["kw_z"]),
            x_pred=torch.from_numpy(x), feats={k: torch.from_numpy(v) for k, v in feats_np.items()},
            pred_distogram_logits=torch.from_numpy(d["kw_pred_distogram_logits"]), multiplicity=2,
            run_sequentially=True,
        )  # fmt: skip
    del t

    feats = {k: mx.array(v) for k, v in feats_np.items()}
    kw = dict(
        s_inputs=mxa(d["kw_s_inputs"]), s=mxa(d["kw_s"]), z=mxa(d["kw_z"]), x_pred=mxa(x), feats=feats,
        pred_distogram_logits=mxa(d["kw_pred_distogram_logits"]), multiplicity=2,
    )  # fmt: skip
    out = conf(run_sequentially=True, **kw)
    _ev(*[v for k, v in out.items() if k != "pair_chains_iptm"])
    for k in HEAD_KEYS:
        r = ref[k].numpy()
        assert out[k].shape == r.shape and out[k].shape[0] == 2, k
        if r.size <= 2:
            m = assert_close(out[k], r, rel=1e-5, cos=0.0, name=f"mult2-seq:{k}")
            assert m["max_abs"] <= 1e-5, (k, m)
        else:
            assert_close(out[k], r, rel=REL, cos=COS, name=f"mult2-seq:{k}")
    for i in ref["pair_chains_iptm"]:
        for j in ref["pair_chains_iptm"][i]:
            m = assert_close(out["pair_chains_iptm"][i][j], ref["pair_chains_iptm"][i][j].numpy(), rel=1e-5, cos=0.0,
                             name=f"mult2-seq:pair[{i}][{j}]")  # fmt: skip
            assert m["max_abs"] <= 1e-5
    # the batched (non-sequential) path must agree with the sequential one
    out_b = conf(run_sequentially=False, **kw)
    _ev(*[v for k, v in out_b.items() if k != "pair_chains_iptm"])
    for k in ["plddt", "pae", "pde", "ptm", "iptm", "ligand_iptm", "complex_plddt", "complex_iplddt", "complex_ipde"]:
        assert_close(out_b[k], ref[k].numpy(), rel=1e-4, cos=0.0, name=f"mult2-batched:{k}")
