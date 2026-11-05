"""MLX port of boltz.model.layers.confidence_utils (the v2 confidence helpers).

Inference only.  The chain bookkeeping in `compute_frame_pred` is data dependent (Python loops over
chains, boolean masks), so those index computations run in numpy on the (small) int feats; all
floating point math stays in MLX fp32 exactly as upstream.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

# boltz.data.const.chain_type_ids
PROTEIN, DNA, RNA, NONPOLYMER = 0, 1, 2, 3


def repeat_interleave(x: mx.array, repeats: int, axis: int = 0) -> mx.array:
    return x if repeats == 1 else mx.repeat(x, repeats, axis=axis)


def cdist(x: mx.array, y: mx.array) -> mx.array:
    """torch.cdist (p=2, compute_mode='use_mm_for_euclid_dist_if_necessary') for fp32 (B,P,3) x (B,R,3).

    torch switches to the |x|^2 + |y|^2 - 2 x.y matmul formulation (clamped at 0) when P > 25 or
    R > 25; we mimic that so that distance-derived bin indices flip at the same places.
    """
    P, R = x.shape[-2], y.shape[-2]
    if P > 25 or R > 25:
        x_norm = (x * x).sum(axis=-1, keepdims=True)
        y_norm = (y * y).sum(axis=-1, keepdims=True)
        x_ = mx.concatenate([x * -2.0, x_norm, mx.ones_like(x_norm)], axis=-1)
        y_ = mx.concatenate([y, mx.ones_like(y_norm), y_norm], axis=-1)
        return mx.sqrt(mx.maximum(mx.matmul(x_, mx.swapaxes(y_, -1, -2)), 0.0))
    diff = x[..., :, None, :] - y[..., None, :, :]
    return mx.sqrt((diff * diff).sum(axis=-1))


def _arange_f32(start: float, end: float, step: float) -> mx.array:
    """torch.arange(start, end, step) in float32 (elements computed in double, then rounded)."""
    n = int(np.ceil((end - start) / step))
    return mx.array((start + np.arange(n, dtype=np.float64) * step).astype(np.float32))


def compute_collinear_mask(v1: mx.array, v2: mx.array) -> mx.array:
    norm1 = mx.sqrt((v1 * v1).sum(axis=1, keepdims=True))
    norm2 = mx.sqrt((v2 * v2).sum(axis=1, keepdims=True))
    v1 = v1 / (norm1 + 1e-6)
    v2 = v2 / (norm2 + 1e-6)
    mask_angle = mx.abs((v1 * v2).sum(axis=1)) < 0.9063
    mask_overlap1 = norm1.reshape(-1) > 1e-2
    mask_overlap2 = norm2.reshape(-1) > 1e-2
    return mask_angle & mask_overlap1 & mask_overlap2


def compute_frame_pred(pred_atom_coords, frames_idx_true, feats, multiplicity, resolved_mask=None, inference=False):
    """Frames (idx triplets) per token + collinear/overlap mask.  Returns (frames_idx_pred, mask) with
    shapes (B//mult, mult, T, 3) and (B//mult, mult, T)."""
    asym_id_token = np.array(feats["asym_id"])
    token_pad_mask = np.array(feats["token_pad_mask"])
    atom_pad_mask = np.array(feats["atom_pad_mask"])
    mol_type = np.array(feats["mol_type"])
    # asym id of every atom (atom_to_token is a one-hot matrix)
    asym_id_atom = np.einsum("bat,bt->ba", np.array(feats["atom_to_token"]).astype(np.float32), asym_id_token.astype(np.float32))

    B, N, _ = pred_atom_coords.shape
    nb = B // multiplicity
    pred_atom_coords = pred_atom_coords.reshape(nb, multiplicity, -1, 3)
    frames_idx_pred = np.array(
        repeat_interleave(frames_idx_true, multiplicity, 0).reshape(nb, multiplicity, -1, 3)
    ).copy()

    for i in range(nb):
        pred_atom_coord = pred_atom_coords[i]  # (mult, N, 3)
        token_idx = 0
        atom_idx = 0
        for cid in np.unique(asym_id_token[i]):
            mask_chain_token = (asym_id_token[i] == cid) * token_pad_mask[i]
            mask_chain_atom = (asym_id_atom[i] == cid) * atom_pad_mask[i]
            num_tokens = int(mask_chain_token.sum())
            num_atoms = int(mask_chain_atom.sum())
            if mol_type[i, token_idx] != NONPOLYMER or num_atoms < 3:
                token_idx += num_tokens
                atom_idx += num_atoms
                continue
            sel = np.nonzero(mask_chain_atom.astype(bool))[0]
            c = pred_atom_coord[:, mx.array(sel)]  # (mult, n, 3)
            diff = c[:, None, :, :] - c[:, :, None, :]
            dist_mat = mx.sqrt((diff * diff).sum(axis=-1))
            if inference:
                pm = atom_pad_mask[i][sel]
            else:
                if resolved_mask is None:
                    resolved_mask = feats["atom_resolved_mask"]
                pm = np.array(resolved_mask)[i][sel]
            resolved_pair = 1 - (pm[None, :] * pm[:, None]).astype(np.float32)
            resolved_pair[resolved_pair == 1] = np.inf
            indices = np.array(mx.argsort(dist_mat + mx.array(resolved_pair), axis=2))
            frames = np.concatenate([indices[:, :, 1:2], indices[:, :, 0:1], indices[:, :, 2:3]], axis=2) + atom_idx
            frames_idx_pred[i, :, token_idx : token_idx + num_atoms, :] = frames
            token_idx += num_tokens
            atom_idx += num_atoms

    frames_idx_pred = mx.array(frames_idx_pred)
    frames_expanded = pred_atom_coords[
        mx.arange(nb)[:, None, None, None],
        mx.arange(multiplicity)[None, :, None, None],
        frames_idx_pred,
    ].reshape(-1, 3, 3)

    mask_collinear_pred = compute_collinear_mask(
        frames_expanded[:, 1] - frames_expanded[:, 0],
        frames_expanded[:, 1] - frames_expanded[:, 2],
    ).reshape(nb, multiplicity, -1)
    return frames_idx_pred, mask_collinear_pred.astype(mx.float32) * feats["token_pad_mask"][:, None, :]


def compute_aggregated_metric(logits: mx.array, end: float = 1.0) -> mx.array:
    num_bins = logits.shape[-1]
    bin_width = end / num_bins
    bounds = _arange_f32(0.5 * bin_width, end, bin_width)
    probs = mx.softmax(logits, axis=-1)
    return mx.sum(probs * bounds, axis=-1)


def tm_function(d: mx.array, Nres: mx.array) -> mx.array:
    d0 = 1.24 * (mx.maximum(Nres, 19.0) - 15) ** (1 / 3) - 1.8
    return 1 / (1 + (d / d0) ** 2)


def _masked_row_max(tm_expected_value: mx.array, mask: mx.array) -> mx.array:
    """max over rows of sum_j(tm * mask) / (sum_j(mask) + 1e-5)."""
    return mx.max(mx.sum(tm_expected_value * mask, axis=-1) / (mx.sum(mask, axis=-1) + 1e-5), axis=1)


def compute_ptms(logits, x_preds, feats, multiplicity):
    _, mask_collinear_pred = compute_frame_pred(x_preds, feats["frames_idx"], feats, multiplicity, inference=True)
    mask_pad = repeat_interleave(feats["token_pad_mask"], multiplicity, 0)
    maski = mask_collinear_pred.reshape(-1, mask_collinear_pred.shape[-1])
    pair_mask_ptm = maski[:, :, None] * mask_pad[:, None, :] * mask_pad[:, :, None]
    asym_id = repeat_interleave(feats["asym_id"], multiplicity, 0)
    diff_chain = (asym_id[:, None, :] != asym_id[:, :, None]).astype(mx.float32)
    pair_mask_iptm = maski[:, :, None] * diff_chain * mask_pad[:, None, :] * mask_pad[:, :, None]

    num_bins = logits.shape[-1]
    bin_width = 32.0 / num_bins
    pae_value = _arange_f32(0.5 * bin_width, 32.0, bin_width)[None]
    N_res = mask_pad.sum(axis=-1, keepdims=True)
    tm_value = tm_function(pae_value, N_res)[:, None, None, :]
    probs = mx.softmax(logits, axis=-1)
    tm_expected_value = mx.sum(probs * tm_value, axis=-1)  # (B, N, N)

    ptm = _masked_row_max(tm_expected_value, pair_mask_ptm)
    iptm = _masked_row_max(tm_expected_value, pair_mask_iptm)

    token_type = repeat_interleave(feats["mol_type"], multiplicity, 0)
    is_ligand_token = (token_type == NONPOLYMER).astype(mx.float32)
    is_protein_token = (token_type == PROTEIN).astype(mx.float32)

    base = maski[:, :, None] * diff_chain * mask_pad[:, None, :] * mask_pad[:, :, None]
    ligand_iptm_mask = base * (
        (is_ligand_token[:, :, None] * is_protein_token[:, None, :])
        + (is_protein_token[:, :, None] * is_ligand_token[:, None, :])
    )
    protein_iptm_mask = base * (is_protein_token[:, :, None] * is_protein_token[:, None, :])
    ligand_iptm = _masked_row_max(tm_expected_value, ligand_iptm_mask)
    protein_iptm = _masked_row_max(tm_expected_value, protein_iptm_mask)

    chain_pair_iptm = {}
    asym_ids_list = [int(a) for a in np.unique(np.array(asym_id))]
    for idx1 in asym_ids_list:
        chain_iptm = {}
        for idx2 in asym_ids_list:
            mask_pair_chain = (
                maski[:, :, None]
                * (asym_id[:, None, :] == idx1).astype(mx.float32)
                * (asym_id[:, :, None] == idx2).astype(mx.float32)
                * mask_pad[:, None, :]
                * mask_pad[:, :, None]
            )
            chain_iptm[idx2] = _masked_row_max(tm_expected_value, mask_pair_chain)
        chain_pair_iptm[idx1] = chain_iptm
    return ptm, iptm, ligand_iptm, protein_iptm, chain_pair_iptm
