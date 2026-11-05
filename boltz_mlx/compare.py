"""Parity metrics between MLX outputs and PyTorch reference dumps."""

from __future__ import annotations

import numpy as np


def to_numpy(x) -> np.ndarray:
    if hasattr(x, "__array__") and not isinstance(x, np.ndarray):
        return np.array(x)
    return np.asarray(x)


def metrics(ours, ref, mask=None) -> dict:
    """max abs err, max rel err (relative to max |ref|), cosine similarity, in float64."""
    a = to_numpy(ours).astype(np.float64)
    b = to_numpy(ref).astype(np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    if mask is not None:
        m = np.broadcast_to(to_numpy(mask).astype(bool), a.shape)
        a, b = a[m], b[m]
    diff = np.abs(a - b)
    scale = max(np.abs(b).max(initial=0.0), 1e-12)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    cos = float(a.ravel() @ b.ravel() / denom) if denom > 0 else (1.0 if diff.max(initial=0) == 0 else 0.0)
    return {
        "max_abs": float(diff.max(initial=0.0)),
        "max_rel": float(diff.max(initial=0.0) / scale),
        "mean_abs": float(diff.mean()) if diff.size else 0.0,
        "cos": cos,
        "ref_absmax": float(scale),
    }


def assert_close(ours, ref, *, rel=1e-4, cos=0.99999, name="", mask=None) -> dict:
    """fp32 tolerance gate: error relative to max |ref| and cosine similarity."""
    m = metrics(ours, ref, mask)
    ok = m["max_rel"] <= rel and m["cos"] >= cos
    msg = f"{name}: max_abs={m['max_abs']:.3e} max_rel={m['max_rel']:.3e} cos={m['cos']:.8f}"
    print(msg)
    assert ok, "parity FAIL " + msg
    return m


def kabsch_rmsd(x, y, mask=None) -> float:
    """RMSD after optimal superposition of x onto y; x,y (N,3)."""
    x = to_numpy(x).astype(np.float64).reshape(-1, 3)
    y = to_numpy(y).astype(np.float64).reshape(-1, 3)
    if mask is not None:
        m = to_numpy(mask).astype(bool).reshape(-1)
        x, y = x[m], y[m]
    xc, yc = x - x.mean(0), y - y.mean(0)
    u, _, vt = np.linalg.svd(xc.T @ yc)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1, 1, d]) @ vt
    return float(np.sqrt(((xc @ r - yc) ** 2).sum(1).mean()))


def rmsd(x, y, mask=None) -> float:
    """Plain RMSD without superposition (same frame, fixed noise)."""
    x = to_numpy(x).astype(np.float64).reshape(-1, 3)
    y = to_numpy(y).astype(np.float64).reshape(-1, 3)
    if mask is not None:
        m = to_numpy(mask).astype(bool).reshape(-1)
        x, y = x[m], y[m]
    return float(np.sqrt(((x - y) ** 2).sum(1).mean()))
