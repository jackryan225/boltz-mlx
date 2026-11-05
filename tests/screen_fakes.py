"""Test doubles for `boltz-mlx screen` (CPU, no checkpoints): a fake model whose outputs are a deterministic function of the
per-record noise (and of the features), a fake engine (no upstream featurization), and a mock for the real engine's model."""

from __future__ import annotations

import json
import os
from pathlib import Path

import boltz_mlx  # noqa: F401  (TF32 env before MLX initialises)
import mlx.core as mx
import numpy as np

from boltz_mlx.screen.engine import LigandFailure

# CPU only (also when imported without tests/conftest.py, e.g. by tools/install_check.sh)
mx.set_default_device(mx.gpu if os.environ.get("BOLTZ_MLX_DEVICE", "cpu") == "gpu" else mx.cpu)


class FakeModel:
    alive = 0        # instances currently referenced (one-checkpoint-resident check)
    max_alive = 0
    calls: list = []  # (name, record tag) per call
    steps_log: list = []  # (name, args.sampling_steps) per call

    def __init__(self, name: str):
        self.name = name
        FakeModel.alive += 1
        FakeModel.max_alive = max(FakeModel.max_alive, FakeModel.alive)

    def __del__(self):
        FakeModel.alive -= 1

    @classmethod
    def reset(cls):
        cls.alive = 0
        cls.max_alive = 0
        cls.calls = []
        cls.steps_log = []

    def __call__(self, feats, args, noise=None, mw_correction=False, timings=None):
        n_tok = int(np.array(feats["token_pad_mask"]).sum())
        n_at = int(np.array(feats["atom_pad_mask"]).shape[-1])
        M = args.diffusion_samples
        draws = np.array(noise.randn((M, n_at, 3)))      # per-record seeded noise
        FakeModel.calls.append((self.name, float(draws.reshape(-1)[0])))
        FakeModel.steps_log.append((self.name, args.sampling_steps))
        base = float(draws.mean()) + 0.001 * n_tok
        ref = np.array(feats["ref_pos"]) if "ref_pos" in feats else np.zeros((1, n_at, 3), np.float32)
        coords = mx.array((ref.reshape(1, n_at, 3) + 0.1 * draws).astype(np.float32))
        conf = {
            "plddt": mx.full((M, n_tok), 0.5 + 0.1 * np.tanh(base)), "pae": mx.zeros((M, n_tok, n_tok)), "pde": mx.zeros((M, n_tok, n_tok)),
            "complex_plddt": mx.array([0.8 + 0.01 * base] * M), "complex_iplddt": mx.array([0.7] * M),
            "complex_pde": mx.array([1.0] * M), "complex_ipde": mx.array([1.0] * M), "ptm": mx.array([0.6] * M),
            "iptm": mx.array([0.5 + 0.01 * k + 0.01 * base for k in range(M)]), "ligand_iptm": mx.array([0.4 + 0.01 * base] * M),
            "protein_iptm": mx.array([0.3] * M),
            "pair_chains_iptm": {0: {0: mx.array([0.6] * M), 1: mx.array([0.5] * M)}, 1: {0: mx.array([0.5] * M), 1: mx.array([0.6] * M)}},
        }
        out = {"sample_atom_coords": coords, "s": mx.zeros((1, n_tok, 4)), "z": mx.zeros((1, n_tok, n_tok, 2)), **conf}
        if self.name == "aff":
            v = 1.0 + base
            p = 1.0 / (1.0 + np.exp(-3.0 * base))
            out.update(affinity_pred_value=mx.array([v]), affinity_probability_binary=mx.array([p]),
                       affinity_pred_value1=mx.array([v - 0.1]), affinity_pred_value2=mx.array([v + 0.1]),
                       affinity_probability_binary1=mx.array([p]), affinity_probability_binary2=mx.array([p]))
        return out


class FakeEngine:
    """Engine API without upstream code. `fail` maps record_id -> phase ('prepare' | 'featurize' | 'affinity') to raise in;
    `interrupt_at` = (phase, record_id) raises KeyboardInterrupt there (a kill)."""

    def __init__(self, work: Path, fail: dict | None = None, interrupt_at: tuple | None = None):
        self.work = Path(work)
        self.fail = fail or {}
        self.interrupt_at = interrupt_at
        self.prepared: list[str] = []
        self.loads: list[str] = []

    def _maybe(self, phase, lig):
        if self.interrupt_at == (phase, lig.record_id):
            self.interrupt_at = None
            raise KeyboardInterrupt
        if self.fail.get(lig.record_id) == phase:
            raise LigandFailure(f"{phase} failed (fake)")

    def prepare(self, lig):
        self._maybe("prepare", lig)
        self.prepared.append(lig.record_id)

    def featurize(self, lig, affinity=False):
        self._maybe("affinity" if affinity else "featurize", lig)
        return {"lig": lig, "affinity": affinity}

    def model_inputs(self, batch):
        lig = batch["lig"]
        n_lig = lig.n_atoms or 5
        n_tok = 20 + n_lig if not batch["affinity"] else min(20 + n_lig, 25)
        asym = np.array([[0] * (n_tok - n_lig) + [1] * n_lig])
        return {"token_pad_mask": mx.ones((1, n_tok)), "atom_pad_mask": mx.ones((1, 3 * n_tok)), "asym_id": mx.array(asym)}

    def affinity_mw(self, batch):
        return batch["lig"].mw or 300.0

    def ligand_asym(self, batch, chain):
        return 1

    def pred_dir(self, lig):
        return self.work / "predictions" / lig.record_id

    def pose_path(self, lig):
        return self.pred_dir(lig) / f"{lig.record_id}_model_0.cif"

    def pre_affinity_path(self, lig):
        return self.pred_dir(lig) / f"pre_affinity_{lig.record_id}.npz"

    def write_structure(self, out, batch):
        lig = batch["lig"]
        self.pred_dir(lig).mkdir(parents=True, exist_ok=True)
        self.pose_path(lig).write_text("data_fake\n")
        self.pre_affinity_path(lig).write_bytes(b"x")

    def write_affinity(self, out, batch):
        lig = batch["lig"]
        (self.pred_dir(lig) / f"affinity_{lig.record_id}.json").write_text(json.dumps({"v": float(np.array(out["affinity_pred_value"])[0])}))

    def cleanup(self, lig):
        self.pre_affinity_path(lig).unlink(missing_ok=True)

    def load_model(self, name):
        self.loads.append(name)
        return FakeModel(name)
