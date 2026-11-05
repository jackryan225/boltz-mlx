"""Featurize a Boltz-2 input YAML with the upstream (PyTorch/NumPy) pipeline and save the batch.

Usage:
  python tools/prep_case.py cases/prot117.yaml            # structure feats -> data/prot117/feats.npz
  python tools/prep_case.py cases/lig_aff.yaml --affinity # affinity feats (needs predictions/ from a reference run)

Featurization stays upstream (boltz.data); MLX receives these arrays at the model boundary.
"""

import os
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
from boltz.data.types import Manifest
from boltz.main import process_inputs

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # the editable install points at the main checkout: use this tree's boltz_mlx (seeding)
    sys.path.insert(0, str(ROOT))
CACHE = Path(os.environ.get("BOLTZ_MLX_CACHE", Path.home() / ".boltz")).expanduser()


def batch_to_numpy(batch: dict) -> tuple[dict, dict]:
    arrays, other = {}, {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            arrays[k] = v.numpy()
        else:
            other[k] = v
    return arrays, other


def load_batch_dataset(case_dir: Path, affinity: bool, pred_dir: Path | None = None):
    """The upstream PredictionDataset for a processed case dir (featurizes lazily on __getitem__)."""
    return _data_module(case_dir, affinity, pred_dir).predict_dataloader().dataset


def load_batch(case_dir: Path, affinity: bool, pred_dir: Path | None = None) -> dict:
    return next(iter(_data_module(case_dir, affinity, pred_dir).predict_dataloader()))


def _data_module(case_dir: Path, affinity: bool, pred_dir: Path | None = None):
    processed = case_dir / "processed"
    manifest = Manifest.load(processed / "manifest.json")
    return Boltz2InferenceDataModule(
        manifest=manifest,
        target_dir=(pred_dir or case_dir / "ref" / "predictions") if affinity else processed / "structures",
        msa_dir=processed / "msa",
        mol_dir=CACHE / "mols",
        num_workers=0,
        constraints_dir=(processed / "constraints") if (processed / "constraints").exists() else None,
        template_dir=(processed / "templates") if (processed / "templates").exists() else None,
        extra_mols_dir=(processed / "mols") if (processed / "mols").exists() else None,
        override_method="other" if affinity else None,
        affinity=affinity,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("yaml", type=Path)
    ap.add_argument("--affinity", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pred-dir", type=Path, default=None, help="structure predictions dir for --affinity (default data/<case>/ref/predictions)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    name = args.yaml.stem
    case_dir = ROOT / "data" / name
    case_dir.mkdir(parents=True, exist_ok=True)

    if not (case_dir / "processed" / "manifest.json").exists():
        from boltz_mlx.seeding import install_conformer_seed

        install_conformer_seed()  # SMILES-ligand conformers seeded (seeding.py); one case per process got seed 42 anyway
        from boltz_mlx.upfix import install_upstream_fixes

        install_upstream_fixes()
        process_inputs(
            data=[args.yaml.resolve()],
            out_dir=case_dir,
            ccd_path=CACHE / "ccd.pkl",
            mol_dir=CACHE / "mols",
            msa_server_url="",
            msa_pairing_strategy="greedy",
            use_msa_server=False,
            boltz2=True,
        )

    batch = load_batch(case_dir, args.affinity, args.pred_dir)
    arrays, other = batch_to_numpy(batch)
    out = case_dir / ("feats_affinity.npz" if args.affinity else "feats.npz")
    np.savez(out, **arrays)
    with open(out.with_suffix(".meta.pkl"), "wb") as f:
        pickle.dump(other, f)
    n_tok = int(arrays["token_pad_mask"].sum())
    n_atom = int(arrays["atom_pad_mask"].sum())
    print(f"{name}: {len(arrays)} arrays -> {out} | tokens={n_tok} atoms={n_atom}")
    for k in sorted(arrays):
        print(f"  {k:32s} {arrays[k].dtype} {arrays[k].shape}")


if __name__ == "__main__":
    main()
