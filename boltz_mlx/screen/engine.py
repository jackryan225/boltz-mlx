"""Upstream glue for the screen: per-ligand YAML + upstream preprocessing (RDKit conformer seeded with
seeding.CONFORMER_SEED, CCD), seeded featurization,
Boltz writers, model loading. The model itself is boltz_mlx; everything here is the same upstream code path that
`boltz-mlx predict` and tools/validation/run_l3.py use (per-record featurizer seed via SeededFeatureDataset).

Offline: upstream process_input is called with use_msa_server=False; a protein chain without an MSA is set to
`msa: empty` by inputs.load_target before it gets here, so no MSA request can be made.

Exactness of the receptor-MSA cache: upstream parses the target's .a3m/.csv once per record into
processed/msa/<record>_<idx>.npz (idx = position in the sorted MSA path list). The screen parses it once into
work/msa_cache/ with the same call (parse_a3m / parse_csv, same max_seqs) and hard-links it to each record's name before
process_input runs, which then skips parsing because the file exists. Same bytes, parsed once.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

from boltz_mlx.screen.inputs import Ligand, Target


class LigandFailure(RuntimeError):
    """A per-ligand failure (preprocessing / featurization / model): becomes an error row, the run continues."""


def _ensure_dirs(work: Path) -> dict[str, Path]:
    d = {
        "inputs": work / "inputs",
        "msa_raw": work / "msa",
        "msa_cache": work / "msa_cache",
        "records": work / "processed" / "records",
        "structures": work / "processed" / "structures",
        "msa": work / "processed" / "msa",
        "constraints": work / "processed" / "constraints",
        "templates": work / "processed" / "templates",
        "mols": work / "processed" / "mols",
        "predictions": work / "predictions",
    }
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    return d


class UpstreamEngine:
    """CPU side of the screen (thread-safe for one prefetch worker + the main thread: the main thread never calls
    prepare/featurize while the worker does)."""

    def __init__(self, work: Path, cache: Path, target: Target, max_msa_seqs: int = 8192, output_format: str = "mmcif",
                 global_seed: int = 0, keep_extras: bool = True):
        self.work = Path(work)
        self.keep_extras = bool(keep_extras)  # False: no pae / pde npz (cleanup() would delete them; --keep-work keeps)
        self.cache = Path(cache)
        self.mol_dir = self.cache / "mols"
        if not self.mol_dir.exists():
            raise FileNotFoundError(
                f"upstream molecule dictionary not found: {self.mol_dir}. Pass --cache <dir containing mols/> or set BOLTZ_MLX_CACHE "
                f"(see tools/install_check.sh); the screen never downloads anything.")
        self.target = target
        self.max_msa_seqs = int(max_msa_seqs)
        self.output_format = output_format
        self.global_seed = int(global_seed)
        self.d = _ensure_dirs(self.work)
        self._canonicals = None
        self._failures: dict[str, str] = {}
        self._swriter = None
        self._awriter = None
        self._install_failure_hook()
        # upstream schema.py calls AllChem.Descriptors.MolWt, which only exists when rdkit.Chem.Descriptors was imported
        # before AllChem was first imported (import-order dependent); make it unconditional
        from rdkit.Chem import AllChem, Descriptors

        if not hasattr(AllChem, "Descriptors"):
            AllChem.Descriptors = Descriptors
        # SMILES-ligand conformers independent of what this process embedded before (seeding.install_conformer_seed)
        from boltz_mlx.seeding import install_conformer_seed

        install_conformer_seed()
        from boltz_mlx.upfix import install_upstream_fixes

        install_upstream_fixes()   # narrow patches of upstream parsing / writers (boltz_mlx/upfix)

    # ------------------------------------------------------------------ preprocessing
    def _install_failure_hook(self) -> None:
        """upstream process_input swallows exceptions and only prints 'Failed to process <path>. Skipping. Error: <e>.':
        record that message (module-level `print` override in boltz.main only) so the row carries the reason."""
        import boltz.main as bm

        failures = self._failures
        builtin_print = print

        def _print(*a, **kw):
            msg = " ".join(str(x) for x in a)
            if msg.startswith("Failed to process "):
                path = msg[len("Failed to process "):].split(". Skipping.")[0]
                failures[Path(path).stem] = msg.split("Error: ", 1)[-1].rstrip(".")
            builtin_print(*a, **kw)

        bm.print = _print

    def canonicals(self):
        if self._canonicals is None:
            from boltz.data.mol import load_canonicals

            self._canonicals = load_canonicals(self.mol_dir)
        return self._canonicals

    def _msa_cached(self, idx: int, path: str) -> Path:
        from boltz.data.parse.a3m import parse_a3m
        from boltz.data.parse.csv import parse_csv

        h = hashlib.sha256(f"{path}|{os.path.getmtime(path)}|{os.path.getsize(path)}|{self.max_msa_seqs}".encode()).hexdigest()[:16]
        out = self.d["msa_cache"] / f"{h}.npz"
        if not out.exists():
            p = Path(path)
            msa = parse_a3m(p, taxonomy=None, max_seqs=self.max_msa_seqs) if p.suffix == ".a3m" else parse_csv(p, max_seqs=self.max_msa_seqs)
            tmp = out.with_name(out.stem + ".tmp.npz")
            msa.dump(tmp)
            os.replace(tmp, out)
        return out

    def record_path(self, lig: Ligand) -> Path:
        return self.d["records"] / f"{lig.record_id}.json"

    def prepare(self, lig: Ligand) -> None:
        """Write inputs/<record>.yaml and run upstream process_input (once per record; reused on resume)."""
        import yaml
        from boltz.main import process_input

        if self.record_path(lig).exists():
            return
        yml = self.d["inputs"] / f"{lig.record_id}.yaml"
        yml.write_text(yaml.safe_dump(self.target.ligand_yaml(lig.smiles_used), sort_keys=False))
        for idx, mp in enumerate(self.target.msa_paths):
            dst = self.d["msa"] / f"{lig.record_id}_{idx}.npz"
            if not dst.exists():
                src = self._msa_cached(idx, mp)
                try:
                    os.link(src, dst)
                except OSError:
                    shutil.copyfile(src, dst)
        self._failures.pop(lig.record_id, None)
        process_input(
            path=yml, ccd=self.canonicals(), msa_dir=self.d["msa_raw"], mol_dir=self.mol_dir, boltz2=True,
            use_msa_server=False, msa_server_url="", msa_pairing_strategy="greedy", msa_server_username=None,
            msa_server_password=None, api_key_header=None, api_key_value=None, max_msa_seqs=self.max_msa_seqs,
            processed_msa_dir=self.d["msa"], processed_constraints_dir=self.d["constraints"],
            processed_templates_dir=self.d["templates"], processed_mols_dir=self.d["mols"],
            structure_dir=self.d["structures"], records_dir=self.d["records"],
        )
        if not self.record_path(lig).exists():
            raise LigandFailure(f"preprocessing failed: {self._failures.get(lig.record_id, 'see log')}")

    # ------------------------------------------------------------------ featurization
    def _dataset(self, lig: Ligand, affinity: bool):
        from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
        from boltz.data.types import Manifest, Record

        rec = Record.load(self.record_path(lig))
        dm = Boltz2InferenceDataModule(
            manifest=Manifest([rec]),
            target_dir=self.d["predictions"] if affinity else self.d["structures"],
            msa_dir=self.d["msa"], mol_dir=self.mol_dir, num_workers=0,
            constraints_dir=self.d["constraints"], template_dir=self.d["templates"], extra_mols_dir=self.d["mols"],
            override_method="other" if affinity else None, affinity=affinity,
        )
        return dm.predict_dataloader().dataset

    def featurize(self, lig: Ligand, affinity: bool = False) -> dict:
        """Collated upstream batch (torch tensors + 'record' [+ 'affinity_mw']) for one record, featurizer seeded per record.

        Upstream PredictionDataset falls back to item 0 when tokenization/cropping/featurization fails; that fallback is
        turned into a LigandFailure carrying the original exception."""
        from boltz.data.module.inferencev2 import collate

        from boltz_mlx.seeding import SeededFeatureDataset

        ds = self._dataset(lig, affinity)

        def _fallback(_idx):
            exc = sys.exc_info()[1]
            raise LigandFailure(f"{'affinity ' if affinity else ''}featurization failed: {exc!r}")

        ds.__getitem__ = _fallback  # instance attribute: only the upstream fallback `self.__getitem__(0)` sees it
        item = SeededFeatureDataset(ds, self.global_seed)[0]
        if item["record"].id != lig.record_id:
            raise LigandFailure(f"featurizer returned record {item['record'].id!r} for {lig.record_id!r}")
        return collate([item])

    def model_inputs(self, batch: dict) -> dict:
        """torch batch -> dict of mx arrays (main thread)."""
        from boltz_mlx.cli import _batch_to_mx, _warn_unsupported_guidance

        feats = _batch_to_mx(batch)
        _warn_unsupported_guidance(feats)
        from boltz_mlx import cachelimit

        cachelimit.adapt_feats(feats)  # no-op up to 1000 tokens and when init_mlx did not arm the policy
        return feats

    def affinity_mw(self, batch: dict) -> float | None:
        mw = batch.get("affinity_mw")
        if mw is None:
            return None
        return float(mw[0])

    def ligand_asym(self, batch: dict, chain: str) -> int | None:
        rec = batch["record"][0]
        for c in rec.chains:
            if c.chain_name == chain:
                return int(c.chain_id)
        return None

    # ------------------------------------------------------------------ writers
    def write_structure(self, out: dict, batch: dict) -> None:
        from boltz.data.write.writer import BoltzWriter

        from boltz_mlx.cli import _structure_pred_dict

        if self._swriter is None:
            self._swriter = BoltzWriter(data_dir=self.d["structures"], output_dir=self.d["predictions"],
                                        output_format=self.output_format, boltz2=True)
        # s / z are only read by a writer with write_embeddings=True (not this one)
        pred = _structure_pred_dict(out, batch, embeddings=False, pae_pde=self.keep_extras)
        self._swriter.write_on_batch_end(None, None, pred, None, batch, 0, 0)

    def write_affinity(self, out: dict, batch: dict) -> None:
        from boltz.data.write.writer import BoltzAffinityWriter

        from boltz_mlx.cli import _to_torch

        if self._awriter is None:
            self._awriter = BoltzAffinityWriter(data_dir=self.d["structures"], output_dir=self.d["predictions"])
        pred = {"exception": False}
        for k in ("affinity_pred_value", "affinity_probability_binary", "affinity_pred_value1",
                  "affinity_probability_binary1", "affinity_pred_value2", "affinity_probability_binary2"):
            pred[k] = _to_torch(out[k])
        self._awriter.write_on_batch_end(None, None, pred, None, batch, 0, 0)

    def pred_dir(self, lig: Ligand) -> Path:
        return self.d["predictions"] / lig.record_id

    def pose_path(self, lig: Ligand) -> Path:
        ext = {"mmcif": "cif", "pdb": "pdb"}[self.output_format]
        return self.pred_dir(lig) / f"{lig.record_id}_model_0.{ext}"

    def pre_affinity_path(self, lig: Ligand) -> Path:
        return self.pred_dir(lig) / f"pre_affinity_{lig.record_id}.npz"

    def cleanup(self, lig: Ligand) -> None:
        """Drop bulky per-ligand intermediates once the row is final (pose, confidence json and affinity json stay)."""
        d = self.pred_dir(lig)
        for pat in ("pae_*.npz", "pde_*.npz", "plddt_*.npz", "pre_affinity_*.npz"):
            for p in d.glob(pat):
                p.unlink(missing_ok=True)

    # ------------------------------------------------------------------ model
    def load_model(self, name: str):
        from boltz_mlx.model import load_model

        return load_model(name)


def config_key(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]
