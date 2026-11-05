"""Per-record seeding.

seed = uint32(sha256(f"{global_seed}|{record_id}|{pass}|{sample}")), pass in {structure, affinity, features}.
Noise and features then depend only on (global seed, record, pass, sample) — not on input order,
resume state, batch composition or the number of samples drawn.

Reference conformers of SMILES ligands (install_conformer_seed): upstream preprocessing embeds every SMILES ligand with
RDKit ETKDG (boltz/data/parse/schema.py compute_3d_conformer, the only ETKDG call upstream) and leaves randomSeed at -1,
so RDKit draws from one process-wide random stream: the first embedding in a process is always the same, every later
one depends on everything embedded before it in that process. A screen's or a multi-input predict's ligand conformer
(featurized ref_pos) therefore depended on the ligands before it (diagnosis 2026-10-29).
install_conformer_seed gives every embedding randomSeed = CONFORMER_SEED (42). RDKit's default stream starts as its
generator seeded with 42, so each ligand gets exactly the conformer it got as the first embedding in a fresh process
(a solo screen, the first ligand of any screen, a single-input `boltz-mlx predict` or upstream `boltz predict`), and
that conformer no longer depends on the order, chunking or resume. A constant, not a per-record seed: the conformer
does not change when a ligand is renamed. Costs, by choice:
  - --seed / the global seed does not change the conformer, so a seed ensemble samples no conformer diversity (as in
    upstream, whose seed never reaches RDKit);
  - a record with 2+ SMILES ligands (co-factors, multi-ligand predict) embeds each with seed 42, whereas upstream's
    single-input run continues the stream for the second one: only the first SMILES ligand of a record matches upstream;
  - "matches today's first-in-process conformer" relies on RDKit's default stream being seed 42
    (tests/test_conformer_seed.py pins it with a fresh subprocess); order independence holds either way.
CCD-code ligands use the CCD's stored conformers and are not embedded, so they are unaffected.
"""

from __future__ import annotations

import hashlib
import threading

PASSES = ("structure", "affinity", "features")
CONFORMER_SEED = 42  # ETKDG randomSeed for every SMILES-ligand reference conformer (see the module docstring)

_install_lock = threading.Lock()
_embed_lock = threading.Lock()


def record_seed(global_seed: int, record_id: str, pass_: str, sample: int = 0) -> int:
    if pass_ not in PASSES:
        raise ValueError(f"pass must be one of {PASSES}, got {pass_!r}")
    h = hashlib.sha256(f"{int(global_seed)}|{record_id}|{pass_}|{int(sample)}".encode()).digest()
    return int.from_bytes(h[:4], "little")


class _SeededAllChem:
    """rdkit.Chem.AllChem whose ETKDGv2 / ETKDGv3 parameter objects carry randomSeed; every other name is AllChem's."""

    def __init__(self, real, seed: int):
        self._real = real
        self._seed = int(seed)

    def ETKDGv3(self):  # noqa: N802 (RDKit's name)
        p = self._real.ETKDGv3()
        p.randomSeed = self._seed
        return p

    def ETKDGv2(self):  # noqa: N802
        p = self._real.ETKDGv2()
        p.randomSeed = self._seed
        return p

    def __getattr__(self, name):
        return getattr(self._real, name)


def install_conformer_seed() -> None:
    """Seed upstream's SMILES-ligand conformers with CONFORMER_SEED in this process (idempotent; upstream is not edited).

    Replaces boltz.data.parse.schema.compute_3d_conformer (looked up as a module global by parse_boltz_schema, which
    yaml / fasta inputs reach) with a wrapper that runs upstream's own function while schema.AllChem hands out seeded
    ETKDG parameters: same options, same fallback to random coordinates, same UFF step, only randomSeed set. Call it
    before upstream preprocessing (screen engine, predict, run_l3, prep_case).

    The swap is module-wide for the duration of each call (under a lock, so seeded calls never overlap): other
    boltz.data.parse.schema code that uses AllChem on another thread during that window also sees the proxy, i.e.
    seeded ETKDGv2/v3 parameters and AllChem's own everything else. Harmless (compute_3d_conformer is upstream's only
    ETKDG user), but not invisible. In this package only the screen's prefetch worker and the main thread exist, and
    only the worker preprocesses; the GUI server never imports upstream boltz (it runs the CLI as a subprocess)."""
    import boltz.data.parse.schema as schema

    with _install_lock:
        if getattr(schema.compute_3d_conformer, "_boltz_mlx_conformer_seed", None) is not None:
            return
        upstream = schema.compute_3d_conformer

        def compute_3d_conformer(mol, version: str = "v3") -> bool:
            with _embed_lock:  # schema.AllChem is swapped for this call only; one embedding at a time
                real = schema.AllChem
                schema.AllChem = _SeededAllChem(real, CONFORMER_SEED)
                try:
                    return upstream(mol, version)
                finally:
                    schema.AllChem = real

        compute_3d_conformer._boltz_mlx_conformer_seed = CONFORMER_SEED
        compute_3d_conformer.__wrapped__ = upstream
        compute_3d_conformer.__doc__ = upstream.__doc__
        schema.compute_3d_conformer = compute_3d_conformer


class SeededFeatureDataset:
    """Wraps an upstream PredictionDataset: seeds torch's global RNG per record before featurizing.

    Upstream seeds its numpy RNG with a constant (42) but the reference-conformer roto-translation
    (featurizerv2 -> center_random_augmentation) draws from torch's global RNG, unseeded. (The conformer itself is
    embedded earlier, in preprocessing: install_conformer_seed.)
    """

    def __init__(self, dataset, global_seed: int):
        self.dataset = dataset
        self.global_seed = int(global_seed)

    def __len__(self):
        return len(self.dataset)

    def record_id(self, idx: int) -> str:
        return self.dataset.manifest.records[idx].id

    def __getitem__(self, idx):
        import torch

        torch.manual_seed(record_seed(self.global_seed, self.record_id(idx), "features"))
        return self.dataset[idx]
