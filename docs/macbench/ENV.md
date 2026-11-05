# macbench environments (built 2026-10-31, nothing run on the GPU)

_Public copy: internal paths replaced by `macbench-env` (`$MACBENCH_ENV`) and the Boltz cache; nothing else changed._

Root: `macbench-env` in the checkout (`$MACBENCH_ENV`; `src/`, `venv-rbvi/`, `venv-stock/`, `msa/`, install logs, freezes). Read-only inputs: the Boltz cache (`$BOLTZ_MLX_CACHE`, default `~/.boltz`)
(Boltz-2 layout: `boltz2_conf.ckpt`, `boltz2_aff.ckpt`, `mols/`), nothing written to it. The ChimeraX app was not installed.

| env | what | python | boltz | torch | other pins |
|---|---|---|---|---|---|
| `venv-rbvi` (PRIMARY, what ChimeraX installs) | https://github.com/RBVI/boltz branch `chimerax_boltz22`, HEAD `2be3519b5642f13b6519c408b5602a938aa8610d` (2026-10-22), cloned to `src/rbvi-boltz`, installed non-editable with `uv pip install` from a copy built outside the source dir | 3.12.13 | 2.2.0 (the fork's pyproject version; base = upstream v2.2.0-27, merge-base `c9b6af1`) | **2.7.1** (the fork pins `torch==2.7.1`: "2.9.1 is 50% slower on Mac") | numpy 1.26.4, pytorch-lightning 2.5.0, rdkit 2026.9.1, gemmi 0.6.5, scikit-learn 1.6.1, cuequivariance-torch 0.10.0 + cuequivariance 0.13.0 (installed on Mac, unused without CUDA), hydra-core 1.3.2 |
| `venv-stock` (secondary) | PyPI `boltz==2.2.1` (tag `cb04aec`) | 3.12.13 | 2.2.1 | 2.15.0 (unpinned: latest wheel) | numpy 1.26.4, pytorch-lightning 2.5.0, rdkit 2026.9.1 |
| `venv-community` (the Mac comparison since 2026-11-02) | https://github.com/Novel-Therapeutics/boltz-community, default branch `main`, HEAD `1431d573c3a02a2cb6b5c3b775e7128c7a98d539` (tag v2.13.0), cloned to `src/boltz-community` (remote `upstream` = jwohlwend/boltz added for diffs; merge-base `cb04aec` = v2.2.1), installed non-editable with `nice uv pip install` run from `~` (not from the source dir); log `install_community.log` | 3.12.13 | distribution `boltz-community` 2.13.0 (package `boltz`) | **2.15.0** (unpinned `torch>=2.4`: latest wheel; the RBVI venv has 2.7.1) | numpy 2.5.4, pytorch-lightning 2.6.6, rdkit 2026.9.1, scikit-learn 1.9.1, click 8.5.0, torchmetrics 1.9.0; `freeze_community.txt` (86 lines) |
| MLX build | the project venv, branch `boltz-macbench`, `boltz_mlx` tree `b3a0ef23d87072e3c3970ea78f81fd191ab42815` = tag `r2-m5max-4` | 3.12 | featurizer: an upstream boltz checkout (v2.2.1-6, `b1ebfc4`, editable) | torch 2.14.1 (featurization only), mlx 0.32.4 | rdkit 2026.9.1 |

An existing install venv (boltz 2.2.1, torch 2.15.0) was NOT reused: its provenance was not checked; `venv-stock` is a fresh install of the
same versions (same boltz 2.2.1 and torch 2.15.0 resolved). Exact package lists: `freeze_rbvi.txt`, `freeze_stock.txt`, `freeze_community.txt` (here and in the env dir).
Smoke tests done (CPU only, `--accelerator cpu`, 2 sampling steps, `nice`, 2 threads): `boltz predict --help` in both venvs, one reduced
prediction in each (prot117; and the affinity path on 8rf4aff in the RBVI venv) through `tools/macbench/up_predict.py`; no GPU prediction.
`torch.backends.mps.is_available()` is True in both venvs (checked, no tensor was placed on it).

## What the RBVI fork changes relative to upstream (read from the diff before installing)

`git diff c9b6af1 2be3519` (its merge-base with upstream): 7 files, +96 / -39. **There is no MPS-specific model or kernel code.** MPS works through
stock PyTorch plus `PYTORCH_ENABLE_MPS_FALLBACK=1` and `--accelerator gpu` (RBVI's Tom Goddard, PR #231 comment 2025-05-23). The changes:

1. `pyproject.toml`: `torch==2.7.1` (was `>=2.2`); the cuequivariance packages pinned to 0.10.0 and (for ChimeraX on Linux) CUDA-only extras.
2. `src/boltz/main.py`:
   * **precision**: new `--precision` option, and the default is 32 unless Linux/Windows + CUDA (then `bf16-mixed`). Stock boltz 2.2.1 hard-codes
     `precision="bf16-mixed"` for Boltz-2 on every accelerator (`precision=32 if model == "boltz1" else "bf16-mixed"`), i.e. an autocast on MPS and CPU. The commit
     is "Only use bfloat16 precision with CUDA" (2025-07-09, the same day as the benchmark source of the ChimeraX table).
   * `--msa_only`; `--write_full_pae/--no_write_full_pae` and `--write_full_pde/--no_write_full_pde` now default to True (PAE and PDE npz are written; stock default False);
     progress messages with timestamps on stderr (loading weights, structure / affinity inference begin and end per record); prints the Boltz version;
     input directories: files that are not fasta / yaml are ignored instead of an error; `filter_inputs_affinity` drops records that do not ask for affinity (multi-input fix).
3. `model/models/boltz2.py`: PAE / PDE only put into the prediction dict when the corresponding write flag is set.
4. MSA / SMILES fixes: paired-sequence handling in `data/feature/featurizerv2.py` and `data/parse/csv.py`; better error text in `schema.py`.
5. It **lacks** upstream 2.2.1's changes: the bf16 dropout fix (`model/layers/dropout.py`; dropout is zero at inference), the cyclic-period padding line, the
   potentials fix (steering off here) and the affinity ligand-size warnings. None of these changes inference numerics at the settings used here (no potentials, no cyclic peptides).

Consequences for the benchmark: the primary arm computes in fp32 (like ChimeraX on a Mac), the stock arm in bf16-mixed autocast; torch versions differ
(2.7.1 vs 2.15.0); the fork writes the extra PAE / PDE files (as ChimeraX does), which is inside W (and deleted afterwards by the driver). The ChimeraX table
was made with Boltz 2.1.1 (July 2025 source) and torch 2.9 (per the help page), so it is not the same software as this HEAD (2026-10-22, torch 2.7.1).

## Tests done without the GPU (2026-10-31)

* `tools/macbench/driver.py`: dry-run of every group and of `t400up`; one real run through `run_one` on CPU (RBVI venv, `up_predict.py`, 8rf4aff_empty, 2 steps, 1 recycle,
  1 affinity sample: run.json, model-call split structure / affinity, checkpoint loads, peak footprint from `/usr/bin/time -l`, cleanup of npz files all worked). The MLX branch of the driver
  (`boltz_mlx_timings.json` parsing) is checked by reading `boltz_mlx/cli.py` only: no MLX prediction was run. `boltz_mlx.cli.parse_args` / `resolve_samplers` accept the flags of all four MLX arms.
* `tools/macbench/accuracy.py --selftest` PASS (9MOJ coordinates against themselves, chain-swapped, noisy; rules S / C / A on toy data); the evaluate path ran on a mock tree of CPU smoke outputs.
* `tools/macbench/summary.py` on mock run files. `tests/test_tools_syspath.py` passes with the new tools.
* The jobs' pin block and environment check were executed (dry-run flag appended).

## boltz-community (Novel-Therapeutics) on Mac MPS: what its code actually does (read 2026-11-02 before installing; treated as data)

Source: `git diff cb04aec 1431d57` (upstream v2.2.1 to the fork's HEAD): 146 files, +15932 / -2772 (61 under `src/`, +4570 / -1097); 640 commits on `main`; the Mac support is
commit `fa8a120` (2026-03-04, PR #1 "Add Apple Silicon MPS support for inference", 23 files, +303 / -48, plus `tests/test_mps.py`). The fork's README documents one Mac command
(`boltz predict input.yaml --accelerator mps --use_msa_server`) and says "MPS mode automatically uses float32 precision and single-device execution. Performance is slower than
CUDA but significantly faster than CPU." The README at HEAD carries **no Mac timing number** (an M3 Air vs T4 table was added in `f783570` and removed in `550c5b8`, 2026-03-26).

* **Kernels: none for MPS.** No Metal / MLX / custom-op code anywhere in the fork. The speed work is CUDA-only: FlashPairformer triangle kernels (`model/layers/flashpairformer.py`, used
  only when `x.is_cuda`, bf16 autocast, capability >= 8.0, and `opt_core` installed; otherwise it returns None and the cuEquivariance / PyTorch path runs), cuEquivariance triangle ops (`--no_kernels` to disable; failing
  kernel imports fall back to PyTorch with a warning), MSA-inference work reuse (`docs/msa-inference.md`: "gradient-free CUDA inference with bf16 autocast"; "non-bf16 contexts use the existing arithmetic"),
  the opt-in denoiser cache (`BOLTZ_DENOISER_CACHE=on`, off by default, measured on A10G only). None of these is active for fp32 on MPS. `--flash_attn` (PyTorch SDPA for the triangle / pair attention, "Requires an Ampere (sm_80+) or newer GPU")
  is opt-in and off by default; it is not in the README's Mac command and is not benchmarked here.
* **Precision**: `main.py` forces `precision = 32` when the accelerator is `mps` or `cpu` (stock 2.2.1 hard-codes `bf16-mixed`, an autocast on MPS and CPU); `autocast_device_type()` (`model/modules/utils.py`) maps the device
  to a valid autocast type so `torch.autocast(..., enabled=False)` blocks work on MPS. So the fork's MPS numerics are fp32, like the RBVI fork's, with torch 2.15.0 instead of 2.7.1.
* **Plumbing**: `--accelerator mps` (checks `torch.backends.mps.is_available()`, prints "Running on Apple Silicon GPU (MPS)."), devices forced to 1, `pin_memory` off on mps / cpu / xpu,
  tensors allocated directly on the device, checkpoints staged on CPU before the move, `torch.mps.empty_cache()` only in the out-of-memory handlers (`boltz2.py`), no setting of `PYTORCH_ENABLE_MPS_FALLBACK` or the MPS watermark
  ratios anywhere (the benchmark sets both, as for the RBVI arm). CPU-side speedups that also apply on a Mac: cached molecule-file reads, vectorized MSA pairing fill, preallocated atom-feature arrays.
* **Affinity on MPS**: yes, the same Lightning path as structure (CPU smoke run of 8rf4aff_empty below ran structure then affinity, both timed by `up_predict.py`; MPS itself was not used). The fork recycles 5 times for affinity (as upstream) and
  runs the affinity samples in one batched pass (`--max_parallel_samples`, default 5, capped at `--diffusion_samples_affinity`; this is a fix of upstream's per-sample loop).
* **Defaults that differ from upstream and matter for this benchmark**: `--subsample_msa` is ON by default with `--num_subsampled_msa 1024` (fork commit `9cca9d6`, "Fix 3 upstream bugs: subsample_msa default"; upstream's
  click flag is `is_flag=True` and therefore OFF; the MLX build raises NotImplementedError if asked to subsample). The 9b3h MSA has 8193 cached rows, so at defaults the fork runs its MSA module on 1024 of them;
  9moj has about 215. Also: structure checks added to `confidence_*.json` (affinity uses the best sample that passes), the csv MSA parser keeps paired duplicate sequences across taxa
  (identical row counts to the RBVI venv on the benchmark's csv MSAs, checked: 9moj 57 / 157, 9b3h 8192 / 53), `--max_msa_seqs` default 8192 as upstream.

CPU-only tests done (2026-11-02, niced, 1 thread, `--accelerator cpu`, no tensor placed on MPS; `torch.backends.mps.is_available()` is True): `boltz predict --help`; one reduced prediction (8rf4aff_empty: structure then affinity, 2 steps, 1 recycle, 1 affinity sample)
through `tools/macbench/up_predict.py` (exit 0; structure_model_s 16.5 and affinity_model_s 45.6 on CPU at those settings, version record `boltz-community 2.13.0`, torch 2.15.0; no libomp error, so `boltz-fix-macos-libomp` was not needed and
`KMP_DUPLICATE_LIB_OK` was not used); MSA parsing of 9moj_msa and 9b3h_msa. The smoke prediction wrote `structure_checks_passed` warnings, which is expected for 2 steps.
