# boltz-mlx

Boltz-2 structure and binding-affinity prediction on your Mac. Free and open source (MIT). Runs locally on any
Apple-silicon (M-series) Mac with macOS 26 or later: no cloud, no account, no per-prediction cost.

```bash
git clone https://github.com/jackryan225/boltz-mlx.git && cd boltz-mlx
./install.sh      # installs `boltz-mlx`, downloads + converts the model (6.2 GB, once), opens the GUI
boltz-mlx gui     # next time
```

Technical report (method, validation, benchmarks, limits): [docs/report.md](docs/report.md).

Headline benchmark (from the report, section 10): seconds of model time for structure + affinity, Fast preset on an
Apple M5 Max, against boltz-community on the same Mac (PyTorch MPS) and NVIDIA's published H100 numbers (NIM v1.6.0
tables; not like-for-like, see the report).

| case | boltz-mlx (M5 Max) | boltz-community (M5 Max) | H100 TensorRT | H100 OSS (PyTorch) |
|---|---|---|---|---|
| 8eil | 5.76 | 37.20 | 6.21 | 18.52 |
| 7qsj | 10.94 | 103.66 | 7.89 | 23.44 |
| 8cpk | 18.64 | 116.86 | 8.89 | 24.43 |
| 8gpp | 39.53 | 223.61 | 13.24 | 34.03 |

## What it is

Boltz-2 (MIT) ported to Apple MLX, verified module by module against the PyTorch reference and tuned for Apple-silicon
GPUs. The model and its weights are the work of the Boltz authors; this project ports and tunes inference only.

- Full model: trunk (template/MSA modules, 64-block pairformer), diffusion structure module,
  confidence head (pLDDT / PAE / PDE / pTM / ipTM) and the affinity ensemble.
- Numerically matched to the CPU fp32 PyTorch implementation (`use_kernels=False`): every module
  within ~1e-6 relative error; end-to-end coordinates with replayed diffusion noise within
  0.0002–0.01 Å on the parity cases and 10/10 validation proteins under 0.1 Å.
- Custom Metal kernels (`mx.fast.metal_kernel`): simdgroup-MMA flash triangle attention, fused transition MLP,
  fused diffusion-transformer layer, narrow-row LayerNorm;
  fused projections, channel-major triangle multiplication, compiled diffusion steps.
- Same command line and output files as `boltz predict` (mmCIF, confidence JSON, pLDDT/PAE/PDE npz,
  affinity JSON).

See [ISSUES.md](ISSUES.md) for the upstream issues that shaped the port and [NOTICE](NOTICE) for credits.

## Install and quick start

Needs an Apple-silicon Mac (M1 or later) with macOS 26+, about 16 GB of free disk and an internet connection for the
first run.

```bash
git clone https://github.com/jackryan225/boltz-mlx.git
cd boltz-mlx
./install.sh
```

That is all: `install.sh` checks the Mac, installs the `boltz-mlx` command (with [uv](https://docs.astral.sh/uv/), in its
own Python 3.12 environment with the tested versions from `constraints.txt`: boltz 2.2.1, mlx 0.32.4, torch 2.15.0, ...),
downloads and converts the model, and opens the interface in your browser (http://127.0.0.1:8765, this Mac only).
Next time just run:

```bash
boltz-mlx gui
```

What happens on the first run (`boltz-mlx setup`; `install.sh` and the first `boltz-mlx gui` run it for you):

- Downloads the Boltz-2 model and molecule dictionary from the same sources upstream `boltz predict` uses (the Boltz model
  gateway / the `boltz-community/boltz-2` Hugging Face repository): `boltz2_conf.ckpt` 2.3 GB, `boltz2_aff.ckpt` 2.1 GB,
  `mols.tar` 1.9 GB, **6.2 GB in all** (about 10 minutes at 100 Mbit/s). Each file's size and sha256 are checked;
  an interrupted download resumes where it stopped when you run `boltz-mlx setup` again.
- Unpacks `mols/` and converts the checkpoints to MLX weights (fp32 + the mixed fp16 copies used by the fast preset), a few
  minutes on the CPU. Disk in total: about 15 GB (`boltz-mlx setup --prune` afterwards deletes the 6.2 GB of downloads
  that are no longer needed).
- Where: the upstream Boltz cache `~/.boltz` (a cache already filled by upstream `boltz predict` is reused, nothing is
  downloaded again) and `~/.boltz/mlx_weights`. Elsewhere: `BOLTZ_MLX_CACHE=/path` (or `--cache`), `BOLTZ_MLX_WEIGHTS=/path`
  (or `--weights`).
- After that, nothing is downloaded again; predictions and screens run offline (except `--use_msa_server`, see below).

If uv is not installed, `install.sh` says so and asks before running uv's official installer
(`curl -LsSf https://astral.sh/uv/install.sh | sh`); or install it yourself with `brew install uv`. If `boltz-mlx` is not
found in a new terminal, run `uv tool update-shell` once (it adds `~/.local/bin` to your PATH). Update: `git pull && ./install.sh`.
Other options: `./install.sh --no-gui`, `--no-setup` (download later), `--editable` (developer install of this checkout).

Developer setup without install.sh (tests, benchmarks):

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -c constraints.txt -e ".[screen,test]"
boltz-mlx setup                              # or: python tools/convert_weights.py --cache DIR --out DIR (conversion only)
```

## Command line

```bash
boltz-mlx check --json                                        # first: macOS 26+, Apple silicon, weights, mols cache (exit 3 = not ready)
boltz-mlx setup                                               # download + convert whatever check reports missing (once)
boltz-mlx predict input.yaml --out_dir out/ --json            # fold one complex (+ affinity if the YAML asks for it)
boltz-mlx screen --target receptor.yaml --ligands lig.csv --out run1/ --json   # rank SMILES by predicted affinity
boltz-mlx gui                                                 # local web interface (below)
```

- `--preset fast` (default): f16, 20 structure + 25 affinity steps; 200 structure steps instead when any ligand has fewer than 10
  heavy atoms (a 4-atom ligand collapsed at 20 steps; `screen` decides per ligand; the reason is in the `--json` settings as
  `structure_steps_reason`). `--preset full`: fp32, 200 + 200 steps, the upstream Boltz-2
  defaults. The earlier names `strict`, `screen`, `turbo`, `f16` still work (hidden). Explicit flags such as `--sampling_steps`
  override the preset. Macs without the M5 neural accelerators default to `full`.
- `--json` (predict, screen, check): one JSON object on stdout, all logs on stderr. It holds the settings used, per target the output
  paths and the key numbers (`confidence_score`, `ptm`, `iptm`, `complex_plddt`, `affinity_pred_value` = log10(IC50 in uM),
  `affinity_probability_binary`) and the wall time; `screen` lists the ranked top `--top N` (default 20) and the path of `results.csv`.
- Exit codes: `0` ok, `1` runtime failure, `2` bad input, `3` setup problem. Errors are one line naming the fix
  (`boltz-mlx: error: ... Fix: ...`), and with `--json` also a `{"status":"error",...}` object.
- Every protein needs an MSA: `msa: empty`, an `.a3m`/`.csv` path, or `--use_msa_server` (predict only; sends the sequence to
  ColabFold). `screen` never contacts a server.
- Inputs are Boltz YAML/FASTA. Not supported: steering potentials (`--use_potentials`), contact/template guidance potentials during
  sampling (a warning is printed when the input has constraints or templates). The old fp32-parity flags (`--fast`, `--turbo`,
  `--sampler`, `--precision`, ...) are hidden from `--help` but still accepted.

AI agents: hand them [AGENTS.md](AGENTS.md) (what the tool does, one predict and one screen example, the JSON shape, exit codes, limits).

## GUI

```bash
boltz-mlx gui                      # http://127.0.0.1:8765 (next free port if busy); opens your browser
boltz-mlx gui --no-browser --port 9000 --runs-dir ~/screens
```

A local web interface for `boltz-mlx predict` and `boltz-mlx screen`, no network needed once set up (on its first start it runs
`boltz-mlx setup` if the weights are missing; `--no-setup` skips that). Two opt-in exceptions: a Predict protein chain whose
MSA is set to **Online** sends its sequence to the ColabFold MSA server, and the protein search box queries UniProt / RCSB.
**Screen** never contacts a server. **Predict** folds one complex: build
the chains (protein / DNA / RNA sequences, ligands by SMILES or CCD code, optional MSA upload per protein), optionally turn
on binding affinity for one ligand, pick a preset (Standard or Reference) and step counts, or import a Boltz YAML / FASTA (a YAML with templates, constraints, modifications or MSA paths can run as written, with the companion files it names); the job page shows the phase, elapsed time and this
chip's estimate, then the result: Mol* coloured by pLDDT (with a legend), pLDDT / pTM / ipTM, the affinity readout
(predicted IC50, pIC50, ΔG, P(binder)), the PAE heatmap with chain boundaries, per-chain pTM / pair ipTM and pocket
residues, and downloads; with several diffusion samples a model selector switches structure, PAE and confidences. Unsent forms survive navigation and reloads; **Use inputs** on a finished run reopens its inputs and settings as a new draft. **Use inputs** keeps a YAML run as written (with its companion files) and offers Undo. Everything downloadable sits in one **Export** menu (results CSV for all, selected or filtered rows, mmCIF / PDB, the ZIP). The ZIP holds the inputs (YAML references rewritten to relative paths, plus a `rerun` command in the manifest and README that runs from the extracted folder), a manifest of the resolved settings, all models and the raw PAE / pLDDT / PDE arrays. Attached MSAs are checked against their chain before Start. A sidebar lists recent runs.
Time estimates come from a per-chip table of bench rows (`boltz_mlx/gui/estimates.py`, chip from `sysctl`); a chip without
rows shows "no estimate yet for this chip".

**Screen**: pick a saved target (or upload / paste one: PDB / mmCIF / FASTA / YAML, kept in the library), an
optional MSA and a ligand list (CSV / SMI / SDF / pasted SMILES), see every ligand checked with the screen's own validators,
pick a preset (provisional ones are flagged) and rank-by, press **Estimate time** (the screen's `--dry-run`) and **Start**.
A run page follows the screen live (progress, ligands done / remaining, s per ligand, ETA, current ligand, errors; **Stop**
finishes the current ligand and exits, **Resume** continues), then shows the ranked, sortable, filterable results with a CSV
download, a Mol* 3D pose view (pLDDT colouring, ligand highlighted, pocket residues within 5 Å) and an RDKit 2D depiction; a ligand whose pose the Top-N setting did not keep says so and can be re-checked to generate it. With Keep PAE, each ligand's raw PAE (.npz) downloads alone and in the ZIP. Past runs are listed.

It listens on 127.0.0.1 only, the pages load nothing from the network (Mol* 4.5.0, MIT, is bundled in
`boltz_mlx/gui/static/vendor/molstar/`), and the model never runs in the server: a screen is a `boltz-mlx screen`
(or `boltz-mlx predict`) subprocess writing into `<runs-dir>/<run-id>/out`. One job runs at a time (the GPU is shared). Python standard library only
(plus the RDKit / PyYAML the screen already needs). See [docs/researcher.md](docs/researcher.md), section 2b.

## Layout

| Path | What |
|---|---|
| `boltz_mlx/layers`, `boltz_mlx/modules` | MLX ports mirroring `boltz/model/{layers,modules}` (same parameter names) |
| `boltz_mlx/kernels.py` | Metal kernels (LayerNorm, triangle attention), precision + chunking knobs |
| `boltz_mlx/sampler.py` | reverse diffusion with replayable noise, weighted rigid alignment |
| `boltz_mlx/model.py` | top-level Boltz-2 (trunk → diffusion → confidence → affinity) |
| `boltz_mlx/cli.py`, `cliout.py`, `check.py` | `boltz-mlx check` / `setup` / `predict` / `screen` / `gui`; `--json`, exit codes, setup check |
| `boltz_mlx/assets.py`, `install.sh`, `constraints.txt` | first run: download (resumable, sha256-checked) + conversion to MLX weights; installer; tested versions |
| `boltz_mlx/screen/` | `boltz-mlx screen` (library screening, resumable) |
| `boltz_mlx/gui/` | `boltz-mlx gui`: local web interface (stdlib server + static pages, bundled Mol*) |
| `tools/make_reference.py` | CPU fp32 PyTorch reference: module dumps + recorded noise |
| `tools/e2e_parity.py`, `tests/` | module and end-to-end parity tests |
| `tools/bench_mlx.py`, `tools/bench_torch.py` | speed/memory benchmarks (MLX; PyTorch CPU/MPS) |
| `tools/validation/` | post-cutoff PDB validation set, lDDT/TM scoring |

## Tests

```bash
BOLTZ_MLX_DEVICE=gpu pytest -q                 # module parity on GPU (custom kernels active)
pytest -q                                      # same on the MLX CPU device (reference ops)
python tools/e2e_parity.py prot117             # end-to-end vs recorded CPU reference
```

Parity data (`data/`) is produced by `tools/prep_case.py` + `tools/make_reference.py` and is not
committed.

## Precision notes

MLX runs fp32 matmuls at TF32 precision by default on M5-class GPUs (7.7e-4 relative error on a
512³ GEMM here). `boltz_mlx` sets `MLX_ENABLE_TF32=0`; TF32 everywhere moves fixed-noise
coordinates by 1.4 Å (error compounds through 64 pairformer blocks × 4 recycles).

## Structure viewer

`docs/viewer/` is a static Mol* page overlaying MLX predictions (pLDDT-coloured) on the
experimental structures of the validation set: `python tools/make_viewer.py`, then
`python -m http.server -d docs/viewer` and open http://localhost:8000.

## Speed modes

The flag-level modes below predate `--preset` (`predict` now defaults to `--preset fast`: f16, 20 + 25 steps, 200 + 25 with a ligand of < 10 heavy atoms; `--preset full` = strict).

| Mode | Flags | prot117 | prot400 | Accuracy |
|---|---|---|---|---|
| strict | `--preset full` | 4.7 s | 30 s | fixed-noise parity with PyTorch fp32 |
| fast | `--fast` | 3.8 s | — | validation lDDT within 0.002 |
| turbo | `--turbo` (= fp16 diffusion + TF32 + 50 steps) | 1.8 s | 18 s | 30-case validation: paired ΔlDDT −0.0005 (95% CI ±0.009) vs strict |

PyTorch MPS (warm, bf16-mixed) on the same machine: 8.0 s / 87 s. Current benchmarks: [docs/report.md](docs/report.md).
