# Running Boltz-2 on your Mac (researcher guide)

_Written 2026-10-28 and updated for the first public release. Sections marked **[pending validation]** were written
before the final validation pass; the validation results are in [report.md](report.md) (sections 7 and 8)._

> Ligand structures never leave the Mac. The target MSA is a one-time, per-target step that sends only the protein sequence (often public; public for every benchmark target here). Fully private targets without local MSA databases (hundreds of GB) run `msa: empty`, at an accuracy cost that this round measures only if pre-flight MSAs exist (strict FEP+ seed 0, MSA vs empty).

## 1. Install (offline after the one-time weight download)
```bash
git clone https://github.com/jackryan225/boltz-mlx.git && cd boltz-mlx
./install.sh            # installs `boltz-mlx`, downloads the Boltz-2 weights + mols/ (6.2 GB, once, resumable) into
                        # ~/.boltz, converts them to MLX (incl. the mixed fp16/fp32 files), opens the GUI
```
`boltz-mlx setup` does the download + conversion on its own (`boltz-mlx check` lists what is missing); details in the
README, "Install and quick start". Paths: `--cache` / `$BOLTZ_MLX_CACHE` (upstream cache with `mols/`, default `~/.boltz`),
`--weights` / `$BOLTZ_MLX_WEIGHTS` (default `~/.boltz/mlx_weights`). No network access at run time after setup, with
two opt-in exceptions: `predict --use_msa_server` and the GUI Predict form's **Online** MSA option send the protein
sequence to the ColabFold MSA server (api.colabfold.com), and the GUI's protein search queries UniProt / RCSB. `screen`
never contacts a server.

## 2. Screen a series: one command
```bash
boltz-mlx screen --target receptor.yaml --ligands series.csv --out results/ --preset strict
```
- `--target`: receptor YAML (an existing complex YAML works: its binder ligand is replaced by each screened ligand),
  FASTA, or an amino-acid sequence. `--msa path.a3m|empty`, `--pocket CHAIN:RES,...` optional.
- `--ligands`: `.csv`/`.tsv` (id, smiles), `.smi`, or `.sdf`.
- Output: `results.csv` (ranked; rewritten after every chunk) and `results.jsonl` (one row per ligand, written as each
  finishes). A killed run resumes where it stopped (`--resume` is the default; per-ligand seeds and a fixed RDKit seed
  for each ligand's 3D reference conformer make results independent of order and of restarts: every ligand gets the
  result it would get screened alone).
- `--rank-by prob` (default; `affinity_probability_binary`, for hit finding) or `value` (`affinity_pred_value`,
  log10 IC50-like, for ranking actives within a lead-optimisation series), following the upstream guidance.
- `--preset`: without it, `screen` (f16, 20 + 25 steps; 200 + 25 for a ligand with < 10 heavy atoms, decided per ligand and
  recorded in each row's `steps` and in `screen_config.json`; `--steps N` switches the rule off) on M5-class Macs and `strict` on M1–M4 (section 5); the
  examples here pin `--preset strict`, the reference build.
- `--dry-run` prints the plan and a time ESTIMATE; `--dry-run --measure 3` runs 3 ligands for real and projects.
- `--save-structures K|all`, `--chunk 50`, `--max-minutes M` (stop cleanly at a ligand boundary).

Worked FEP+ example **[pending validation: final numbers]**:
```bash
boltz-mlx screen --target fep/yaml/tyk2/tyk2_ejm_31.yaml \
                 --ligands tyk2_ligands.csv --out tyk2_screen/ --preset strict --rank-by value
```

## 2b. The same screen in a browser window: `boltz-mlx gui`
```bash
boltz-mlx gui                                  # http://127.0.0.1:8765 (the next free port if busy); opens your browser
boltz-mlx gui --no-browser --port 9000 --runs-dir ~/screens --cache ~/.boltz --weights ~/weights
python -m boltz_mlx.gui                        # same, without importing the model code
```
Runs are kept in `~/boltz-mlx-runs/<run-id>/` (`--runs-dir` to change): `inputs/` (your files, copied), `out/` (exactly the
`--out` directory of `boltz-mlx screen`), `screen.log`, `gui_run.json`. A folder of CLI screen outputs can be listed too
(read-only). Everything stays on this computer: the server listens on 127.0.0.1 only and refuses other interfaces, the pages
load no script, font or image from the network (the 3D viewer, Mol* 4.5.0 under the MIT licence, is bundled), and a screen
never contacts the MSA server (only a Predict chain whose MSA is set to **Online** does). Every API call carries a random per-session token, and the Host / Origin headers are checked, so other web
pages in your browser cannot drive it. Uploads are size-limited (target 2 MB, MSA 64 MB, ligands 64 MB) and stored under fixed names.

What you see:
- **New screen.** Four cards and a "Run plan" panel. (1) *Target*: upload a Boltz YAML / FASTA or paste a sequence; the chains, the
  MSA state and the ligand chain are shown, with the same warnings as the command line (for example "no MSA: running
  single-sequence"). (2) *MSA*: as in the target file, none, or an uploaded .a3m / .csv. (3) *Ligands*: upload CSV / TSV / SMI / SDF or
  paste SMILES; a preview table lists every ligand with atoms, MW and a status chip (OK, Warning, Large above 56 atoms, Invalid for
  unparseable, duplicate or over 128 atoms); click a row for the RDKit 2D drawing. (4) *Settings*: preset cards marked
  **Validated** or **Provisional** (with the caution text for the latter) and a per-ligand time for your target, *rank by*
  probability (hit finding) or value (lead optimisation) with a one-line explanation of each, and advanced options (ligands per
  chunk, save top poses, seed, time budget, pocket residues, ligand chain, id / SMILES columns). **Estimate time** runs the screen's
  `--dry-run` (an ESTIMATE from M5 Pro bench rows, not a measurement); **Start screen** launches it.
- **Run page.** A progress bar (solid = finished ligands, light = poses waiting for the affinity pass), tiles for done / remaining /
  seconds per ligand / ETA / elapsed / failed, the current ligand and phase (structure or affinity pass, chunk k/n), the list of
  failed and invalid ligands with reasons, and the run log. **Stop** sends SIGTERM: the screen finishes the ligand it is on, writes
  results.csv and exits (a second SIGTERM, "Stop now", interrupts immediately; finished ligands are kept either way).
  **Resume** relaunches the same command; per-ligand seeds make resumed results identical. A screen survives closing the page or
  quitting `boltz-mlx gui`; reopening the GUI re-attaches to it. Only one screen runs at a time.
- **Results.** Ranked table (rank, id, SMILES, value, probability, ipTM, ligand ipTM, pLDDT, status; more columns in the Columns
  menu), sortable by any column, with search and thresholds, switchable between the two rankings, and a CSV download. Click a
  row for the pose: 3D (Mol*: cartoon coloured by pLDDT, the ligand in magenta, residues within 5 Å and contacts shown;
  switch to chain colours, hide the pocket), 2D, and a details tab with all numbers and the pocket residue list. Poses outside
  the saved top-K are shown too (they are read from the run's work folder).
- **Runs.** All past runs with state, preset, counts and date.

## 3. Ligand size limits
Counted as upstream does (heavy atoms plus the hydrogens RDKit keeps). Above **128** atoms a ligand is rejected
(upstream limit for the affinity binder); above **56** it runs with a warning (training maximum; affinity may be
inaccurate). Salts are reduced to the largest fragment (with a warning); unassigned stereocentres are flagged.

## 4. Speed (M5 Pro, 24 GB; measured, not yet a validated preset)
Build R4-combined (git da1f229), `--precision f16`, warm model predict, median of
runs 2..N:
| | strict (fp32, 200 steps) | f16 @50 structure steps |
|---|---|---|
| 117-residue protein | 4.65 s | 1.03 s |
| 400-residue protein | 30.28 s | 7.09 s |
| affinity pass (213-token crop, 200 steps × 5 samples) | 29.8 s | 8.31 s |
| T400 screen (384-aa receptor + 249-row MSA, per ligand, cold, measured on 3 ligands) | — | 17.8 s |
The fp16/NAX speedups use the M5 GPU's neural accelerators (NAX): **M5-only; no speed projection to M1–M4** (no NAX,
different TF32 behaviour). The strict preset is the reference build and runs on any Apple-silicon Mac with enough memory.

## 5. Presets and what is validated
- `strict`: fp32 everywhere; numerically the reference (fixed-noise parity with upstream PyTorch, round 1). The default
  on Macs without NAX (M1–M4): there it reproduces the M5 results to rung-1 tolerance and the f16 path does not.
- `screen` (the default on M5-class GPUs): f16 numerics, a 20-step structure pass and a 25-step affinity pass (5
  samples; recycles as strict). **Validated on the Apple M5 Max:** sampler confirmation attempt 1 under PREREG-B passed
  all 8 gated gates against strict (L3-P lDDT and TM on 56 protein chains; L3-L median and p90 ligand RMSD, A1, A2p and
  A3 Pearson on 107 FEP+ ligands of CDK2, JNK1, P38 and BACE1), plus the rung 5 screen gates G1–G6
  (throughput, accuracy, memory, resume, install, offline). **Not validated:** ligand chirality and the affinity rank agreement (Kendall tau) are
  untestable on this data and are reported only, by decision: chirality reads 10 stereocentre flips against 3
  for strict over the same poses, Kendall tau 0.88. On M1–M4 the f16 path stays available by name
  (`--preset screen` runs as `screen-tensorops`) but is not validated there. MEASURED on the M5 Max (T400 receptor,
  384 aa, AC power): 6.8 s per ligand over 4 full cold screens of 199 ligands (6.3–7.2 s per screen), 5.4 s on a
  5-ligand measure; peak memory footprint 10.5 GB.
- `f16` (200 + 200 steps) and `turbo`: not gated presets.

## 6. What the numbers mean
- `affinity_pred_value`: predicted log10(IC50 in µM)-like value; lower = tighter. Use within a congeneric series.
- `affinity_probability_binary`: probability the ligand is a binder; use for hit finding.
- `iptm` / `ligand_iptm` / `complex_plddt`: pose confidence; low values mean the affinity number is less trustworthy.
- MSA label: every reported accuracy number says whether an MSA was used. With `msa: empty`: "consistency with
  published results NOT tested (single-sequence)". MSA-vs-empty penalty: **[pending: measured only if pre-flight MSAs
  exist]**.

## 7. Input-size envelope (`--precision f16`)
Activation ranges were measured up to 1000 tokens (prot1000), 964 tokens (ex_pocket) and a 4096-row MSA (msa4096)
(`tools/prof/act_ranges.py`). Beyond those, f16 relies on the overflow guard: inf/nan in a recycle triggers a recompute of that
recycle with fp32 GEMMs (reported as `fp16_fallbacks` in the output). Memory: prot400 f16 peaks at ~4 GB (MLX), so 16 GB
Macs are expected to fit ~400-residue targets (projected; not run on one).

## 8. Other Macs (M1–M4, M5 Max / Ultra): kernel paths and tuning on another Mac
One codebase runs on every Apple-silicon Mac with **macOS 26 or later** (Metal 4 is required; older macOS is refused at
start-up and by `tools/install_check.sh`). `python -m boltz_mlx.capability` (also printed by `tools/install_check.sh`)
shows the chip, the GPU architecture and the **kernel path** `--precision f16` takes:
- `nax` (M5-class GPU): the MetalPerformancePrimitives (Metal 4 tensor-op) kernels on the neural accelerators — rows say
  `f16`.
- `tensorops` (M1–M4): the same kernels as Metal 4 tensor ops on the shader cores — rows say `f16-tensorops` (screen
  presets `f16-tensorops` / `screen-tensorops`). Speed and accuracy were measured on the M5; the M1 checks are
  summarised in [report.md](report.md) section 5 (`tools/portable_check.sh` is the test kit), and on other chips the dry-run estimate says "not measured on this chip".
Both run only after an on-device check of the tensor-op fragment layout and arithmetic passes (once per device + OS,
cached). If it fails, a loud warning is printed and an internal fallback path runs (rows say `f16-portable`; not tuned,
not validated — please report it). `BOLTZ_MLX_KERNEL_PATH=portable` forces that fallback for debugging. `strict` is the
same fp32 code on every chip. Fingerprints and timings are per chip: results rows record chip, architecture and kernel
path.

**Tuning on another Mac** (e.g. an M5 Max: ~2× the GPU of the M5 Pro the configs were tuned on). Tile configs live in
`boltz_mlx/nax3_cfg.json`, `atom_cfg.json`, `msa_cfg.json`, keyed by chip name (`"Apple M5 Pro"`, …) with a `"default"`;
a chip without an entry uses the default and prints one note line. To tune (GPU, ~10–20 min each; needs the L1 captures
in `data/cache/l1` for the pair / MSA parts), on that Mac:
```bash
python tools/prof/nax3_ab.py all        && python tools/prof/nax3_ab.py suggest all --write   # token transformer + pair epilogues
python tools/prof/pair_profile.py r5 && python tools/prof/pair_profile.py r6 \
  && python tools/prof/pair_profile.py suggest --write                                        # tri-mult tail / transition
python tools/prof/atom_ab.py            && python tools/prof/atom_ab.py suggest --write --require-gate
python tools/prof/msa_kernels.py        && python tools/prof/msa_kernels.py suggest --write   # MSA OPM / PWA kernels
python tools/fingerprint.py --mode f16 --runs 2 && git diff boltz_mlx/*_cfg.json            # review, then commit
```
Each `--write` stores only this chip's entry (rows from other chips are ignored); config choices inside a kernel family
are bitwise-neutral except where the tool says otherwise, and the fingerprint row records what changed.

## 9. Not validated
- Ligand chirality and affinity rank agreement (Kendall τ): reported only (section 5).
- Targets and modalities outside the test sets: nucleic acids, glycans, covalent ligands, multimers beyond the test set.
- The f16 presets on M1–M4 Macs. See [report.md](report.md) sections 13 and 14 for the limits of this release.
