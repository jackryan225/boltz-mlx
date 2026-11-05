# boltz-mlx: Boltz-2 on Apple Silicon, free and open source

_Every number carries a label: **MEASURED** (we ran it; a log exists), **PUBLISHED** (someone else's number, quoted with its source), or **ESTIMATE** (our own arithmetic or inference, with the rule stated)._

## 1. The claim

boltz-mlx is a free, MIT-licensed port of Boltz-2 (structure and binding-affinity prediction) to Apple's MLX framework. It runs locally on any M-series Mac with macOS 26 or later. On an M5 Max it predicts a 384-residue protein–ligand complex (8cpk), structure and affinity, in 18.64 s of model time (MEASURED, median of 3).

## 2. Why it matters

- **No cloud.** Prediction runs on the Mac's GPU. Nothing has to be rented, queued or uploaded.
- **Your data stays on your machine.** At run time the model makes no network calls. The local GUI listens on 127.0.0.1 only and loads no scripts from the network. Ligand structures never leave the Mac. Two things do use the network, and only when you ask: `--use_msa_server` sends the protein sequence to the ColabFold MSA server (off by default), and the GUI's UniProt/PDB lookup queries those services.
- **Free.** No licence fee, no per-prediction charge, no account.

## 3. Free and open source

boltz-mlx is released under the MIT licence, the same licence as Boltz-2. The model, its weights and its training are the work of the Boltz authors. This project ports and tunes inference only. We did not retrain or change the model, and the converted weights come from the official Boltz-2 checkpoints.

The software costs $0. Renting the GPU that NVIDIA's published numbers use, an H100, costs $3.41/hour on average (PUBLISHED, SemiAnalysis GPU pricing index, https://semianalysis.com/gpu-pricing-index/). The H100 we rented for our own baseline cost $2.75/hour (MEASURED, rented cloud H100). The $0 applies to a Mac you already own.

## 4. Starting point

**What Boltz-2 computes.** A 3D complex from protein, nucleic-acid and ligand inputs, with confidence scores (pLDDT, PAE, pTM, ipTM), and for one ligand a binding affinity (a log10(IC50)-like value and a binder probability). It has a trunk (MSA module and 64-block pairformer), a diffusion structure module, a confidence head and an affinity ensemble.

**Why it is expensive.** The trunk's pair representation makes memory grow as N² and compute roughly as N³. It takes 92–96% of structure time from 800 tokens upward (MEASURED, M5 Max, f16, 50 steps).

**The Mac baseline.** The boltz-community fork, which MoleculeDesk installs on Apple Silicon, runs on stock PyTorch MPS in fp32 with no Mac-specific kernels. On our M5 Max it takes 116.86 s of model time for the 8cpk complex above (MEASURED, single run).

## 5. Correctness before speed

- **Module parity.** Each MLX module matches the CPU fp32 PyTorch implementation (`use_kernels=False`) to about 1e-6 relative error (MEASURED).
- **End-to-end agreement.** With PyTorch's diffusion noise replayed in MLX, the strict build reproduces PyTorch coordinates within 0.1 Å on all 10 validation proteins (0.0002–0.01 Å on the parity cases). Affinity parity passes on 8 of 8 ligands, worst difference 5.4e-4 against a limit of 0.01 (MEASURED).
- **The TF32 finding.** On M5-class GPUs, MLX runs fp32 matrix multiplies at TF32 precision by default (7.7e-4 relative error on a 512³ GEMM, MEASURED). Through 64 pairformer blocks and 4 passes this moves fixed-noise coordinates by 1.4 Å (MEASURED). boltz-mlx sets `MLX_ENABLE_TF32=0`; anyone porting a deep pair model to MLX should check this.
- **Other Macs.** In strict mode an M1 Air matches the M5 Max to 0.003 Å and 0.0006 in scores, against a bar of 0.1 Å and 0.01 set beforehand (MEASURED). In f16 it fails that bar (atoms differ by up to 2.6 Å), so M1–M4 Macs default to the Full preset. f16 differs from strict by a similar amount on one Mac, so it is judged statistically (section 7).

## 6. Where the speed comes from

1. **Fewer affinity steps.** Cutting the affinity pass from 200 to 25 diffusion steps reduces it from 3.88 s to 1.63 s (MEASURED, M5 Max). Cutting structure steps from 50 to 20 saves only about 6% of structure time (MEASURED), because the trunk dominates.
2. **f16 on the M5's neural accelerators** (per-core matrix units, via Metal 4 tensor ops). On the M1, f16 is only 1.07–1.4× faster than strict (MEASURED).
3. **Custom Metal kernels**: flash-style triangle attention, fused transition MLPs, a fused diffusion-transformer layer, channel-major triangle multiplication and compiled diffusion steps, tuned per chip.
4. **Memory handling for large inputs.** A buffer cache scaled to input size cut 1,600 tokens from 126.1 s to 103.2 s and 2,400 tokens from 381.1 s to 301.4 s, with byte-identical outputs (MEASURED, pre-registered A/B).

## 7. The Fast preset and its evidence

**Fast** is f16 with 20 structure + 25 affinity steps; if any ligand has fewer than 10 heavy atoms, it uses 200 structure steps. Fast is the default on M5 (neural-accelerator) Macs. **Full** is strict fp32 at 200 + 200 steps, the PyTorch-matching build, and the default on M1–M4.

**Why the small-ligand rule.** On 8gpp, the 4-atom ligand BCT collapsed in every 20-step run (f16 seeds 0, 1, 2 and fp32): 3–6 of its 6 atom pairs closer than 1.0 Å, 14.4–14.5 Å from the experimental site. f16 at 50 and 200 steps did not collapse (1.47 and 1.84 Å from experiment), nor did Full or boltz-community (MEASURED, one run per row). The cause is too few structure steps, not precision or seed. Protein TM stayed 0.997–0.998, and the affinity value (2.26–2.46) did not flag the bad pose. We chose 200 steps over 50 because 200 + 25 passed every held-out check below and the 50-step structure pass did not. **The 10-atom cutoff is a judgement call, not measured.** The evidence is one 4-atom ligand; 8eil's 8-atom SIN did not collapse at 20 steps, and no other ligand under 10 atoms was tested.

**Held-out validation (PREREG-B).** Rules were fixed before any candidate ran. The 8 gated checks cover lDDT and TM on 56 protein chains, and ligand RMSD and affinity on 107 FEP+ ligands (CDK2, JNK1, P38, BACE1), all against Full (MEASURED):

| setting | gated checks | stereocentre flips, candidate vs Full (reported, not gated) |
|---|---|---|
| f16, 20 structure + 25 affinity (Fast) | PASS 8/8 | 10 vs 3 |
| f16, 200 structure + 25 affinity (Fast, small ligands) | PASS 8/8 | 3 vs 3 (no excess) |
| f16, 50 structure + 200 affinity | FAIL: 2 of 8 failed (lDDT, TM) | 8 vs 3 |

Fast's affinity values stay within ±0.03 of Full against an allowed ±0.05 (MEASURED).

**Two larger proteins (pre-registered, PREREG_MACBENCH).** On 9moj (660 tokens) and 9b3h (911 tokens), with and without an MSA, 3 seeds each, Fast at 20 steps and f16 at 200 steps matched Full on all 4 inputs under all rules (agreement, accuracy against experiment, confidence). f16 at 50 steps failed ipTM on 9b3h without MSA: seed mean 0.270 against an allowed 0.305–0.368 (MEASURED). With 3 seeds that may be noise; we report it as a failure.

## 8. Accuracy against experiment

**Affinity.** On the FEP+ confirmation set (n = 71), predicted affinity correlates with experimental affinity at Pearson r = 0.667 for Fast at 20 + 25 and 0.681 for Full (MEASURED); 200 + 25 gives 0.674 (MEASURED). So Fast keeps Boltz-2's affinity signal on these four targets. Whether Boltz-2's affinity is right for your target is a question for the Boltz-2 paper.

**The NVIDIA cases** (consistency check, one seed per build, no threshold applied; MEASURED):

- Protein TM against the deposited structure is 0.974–0.998 for every build (Fast, Full, boltz-community).
- confidence_score agrees within 0.013 across the three builds in every case.
- Fast and Full affinity values differ by 0.02–0.12.
- boltz-community's affinity values are 0.3–0.7 lower than Full's on 7qsj, 8cpk and 8gpp. A possible cause is its SMILES atom-naming change; untested.
- The one physical defect was the BCT collapse (section 7), which led to the small-ligand rule. 8eil's SIN is placed 5.3, 11.1 and 12.2 Å from experiment by Fast, Full and boltz-community; no build finds a stable site.

## 9. Benchmark design

- **Machine.** Apple M5 Max, 128 GB, on AC power, one job at a time, 120 s GPU idle before each job.
- **Cases.** Four test IDs from NVIDIA's Boltz-2 NIM performance page (v1.6.0 tables), sequences from RCSB:

| case | NVIDIA "sequence length" | what we modelled | tokens | ligand (heavy atoms) |
|---|---|---|---|---|
| 8eil | 186 | 1 chain, 186 aa | 194 | SIN, from the entry (8) |
| 7qsj | 375 | 1 chain, 373 aa (RCSB entity) | 399 | celecoxib, entry has none (26) |
| 8cpk | 384 | 1 chain, 384 aa | 410 | celecoxib, entry has none (26) |
| 8gpp | 628 | 3 × 208 aa trimer | 628 | BCT, from the entry (4) |

- **MSAs.** One ColabFold MSA per case, shared by both Mac builds, capped at 8,192 rows (boltz-community with `--no_subsample_msa`).
- **Arms.** *Ours*: Fast, so 8eil and 8gpp run 200 + 25 steps and 7qsj and 8cpk run 20 + 25. *boltz-community* 2.13.0 on MPS at its defaults (fp32, 200 / 200). Both use 3 recycles, 1 diffusion sample, affinity at 5 recycles × 5 samples, seed 0.
- **Timing.** Model-call seconds for structure and affinity separately, as NVIDIA reports them. Ours: median of 3 interleaved repeats, with min–max. boltz-community: one run after an untimed 8eil warm-up.
- **What NVIDIA does not disclose:** step counts, recycles, whether MSA or featurisation time is included, warm or cold, and which ligand. NVIDIA's numbers are PUBLISHED (https://docs.nvidia.com/nim/bionemo/boltz2/1.7.0/performance.html, Boltz2 NIM v1.6.0 tables, H100, accessed 2026-10-27). We show them beside ours, not as like-for-like.

## 10. Results: the 8 comparisons

Seconds. Ours and boltz-community MEASURED on the same M5 Max; H100 PUBLISHED. "OSS" is NVIDIA's open-source PyTorch column.

**Structure**

| case | ours, Fast: median [min–max] | boltz-community | H100 TensorRT | H100 OSS (PyTorch) |
|---|---|---|---|---|
| 8eil (200 steps) | 2.88 [2.88–3.01] | 11.23 | 1.72 | 11.07 |
| 7qsj | 7.06 [6.43–7.46] | 62.30 | 3.81 | 7.76 |
| 8cpk | 12.21 [11.88–13.38] | 69.82 | 5.47 | 7.93 |
| 8gpp (200 steps) | 34.08 [33.31–34.85] | 185.54 | 8.23 | 15.40 |

**Affinity.** NVIDIA's affinity times exceed its structure times, which suggests they include structure (inference). So we compare them with our structure + affinity (sum of the two medians).

| case | ours, affinity: median [min–max] | ours, structure + affinity | boltz-community, affinity | boltz-community, structure + affinity | H100 TensorRT | H100 OSS (PyTorch) |
|---|---|---|---|---|---|---|
| 8eil | 2.88 [2.84–2.90] | 5.76 | 25.97 | 37.20 | 6.21 | 18.52 |
| 7qsj | 3.88 [3.26–4.32] | 10.94 | 41.36 | 103.66 | 7.89 | 23.44 |
| 8cpk | 6.43 [6.09–6.76] | 18.64 | 47.04 | 116.86 | 8.89 | 24.43 |
| 8gpp | 5.45 [5.21–5.62] | 39.53 | 38.07 | 223.61 | 13.24 | 34.03 |

**Ratios** (other time ÷ ours; above 1 means ours is faster; ESTIMATE, arithmetic on the rows above):

| case | vs community, structure + affinity | vs TensorRT, structure | vs TensorRT, affinity (vs our sum) | vs OSS, structure | vs OSS, affinity (vs our sum) |
|---|---|---|---|---|---|
| 8eil | 6.46 | 0.60 | 1.08 | 3.84 | 3.22 |
| 7qsj | 9.48 | 0.54 | 0.72 | 1.10 | 2.14 |
| 8cpk | 6.27 | 0.45 | 0.48 | 0.65 | 1.31 |
| 8gpp | 5.66 | 0.24 | 0.33 | 0.45 | 0.86 |

How to read this:

- **An H100 with TensorRT is faster on structure in every case**, by up to 4× at 628 tokens. Our structure + affinity beats TensorRT's affinity time only on 8eil (5.76 vs 6.21 s).
- **Against NVIDIA's own PyTorch**, our structure + affinity is faster on 8eil, 7qsj and 8cpk, and slower on 8gpp.
- **Against the best other Mac option** we are 5.7–9.5× faster on structure + affinity. On structure alone the range is 3.9–8.8×; 8eil is the low end because it now runs 200 steps. The gap mixes step count and precision; this benchmark does not separate them.
- **Caveats.** Our 7qsj → 8cpk structure time nearly doubles for 11 more tokens, likely because 8cpk's MSA is at the 8,192-row cap and 7qsj's has 4,024 rows (inference, not tested). NVIDIA's OSS 8eil structure time (11.07 s) exceeds its time for the larger 8c4d (6.80 s, PUBLISHED), and our log flags boltz-community's 8eil structure time as unexplained; treat the 8eil row with caution.

## 11. What's new in this release

- **Opt-in MSA server.** `--use_msa_server` (ColabFold), off by default. Otherwise bring your own MSA or run single-sequence.
- **Agent CLI.** `--json` output, a `check` command, documented exit codes, and an AGENTS.md for coding agents.
- **GUI** with UniProt and PDB lookup, served on 127.0.0.1.
- **Upstream fixes carried over.** Long SMILES atom names; a SMILES validation guard; and output-writer fixes for wrong elements on charged-ion SMILES ligands, a missing cyclic ring-closure bond, a column shift with multi-character PDB chain IDs, and PDB bond orders. Credit to the boltz-community fork (MIT) for identifying them.

## 12. Screening throughput

A full cold screen of 199 ligands against a 384-residue receptor (the "T400" set) at Fast's 20 + 25 steps runs at **6.84 s/ligand, about 530 ligands/hour** on the M5 Max (MEASURED, pooled over 4 back-to-back screens at 6.32–7.21 s/ligand; 796 of 796 completed; peak memory 10.5 GB). The 5-ligand warm rate is 5.38 s/ligand (MEASURED). These screens predate the ligand-conformer seeding fix; throughput was not re-measured on the release build.

On our rented H100, upstream Boltz with cuEquivariance at the same step counts ran the 199 ligands at **5.86 s/ligand, about 614/hour** (MEASURED rate; ligands/hour ESTIMATE), and at 16.39 s/ligand at 200 + 200 steps (MEASURED). That is one unbatched process, not NVIDIA NIM (which needs an NGC licence key). **A batched H100 very likely screens more ligands per hour than the M5 Max; we have not measured that.** The Mac gains little from running jobs in parallel: two strict jobs at once took 55–61 s each against about 30 s for one alone (MEASURED).

## 13. Scale, cost and limits

**Requirements:** any Apple Silicon Mac, M1 to M5, on macOS 26 or later (Metal 4). RAM sets the largest input.

Structure only, M5 Max, f16, 50 steps, before the memory-cache change in section 6 (MEASURED):

| tokens | time | MLX peak memory |
|---|---|---|
| 117 | 0.48 s | 2.8 GB |
| 400 | 2.99 s | 4.0 GB |
| 800 | 22.5 s | 6.3 GB |
| 1,000 | 38.2 s | 8.6 GB |
| 1,600 | 139 s | 19.9 GB |
| 2,400 | 403 s | 43.1 GB |

_Not yet measured: the largest input that fits on 16 / 24 / 36 / 48 / 64 / 128 GB of memory._

**Older Macs.** On an M1 Air (16 GB) the port runs, slowly: about 24 s for a 117-residue protein, 249 s for a 400-residue protein, and 126 s for the affinity pass at 25 steps (MEASURED). Fine for single predictions, not for screening. _Not yet measured: M1 timings on AC power for both presets, and M2/M3/M4 timings._

**Cost.** At $3.41/hour, the H100's measured 5.86 s/ligand is about $5.55 per 1,000 ligands, and $15.52 per 1,000 at 200 + 200 steps (ESTIMATE: rate × time, unbatched, excluding setup and idle time). A Mac you already own has no hourly charge; we have not costed buying one. Under heavy load the M5 Max drew 129–131 W (MEASURED). The point is not that the Mac beats an H100; it is that a Mac you own runs Boltz-2 at useful speed with no rental, no queue and no data leaving the building.

**Limits of this release:**

- **Above about 2,900 tokens**, a tri-mult kernel defect (it misreads inputs over 2³¹ bytes) forces an fp32 recompute, so large inputs finish more slowly. The fix exists and is not in this release; at 3,200 tokens it took 460.4 s against 577.5 s with the recompute (MEASURED, diagnostic run).
- **No steering potentials** (`--use_potentials`) and **no method conditioning**.
- **Contact and template guidance during sampling is skipped**, with a warning. Constraints, templates and modifications are accepted through the YAML input.
- **MSA subsampling** (`--subsample_msa`) is not implemented.

## 14. Threats to validity

- **Step counts differ** (ours 20 or 200 + 25, boltz-community 200/200, NVIDIA undisclosed), so the ratios mix implementation, precision and steps.
- **Thin repeats.** Our times are medians of 3; each boltz-community cell is one run. Accuracy on the NVIDIA cases is one seed per build.
- **NVIDIA's timing boundaries and ligand are unknown.** Size labels differ too: 7qsj is 375 residues in NVIDIA's table and 373 at RCSB; 8gpp's 628 matches only if the 4 BCT atoms are counted.
- **Sustained load.** Long heavy loads slow memory-heavy work by 14–29%, more likely a power limit than heat (ESTIMATE). On battery this Mac runs structure about 2.5× slower (MEASURED); all headline rows ran on AC.
- **One machine.** Everything was measured on one M5 Max, plus the M1 Air checks above.

## 15. Reproduce it

```bash
# install (needs macOS 26+, Apple Silicon): installs boltz-mlx, downloads and converts the Boltz-2 weights (MIT)
git clone https://github.com/jackryan225/boltz-mlx.git && cd boltz-mlx && ./install.sh

# one prediction
boltz-mlx predict input.yaml --out_dir out/

# screen a ligand library (resumable; per-ligand seeds make results order-independent)
boltz-mlx screen --target receptor.yaml --ligands ligands.csv --out results/

# local GUI (127.0.0.1 only)
boltz-mlx gui
```

The benchmark of sections 9 and 10 is in `tools/macbench/`: `driver.py` runs the arms, `accuracy.py` and `nim_accuracy.py` apply the pre-registered rules, `summary.py` builds the tables. The case inputs are `tools/macbench/cases/nim_<id>_msa.yaml` (and `_empty.yaml`). The pre-registration with its dated amendments, written before the data, is [macbench/PREREG_MACBENCH.md](macbench/PREREG_MACBENCH.md); environments in [macbench/ENV.md](macbench/ENV.md); our accuracy tables in [macbench/nim_accuracy.md](macbench/nim_accuracy.md). The ColabFold MSAs are not redistributed: `fetch_msa.sh` sends the public sequences to the ColabFold server and caches them in `macbench-env/msa/`; the hashes of the files we used are in `tools/macbench/msa_manifest.json` (the ColabFold databases change, so a fresh fetch can differ). Run from the repository root, in the developer environment of the README (`uv pip install -c constraints.txt -e ".[screen,test]"`; `python` must import `boltz_mlx`, or set `MACBENCH_MLX_PY`):

```bash
# the four NVIDIA-size MSAs (sends the public RCSB sequences to api.colabfold.com) and the experimental structures (RCSB)
MSA_CASES="nim_8eil nim_7qsj nim_8cpk nim_8gpp" zsh tools/macbench/fetch_msa.sh
python tools/macbench/check_env.py --need msa                 # compare with the pinned hashes
mkdir -p data/external/nim && for id in 8eil 7qsj 8cpk 8gpp; do
  curl -fsSL -o data/external/nim/$id.cif https://files.rcsb.org/download/$id.cif; done   # experimental structures (RCSB, CC0)

# one job at a time, on AC power; each waits 120 s GPU idle first; --dry-run prints the commands without running them.
# The tags are the ones nim_accuracy.py reads (its BUILDS list); results go to logs/macbench/<tag>/.
python tools/macbench/driver.py run --group nim-ours-rep    --tag nim-ours-rep-20261103a     # Fast 20 + 25, 3 interleaved repeats
python tools/macbench/driver.py run --group nim-ours-rep200 --tag nim-ours-rep200-20261103a  # Fast for 8eil, 8gpp (< 10 heavy atoms): 200 + 25
python tools/macbench/driver.py run --group nim-ours-screen --tag nim-ours-screen-20261102c  # Fast 20 + 25, one run (accuracy reference)
python tools/macbench/driver.py run --group nim-full        --tag nim-full-20261103a         # Full: fp32, 200 + 200
# boltz-community needs its own venv in macbench-env/venv-community (docs/macbench/ENV.md)
python tools/macbench/driver.py run --group nim-comm        --tag nim-comm-20261102b

python tools/macbench/summary.py --md summary.md                # timing tables
python tools/macbench/nim_accuracy.py --md nim_accuracy.md      # accuracy against Full, boltz-community and the experimental structures
```

PyTorch equivalence is checked by `tools/e2e_parity.py` and `pytest`.
