# NIM-size cases: Fast vs Full vs boltz-community vs experiment (consistency check, n = 1 seed)

**Read this first.** One seed per build per case (seed 0), 1 diffusion sample: these numbers are a *consistency check*, not a pass/fail gate. No threshold is applied, before or after seeing the data. A difference between two builds of one seed cannot be separated from seed-to-seed noise here (the repeats of Fast share the seed and only measure run-to-run determinism). Builds: Fast = ours `screen` (f16, 20 structure + 25 affinity steps); Full = ours `strict200` (fp32, 200 + 200); Community = boltz-community `comm200_full` (fp32, 200 + 200, MPS, `--no_subsample_msa`); all with the same ColabFold MSA csv, the same ligand, 3 recycles, 1 sample. Fast and Full differ in precision AND steps. Experimental structures: RCSB mmCIFs in `data/external/nim/` (tools/data/nim_cases.py) (sha256 in `tools/macbench/ref/nim_sources.json`); reference chain = the copy carrying the scored ligand (8eil: chain D), else chain A; 8gpp = the A, B, C trimer. TM / lDDT / CA RMSD are over protein CA atoms and normalised by the residues modelled in the reference. Ligand RMSD: heavy atoms, symmetry-aware, after superposing the 10 A pocket CA atoms, lowest over the reference ligand copies; against the experiment only where the entry contains the ligand (8eil SIN, 8gpp BCT; 7qsj and 8cpk have no celecoxib). The experimental ligand of 8eil sits at chain D and 8gpp has one BCT per chain, so those ligand RMSDs also depend on which site the model chose.

## 1. Confidence and affinity side by side (model 0)

| case | build | confidence_score | iptm | ligand_iptm | ptm | complex_plddt | affinity_pred_value | affinity_probability_binary |
|---|---|---|---|---|---|---|---|---|
| 8eil | Fast | 0.872 | 0.738 | 0.738 | 0.958 | 0.906 | 1.836 | 0.265 |
| 8eil | Fast r1 | 0.872 | 0.738 | 0.738 | 0.958 | 0.906 | 1.836 | 0.265 |
| 8eil | Fast r2 | 0.872 | 0.738 | 0.738 | 0.958 | 0.906 | 1.836 | 0.265 |
| 8eil | Fast r3 | 0.872 | 0.738 | 0.738 | 0.958 | 0.906 | 1.836 | 0.265 |
| 8eil | Full | 0.872 | 0.761 | 0.761 | 0.962 | 0.900 | 1.859 | 0.291 |
| 8eil | Community | 0.871 | 0.773 | 0.773 | 0.961 | 0.896 | 1.931 | 0.281 |
| 7qsj | Fast | 0.927 | 0.855 | 0.855 | 0.928 | 0.944 | 1.516 | 0.283 |
| 7qsj | Fast r1 | 0.927 | 0.855 | 0.855 | 0.928 | 0.944 | 1.516 | 0.283 |
| 7qsj | Fast r2 | 0.927 | 0.855 | 0.855 | 0.928 | 0.944 | 1.516 | 0.283 |
| 7qsj | Fast r3 | 0.927 | 0.855 | 0.855 | 0.928 | 0.944 | 1.516 | 0.283 |
| 7qsj | Full | 0.931 | 0.893 | 0.893 | 0.937 | 0.940 | 1.532 | 0.284 |
| 7qsj | Community | 0.929 | 0.862 | 0.862 | 0.939 | 0.946 | 1.112 | 0.333 |
| 8cpk | Fast | 0.928 | 0.895 | 0.895 | 0.965 | 0.936 | 1.684 | 0.175 |
| 8cpk | Fast r1 | 0.928 | 0.895 | 0.895 | 0.965 | 0.936 | 1.684 | 0.175 |
| 8cpk | Fast r2 | 0.928 | 0.895 | 0.895 | 0.965 | 0.936 | 1.684 | 0.175 |
| 8cpk | Fast r3 | 0.928 | 0.895 | 0.895 | 0.965 | 0.936 | 1.684 | 0.175 |
| 8cpk | Full | 0.918 | 0.840 | 0.840 | 0.961 | 0.937 | 1.620 | 0.172 |
| 8cpk | Community | 0.915 | 0.896 | 0.896 | 0.966 | 0.920 | 0.891 | 0.307 |
| 8gpp | Fast | 0.937 | 0.947 | 0.677 | 0.951 | 0.935 | 2.383 | 0.300 |
| 8gpp | Fast r1 | 0.937 | 0.947 | 0.677 | 0.951 | 0.935 | 2.383 | 0.300 |
| 8gpp | Fast r2 | 0.937 | 0.947 | 0.677 | 0.951 | 0.935 | 2.383 | 0.300 |
| 8gpp | Fast r3 | 0.937 | 0.947 | 0.677 | 0.951 | 0.935 | 2.383 | 0.300 |
| 8gpp | Full | 0.935 | 0.954 | 0.783 | 0.958 | 0.930 | 2.265 | 0.287 |
| 8gpp | Community | 0.939 | 0.959 | 0.817 | 0.963 | 0.935 | 1.973 | 0.283 |

## 2. Prediction against prediction (same case, same MSA and ligand)

| case | pair | TM | CA-lDDT | CA RMSD A | ligand RMSD A (pocket-superposed) | max atom shift A |
|---|---|---|---|---|---|---|
| 8eil | Fast vs Fast r1 | 1.000 | 1.000 | 0.00 | 0.00 | 0.00 |
| 8eil | Fast vs Fast r2 | 1.000 | 1.000 | 0.00 | 0.00 | 0.00 |
| 8eil | Fast vs Fast r3 | 1.000 | 1.000 | 0.00 | 0.00 | 0.00 |
| 8eil | Fast vs Full | 0.993 | 0.979 | 0.47 | 14.64 | - |
| 8eil | Fast vs Community | 0.995 | 0.988 | 0.38 | 16.15 | - |
| 8eil | Full vs Community | 0.992 | 0.980 | 0.50 | 3.41 | - |
| 7qsj | Fast vs Fast r1 | 1.000 | 1.000 | 0.00 | 0.00 | 0.00 |
| 7qsj | Fast vs Fast r2 | 1.000 | 1.000 | 0.00 | 0.00 | 0.00 |
| 7qsj | Fast vs Fast r3 | 1.000 | 1.000 | 0.00 | 0.00 | 0.00 |
| 7qsj | Fast vs Full | 0.963 | 0.979 | 2.92 | 3.59 | - |
| 7qsj | Fast vs Community | 0.980 | 0.985 | 1.32 | 3.29 | - |
| 7qsj | Full vs Community | 0.969 | 0.977 | 2.26 | 0.96 | - |
| 8cpk | Fast vs Fast r1 | 1.000 | 1.000 | 0.00 | 0.00 | 0.00 |
| 8cpk | Fast vs Fast r2 | 1.000 | 1.000 | 0.00 | 0.00 | 0.00 |
| 8cpk | Fast vs Fast r3 | 1.000 | 1.000 | 0.00 | 0.00 | 0.00 |
| 8cpk | Fast vs Full | 0.994 | 0.980 | 0.71 | 1.27 | - |
| 8cpk | Fast vs Community | 0.990 | 0.975 | 0.87 | 2.57 | - |
| 8cpk | Full vs Community | 0.991 | 0.973 | 0.79 | 1.72 | - |
| 8gpp | Fast vs Fast r1 | 1.000 | 1.000 | 0.00 | 0.00 | 0.00 |
| 8gpp | Fast vs Fast r2 | 1.000 | 1.000 | 0.00 | 0.00 | 0.00 |
| 8gpp | Fast vs Fast r3 | 1.000 | 1.000 | 0.00 | 0.00 | 0.00 |
| 8gpp | Fast vs Full | 0.976 | 0.938 | 1.81 | 15.68 | - |
| 8gpp | Fast vs Community | 0.962 | 0.921 | 2.51 | 16.16 | - |
| 8gpp | Full vs Community | 0.981 | 0.957 | 1.96 | 27.27 | - |

Rows `Fast vs Fast rK` compare repeats of the same seed (determinism); `Fast vs Full` is f16 20/25 steps against fp32 200/200; `Fast vs Community` and `Full vs Community` are the two implementations. 'max atom shift' is the largest coordinate difference over all atoms in the CIFs in place (no superposition), so it is only meaningful when two outputs are near-identical.


## 3. Each build against the experimental structure

| case | build | TM | CA-lDDT | CA RMSD A | modelled / reference residues | ligand RMSD A | ligand pocket CA used (ref copy) |
|---|---|---|---|---|---|---|---|
| 8eil | Fast | 0.978 | 0.944 | 0.76 | 177 / 177 | 5.29 | 29 (D) |
| 8eil | Fast r1 | 0.978 | 0.944 | 0.76 | 177 / 177 | 5.29 | 29 (D) |
| 8eil | Fast r2 | 0.978 | 0.944 | 0.76 | 177 / 177 | 5.29 | 29 (D) |
| 8eil | Fast r3 | 0.978 | 0.944 | 0.76 | 177 / 177 | 5.29 | 29 (D) |
| 8eil | Full | 0.974 | 0.937 | 0.84 | 177 / 177 | 11.05 | 29 (D) |
| 8eil | Community | 0.978 | 0.942 | 0.76 | 177 / 177 | 12.16 | 29 (D) |
| 7qsj | Fast | 0.997 | 0.990 | 0.40 | 346 / 346 | n/a (ligand not in entry) | - (-) |
| 7qsj | Fast r1 | 0.997 | 0.990 | 0.40 | 346 / 346 | n/a (ligand not in entry) | - (-) |
| 7qsj | Fast r2 | 0.997 | 0.990 | 0.40 | 346 / 346 | n/a (ligand not in entry) | - (-) |
| 7qsj | Fast r3 | 0.997 | 0.990 | 0.40 | 346 / 346 | n/a (ligand not in entry) | - (-) |
| 7qsj | Full | 0.997 | 0.990 | 0.41 | 346 / 346 | n/a (ligand not in entry) | - (-) |
| 7qsj | Community | 0.997 | 0.990 | 0.36 | 346 / 346 | n/a (ligand not in entry) | - (-) |
| 8cpk | Fast | 0.982 | 0.942 | 1.20 | 384 / 384 | n/a (ligand not in entry) | - (-) |
| 8cpk | Fast r1 | 0.982 | 0.942 | 1.20 | 384 / 384 | n/a (ligand not in entry) | - (-) |
| 8cpk | Fast r2 | 0.982 | 0.942 | 1.20 | 384 / 384 | n/a (ligand not in entry) | - (-) |
| 8cpk | Fast r3 | 0.982 | 0.942 | 1.20 | 384 / 384 | n/a (ligand not in entry) | - (-) |
| 8cpk | Full | 0.984 | 0.946 | 1.13 | 384 / 384 | n/a (ligand not in entry) | - (-) |
| 8cpk | Community | 0.977 | 0.935 | 1.50 | 384 / 384 | n/a (ligand not in entry) | - (-) |
| 8gpp | Fast | 0.997 | 0.987 | 0.44 | 569 / 569 | 14.46 | 51 (B) |
| 8gpp | Fast r1 | 0.997 | 0.987 | 0.44 | 569 / 569 | 14.46 | 51 (B) |
| 8gpp | Fast r2 | 0.997 | 0.987 | 0.44 | 569 / 569 | 14.46 | 51 (B) |
| 8gpp | Fast r3 | 0.997 | 0.987 | 0.44 | 569 / 569 | 14.46 | 51 (B) |
| 8gpp | Full | 0.997 | 0.986 | 0.44 | 569 / 569 | 1.84 | 51 (B) |
| 8gpp | Community | 0.998 | 0.988 | 0.39 | 569 / 569 | 1.65 | 52 (C) |

## 4. Predicted ligand internal geometry (sanity, not a metric)

Smallest and largest distance between any two ligand heavy atoms and the number of pairs closer than 1.0 A (no covalent bond is that short: such pairs mean a collapsed ligand).

| case | build | heavy atoms | min pair distance A | pairs < 1.0 A | max pair distance A |
|---|---|---|---|---|---|
| 8eil | Fast | 8 | 1.23 | 0 | 5.22 |
| 8eil | Fast r1 | 8 | 1.23 | 0 | 5.22 |
| 8eil | Fast r2 | 8 | 1.23 | 0 | 5.22 |
| 8eil | Fast r3 | 8 | 1.23 | 0 | 5.22 |
| 8eil | Full | 8 | 1.22 | 0 | 5.41 |
| 8eil | Community | 8 | 1.23 | 0 | 5.60 |
| 7qsj | Fast | 26 | 1.30 | 0 | 10.95 |
| 7qsj | Fast r1 | 26 | 1.30 | 0 | 10.95 |
| 7qsj | Fast r2 | 26 | 1.30 | 0 | 10.95 |
| 7qsj | Fast r3 | 26 | 1.30 | 0 | 10.95 |
| 7qsj | Full | 26 | 1.30 | 0 | 11.00 |
| 7qsj | Community | 26 | 1.31 | 0 | 10.89 |
| 8cpk | Fast | 26 | 1.31 | 0 | 10.94 |
| 8cpk | Fast r1 | 26 | 1.31 | 0 | 10.94 |
| 8cpk | Fast r2 | 26 | 1.31 | 0 | 10.94 |
| 8cpk | Fast r3 | 26 | 1.31 | 0 | 10.94 |
| 8cpk | Full | 26 | 1.31 | 0 | 10.77 |
| 8cpk | Community | 26 | 1.33 | 0 | 10.89 |
| 8gpp | Fast | 4 | 0.22 | 6 | 0.69 |
| 8gpp | Fast r1 | 4 | 0.22 | 6 | 0.69 |
| 8gpp | Fast r2 | 4 | 0.22 | 6 | 0.69 |
| 8gpp | Fast r3 | 4 | 0.22 | 6 | 0.69 |
| 8gpp | Full | 4 | 1.26 | 0 | 2.20 |
| 8gpp | Community | 4 | 1.25 | 0 | 2.18 |

Ligand RMSD by element-wise optimal assignment (the predicted ligand's geometry was too distorted to perceive a bond graph; ignores connectivity, adequate only for the 3-oxygen BCT): 8gpp Fast vs Fast r1; 8gpp Fast vs Fast r2; 8gpp Fast vs Fast r3; 8gpp Fast vs Full; 8gpp Fast vs Community; 8gpp Fast vs experiment; 8gpp Fast r1 vs experiment; 8gpp Fast r2 vs experiment; 8gpp Fast r3 vs experiment. All other ligand RMSDs use the SMILES graph.

Missing rows mean the run did not exist or failed (see the build's `run.json`). Sources of every number: `logs/macbench/<job tag>/<arm>/nim_<id>_msa/seed0[_rK]/` (not committed); code: `tools/macbench/nim_accuracy.py`.


## 5. Reading (written by hand after the run of the 2026-11-03 job outputs; descriptive only, n = 1 seed, no threshold applied)

* **Repeats are deterministic.** The three `nim-ours-rep` repeats are bit-identical to the single `nim-ours-screen` run on all four cases (TM 1.000, 0.00 A atom shift, identical confidence and affinity digits), so the timing repeats also repeat the output.
* **Protein structure: the three builds agree closely.** Fast vs Full TM 0.963-0.994 (CA RMSD 0.47-2.92 A, largest on 7qsj); Fast vs Community TM 0.962-0.995; Full vs Community TM 0.969-0.992. The two implementations at the same 200 / 200 fp32 settings (Full vs Community) differ about as much as our own Fast and Full do. With one seed these differences cannot be told apart from seed-to-seed noise.
* **Against experiment** every build is within TM 0.974-0.998 and lDDT 0.935-0.990; within a case the builds are within 0.007 TM of each other (8eil 0.974-0.978, 7qsj 0.997 for all, 8cpk 0.977-0.984, 8gpp 0.997-0.998).
* **Confidence**: confidence_score is within 0.013 across builds in every case; iptm differs more (7qsj 0.855 / 0.893 / 0.862, 8cpk 0.895 / 0.840 / 0.896 for Fast / Full / Community).
* **Ligand pose.** 8gpp BCT: Full and Community sit 1.84 and 1.65 A from the experimental site; the **Fast prediction's BCT is collapsed** (all six atom pairs closer than 1.0 A, largest distance 0.69 A, table 4), so its ligand RMSD (14.46 A) and the affinity computed on it (2.383) are not comparable. This is the only physical defect found in Fast's outputs; all other ligands have sane geometry in all builds. 8eil SIN is placed 5.3 A (Fast), 11.1 A (Full) and 12.2 A (Community) from the experimental ligand, and the builds put it in different places (pairwise 3-16 A): a single small dicarboxylate with no stable site in this model.
* **Affinity is where Community looks different.** affinity_pred_value, Fast / Full / Community: 8eil 1.836 / 1.859 / 1.931; 7qsj 1.516 / 1.532 / 1.112; 8cpk 1.684 / 1.620 / 0.891; 8gpp 2.383 (collapsed pose) / 2.265 / 1.973. Fast vs Full differ by 0.02-0.12; Community vs Full by +0.07 (8eil), -0.42 (7qsj), -0.73 (8cpk), -0.29 (8gpp), and its binary probabilities are higher on 7qsj and 8cpk (0.333 and 0.307 against 0.284 and 0.172 for Full). One seed cannot say whether that exceeds seed noise; the fork's different affinity handling (best sample that passes its structure checks, batched samples) is a candidate explanation that was not tested.

## 6. BCT collapse diagnostic on 8gpp (job `bct-diag-20261103b`; rule pre-registered in PREREG_MACBENCH.md before the data)

Collapse = any pair of ligand heavy atoms closer than 1.0 A in the predicted BCT (4 atoms, 6 pairs). Pose error = ligand RMSD against the experimental BCT after pocket-CA superposition (lowest over the three experimental copies). n = 1 run per row, a diagnostic, no gate. Protein TM against the experimental trimer (A, B, C). `screen` is f16 with 20 structure / 25 affinity steps; `strict20` is fp32 at the same 20 / 25 steps (precision isolated from steps).

| run | collapse | min pair A | pairs < 1.0 A | ligand RMSD vs exp A | protein TM vs exp | affinity_pred_value | affinity_probability_binary | peak GB |
|---|---|---|---|---|---|---|---|---|
| screen seed 0 (reference) | YES | 0.22 | 6 / 6 | 14.46 | 0.997 | 2.383 | 0.300 | 28.3 |
| strict200 seed 0 (reference) | no | 1.26 | 0 / 6 | 1.84 | 0.997 | 2.265 | 0.287 | 34.4 |
| boltz-community 200 seed 0 (reference) | no | 1.25 | 0 / 6 | 1.65 | 0.998 | 1.973 | 0.283 | 32.8 |
| f16_50 seed 0: f16, 50 / 50 | no | 1.24 | 0 / 6 | 1.47 | 0.998 | 2.287 | 0.298 | 28.3 |
| f16_200 seed 0: f16, 200 / 200 | no | 1.26 | 0 / 6 | 1.84 | 0.997 | 2.262 | 0.287 | 28.3 |
| screen seed 1: f16, 20 / 25 | YES | 0.41 | 4 / 6 | 14.36 | 0.998 | 2.462 | 0.300 | 28.3 |
| screen seed 2: f16, 20 / 25 | YES | 0.70 | 3 / 6 | 14.43 | 0.998 | 2.387 | 0.303 | 28.3 |
| strict20 seed 0: fp32, 20 / 25 | YES | 0.22 | 6 / 6 | 14.46 | 0.997 | 2.279 | 0.287 | 34.4 |

**Reading (section 6, n = 1 per row, descriptive):** the collapse follows the number of structure steps, not precision or seed: every 20-step run collapsed (f16 seeds 0, 1, 2 and fp32 seed 0, with 3-6 of 6 pairs below 1.0 A and ligand RMSD 14.4-14.5 A), and every 50- and 200-step run (f16) and the 200-step fp32 and community references did not (min pair 1.24-1.26 A, RMSD 1.5-1.8 A). Protein TM against experiment is 0.997-0.998 in all rows, and affinity_pred_value is 2.26-2.46 in all of our rows (community 1.97), so the collapse shows in the ligand pose and not visibly in the affinity value. The `screen` (Standard) preset is the 20-step setting.
