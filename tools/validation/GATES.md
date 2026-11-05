# gates.py: statistical gates, calibration, and the jsonl schema

`tools/validation/gates.py` evaluates the Level 2 / Level 3 gates from jsonl rows, and (`--calibrate`) estimates each
gate's null pass probability, power and minimum detectable effect from strict-only data. `tools/validation/score.py`
supplies the metrics (CA-lDDT, TM, pocket-frame ligand RMSD, chirality, Pearson/Kendall, paired bootstrap).
Thresholds live in `validation/gates_config.json` (PREREG-A defaults = the SPEC values; PREREG-B edits that file, not code).

```
# run the gates for one candidate against strict
python tools/validation/gates.py logs/l3.jsonl --candidate f16 --strict strict [--candidate-type precision|sampler]
       [--cand-steps 50 --strict-steps 200] [--config validation/gates_config.json] [--n-boot 10000] [--seed 0] [--json out.json]
# A4 report only (never a gate)
python tools/validation/gates.py logs/l3.jsonl --a4 --candidate strict
# null calibration on strict-only data (V36 seeds 5-9 vs 0-4, TYK2 seeds 3-5 vs 0-2)
python tools/validation/gates.py logs/strict.jsonl --calibrate --mode strict \
       --protein-cand-seeds 5-9 --protein-ref-seeds 0-4 --ligand-cand-seeds 3-5 --ligand-ref-seeds 0-2 \
       --design-ligands 87 --mc-reps 1000 --boot-reps 300 --power-reps 400 --inner-boot 300 --out logs/calibration.json
```
Exit code of a gate run is 0 only if L2 has no hard failure and every gate of the candidate type has data and PASSes
(`UNDETERMINED` and `N/A` count as not passed). Everything is deterministic for a fixed `--seed`.

## Row schema (one JSON object per line, one row per (case, seed, mode))

Fields are optional unless marked; a gate is evaluated only if its data are present (`N/A` otherwise). Lines starting with `#` are ignored.

| field | type | meaning |
|---|---|---|
| `case` (required) | str | unit id: V36 chain (`1abc_A`) or ligand (`<target>_<name>`, e.g. `tyk2_ejm_31`) |
| `target` | str | grouping for within-target bootstrap, Pearson/Kendall pooling and the optional-target rule (`tyk2`, `cdk2`, `jnk1`, `p38`, `bace1`, `chymase`). Default: `case` |
| `seed` (required) | int | the per-record seed index (global seed of the SPEC seeding scheme) |
| `mode` (required) | str | `strict`, `tf32`, `f16`, `screen`, ...; sampler runs of `run_l3.py --sampler P` are labelled `<mode>+P` (e.g. `strict+s25`, `f16+heun11_n8`, `strict+aff50`): pass that label as `--candidate` |
| `steps` | int | structure-pass sampling steps (filtered by `--cand-steps/--strict-steps`) |
| `precision_mode`, `sampler` | str | (run_l3) the precision mode (environment preset) and the sampler preset name (`null` = upstream sampler) |
| `sampler_structure`, `sampler_affinity` | dict | (run_l3) `{steps, rho, step_scale, solver, heun_euler_first, heun_sigma_min, nfe, gamma_0, gamma_min, noise_scale}` of each pass; not used by the gates |
| `n_res`, `large_chain` | int, bool | residues; the large-chain subset is `large_chain` or `n_res >= l3p.large_chain_min_residues` (380) |
| `ca_lddt`, `tm` | float | L3-P metrics from `score.py` against the experimental structure |
| `affinity_pred_value` | float | log10 IC50 (uM), as produced by the run (MW setting in `mw_correction`) |
| `affinity_probability_binary` | float | |
| `value_mw_corrected` (alias `affinity_pred_value_mw_on`), `affinity_pred_value_mw_off` | float | explicit MW columns for A4 (results.csv column names). If absent, `affinity_pred_value` is taken as MW-off unless `mw_correction: true` |
| `mw_correction` | bool | MW setting that produced `affinity_pred_value` |
| `exp_value` | float | experimental value **in the same sign convention as `affinity_pred_value`** (FEP+ benchmark: Exp. dG kcal/mol, higher = weaker, so a good model correlates positively). Needed for A3/A4 |
| `msa` (alias `msa_source`) | str | `empty`, `server`, ... printed on every A4 number |
| `pose_cif` | path | Boltz output mmCIF of this run (ligand RMSD/chirality are computed from it) |
| `ligand_chain`, `smiles` | str | ligand chain id in `pose_cif` (default `B`) and the input SMILES (bond orders + stereo) |
| `chirality_flips` | int | precomputed stereocentre flips of this pose (otherwise computed from `pose_cif` + `smiles`) |
| `ref_cif`, `ref_ligand_chain`, `ref_pose`, `ref_pose_kind` | | optional reference complex (and an SDF/PDB reference pose in its frame) for the success rate (< 2 A); `ref_pose_kind` = `crystal`/`modelled`. Or give `ligand_rmsd_ref` directly |
| `model_cif` | path | structure for the optional L2 coordinate-RMSD spread check |
| `ptm`, `iptm`, `plddt`, ... | float | any scalar metric: L2 flags compare candidate-vs-strict deltas to the strict seed spread (`ca_lddt tm ptm iptm plddt affinity_* ligand_rmsd_ref`) |
| `nan`, `inf` | bool | any NaN/inf in the run: L2 hard failure |
| `guard_counts` (or `guard_count`) | int or {name: int} | fp16 overflow guard counts; any nonzero is an L2 hard failure |
| `affinity_crop_hash` | str | hash of the affinity crop residue set (O4: reported fraction that differs strict vs candidate) |
| `fingerprint`, `fp16_fallbacks` | | carried through, not used by the gates |

**Precomputed pose-pair rows** (alternative to `pose_cif`; they take precedence):
`{"type": "ligand_rmsd_pair", "case": ..., "a_mode": "f16", "a_seed": 0, "b_mode": "strict", "b_seed": 0, "rmsd": 1.7}`
meaning RMSD of pose `a` against reference pose `b` in the pocket frame of `b` (symmetric lookups are allowed).

Seed handling in a gate run: candidate and strict rows are matched on equal `seed`; only seeds present in both arms are used.
`ca_lddt`/`tm`/affinity use the matched seeds per case; strict seeds that have no candidate partner are ignored.

## Gates (id: statistic -> PASS rule; all thresholds from the config)

| id | statistic | PASS (PREREG-A) |
|---|---|---|
| L2 | hard failures (NaN/inf/guards); flags (delta > strict seed-to-seed range) | no hard failure (flags are reported, must be explained in the report) |
| L3P.lddt | per-case delta = mean over matched seeds (cand - strict); mean and t CI (df = n-1) over cases | CI lower >= -0.010 **and** mean >= -0.005 |
| L3P.tm | same for TM-score | CI lower >= -0.015 |
| L3L.median | ratio of medians: RMSD(cand, strict, same seed) over RMSD(strict i, strict j); ligands resampled within target | 95% one-sided bootstrap upper bound <= 1.25 |
| L3L.p90 | ratio of 90th percentiles, same | upper bound <= 1.5 |
| L3L.chirality | sum of stereocentre flips: candidate minus strict, over the matched poses | <= 0 |
| A1.value / A1.prob | t CI of the mean over ligands of the seed-averaged delta | CI within +-0.05 (log10) / +-0.02 |
| A2.value / A2.prob (precision) | RMS(seed-matched per-run delta) / RMS(strict seed i - strict seed j) over the same ligands | <= 0.5 |
| A2p.value (sampler) | RMS(seed-averaged per-ligand delta) / (sigma_seed * sqrt(1/n_cand + 1/n_strict)); sigma_seed = pooled per-ligand strict seed SD. For 3 vs 3 seeds the factor is sqrt(2/3) | <= 1.25 |
| A3.pearson | paired bootstrap (10,000, ligands within target) of delta of the ligand-count-weighted mean per-target Pearson vs `exp_value`, seed-averaged predictions | CI lower >= -0.05 |
| A3.kendall | ligand-weighted pooled Kendall tau-b(candidate, strict), bootstrap | CI lower >= tau_null - 0.05 (`l3a.tau_null` from calibration; `UNDETERMINED` while unset) |

Gate set of a candidate: `precision` = L3P, L3L, A1, A2, A3; `sampler` = L3P, L3L, A1, A2p, A3 (the other of A2/A2p is printed with `*`, reported only).
Chymase (config `l3a.optional_targets`) is pooled into the gates only with >= 15 complexes. Reported alongside (not gates): per-case |delta| > 0.05 outliers and the
large-chain subset (L3-P); success rate < 2 A and O4 crop-set differences (L3-L); per-target Pearson/tau, per-target tau(cand, strict) and the
probability distribution with a near-ceiling warning (L3-A); A4 (`a4_report`: unweighted/weighted mean per-target Pearson, MW on/off, BACE1 excluded,
bootstrap CI, MSA label, published 0.66 and a coarse-CI note).

## Calibration (`--calibrate`)

Inputs: strict-only rows. The "candidate" arm = `--*-cand-seeds`, the reference arm = `--*-ref-seeds`, paired by position (5-9 vs 0-4 pairs 5-0, 6-1, ...).
Protein rows (those with `ca_lddt`) use the protein seed sets; all other rows (affinity, poses) use the ligand seed sets.

* **Observed null**: the full gate set on the two strict subsets (`observed_null`).
* **Bootstrap estimator**: cases/ligands resampled within target from the observed subsets (`--boot-reps`); the gate's own CIs are nested bootstraps (`--inner-boot`).
* **Monte Carlo estimator**: fixed cases, new seeds. Each case/ligand is N(mu, sigma^2) over seeds (mu, sigma from **all** strict seeds of that mode; `--sd-mode pooled` = one
  pooled sigma, `per_case` = each case's own SD, `auto` = per-case when >= 6 seeds); CA-lDDT/TM and value/prob share their pooled residual correlation; pose RMSDs are drawn
  from each ligand's own pool of strict cross-seed RMSDs; chirality flips are Poisson with the observed strict flip rate. `--design-ligands 87` simulates the L3-A design size
  by drawing ligands with replacement within target (use it for PREREG-B: TYK2 alone has 16 ligands and cannot reach the A1 half-width).
* **Null model**: candidate seeds independent of strict seeds (kappa = 1), as the SPEC specifies. **A2 is the exception**: its ratio is ~1 by construction under independent
  seeds, so A2's null uses a noise-matched model with RMS(matched delta)/RMS(strict i-j) = `calibration.a2_kappa_null` (0.25; round 1 measured 0.02-0.05).
* **Null pass** per gate = min(bootstrap, MC) (`--decision min`, default; `--decision mc` uses MC only). Required >= 0.9; the conjunction of each gate set >= 0.8.
  `tau_null` = MC mean of the strict-vs-strict pooled Kendall tau.
* **Re-specification**: the margin factor f (threshold distance from its null reference: 0 for differences and CI bounds, 1 for ratios, tau_null for tau) is the smallest value on a 0.05 grid
  in [1, 2] with null pass >= 0.9; if none, `untestable` (the 2x-capped value is reported as the final threshold and `recommended_config.untestable` lists the gate). Chirality's margin is a
  count (0), so scaling does nothing: the entry carries `info.excess_slack_needed_for_null_pass` instead.
* **Power / MDE** (MC, at the final margins): bad candidates (SPEC-defined: +0.10 log10 value bias, weighted Pearson -0.10, CA-lDDT -0.02; defined here and to be confirmed in PREREG-A:
  TM -0.02, prob bias +0.04, candidate-vs-strict RMSD x1.5, +0.05 extra chirality flips per pose). The Pearson bad candidate adds persistent per-ligand noise whose SD is solved so the
  expected delta of the weighted Pearson is -0.10. `power_curve` is over `effect_multiples` x bad; `mde_80` is the smallest effect with power >= 0.8 (linear interpolation; null if not reached).
* Output JSON keys: `meta, estimated, observed_null, tau_null, gates{id: null_pass_boot, null_pass_mc, null_pass, respec{status,factor}, final_factor, final_threshold, power{...}},
  conjunction{precision,sampler}, recommended_config, verdict`. `recommended_config` is a ready-to-paste overlay for `validation/gates_config.json` (PREREG-B).

## Producing the rows: `run_l3.py`

`tools/validation/run_l3.py` runs the models and appends rows in this schema (docstring has the full flow; `--help` the CLI):

```
python tools/validation/run_l3.py --set v36|fep|parity|chymase|thrombin [--targets tyk2 cdk2 ...] [--cases ...] --seeds 0 1 2 \
       --mode strict|fast|turbo|<--modes-file name> --steps 200 [--affinity-steps 200] --out logs/l3/<name>.jsonl [--max-minutes 120]
       [--collect-only | --prefeaturize | --dry-run]
```
Resume key = (`case`, `seed`, `mode`, `steps`, `affinity_steps`). Extra fields beyond the table above: `recycling`, `device`, `env` (mode preset), `pkg_tree`, `git`, `mw`,
`affinity_pred_value_mw_off`, per-member `affinity_pred_value1/2` and `affinity_probability_binary1/2`, `affinity_best_sample`, `affinity_iptm` (5 samples),
`affinity_crop_tokens`, `seconds_structure`, `seconds_affinity`, `rmsd_ca`. `guard_counts` is written only if the model returns `guard_counts`
(ligand rows: flat `{"structure": n, "affinity.<name>": n}`). `affinity_pred_value` is always MW-off (`mw_correction: false`); `value_mw_corrected` is
`1.03525938*value - 0.59992683*MW**0.3 + 2.83288489`. Ligand `target` for BACE is `bace1`.
