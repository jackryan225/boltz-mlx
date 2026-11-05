# boltz-mlx for agents

Boltz-2 on Apple silicon (MLX): `predict` folds a complex (protein, DNA/RNA, ligands) and can score binding affinity;
`screen` ranks a SMILES list against one protein. Local and offline; `--use_msa_server` and the one-time `boltz-mlx setup`
download are the only network use.

## First: check the setup
```
boltz-mlx check --json        # exit 0 ready, 3 not ready; "checks[].fix" and "next" say what to do
```
Needs macOS 26+, an Apple-silicon Mac, the converted weights, and the upstream `mols/` cache (`--cache DIR`, `$BOLTZ_MLX_CACHE`).
If weights or `mols/` are missing, `boltz-mlx setup --json` downloads (6.2 GB, network, resumable) and converts them once;
`boltz-mlx setup --dry-run --json` lists the steps without downloading. `predict` / `screen` never download anything.

## Predict one complex (protein + ligand + affinity)
`complex.yaml` (Boltz YAML; every protein needs `msa:`):
```yaml
version: 1
sequences:
  - protein: {id: A, sequence: MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDG, msa: empty}
  - ligand:  {id: B, smiles: 'CC(=O)Nc1ccc(O)cc1'}
properties:
  - affinity: {binder: B}      # one small-molecule binder; omit for structure only
```
```
boltz-mlx predict complex.yaml --out_dir out/ --json            # --diffusion_samples 5 for more poses
```

## Screen a ligand list against one target
`target.yaml` is the same YAML without the ligand. `ligands.csv` has `id,smiles` (also `.smi`, `.sdf`).
```
boltz-mlx screen --target target.yaml --ligands ligands.csv --out run1/ --json --top 20
boltz-mlx screen ... --dry-run     # plan + time estimate, no model; rerun the same command to resume
```
Ranked by `affinity_probability_binary` (hit finding); `--rank-by value` ranks by `affinity_pred_value` (lead optimisation).

## MSA
- `msa: empty`: single sequence, fastest, less accurate. `msa: path/to.a3m` (or `.csv`): your own MSA.
- `--use_msa_server` (predict only): sends the protein sequence to ColabFold (https://api.colabfold.com) and writes
  `out/boltz_results_<name>/msa/*.csv`. Only with the user's consent. `screen` never contacts a server: pass `--msa file.csv`.

## JSON output (stdout; logs and progress go to stderr)
```
predict: {"status":"ok","command":"predict","exit_code":0,"inputs":{...},
  "settings":{"preset":"fast","precision":"f16","structure_steps":20,"structure_steps_reason":null,"affinity_steps":25,"recycles":3,"samples":1,"seed":0,...},
  "targets":[{"id":"complex","status":"ok","structure":".../complex_model_0.cif","confidence_json":"...",
    "confidence_score":0.88,"ptm":0.9,"iptm":0.85,"complex_plddt":0.87,"affinity_json":"...",
    "affinity":{"affinity_pred_value":-1.2,"unit":"log10(IC50 in uM)","ic50_uM":0.063,"affinity_probability_binary":0.91},
    "wall_s":14.2}],"wall_s":14.2}
screen:  {"status":"ok","settings":{...},"counts":{"ligands":..,"ok":..,"invalid":..,"error":..},"top_n":20,
  "top":[{"rank":1,"ligand_id":"..","smiles":"..","affinity":{...as above},"confidence_score":..,"iptm":..}],
  "results_csv":"run1/results.csv","results_jsonl":"...","structures_dir":"run1/structures","wall_s":..}
errors:  {"status":"error","exit_code":2,"error":{"kind":"bad_input","message":"..","fix":".."}}
```
`affinity_pred_value` is log10(IC50 in uM): lower is stronger (-1 = 0.1 uM). `affinity_probability_binary` is P(binder), 0 to 1.
`status` can also be `partial` / `incomplete` (stopped early: rerun) / `dry_run`.

## Exit codes
| code | meaning |
|---|---|
| 0 | ok |
| 1 | runtime failure (message on stderr; `BOLTZ_MLX_TRACEBACK=1` for the traceback) |
| 2 | bad input or arguments (the message names the fix) |
| 3 | setup problem: run `boltz-mlx check` |

## Presets (`--preset`)
- `--preset fast` (default): f16, 20 structure + 25 affinity steps, 3 recycles. Use for screening and everyday predictions.
  If any ligand has fewer than 10 heavy atoms (atoms with atomic number > 1; SMILES via RDKit, CCD codes via `mols/`) and you did not
  set the structure steps, `predict` runs 200 structure steps (affinity stays 25), and `screen` does so for those ligands only. A 4-atom
  ligand collapsed at 20 steps. `settings.structure_steps_reason` in `--json` says why (null otherwise); a note goes to stderr.
- `--preset full`: fp32, 200 + 200 steps (upstream Boltz-2 defaults). Use for the final answer on a shortlist or when stereochemistry matters.
- Explicit `--sampling_steps` / `--steps` etc. override the preset (and the tiny-ligand rule). Macs without the M5 neural accelerators default to `full`.

## Limits
- macOS 26+, Apple silicon only; one GPU job at a time (do not start two).
- Memory is unified RAM: about 400 tokens peaks near 11-15 GB, 1000 tokens near 16 GB (measured on a 24 GB M5 Pro).
  Above ~1500 tokens on 24 GB expect memory failure (exit 1); split the complex or use a bigger Mac.
- Speed on an M5 Max: `fast` screening ~7 s per ligand (400-residue target); `full` ~30 s per ligand.
- No steering potentials; contact constraints and templates are used by the trunk but not as sampling guidance (a warning is printed).
- One affinity binder per prediction. Affinity values are predictions, not measurements.
