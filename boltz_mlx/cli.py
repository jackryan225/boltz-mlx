"""`boltz-mlx {check,setup,predict,screen,gui}`: Boltz-2 inference on Apple MLX with Boltz-compatible outputs.

Agent-friendly surface (AGENTS.md): `check` verifies the setup, `setup` downloads + converts what is missing
(boltz_mlx/assets.py; `gui` runs it on first start); `predict` / `screen` take `--preset fast|full` and `--json`
(one JSON object on stdout, logs on stderr); exit codes 0 ok, 1 runtime failure, 2 bad input, 3 setup problem
(boltz_mlx/cliout.py). The flag-level behaviour below is unchanged: parse_args() returns the raw flags, the preset is
applied in main() (apply_preset), so explicit flags (what the GUI passes) always win.

Featurization, CCD/molecule handling and output writing reuse the upstream `boltz` package
(PyTorch/NumPy); only the model runs in MLX. Output layout matches `boltz predict`:
  <out_dir>/boltz_results_<stem>/predictions/<id>/{<id>_model_<k>.cif, confidence_<id>_model_<k>.json,
  plddt_*.npz, pae_*.npz, pde_*.npz, affinity_<id>.json}

Not supported (raises/warns): steering potentials (--use_potentials), contact/template
guidance potentials (inputs with contact constraints or templates fall back to unguided sampling,
with a warning), multi-device.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import boltz_mlx  # noqa: F401  (TF32 off unless BOLTZ_MLX_TF32=1)
import mlx.core as mx
import numpy as np

from boltz_mlx import cachelimit, tiny_ligand
from boltz_mlx.cliout import (CliError, bad_input, canonical_preset, classify_exception, emit, error_object, json_stdout,
                              preset_settings, public_name)
from boltz_mlx.model import PredictArgs, load_model
from boltz_mlx.paths import default_cache
from boltz_mlx.sampler import SAMPLER_PRESETS, ReplayNoise, SamplerConfig, affinity_config, record_noise, resolve_sampler
from boltz_mlx.seeding import SeededFeatureDataset, install_conformer_seed
from boltz_mlx.upfix import install_upstream_fixes

# --cache > $BOLTZ_MLX_CACHE > $BOLTZ_CACHE > ~/.boltz (boltz_mlx/paths.py)
DEFAULT_CACHE = default_cache()


def _to_torch(x):
    import torch

    if isinstance(x, dict):
        return {k: _to_torch(v) for k, v in x.items()}
    if isinstance(x, mx.array):
        return torch.from_numpy(np.array(x))
    return x


def _batch_to_mx(batch: dict) -> dict:
    import torch

    out = {k: mx.array(v.numpy()) for k, v in batch.items() if isinstance(v, torch.Tensor)}
    if "affinity_mw" in batch and "affinity_mw" not in out:  # collated as a list of floats, not a tensor
        out["affinity_mw"] = mx.array([float(x) for x in batch["affinity_mw"]], mx.float32)
    return out


def _warn_unsupported_guidance(feats: dict) -> None:
    has_contacts = feats["contact_pair_index"].shape[-1] > 0
    has_templates = bool(np.array(feats["template_mask"]).any())
    if has_contacts or has_templates:
        warnings.warn(
            "Input has contact constraints or templates: upstream applies contact/template guidance potentials "
            "during sampling; boltz-mlx does not port the potentials and samples without guidance "
            "(template features are still used by the trunk).",
            stacklevel=2,
        )


_CONF_KEYS = ("pde", "plddt", "complex_plddt", "complex_iplddt", "complex_pde", "complex_ipde",
              "pae", "ptm", "iptm", "ligand_iptm", "protein_iptm", "pair_chains_iptm")


def _structure_pred_dict(out: dict, batch: dict, embeddings: bool = True, pae_pde: bool = True) -> dict:
    """Assemble the dict that upstream Boltz2.predict_step hands to BoltzWriter.

    Only the keys BoltzWriter reads are converted to torch (the pae / pde / plddt logits, ~2 x N^2 x 64 floats, are
    not). embeddings=False leaves out s / z (read only by a writer with write_embeddings=True); pae_pde=False leaves out
    pae / pde, so their npz files are not written (the screen deletes them unless --keep-work)."""
    pred = {
        "exception": False,
        "masks": batch["atom_pad_mask"],
        "token_masks": batch["token_pad_mask"],
        "coords": _to_torch(out["sample_atom_coords"]),
    }
    if embeddings:
        pred["s"] = _to_torch(out["s"])
        pred["z"] = _to_torch(out["z"])
    if "plddt" in out:
        import torch

        keys = _CONF_KEYS if pae_pde else tuple(k for k in _CONF_KEYS if k not in ("pae", "pde"))
        t = {k: _to_torch(out[k]) for k in keys}
        iptm = t["iptm"]
        pred["confidence_score"] = (
            4 * t["complex_plddt"] + (iptm if not torch.allclose(iptm, torch.zeros_like(iptm)) else t["ptm"])
        ) / 5
        pred.update(t)
    return pred


def resolve_samplers(args) -> tuple[SamplerConfig, SamplerConfig]:
    """(structure, affinity) sampler configs from the step / preset flags.

    Structure pass: --sampler's structure part, else --sampling_steps (default 200; 50 with --turbo).
    Affinity pass:  --sampler_affinity (its affinity part, else its structure part: 's25' == 'aff25'), else --sampler's
                    affinity part, else --sampling_steps_affinity (default 200; --turbo does not change it).
    --step_scale (upstream flag, both passes) applies to passes not set by a preset. Conflicting flags are errors.
    """
    from dataclasses import replace

    pre = resolve_sampler(args.sampler) if args.sampler else None
    st = pre.structure if pre else None
    if st is not None:
        if args.sampling_steps is not None and args.sampling_steps != st.steps:
            raise ValueError(f"--sampler {args.sampler} sets {st.steps} structure steps; --sampling_steps {args.sampling_steps} conflicts")
    else:
        steps = args.sampling_steps if args.sampling_steps is not None else (50 if args.turbo else 200)
        st = SamplerConfig(steps)
        if args.step_scale is not None:
            st = replace(st, step_scale=args.step_scale)
    af = pre.affinity if pre else None
    if args.sampler_affinity:
        if af is not None:
            raise ValueError(f"--sampler {args.sampler} already sets the affinity pass; drop --sampler_affinity")
        af = affinity_config(args.sampler_affinity)
    if af is not None:
        if args.sampling_steps_affinity is not None and args.sampling_steps_affinity != af.steps:
            raise ValueError(f"affinity sampler sets {af.steps} steps; --sampling_steps_affinity {args.sampling_steps_affinity} conflicts")
    else:
        af = SamplerConfig(args.sampling_steps_affinity if args.sampling_steps_affinity is not None else 200)
        if args.step_scale is not None:
            af = replace(af, step_scale=args.step_scale)
    return st, af


def msa_server_kwargs(args) -> dict:
    """process_inputs() arguments for the MSA server (upstream boltz.main does the work: compute_msa -> run_mmseqs2 writes
    <out>/msa/<id>_<entity>.csv). Off by default: then nothing is read from the environment and no request is ever made.
    On: credentials fall back to the same environment variables as upstream's `boltz predict`."""
    on = bool(getattr(args, "use_msa_server", False))
    kw = dict(use_msa_server=on, msa_server_url=getattr(args, "msa_server_url", "") or "",
              msa_pairing_strategy=getattr(args, "msa_pairing_strategy", "greedy"))
    if on:
        env = os.environ.get
        kw.update(msa_server_username=args.msa_server_username or env("BOLTZ_MSA_USERNAME"),
                  msa_server_password=args.msa_server_password or env("BOLTZ_MSA_PASSWORD"),
                  api_key_header=args.api_key_header,
                  api_key_value=args.api_key_value or env("MSA_API_KEY_VALUE"))
    return kw


def predict(args) -> None:
    from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
    from boltz.data.types import Manifest
    from boltz.data.write.writer import BoltzAffinityWriter, BoltzWriter
    from boltz.main import check_inputs, filter_inputs_affinity, filter_inputs_structure, process_inputs

    mx.set_default_device(mx.gpu if args.device == "gpu" else mx.cpu)
    # bound MLX's buffer cache so large inputs (800-1000+ tokens) leave headroom on 24 GB machines: 4 GB (or
    # $BOLTZ_MLX_CACHE_GB) now; cachelimit.adapt() below raises it for big inputs once the token count is known
    cachelimit.install_startup()
    if args.fast:
        from boltz_mlx import kernels

        kernels.set_diff_dtype("fp16")
    cache = default_cache(args.cache)
    data = Path(args.data).expanduser()
    out_dir = Path(args.out_dir).expanduser() / f"boltz_results_{data.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)
    s_cfg, a_cfg = resolve_samplers(args)
    settings = {"precision": os.environ.get("BOLTZ_MLX_PRECISION", "fp32"), "fast": bool(args.fast),
                "tf32": os.environ.get("BOLTZ_MLX_TF32") == "1", "sampler": args.sampler, "sampler_affinity": args.sampler_affinity,
                "structure": s_cfg.describe(), "affinity": a_cfg.describe(),
                "structure_steps": s_cfg.steps, "structure_steps_reason": getattr(args, "structure_steps_reason", None),
                "msa_server": args.msa_server_url if args.use_msa_server else None}
    if args.use_msa_server:
        print(f"boltz-mlx: MSA server enabled: {args.msa_server_url} (protein sequences without an MSA are sent there)", file=sys.stderr)
    print(f"boltz-mlx: structure sampler {s_cfg.describe()}; affinity sampler {a_cfg.describe()}", file=sys.stderr)
    (out_dir / "boltz_mlx_settings.json").write_text(json.dumps(settings, indent=1))

    install_conformer_seed()  # SMILES-ligand conformers independent of the input order (seeding.py)
    install_upstream_fixes()  # narrow patches of upstream parsing / writers (boltz_mlx/upfix)
    process_inputs(
        data=check_inputs(data),
        out_dir=out_dir,
        ccd_path=cache / "ccd.pkl",
        mol_dir=cache / "mols",
        **msa_server_kwargs(args),
        boltz2=True,
        preprocessing_threads=1,
        max_msa_seqs=args.max_msa_seqs,
    )
    processed = out_dir / "processed"
    manifest = Manifest.load(processed / "manifest.json")
    opt = lambda p: p if p.exists() else None  # noqa: E731

    def loader(m, target_dir, affinity):
        """Upstream dataset; with --num_workers > 0, featurization of the next input runs in spawned
        worker processes while the GPU predicts the current one (spawn: Metal is not fork-safe)."""
        from torch.utils.data import DataLoader

        from boltz.data.module.inferencev2 import collate

        dl = data_module(m, target_dir, affinity).predict_dataloader()
        ds = SeededFeatureDataset(dl.dataset, args.seed)  # per-record featurizer seed (conformer rotation)
        extra = {} if args.num_workers <= 0 else dict(
            num_workers=args.num_workers, multiprocessing_context="spawn", prefetch_factor=2, persistent_workers=False
        )
        return DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate, **extra)

    def data_module(m, target_dir, affinity):
        return Boltz2InferenceDataModule(
            manifest=m,
            target_dir=target_dir,
            msa_dir=processed / "msa",
            mol_dir=cache / "mols",
            num_workers=0,
            constraints_dir=opt(processed / "constraints"),
            template_dir=opt(processed / "templates"),
            extra_mols_dir=opt(processed / "mols"),
            override_method="other" if affinity else args.method,
            affinity=affinity,
        )

    timings = {}
    t_start = time.perf_counter()
    # ---------------------------------------------------------------- structure
    todo = filter_inputs_structure(manifest=manifest, outdir=out_dir, override=args.override)
    if todo.records:
        writer = BoltzWriter(
            data_dir=processed / "structures",
            output_dir=out_dir / "predictions",
            output_format=args.output_format,
            boltz2=True,
            write_embeddings=args.write_embeddings,
        )
        t0 = time.perf_counter()
        model = load_model("conf")
        timings["load_conf_s"] = time.perf_counter() - t0
        pargs = PredictArgs.from_sampler(s_cfg, args.recycling_steps, args.diffusion_samples)
        for i, batch in enumerate(loader(todo, processed / "structures", False)):
            feats = _batch_to_mx(batch)
            _warn_unsupported_guidance(feats)
            cachelimit.adapt_feats(feats)  # N known, trunk not started
            t0 = time.perf_counter()
            rid = batch["record"][0].id
            noise = (ReplayNoise(args.replay_noise.format(id=rid)) if args.replay_noise
                     else record_noise(args.seed, rid, "structure", pargs.diffusion_samples))
            out = model(feats, pargs, noise)
            timings[f"predict_{batch['record'][0].id}_s"] = time.perf_counter() - t0
            writer.write_on_batch_end(None, None, _structure_pred_dict(out, batch), None, batch, i, 0)
        del model
        mx.clear_cache()

    # ---------------------------------------------------------------- affinity
    if any(r.affinity for r in manifest.records):
        todo = filter_inputs_affinity(manifest=manifest, outdir=out_dir, override=args.override)
        if todo.records:
            writer = BoltzAffinityWriter(data_dir=processed / "structures", output_dir=out_dir / "predictions")
            t0 = time.perf_counter()
            model = load_model("aff")
            timings["load_aff_s"] = time.perf_counter() - t0
            pargs = PredictArgs.from_sampler(a_cfg, 5, args.diffusion_samples_affinity)
            for i, batch in enumerate(loader(todo, out_dir / "predictions", True)):
                feats = _batch_to_mx(batch)
                cachelimit.adapt_feats(feats)  # N known, trunk not started
                t0 = time.perf_counter()
                rid = batch["record"][0].id
                noise = record_noise(args.seed, rid, "affinity", pargs.diffusion_samples)
                out = model(feats, pargs, noise, mw_correction=args.affinity_mw_correction)
                timings[f"affinity_{batch['record'][0].id}_s"] = time.perf_counter() - t0
                pred = {"exception": False}
                for k in ("affinity_pred_value", "affinity_probability_binary", "affinity_pred_value1",
                          "affinity_probability_binary1", "affinity_pred_value2", "affinity_probability_binary2"):
                    pred[k] = _to_torch(out[k])
                writer.write_on_batch_end(None, None, pred, None, batch, i, 0)
            del model
            mx.clear_cache()

    timings["total_wall_s"] = time.perf_counter() - t_start
    (out_dir / "boltz_mlx_timings.json").write_text(json.dumps({k: round(v, 3) for k, v in timings.items()}, indent=1))
    print(json.dumps({k: round(v, 2) for k, v in timings.items()}))
    return timings


class _Parser(argparse.ArgumentParser):
    """Usage errors are CliErrors (exit 2, one line, and a JSON error object with --json) instead of argparse's usage dump."""

    def error(self, message):
        raise bad_input(message, f"see `{self.prog} --help`.")


TOP_EPILOG = """\
first run:   boltz-mlx setup    (downloads the Boltz-2 weights + molecule dictionary, ~6.2 GB, and converts them)
             boltz-mlx gui      (local web interface; runs setup itself when something is missing)
start with:  boltz-mlx check --json
presets (predict, screen):  --preset fast (default)  f16, 20 structure + 25 affinity steps
                                                    (200 structure steps if a ligand has < 10 heavy atoms)
                            --preset full            fp32, 200 + 200 steps (upstream Boltz-2 defaults)
--json prints one JSON object on stdout (logs on stderr). Exit codes: 0 ok, 1 runtime failure, 2 bad input, 3 setup problem.
Guide for AI agents: AGENTS.md in the repository root."""

PREDICT_EPILOG = """\
example:  boltz-mlx predict complex.yaml --out_dir out/ --json
Every protein needs an MSA: `msa: empty` (single sequence), an .a3m/.csv path, or --use_msa_server (sends the sequence to ColabFold).
Output: <out_dir>/boltz_results_<name>/predictions/<id>/ with <id>_model_0.cif, confidence_<id>_model_0.json, affinity_<id>.json.
Expert flags (hidden here, still accepted): --sampler --sampler_affinity --step_scale --precision --fast --tf32 --turbo
--num_workers --method --write_embeddings --replay_noise --max_msa_seqs --diffusion_samples_affinity and the MSA-server credentials.
Exit codes: 0 ok, 1 runtime failure, 2 bad input, 3 setup problem."""


def build_parser() -> argparse.ArgumentParser:
    H = argparse.SUPPRESS  # expert flags keep working but stay out of --help
    ap = _Parser(prog="boltz-mlx", description="Boltz-2 structure and binding-affinity prediction on Apple silicon (MLX).",
                 epilog=TOP_EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="{check,setup,predict,screen,gui}")
    c = sub.add_parser("check", help="verify the setup: macOS 26+, Apple silicon, weights, molecule cache",
                       description="Check that this Mac can run boltz-mlx. Never touches the GPU. Exit 0 ready, 3 not ready.")
    c.add_argument("--json", action="store_true", help="print one JSON object on stdout")
    c.add_argument("--cache", default=None, help="dir with the upstream mols/ dictionary (default $BOLTZ_MLX_CACHE, $BOLTZ_CACHE, ~/.boltz)")
    c.add_argument("--weights", default=None, help="dir with boltz2_{conf,aff}.safetensors (default $BOLTZ_MLX_WEIGHTS)")
    s = sub.add_parser("setup", help="download the Boltz-2 weights + molecule dictionary and convert them (first run)",
                       description="Download whatever is missing (Boltz-2 checkpoints and mols/ from the official Boltz / "
                       "Hugging Face sources, ~6.2 GB on a fresh Mac, resumable, sha256 checked) into the cache and convert "
                       "the checkpoints to MLX weights in <cache>/mlx_weights. Does nothing when everything is there. "
                       "Exit 0 ready, 3 failed (e.g. offline).")
    s.add_argument("--cache", default=None, help="Boltz cache dir (default $BOLTZ_MLX_CACHE, $BOLTZ_CACHE, ~/.boltz)")
    s.add_argument("--weights", default=None, help="where the MLX weights go (default $BOLTZ_MLX_WEIGHTS, else <cache>/mlx_weights)")
    s.add_argument("--prune", action="store_true", help="afterwards delete the downloaded .ckpt files and mols.tar (frees "
                   "~6 GB; upstream `boltz predict` with the same cache would download them again)")
    s.add_argument("--dry-run", action="store_true", help="print the steps and sizes, download nothing")
    s.add_argument("--json", action="store_true", help="print one JSON object on stdout (progress on stderr)")
    p = sub.add_parser("predict", help="predict a structure (and binding affinity) from a Boltz YAML/FASTA",
                       description="Predict structures, confidence and (if the YAML asks for it) binding affinity with Boltz-2.",
                       epilog=PREDICT_EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data", help="Boltz YAML or FASTA file, or a directory of them")
    p.add_argument("--out_dir", default="./", help="output directory (default ./)")
    p.add_argument("--preset", default=None, metavar="{fast,full}",
                   help="fast: f16, 20 structure + 25 affinity steps (200 structure steps when a ligand has < 10 heavy atoms and "
                        "the steps are not set; default; fp32 'full' on Macs without the M5 neural "
                        "accelerators). full: fp32, 200 + 200 steps (upstream Boltz-2 defaults)")
    p.add_argument("--json", action="store_true", help="print one JSON result object on stdout; logs go to stderr")
    p.add_argument("--cache", default=None,
                   help="dir with the upstream mols/ dictionary (default $BOLTZ_MLX_CACHE, $BOLTZ_CACHE, ~/.boltz)")
    p.add_argument("--recycling_steps", type=int, default=3, help="structure-pass recycles (default 3)")
    p.add_argument("--diffusion_samples", type=int, default=1, help="structures to sample (default 1; best is model_0)")
    p.add_argument("--seed", type=int, default=0, help="random seed (default 0)")
    p.add_argument("--sampling_steps", type=int, default=None, help="override the preset's structure-pass steps")
    p.add_argument("--sampling_steps_affinity", type=int, default=None, help="override the preset's affinity-pass steps")
    p.add_argument("--output_format", choices=["mmcif", "pdb"], default="mmcif", help="structure format (default mmcif)")
    p.add_argument("--override", action="store_true", help="recompute outputs that already exist")
    # MSA server (names and defaults as upstream `boltz predict`); OFF by default
    p.add_argument("--use_msa_server", action="store_true",
                   help="Send protein sequences lacking an MSA to the MMSeqs2 server (default off: sequences stay on this machine).")
    p.add_argument("--msa_server_url", default="https://api.colabfold.com", help="MSA server URL (only with --use_msa_server).")
    p.add_argument("--msa_pairing_strategy", default="greedy", help=H)
    p.add_argument("--msa_server_username", default=None, help=H)
    p.add_argument("--msa_server_password", default=None, help=H)
    p.add_argument("--api_key_header", default=None, help=H)
    p.add_argument("--api_key_value", default=None, help=H)
    p.add_argument("--device", choices=["gpu", "cpu"], default="gpu", help=H)
    # expert flags (hidden from --help)
    p.add_argument("--step_scale", type=float, default=None, help=H)
    p.add_argument("--sampler", default=None, help=H)
    p.add_argument("--sampler_affinity", default=None, help=H)
    p.add_argument("--precision", choices=["fp32", "f16"], default=None, help=H)
    p.add_argument("--diffusion_samples_affinity", type=int, default=5, help=H)
    p.add_argument("--affinity_mw_correction", action="store_true", help=H)
    p.add_argument("--write_embeddings", action="store_true", help=H)
    p.add_argument("--max_msa_seqs", type=int, default=8192, help=H)
    p.add_argument("--num_workers", type=int, default=0, help=H)
    p.add_argument("--method", default=None, help=H)
    p.add_argument("--fast", action="store_true", help=H)   # fp16 matmuls in the diffusion transformers
    p.add_argument("--tf32", action="store_true", help=H)
    p.add_argument("--turbo", action="store_true", help=H)  # = --fast --tf32 --sampling_steps 50 (structure pass)
    p.add_argument("--replay_noise", default=None, help=H)
    from boltz_mlx.screen.main import add_parser as add_screen_parser

    add_screen_parser(sub)
    from boltz_mlx.gui.main import add_parser as add_gui_parser

    add_gui_parser(sub)
    return ap


def parse_args(argv=None):
    """Parse and normalise flags: --turbo implies --fast --tf32; sampler flags are validated (errors exit before any model load).
    `screen`, `gui` and `check` run here and exit. Presets are NOT applied here (apply_preset, called by main())."""
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.cmd == "screen":
        from boltz_mlx.screen.main import command as run_screen

        full = list(argv if argv is not None else sys.argv[1:])
        sys.exit(run_screen(args, full[full.index("screen") + 1:]))
    if args.cmd == "gui":
        from boltz_mlx.gui.main import run as run_gui

        sys.exit(run_gui(args))
    if args.cmd == "check":
        sys.exit(run_check(args))
    if args.cmd == "setup":
        sys.exit(run_setup(args))
    if args.cmd == "predict" and args.turbo:
        args.fast, args.tf32 = True, True  # structure steps: 50 unless set (resolve_samplers)
    if args.cmd == "predict":
        try:
            resolve_samplers(args)
        except ValueError as ex:
            ap.error(str(ex))
    return args


# ------------------------------------------------------------------------------------------------------- check
def run_check(args) -> int:
    from boltz_mlx import check

    r = check.run_checks(args.cache, args.weights)
    if args.json:
        emit({**r, "exit_code": 0 if r["ok"] else 3})
    else:
        print(check.format_text(r))
    return 0 if r["ok"] else 3


# ------------------------------------------------------------------------------------------------------- setup
def run_setup(args) -> int:
    from boltz_mlx import assets
    from boltz_mlx.cliout import setup_problem

    log = lambda m: print(m, file=sys.stderr, flush=True)  # noqa: E731  (stdout stays free for --json)
    if args.dry_run:
        p = assets.plan(args.cache, args.weights)
        if args.json:
            emit({"status": "ok", "command": "setup", "dry_run": True, **p, "exit_code": 0})
        else:
            print(f"boltz-mlx setup: {assets.describe(p)}")
            for s in p["steps"]:
                size = s.get("bytes") or s.get("bytes_written") or 0
                print(f"  {s['step']:<8} {s.get('file') or s.get('model') or '':<17} {size / 1e9:5.2f} GB  {s['detail']}")
        return 0
    try:
        p = assets.ensure(args.cache, args.weights, log=log, prune=args.prune)
    except assets.SetupError as ex:
        raise setup_problem(ex.message, ex.fix) from None
    if p["ready"]:
        log(f"boltz-mlx setup: {assets.describe(p)}")
    if args.json:
        emit({"status": "ok", "command": "setup", **assets.plan(args.cache, args.weights), "done": p["steps"], "exit_code": 0})
    else:
        print("boltz-mlx is ready. Start the interface with: boltz-mlx gui")
    return 0


# ------------------------------------------------------------------------------------------------------ predict
def validate_predict_input(data: Path, use_msa_server: bool) -> None:
    """Cheap input checks before any model work (exit 2 with the fix): the path exists, YAML parses, every protein has an MSA."""
    import yaml

    if not data.exists():
        raise bad_input(f"input not found: {data}.", "pass a Boltz YAML/FASTA file, or a directory of them.")
    files = sorted(f for f in data.iterdir() if f.suffix.lower() in _EXT) if data.is_dir() else [data]
    if not files:
        raise bad_input(f"no .yaml/.yml/.fasta files in {data}.", "put the Boltz inputs in that directory.")
    no_msa_fix = ("add `msa: empty` (single sequence, lower accuracy) or an .a3m/.csv path to each protein, or pass "
                  "--use_msa_server (sends the sequence to ColabFold).")
    for f in files:
        if f.suffix.lower() not in _EXT:
            raise bad_input(f"unsupported input type {f.suffix!r}: {f}.", "use a Boltz .yaml/.yml or .fasta/.fa file.")
        if f.suffix.lower() in (".yaml", ".yml"):
            try:
                doc = yaml.safe_load(f.read_text())
            except (yaml.YAMLError, UnicodeDecodeError) as ex:
                raise bad_input(f"{f.name} is not valid YAML: {str(ex).splitlines()[0] if str(ex) else ex}", "fix the YAML syntax.") from None
            seqs = doc.get("sequences") if isinstance(doc, dict) else None
            if not isinstance(seqs, list) or not seqs:
                raise bad_input(f"{f.name} has no `sequences:` list.", "Boltz YAML needs `version: 1` and `sequences:` with "
                                "protein / ligand entries (see AGENTS.md).")
            bare = []
            for e in seqs:
                prot = e.get("protein") if isinstance(e, dict) else None
                if isinstance(prot, dict) and "msa" not in prot:
                    bare.append(str(prot.get("id")))
            if bare and not use_msa_server:
                raise bad_input(f"{f.name}: protein chain(s) {', '.join(bare)} have no `msa:`.", no_msa_fix)
            _check_ligand_smiles(f.name, _yaml_smiles(doc))
        else:  # FASTA: >CHAIN|protein|msa_path
            for line in f.read_text().splitlines():
                if line.startswith(">"):
                    parts = line[1:].strip().split("|")
                    if len(parts) >= 2 and parts[1].lower() == "protein" and (len(parts) < 3 or not parts[2]) and not use_msa_server:
                        raise bad_input(f"{f.name}: protein {parts[0]} has no MSA in its header.",
                                        "use `>A|protein|path/to/msa.a3m` (or `>A|protein|empty`), or pass --use_msa_server.")
            _check_ligand_smiles(f.name, _fasta_smiles(f.read_text()))


_EXT = (".yaml", ".yml", ".fasta", ".fa", ".fas")


def _yaml_smiles(doc: dict) -> list[tuple[str, object, bool]]:
    """(chain id, smiles, is the affinity binder) of every SMILES ligand of a Boltz YAML document."""
    props = doc.get("properties") if isinstance(doc.get("properties"), list) else []
    binders = {str(p["affinity"].get("binder")) for p in props
               if isinstance(p, dict) and isinstance(p.get("affinity"), dict)}
    out = []
    for e in doc["sequences"]:
        lig = e.get("ligand") if isinstance(e, dict) else None
        if isinstance(lig, dict) and "smiles" in lig:
            ids = lig.get("id")
            ids = [str(i) for i in ids] if isinstance(ids, list) else [str(ids)]
            out.append((",".join(ids), lig["smiles"], any(i in binders for i in ids)))
    return out


def _fasta_smiles(text: str) -> list[tuple[str, object, bool]]:
    """(chain id, smiles, False) of the `>ID|smiles` records of a Boltz FASTA (affinity needs a YAML property)."""
    out, lines = [], text.splitlines()
    for k, line in enumerate(lines):
        parts = line[1:].strip().split("|") if line.startswith(">") else []
        if len(parts) >= 2 and parts[1].lower() == "smiles":
            out.append((parts[0], lines[k + 1].strip() if k + 1 < len(lines) else "", False))
    return out


def _check_ligand_smiles(fname: str, ligands: list[tuple[str, object, bool]]) -> None:
    """Exit 2 with a per-ligand message for a SMILES that RDKit refuses (or could abort on), before upstream parses it.
    Ordinary molecules are checked in-process; unusual ones in a child process (boltz_mlx/smiles_guard.py)."""
    from boltz_mlx import smiles_guard

    fix = "fix the SMILES, or give the ligand as a CCD code."
    bad = [(cid, smi) for cid, smi, _ in ligands if not isinstance(smi, str)]
    if bad:
        raise bad_input(f"{fname}: ligand {bad[0][0]}: the SMILES must be a string.", fix)
    if not ligands:
        return
    verdict = smiles_guard.check_many([smi for _, smi, _ in ligands], [aff for _, _, aff in ligands])
    for cid, smi, _ in ligands:
        if verdict[smi]:
            raise bad_input(f"{fname}: ligand {cid}: invalid SMILES {smi[:60]!r}: {verdict[smi]}.", fix)


def apply_preset(args) -> dict:
    """Fill the flags the user left unset from --preset (default fast; full on Macs without NAX). Explicit flags win.
    Names: fast / full, plus the old screen / strict / turbo / f16 (hidden aliases). Returns {preset, mode, source}."""
    from boltz_mlx.screen.presets import PRESETS, default_preset

    source = "flag" if args.preset else "default"
    internal = canonical_preset(args.preset, PRESETS) if args.preset else default_preset()
    st = preset_settings(internal)
    if internal == "turbo":
        args.turbo, args.fast, args.tf32 = True, True, True
    pre = resolve_sampler(args.sampler) if args.sampler else None
    if args.precision is None and not args.turbo:
        args.precision = st["precision"]
    args.structure_steps_reason = None
    if args.sampling_steps is None and not args.turbo and not (pre and pre.structure):
        args.sampling_steps = st["structure_steps"]
        if internal == tiny_ligand.RULE_PRESET:   # Fast, steps not set by the user: tiny ligands get 200 (boltz_mlx/tiny_ligand.py)
            sizes = tiny_ligand.sizes_from_path(args.data, default_cache(args.cache) / "mols")
            args.sampling_steps, args.structure_steps_reason = tiny_ligand.decide(sizes, args.sampling_steps)
            args.uncounted_ligands = [s.label for s in tiny_ligand.uncounted(sizes)]
    if args.sampling_steps_affinity is None and not args.sampler_affinity and not (pre and pre.affinity):
        args.sampling_steps_affinity = st["affinity_steps"]
    for name, lo in (("recycling_steps", 0), ("diffusion_samples", 1), ("diffusion_samples_affinity", 1)):
        if getattr(args, name) < lo:
            raise bad_input(f"--{name} must be >= {lo}.", f"pass --{name} {max(lo, 1)} or more.")
    try:
        resolve_samplers(args)
    except ValueError as ex:
        raise bad_input(str(ex), "drop one of the conflicting flags.") from None
    return {"preset": public_name(internal), "mode": internal, "source": source, "structure_steps_reason": args.structure_steps_reason}


def predict_result(args, info: dict, timings: dict) -> dict:
    """The one JSON object of `boltz-mlx predict --json`, read back from the output directory."""
    from boltz_mlx.cliout import collect_predict

    data = Path(args.data).expanduser()
    results = Path(args.out_dir).expanduser().resolve() / f"boltz_results_{data.stem}"
    ids = None
    try:
        ids = [r["id"] for r in json.loads((results / "processed" / "manifest.json").read_text())["records"]]
    except (OSError, ValueError, KeyError, TypeError):
        pass
    targets = collect_predict(results, ids, timings)
    s_cfg, a_cfg = resolve_samplers(args)
    precision = args.precision or os.environ.get("BOLTZ_MLX_PRECISION", "fp32")
    ok = bool(targets) and all(t["status"] == "ok" for t in targets)
    return {
        "status": "ok" if ok else ("partial" if any(t["status"] == "ok" for t in targets) else "error"),
        "command": "predict", "exit_code": 0 if ok else 1,
        "inputs": {"data": str(data.resolve()), "out_dir": str(Path(args.out_dir).expanduser().resolve()),
                   "results_dir": str(results), "msa_server": args.msa_server_url if args.use_msa_server else None},
        "settings": {"preset": info["preset"], "preset_source": info["source"], "mode": info["mode"], "precision": precision,
                     "structure_steps": s_cfg.steps, "structure_steps_reason": info.get("structure_steps_reason"),
                     "affinity_steps": a_cfg.steps, "recycles": args.recycling_steps,
                     "affinity_recycles": 5, "samples": args.diffusion_samples,
                     "affinity_samples": args.diffusion_samples_affinity, "seed": args.seed,
                     "tf32": bool(args.tf32), "diffusion_fp16": bool(args.fast)},
        "targets": targets,
        "wall_s": round(timings["total_wall_s"], 2) if "total_wall_s" in timings else None,
    }


def run_predict(args, info: dict) -> int:
    from boltz_mlx.check import preflight

    preflight(args.cache)
    if info.get("structure_steps_reason"):
        print(f"boltz-mlx: Fast preset: {resolve_samplers(args)[0].steps} structure steps instead of "
              f"{preset_settings(info['mode'])['structure_steps']}: {info['structure_steps_reason']}. "
              "Set --sampling_steps to choose the steps yourself.", file=sys.stderr)
    for lab in getattr(args, "uncounted_ligands", None) or []:
        print(f"boltz-mlx: could not count the heavy atoms of {lab} (unreadable SMILES or CCD code not in mols/); "
              "it does not trigger the tiny-ligand rule.", file=sys.stderr)
    with json_stdout(args.json):
        timings = predict(args)
    obj = predict_result(args, info, timings)
    for t in obj["targets"]:
        if t["status"] == "ok":
            print(f"boltz-mlx: {t['id']}: {t['structure']} (confidence_score {t['confidence_score']})", file=sys.stderr)
        else:
            print(f"boltz-mlx: {t['id']}: {t.get('error')}", file=sys.stderr)
    if args.json:
        emit(obj)
    if not obj["targets"]:
        raise bad_input("no record was predicted.", "fix the input (the reason is printed above) and run again.")
    return obj["exit_code"]


def _fail(err: CliError, command: str | None, use_json: bool) -> int:
    print(err.line(), file=sys.stderr)
    if use_json:
        emit(error_object(command, err))
    return err.code


def main(argv=None) -> None:
    argv = list(argv if argv is not None else sys.argv[1:])
    use_json = "--json" in argv
    command = next((a for a in argv if not a.startswith("-")), None)
    try:
        rc = _main(argv)
    except CliError as ex:
        rc = _fail(ex, command, use_json)
    except KeyboardInterrupt:
        rc = 130
    except Exception as ex:  # noqa: BLE001  (every failure becomes a one-line message + exit code)
        if os.environ.get("BOLTZ_MLX_TRACEBACK") == "1":
            import traceback

            traceback.print_exc()
        rc = _fail(classify_exception(ex), command, use_json)
    if rc:
        sys.exit(rc)


def _main(argv) -> int:
    args = parse_args(argv)
    info = None
    if args.cmd == "predict":
        validate_predict_input(Path(args.data).expanduser(), args.use_msa_server)
        info = apply_preset(args)
    need = {}
    if getattr(args, "tf32", False) and os.environ.get("BOLTZ_MLX_TF32") != "1":
        need["BOLTZ_MLX_TF32"] = "1"
    if getattr(args, "precision", None) and os.environ.get("BOLTZ_MLX_PRECISION", "fp32") != args.precision:
        need["BOLTZ_MLX_PRECISION"] = args.precision
    if need:
        # TF32 and the precision mode must be configured before MLX / boltz_mlx.kernels initialize: re-exec with them set
        env = {**os.environ, **need}
        os.execve(sys.executable, [sys.executable, "-m", "boltz_mlx.cli", *argv], env)
    if args.cmd == "predict":
        return run_predict(args, info)
    return 0


if __name__ == "__main__":
    main(sys.argv[1:])
