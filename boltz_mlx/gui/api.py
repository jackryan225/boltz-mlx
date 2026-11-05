"""The GUI's API (no HTTP in here): presets, input validation, the dry-run estimate, run creation and the run views.

Everything that starts work does it through RunStore.launch (a `boltz-mlx screen` or `boltz-mlx predict` subprocess). Screen
inputs are validated with the screen's own code (boltz_mlx.screen.inputs), and the screen estimate is the screen's own
`--dry-run` plan (boltz_mlx.screen.runner.plan_lines) with this chip's bench numbers (boltz_mlx/gui/estimates.py) in place of
the M5 Pro defaults; a chip without numbers gets "no estimate yet for this chip". Structure jobs: boltz_mlx/gui/predict.py."""

from __future__ import annotations

import dataclasses
import json
import math
import re
import statistics
import tempfile
from pathlib import Path

from boltz_mlx.gui import estimates as EST
from boltz_mlx.gui import inputs as GI
from boltz_mlx.gui import predict as PR
from boltz_mlx.gui import presets as GP
from boltz_mlx.gui.common import LIMITS, PRIVACY, PRIVACY_GUI, ApiError, is_within, new_token, valid_run_id
from boltz_mlx.gui.depict import depict_svg
from boltz_mlx.gui.runs import RunStore
from boltz_mlx.gui.targets import TargetLibrary
from boltz_mlx.screen.presets import default_preset, est_overhead

MOLSTAR = {"name": "Mol*", "version": "4.5.0", "license": "MIT", "url": "https://molstar.org",
           "citation": "Sehnal D, et al. Mol* Viewer: modern web app for 3D visualization and analysis of large biomolecular "
                       "structures. Nucleic Acids Res. 2021;49(W1):W431-W437."}

RANK_HELP = {
    "prob": "affinity_probability_binary: the chance the ligand is a binder. Use it to find hits in a library "
            "(high is better).",
    "value": "affinity_pred_value: a log10(IC50 in uM)-like number. Use it to rank actives within one congeneric "
             "series (lower is stronger).",
}


def package_version() -> str:
    try:
        from importlib.metadata import version

        return version("boltz-mlx")
    except Exception:  # noqa: BLE001
        return "dev"


class GuiApp:
    def __init__(self, runs_dir, *, token: str | None = None, screen_cmd: list[str] | None = None, cache: str | None = None,
                 weights: str | None = None, predict_cmd: list[str] | None = None):
        from boltz_mlx.screen.main import load_presets

        self.token = token or new_token()
        self.cache = cache
        self.weights = weights
        extra = []
        if cache:
            extra.append(f"--cache={Path(cache).expanduser().resolve()}")
        if weights:
            extra.append(f"--weights={Path(weights).expanduser().resolve()}")
        pargs = [f"--cache={Path(cache).expanduser().resolve()}"] if cache else []
        penv = {"BOLTZ_MLX_WEIGHTS": str(Path(weights).expanduser().resolve())} if weights else {}
        self.runs = RunStore(runs_dir, screen_cmd=screen_cmd, extra_args=extra, predict_cmd=predict_cmd, predict_args=pargs,
                             predict_env=penv)
        self.presets = load_presets(None)
        self.targets = TargetLibrary(self.runs.root)

    # ---------------------------------------------------------------------------------------------- info
    def checks(self) -> dict:
        from boltz_mlx.paths import default_cache, weights_dir

        w = Path(self.weights).expanduser() if self.weights else weights_dir(self.cache)
        c = default_cache(self.cache)
        return {
            "weights": {"dir": str(w), "conf": (w / "boltz2_conf.safetensors").exists(), "aff": (w / "boltz2_aff.safetensors").exists(),
                        "conf_cfg": (w / "boltz2_conf.json").exists(), "aff_cfg": (w / "boltz2_aff.json").exists(),
                        "mixed": (w / "boltz2_conf_mixed.safetensors").exists() and (w / "boltz2_aff_mixed.safetensors").exists()},
            "cache": {"dir": str(c), "mols": (c / "mols").is_dir()},
        }

    @staticmethod
    def missing(ck: dict) -> dict:
        """What each workflow needs and lacks: structure (a prediction), affinity (the affinity pass; every screen), molecules (a
        screen's molecule dictionary; Boltz-2 reads molecules from mols/, never ccd.pkl)."""
        w, c = ck["weights"], ck["cache"]
        return {"structure": [n for n, ok in (("boltz2_conf.safetensors", w["conf"]), ("boltz2_conf.json", w["conf_cfg"])) if not ok],
                "affinity": [n for n, ok in (("boltz2_aff.safetensors", w["aff"]), ("boltz2_aff.json", w["aff_cfg"])) if not ok],
                "mols": [] if c["mols"] else ["mols/"]}

    def problems_for(self, kind: str, affinity: bool = False) -> list[str]:
        """Setup problems that stop this workflow ('predict' with or without the affinity pass, or 'screen'), each naming the
        component: "Structure weights missing ...". Empty when the workflow can run."""
        ck = self.checks()
        m = self.missing(ck)
        wd, cd = ck["weights"]["dir"], ck["cache"]["dir"]
        out = []
        if m["structure"]:
            out.append(f"Structure weights missing in {wd}: {', '.join(m['structure'])} (run `boltz-mlx setup`, or pass --weights / set BOLTZ_MLX_WEIGHTS).")
        if m["affinity"] and (kind == "screen" or affinity):
            out.append(f"Affinity weights missing in {wd}: {', '.join(m['affinity'])} (run `boltz-mlx setup`, or pass --weights / set BOLTZ_MLX_WEIGHTS).")
        if m["mols"]:
            out.append(f"Molecule dictionary missing: {Path(cd) / 'mols'} (run `boltz-mlx setup`, or pass --cache / set BOLTZ_MLX_CACHE).")
        return out

    def info(self) -> dict:
        ck = self.checks()
        by = {"predict": self.problems_for("predict"), "affinity": self.problems_for("predict", True), "screen": self.problems_for("screen")}
        problems = list(dict.fromkeys(by["affinity"] + by["screen"]))   # everything that stops some workflow, once each
        return {
            "version": package_version(), "runs_dir": str(self.runs.root), "checks": ck, "problems": problems, "problems_by": by,
            "privacy": PRIVACY, "privacy_gui": PRIVACY_GUI, "molstar": MOLSTAR, "default_preset": default_preset(),
            "limits": {k: LIMITS[k] for k in ("target", "msa", "ligands", "body")}, "rank_help": RANK_HELP,
            "active_run": self.runs.active_run_id(), "chip": EST.chip_info(),
            "gui_presets": GP.payload(EST.chip_name()), "default_screen_preset": GP.default_key(),
            "default_predict_preset": GP.default_key(),
        }

    def presets_payload(self) -> list[dict]:
        out = []
        chip = EST.chip_name()
        k = EST.chip_constants(chip)
        default = default_preset()  # by kernel path: screen on NAX (M5-class), strict on M1-M4
        for name, p in self.presets.items():
            e = EST.entry(name, chip)
            ref = EST.reference_preset(p, chip)
            est = None
            if ref is not None:  # this preset's own step counts, scaled from the chip entry's
                from boltz_mlx.screen.presets import affinity_estimate, struct_estimate

                est = {"struct_117_s": round(struct_estimate(ref, 117, p.steps), 2),
                       "struct_400_s": round(struct_estimate(ref, 400, p.steps), 2),
                       "affinity_s": round(affinity_estimate(ref, p.affinity_steps), 2), "overhead_s": k["overhead"],
                       "source": e["source"]}
            out.append({
                "name": name, "validated": p.validated, "status": "validated" if p.validated else "provisional",
                "note": p.note, "default": name == default, "env": dict(p.env),
                "steps": p.steps, "recycling": p.recycling, "affinity_steps": p.affinity_steps,
                "affinity_recycling": p.affinity_recycling, "affinity_samples": p.affinity_samples,
                "caution": None if p.validated else
                "Not a gated preset: its accuracy has not been validated on ligands / affinity (docs/researcher.md, section 5). "
                "Use it for exploration, and confirm hits with the strict preset.",
                "est": est, "chip": chip,
            })
        out.sort(key=lambda d: (not d["validated"], not d["default"], d["name"]))
        return out

    # ------------------------------------------------------------------------------------------- validation
    def _preset_times(self, args, target, ligs) -> dict:
        """Seconds per ligand for this target on this chip, per preset (None: no bench numbers for this chip / preset)."""
        from boltz_mlx.screen.presets import affinity_estimate, struct_estimate

        sizes = [l.n_atoms for l in ligs if l.status != "invalid" and l.n_atoms]
        atoms = int(statistics.median(sizes)) if sizes else 30
        tokens = target.n_residues + atoms
        over = EST.chip_constants()["overhead"]
        out = {}
        for name, p in self.presets.items():
            ref = EST.reference_preset(p)
            out[name] = None if ref is None else round(
                struct_estimate(ref, tokens, p.steps) + affinity_estimate(ref, p.affinity_steps) + over, 1)
        return out

    def validate(self, payload: dict) -> dict:
        """Check the target and/or the ligands of a payload (each independently; a section that is absent is skipped)."""
        if not isinstance(payload, dict):
            raise ApiError(400, "expected a JSON object")
        options = GI.normalize_options(payload.get("options"), self.presets)
        res: dict = {"target": None, "ligands": None, "preset_times": None, "limits": dict(LIMITS)}
        with tempfile.TemporaryDirectory(prefix="boltz-gui-") as td:
            rd = Path(td)
            inp = rd / "inputs"
            target = ligs = None
            t_path = l_path = None
            msa_mode, m_path = "default", None
            if payload.get("target") is not None:
                try:
                    rr = self._payload_rerun(payload)
                    if rr is not None:
                        t_path, _, msa_mode, m_path, _ = GI.stage_rerun(rr[0], rr[1], inp)
                    else:
                        t_path, _ = GI.stage_target(payload["target"], inp, self.targets)
                        if not GI.is_saved(payload["target"]):
                            msa_mode, m_path, _ = GI.stage_msa(payload.get("msa"), inp)
                except ApiError as ex:
                    res["target"] = {"ok": False, "error": ex.message}
            if payload.get("ligands") is not None:
                try:
                    l_path, _ = GI.stage_ligands(payload["ligands"], inp)
                except ApiError as ex:
                    res["ligands"] = {"ok": False, "error": ex.message}

            def argv_for(t, l):
                a = [f"--target={t or inp / 'none.fasta'}", f"--ligands={l or inp / 'none.csv'}", f"--out={rd / 'out'}",
                     f"--preset={options['preset']}", f"--rank-by={options['rank_by']}", f"--chunk={options['chunk']}",
                     f"--seed={options['seed']}", f"--save-structures={options['save_structures']}"]
                if msa_mode == "empty":
                    a.append("--msa=empty")
                elif msa_mode == "file" and m_path:
                    a.append(f"--msa={m_path}")
                a += [f"--pocket={p}" for p in options["pocket"]]
                if options["ligand_chain"]:
                    a.append(f"--ligand-chain={options['ligand_chain']}")
                for k, flag in (("id_col", "--id-col"), ("smiles_col", "--smiles-col")):
                    if options[k]:
                        a.append(f"{flag}={options[k]}")
                for k, flag in (("steps", "--steps"), ("recycling", "--recycling"), ("affinity_steps", "--affinity-steps")):
                    if options[k] is not None:
                        a.append(f"{flag}={options[k]}")
                return a

            args = GI.parse_screen_argv(argv_for(t_path, l_path))
            if t_path is not None and res["target"] is None:
                try:
                    target = GI.load_target_checked(args)
                    res["target"] = {"ok": True, **GI.target_summary(target)}
                except ApiError as ex:
                    res["target"] = {"ok": False, "error": ex.message}
            if l_path is not None and res["ligands"] is None:
                try:
                    ligs = GI.load_ligands_checked(args)
                    rows, counts = GI.ligand_rows(ligs, LIMITS["preview_rows"])
                    res["ligands"] = {"ok": True, "rows": rows, "counts": counts}
                except ApiError as ex:
                    res["ligands"] = {"ok": False, "error": ex.message}
            if target is not None and ligs is not None:
                res["preset_times"] = self._preset_times(args, target, ligs)
        res["ok"] = bool(res["target"] and res["target"]["ok"] and res["ligands"] and res["ligands"]["ok"]
                         and res["ligands"]["counts"]["valid"] > 0)
        return res

    # -------------------------------------------------------------------------------------------- estimate
    def _estimate(self, args, target, ligs, done: set | None = None, label: str | None = None) -> dict:
        from boltz_mlx.screen.main import build_config
        from boltz_mlx.screen.runner import _dur, estimate_seconds, est_tokens, plan_lines
        from boltz_mlx.screen.presets import affinity_estimate, for_device, struct_estimate
        from boltz_mlx import kernels as K

        # as `screen` does (screen/main.py): hardware line in the plan, preset labelled for this kernel path
        hw = K.device_tag()
        presets = dict(self.presets)
        presets[args.preset] = for_device(presets[args.preset], hw["path"])
        try:
            cfg = build_config(args, presets)
        except SystemExit as ex:
            raise ApiError(400, str(ex)) from None
        cfg.hw = hw
        cfg.env = dict(presets[args.preset].env)
        chip = EST.chip_name()
        ref = EST.reference_preset(cfg.preset, chip)
        k = EST.chip_constants(chip)
        # the estimate functions read est_* (and the steps they were measured at) from cfg.preset: a copy of the config
        # carries this chip's numbers; the plan itself is printed from the real config (its mode / steps are the run's)
        ecfg = dataclasses.replace(cfg, preset=ref) if ref is not None else None
        done = done or set()
        lines = [re.sub(r", (\d+) already done in [^,]+,", r", \1 already done,", ln) for ln in plan_lines(cfg, target, ligs, done)]
        if label and lines and lines[0].startswith(f"target: {target.source}"):
            lines[0] = f"target: {label}" + lines[0][len(f"target: {target.source}"):]  # not the temporary staging path
        todo = [l for l in ligs if l.status != "invalid" and l.ligand_id not in done]
        p0 = presets[args.preset]  # the benchmark and the measured time are for the preset's own step counts only
        custom = (cfg.steps, cfg.recycling, cfg.affinity_steps) != (p0.steps, p0.recycling, p0.affinity_steps)
        est: dict = {"n_to_run": len(todo), "preset": args.preset, "validated": cfg.preset.validated, "w_struct": 0.4,
                     "chip": chip, "available": ref is not None}
        if todo and ref is not None:
            per = [estimate_seconds(ecfg, target, l) for l in todo]
            # estimate_seconds added the screen's overhead for this point; the GUI table has its own per-chip overhead
            hw_chip = cfg.hw.get("chip") if cfg.hw and cfg.hw.get("path") == "nax" else None  # as estimate_seconds
            o = est_overhead(ecfg.preset, hw_chip, cfg.steps, cfg.affinity_steps)
            per = [x - o + k["overhead"] for x in per]
            n_chunks = math.ceil(len(todo) / cfg.chunk)
            loads = n_chunks * 2 * k["load"]
            total = sum(per) + loads
            toks = statistics.median(est_tokens(target, l) for l in todo)
            s, a = struct_estimate(ref, toks, cfg.steps), affinity_estimate(ref, cfg.affinity_steps)
            est.update(per_ligand_median_s=round(statistics.median(per), 1), per_ligand_min_s=round(min(per), 1),
                       per_ligand_max_s=round(max(per), 1), n_chunks=n_chunks, n_loads=n_chunks * 2, load_s=round(loads, 1),
                       total_s=round(total), ligands_per_hour=round(3600 * len(todo) / total, 1),
                       tokens_min=min(est_tokens(target, l) for l in todo), tokens_max=max(est_tokens(target, l) for l in todo),
                       w_struct=round(s / (s + a), 3))
            src = EST.entry(args.preset, chip)["source"]
            eline = (f"ESTIMATE (not a measurement; {src}; other Macs differ): ~{statistics.median(per):.0f} s/ligand (range "
                     f"{min(per):.0f}-{max(per):.0f} s over {est['tokens_min']}-{est['tokens_max']} tokens), total ~{_dur(total)} "
                     f"({3600 * len(todo) / total:.0f} ligands/h) incl. {n_chunks * 2} checkpoint loads")
            lines = [eline if ln.startswith("ESTIMATE") else ln for ln in lines]
        elif todo and custom:
            est["custom"] = True
            est["n_chunks"] = math.ceil(len(todo) / cfg.chunk)
            lines = [f"ESTIMATE: none for custom step counts ({chip}); the run measures its own rate" if ln.startswith("ESTIMATE") else ln for ln in lines]
        elif todo and GP.measured(chip, args.preset):
            m = GP.measured(chip, args.preset)
            n = len(todo)
            mid = (m["lo"] + m["hi"]) / 2
            est.update(available=True, measured=m, per_ligand_median_s=round(mid, 1), per_ligand_min_s=m["lo"],
                       per_ligand_max_s=m["hi"], n_chunks=math.ceil(n / cfg.chunk), total_s=round(n * mid),
                       total_lo_s=round(n * m["lo"]), total_hi_s=round(n * m["hi"]), ligands_per_hour=round(3600 / mid, 1))
            mline = (f"ESTIMATE from a measurement on this chip ({m['what']}; {m['source']}): {m['lo']:g}-{m['hi']:g} s/ligand, "
                     f"total ~{_span(n * m['lo'], n * m['hi'])} for {n} ligands")
            lines = [mline if ln.startswith("ESTIMATE") else ln for ln in lines]
        elif todo:
            est["n_chunks"] = math.ceil(len(todo) / cfg.chunk)
            why = GP.MEASURING.get((chip, args.preset))
            if why:
                est["measuring"] = why
            lines = [f"ESTIMATE: {why or EST.NO_ESTIMATE} ({chip}, preset {args.preset}); the first ligands of the run give the "
                     "real rate" if ln.startswith("ESTIMATE") else ln for ln in lines]
        from boltz_mlx.screen.main import steps_info, steps_note

        est["steps_note"] = steps_note(steps_info(cfg, ligs))   # Fast + no --steps: ligands with < 10 heavy atoms run 200 steps
        if est["steps_note"] and est.get("measured"):
            est["steps_note"] += "; the measured time shown is for ordinary ligands and does not include them"
        if est.get("measured"):
            m = est["measured"]
            est["caveat"] = (f"From whole ligands measured on this chip ({m['what']}). Per-ligand time varies with the target "
                             "and ligand size; the first ligands of a real run give the real rate.")
        elif est.get("custom"):
            est["available"] = False
            est["caveat"] = "Custom step counts: no estimate. The run measures its own rate."
        elif ref is not None:
            est["caveat"] = ("ESTIMATE from benchmark runs on this chip, not a measurement. Per-ligand time varies with ligand and "
                             "target size; the first ligands of a real run give the real rate.")
            est["reference"] = EST.chip_info(chip).get("reference")
        elif est.get("measuring"):
            est["caveat"] = (f"No time yet for this preset on {chip}: {est['measuring']}. The run measures its own rate: the time "
                             "left appears after the first ligands finish.")
        else:
            est["caveat"] = (f"{EST.NO_ESTIMATE} ({chip}). The run measures its own rate: the time left appears after the "
                             "first ligands finish.")
        return {"lines": lines, "estimate": est}

    def _payload_rerun(self, payload: dict):
        """A target given as {"kind": "run", "id": ...} is that screen's own staged target bundle (YAML / FASTA with its MSA and
        template files, as for a re-check): (run dir, inputs record), or None for any other target."""
        t = payload.get("target") if isinstance(payload, dict) else None
        if not (isinstance(t, dict) and t.get("kind") == "run"):
            return None
        rid = str(t.get("id") or "")
        meta = self.runs.read_meta(rid) if valid_run_id(rid) else None
        if not meta or (meta.get("kind") or "screen") != "screen":
            raise ApiError(400, "target: the run this target came from is not available")
        return self.runs.run_dir(rid), meta.get("inputs") or {}

    def _staged_args(self, payload: dict, rd: Path, rerun=None):
        """Stage the payload into rd/inputs, parse the argv with the screen's parser, load + validate target and ligands."""
        rerun = rerun or self._payload_rerun(payload)
        options = GI.normalize_options(payload.get("options"), self.presets)
        staged = GI.stage_inputs(payload, rd / "inputs", self.targets, rerun=rerun)
        meta_inputs = GI.inputs_meta(staged, rd)
        argv = GI.screen_argv(rd, meta_inputs, options, rd / "out", self.runs.extra_args)
        args = GI.parse_screen_argv(argv)
        target, ligs = GI.load_inputs(args)
        return options, meta_inputs, args, target, ligs

    def estimate(self, payload: dict) -> dict:
        """The screen's --dry-run for this payload: plan lines + a structured time ESTIMATE (no model, nothing written)."""
        with tempfile.TemporaryDirectory(prefix="boltz-gui-") as td:
            options, meta_inputs, args, target, ligs = self._staged_args(payload, Path(td))
            valid = [l for l in ligs if l.status != "invalid"]
            if not valid:
                raise ApiError(400, "no valid ligands to run")
            out = self._estimate(args, target, ligs, label=meta_inputs["target"]["name"])
            out["target"] = GI.target_summary(target)
            return out

    # ----------------------------------------------------------------------------------------------- runs
    def create_run(self, payload: dict, _rerun=None, _extra_meta: dict | None = None) -> dict:
        """Validate, create the run directory (inputs + gui_run.json) and start the screen. 409 if another one is running.
        _rerun / _extra_meta are set by recheck() only (never from a request body)."""
        if isinstance(payload, dict) and payload.get("kind") == "predict":
            return self.create_predict_run(payload)
        if self.runs.active_run_id():
            raise ApiError(409, f"another screen is running ({self.runs.active_run_id()}): only one screen runs at a time "
                                f"(the GPU is shared). Wait for it or stop it first.")
        problems = self.problems_for("screen")
        if problems:
            raise ApiError(400, problems[0])
        name = re.sub(r"[\x00-\x1f]+", " ", str(payload.get("name") or "")).strip()[: LIMITS["name"]]
        tspec = payload.get("target") or {}
        if _rerun is not None:
            label = name or "recheck"
        elif isinstance(tspec, dict) and tspec.get("kind") == "run":
            label = name or "screen"
        elif GI.is_saved(tspec):
            label = name or self.targets.get(str(tspec.get("id") or "")).get("name") or "screen"
        else:
            label = name or Path(str(tspec.get("filename") or "screen")).stem
        run_id, rd = self.runs.allocate(label)
        try:
            options, meta_inputs, args, target, ligs = self._staged_args(payload, rd, rerun=_rerun)
            valid = [l for l in ligs if l.status != "invalid"]
            if not valid:
                raise ApiError(400, "no valid ligands to run")
            est = self._estimate(args, target, ligs, label=meta_inputs["target"]["name"])
            _, counts = GI.ligand_rows(ligs, 0)
            summary = {
                "total": counts["unique"], "n_valid": counts["valid_unique"], "n_invalid": counts["invalid"],
                "n_warn": counts["warn"], "target": GI.target_summary(target), "estimate": est["estimate"],
            }
            import time

            meta = {"version": 1, "kind": "screen", "id": run_id, "name": name or label, "created": time.time(), "inputs": meta_inputs,
                    "options": options, "summary": summary, "launches": []}
            meta.update(_extra_meta or {})
            self.runs.write_meta(run_id, meta)
        except Exception:
            self.runs.discard_new(run_id)
            raise
        try:
            self.runs.launch(run_id)
        except ApiError:
            raise
        return {"id": run_id, "state": self.runs.state(run_id)}

    # ------------------------------------------------------------------------------------------ re-check shortlist
    RECHECK_MAX = 200

    def recheck(self, run_id: str, payload: dict) -> dict:
        """Re-run ticked ligands of a screen with the Reference (strict) preset: upstream Boltz-2's own defaults at fp32
        (200 structure steps, 3 recycles, affinity 200 x 5). Same target, MSA, seed and options; a new run that records
        recheck_of, so compare() can put the two results side by side."""
        import csv as _csv
        import io as _io

        meta = self.runs.read_meta(run_id)
        if not meta:
            raise ApiError(400, "re-check: works on screens started from this interface (a command-line run has no staged "
                                "inputs here)")
        if (meta.get("kind") or "screen") != "screen":
            raise ApiError(400, "re-check: only a screen's ligands can be re-checked")
        ids = (payload or {}).get("ligand_ids")
        if not isinstance(ids, list) or not ids or not all(isinstance(x, str) for x in ids):
            raise ApiError(400, "re-check: tick one or more ligands")
        ids = list(dict.fromkeys(ids))
        if len(ids) > self.RECHECK_MAX:
            raise ApiError(400, f"re-check: at most {self.RECHECK_MAX} ligands at a time")
        rows = {str(r.get("ligand_id")): r for r in self.runs.rows(run_id).rows()}
        missing = [x for x in ids if x not in rows]
        if missing:
            raise ApiError(400, f"re-check: not in this run: {', '.join(missing[:5])}")
        buf = _io.StringIO()
        w = _csv.writer(buf, lineterminator="\n")
        w.writerow(["id", "smiles"])
        for x in ids:
            w.writerow([x, rows[x].get("smiles") or ""])
        o = {k: v for k, v in (meta.get("options") or {}).items() if k in GI.OPTION_DEFAULTS}
        o.update(preset="strict", steps=None, recycling=None, affinity_steps=None, id_col=None, smiles_col=None,
                 save_structures="all", max_minutes=None)
        parent_name = meta.get("name") or run_id
        new = {"name": f"Recheck · {parent_name}"[: LIMITS["name"]],
               "ligands": {"kind": "file", "filename": "recheck.csv", "content": buf.getvalue()}, "options": o}
        return self.create_run(new, _rerun=(self.runs.run_dir(run_id), meta.get("inputs") or {}),
                               _extra_meta={"recheck_of": run_id, "recheck_of_preset": (meta.get("options") or {}).get("preset")})

    def compare(self, run_id: str) -> dict:
        """A re-check run beside its original: per ligand, the original preset's and Reference's P(binder) and value,
        their difference and both ranks (each run ranked on its own rows)."""
        from boltz_mlx.screen import results as R

        meta = self.runs.read_meta(run_id) or {}
        parent = meta.get("recheck_of")
        if not parent:
            raise ApiError(404, "this run is not a re-check")
        pmeta = self.runs.read_meta(parent) or {}
        rb = (meta.get("options") or {}).get("rank_by") or "prob"
        mine = R.rank_rows(self.runs.rows(run_id).rows(), rb)
        ranked_parent = R.rank_rows(self.runs.rows(parent).rows(), rb)
        theirs = {str(r.get("ligand_id")): r for r in ranked_parent}
        out = []
        for r in mine:
            p = theirs.get(str(r.get("ligand_id"))) or {}
            d = {"ligand_id": r.get("ligand_id"), "record_id": r.get("record_id"), "status": r.get("status"),
                 "rank": r.get("rank"), "orig_rank": p.get("rank"), "orig_record_id": p.get("record_id")}
            for k in ("affinity_probability_binary", "affinity_pred_value"):
                a, b = p.get(k), r.get(k)
                d["orig_" + k], d[k] = a, b
                d["delta_" + k] = (b - a) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None
            out.append(d)
        orig_preset = meta.get("recheck_of_preset") or (pmeta.get("options") or {}).get("preset")
        return {"run": run_id, "original": parent, "original_name": pmeta.get("name") or parent,
                "original_preset": orig_preset, "original_label": GP.run_label(self.runs.screen_config(parent), orig_preset),
                "preset": (meta.get("options") or {}).get("preset"), "rank_by": rb, "rows": out,
                "n_original": sum(1 for r in ranked_parent if r.get("rank")), "n_recheck": sum(1 for r in mine if r.get("rank"))}

    # ------------------------------------------------------------------------------------------ targets
    def _check_target(self, yaml_path: Path) -> dict:
        """Load a prepared target.yaml with the screen's own loader; its summary (raises ApiError 400)."""
        args = GI.parse_screen_argv([f"--target={yaml_path}", f"--ligands={yaml_path.parent / 'none.csv'}",
                                     f"--out={yaml_path.parent / 'out'}"])
        t = GI.load_target_checked(args)
        return GI.target_summary(t)

    def prepare_target(self, payload: dict) -> dict:
        return self.targets.prepare(payload, self._check_target)

    def list_targets(self) -> list[dict]:
        return self.targets.list()

    def get_target(self, tid: str) -> dict:
        return self.targets.get(tid)

    def delete_target(self, tid: str) -> dict:
        return self.targets.delete(tid)

    # ------------------------------------------------------------------------------------- structure jobs
    def _predict_check(self, payload: dict) -> tuple[dict, dict]:
        from boltz_mlx.paths import default_cache

        if not isinstance(payload, dict):
            raise ApiError(400, "expected a JSON object")
        options = PR.normalize_predict_options(payload.get("options"), self.presets, GP.default_key())
        if payload.get("yaml") is not None:
            if not isinstance(payload["yaml"], dict):
                raise ApiError(400, "YAML input: expected an object")
            checked = PR.check_yaml(payload["yaml"], default_cache(self.cache))
        else:
            from boltz_mlx.gui import constraints as CX

            checked = PR.check_chains(payload, default_cache(self.cache))
            CX.check_extras(payload, checked, default_cache(self.cache) / "mols")   # constraints and templates of the form
        PR.apply_tiny_ligand_rule(options, checked)   # Fast + no explicit steps + a ligand with < 10 heavy atoms -> 200 steps
        return options, checked

    def _predict_estimate(self, options: dict, checked: dict) -> dict:
        p = self.presets[options["preset"]]
        chip = EST.chip_name()
        e = EST.predict_seconds(p, checked["n_tokens"], steps=options["steps"], affinity=checked["affinity"],
                                affinity_steps=options["affinity_steps"], samples=options["samples"], chip=chip)
        custom = [k for k in options["changed"] if k == "recycling" or (k == "affinity_samples" and checked["affinity"])]
        if custom:   # the bench rows were measured at the preset's recycles / affinity samples: no number for other values
            e = None
        out = {"chip": chip, "available": e is not None, "preset": p.name, "validated": p.validated and not options["changed"],
               "n_tokens": checked["n_tokens"]}
        # memory does not depend on the step counts: shown whenever there is a job, with the precision the preset runs at
        out["memory"] = EST.predict_memory_gb(checked["n_tokens"], "f16" if "--precision=f16" in PR.PRESET_FLAGS.get(p.name, []) else "fp32")
        if options.get("steps_reason"):
            out["steps_note"] = (f"Fast runs {options['steps']} structure steps here, not {p.steps}: {options['steps_reason']}. "
                                 "Set the structure steps yourself to override.")
        if custom:
            out["caveat"] = "No time estimate for custom recycles or affinity samples: the job page shows the elapsed time."
        elif e is not None:
            out.update(e)
            out["caveat"] = ("ESTIMATE from benchmark runs on this chip, including model loading; the first job after a "
                             "restart takes a little longer.")
        else:
            out["caveat"] = f"{EST.NO_ESTIMATE} ({chip}): the job page shows the elapsed time instead."
        return out

    def validate_predict(self, payload: dict) -> dict:
        """Check a structure job without starting it: normalised chains, the YAML that would run, warnings, estimate."""
        try:
            options, checked = self._predict_check(payload)
        except ApiError as ex:
            return {"ok": False, "error": ex.message}
        times = {}
        for name in self.presets:
            if name not in PR.PRESET_FLAGS:
                continue
            p = self.presets[name]
            e = EST.predict_seconds(p, checked["n_tokens"], steps=p.steps, affinity=checked["affinity"],
                                    affinity_steps=p.affinity_steps, samples=options["samples"])
            times[name] = None if e is None else e["total_s"]
        return {"ok": True, "chains": PR.public_chains(checked), "n_tokens": checked["n_tokens"], "warnings": checked["warnings"],
                "affinity": checked["affinity"], "binder": checked["binder"], "yaml": PR.preview_yaml(checked),
                "constraints": checked.get("constraints") or [], "templates": PR.public_templates(checked),
                "options": options, "estimate": self._predict_estimate(options, checked), "preset_times": times}

    def create_predict_run(self, payload: dict) -> dict:
        if self.runs.active_run_id():
            raise ApiError(409, f"another job is running ({self.runs.active_run_id()}): only one job runs at a time "
                                f"(the GPU is shared). Wait for it or stop it first.")
        options, checked = self._predict_check(payload)
        problems = self.problems_for("predict", checked["affinity"])
        if problems:
            raise ApiError(400, problems[0])
        name = re.sub(r"[\x00-\x1f]+", " ", str(payload.get("name") or "")).strip()[: LIMITS["name"]]
        chains = PR.public_chains(checked)
        short = " + ".join(f"{c['type']} {c['id']}" + (f" ({c['length']})" if c.get("length") else "") for c in chains)
        label = name or ("structure" if not checked["affinity"] else "affinity")
        run_id, rd = self.runs.allocate(label)
        try:
            import time

            inputs = PR.stage(checked, rd / "inputs")
            est = self._predict_estimate(options, checked)
            summary = {"chains": chains, "chains_short": short, "n_tokens": checked["n_tokens"], "affinity": checked["affinity"],
                       "binder": checked["binder"], "warnings": checked["warnings"], "estimate": est,
                       "label": short, "constraints": checked.get("constraints") or [], "templates": PR.public_templates(checked)}
            meta = {"version": 1, "kind": "predict", "id": run_id, "name": name or short[:60] or run_id, "created": time.time(),
                    "inputs": inputs, "options": options, "summary": summary, "launches": []}
            self.runs.write_meta(run_id, meta)
        except Exception:
            self.runs.discard_new(run_id)
            raise
        self.runs.launch(run_id)
        return {"id": run_id, "kind": "predict", "state": self.runs.state(run_id)}

    def run_inputs(self, run_id: str) -> dict:
        """What a GUI run was started from, as a form draft ('Use inputs'): chains / target / ligands with their file contents,
        and the options it resolved to. Runs made by the command line have no staged inputs."""
        meta = self.runs.read_meta(run_id)
        if not meta:
            raise ApiError(400, "this run was not started from this interface: its inputs are not stored here")
        rd = self.runs.run_dir(run_id)

        def read(rel):
            p = rd / str(rel or "")
            return p.read_text(encoding="utf-8", errors="replace") if rel and is_within(p, rd) and p.is_file() and not p.is_symlink() else None

        inputs, opts = meta.get("inputs") or {}, meta.get("options") or {}
        if meta.get("kind") == "predict":
            sm = meta.get("summary") or {}
            msas = inputs.get("msa") or {}
            chains = []
            for c in sm.get("chains") or []:
                d = {k: c.get(k) for k in ("type", "id", "sequence", "cyclic", "smiles", "ccd", "copies", "modifications")}
                m = msas.get(c.get("id"))
                if m and read(m.get("file")) is not None:
                    d["msa"] = {"filename": m.get("name") or Path(m["file"]).name, "content": read(m["file"])}
                elif c.get("msa") == "server":  # fetched online by the job: the draft asks for it again
                    d["msa"] = {"mode": "server"}
                chains.append(d)
            tpl = [{"filename": t.get("from") or n, "content": read(t.get("file")), "chain_id": t.get("chain_id")}
                   for n, t in (inputs.get("templates") or {}).items() if read(t.get("file")) is not None]
            out = {"kind": "predict", "name": meta.get("name"), "chains": chains, "binder": sm.get("binder"), "options": opts,
                   "constraints": sm.get("constraints") or [], "templates": tpl}
            if inputs.get("mode") == "yaml" or "files" in inputs:  # an advanced YAML input (also with no companion files): the YAML as run + companions
                out["yaml"] = {"filename": (inputs.get("yaml") or {}).get("name") or "job.yaml", "content": read((inputs.get("yaml") or {}).get("file")),
                               "files": [{"filename": n, "content": read(f)} for n, f in (inputs.get("files") or {}).items() if read(f) is not None]}
            return out
        t, m, lg = inputs.get("target") or {}, inputs.get("msa") or {}, inputs.get("ligands") or {}
        return {"kind": "screen", "name": meta.get("name"), "options": opts,
                "target": {"saved": t.get("saved"), "filename": t.get("name"), "content": read(t.get("file")),
                           "bundle": bool(t.get("file")) and (rd / str(t["file"])).is_file()},
                "msa": {"mode": m.get("mode") or "default", "filename": m.get("name"), "content": read(m.get("file"))},
                "ligands": {"filename": lg.get("name"), "content": read(lg.get("file"))}}

    def start_existing(self, run_id: str, retry_errors: bool = False) -> dict:
        self.runs.launch(run_id, retry_errors=retry_errors)
        return {"id": run_id, "state": self.runs.state(run_id)}

    def stop(self, run_id: str, force: bool = False) -> dict:
        self.runs.stop(run_id, force=force)
        return {"id": run_id, "state": self.runs.state(run_id)}

    def depict(self, smiles: str, dark: bool = False, w: int = 320, h: int = 240) -> str:
        return depict_svg(smiles, w, h, dark)

    def depict_ligand(self, run_id: str, record_id: str, dark: bool = False, w: int = 320, h: int = 240) -> str:
        row = self.runs.row_by_record(run_id, record_id)
        return depict_svg(row.get("smiles_used") or row.get("smiles") or "", w, h, dark)

    def depict_vs(self, run_id: str, record_id: str, ref_record: str, dark: bool = False, w: int = 320, h: int = 240):
        """2D of a ligand in the reference compound's layout, with what differs from it highlighted: (svg, info)."""
        from boltz_mlx.gui.leadopt import depict_vs

        row, ref = self.runs.row_by_record(run_id, record_id), self.runs.row_by_record(run_id, ref_record)
        return depict_vs(row.get("smiles_used") or row.get("smiles") or "", ref.get("smiles_used") or ref.get("smiles") or "", w, h, dark)

    def aligned_pose(self, run_id: str, record_id: str, ref_record: str):
        """A ligand's pose superposed (protein C-alpha) onto another ligand's pose of the same run: (text, format, rmsd, n)."""
        from boltz_mlx.gui.leadopt import aligned_pose

        return aligned_pose(self.runs.pose_file(run_id, record_id), self.runs.pose_file(run_id, ref_record))

    @staticmethod
    def json(obj) -> bytes:
        return json.dumps(_finite(obj), separators=(",", ":"), default=str, allow_nan=False).encode()


def _finite(o):
    """NaN / Infinity are not valid JSON (the browser's JSON.parse rejects them): they become null."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite(v) for v in o]
    return o


def _span(lo_s: float, hi_s: float) -> str:
    """'20-25 s', '3-4 min' or '1.2-1.5 h' for a measured total range ('16 s' when both ends round alike)."""
    if hi_s < 120:
        lo, hi, unit = f"{lo_s:.0f}", f"{hi_s:.0f}", "s"
    elif hi_s < 7200:
        lo, hi, unit = f"{lo_s / 60:.0f}", f"{hi_s / 60:.0f}", "min"
    else:
        lo, hi, unit = f"{lo_s / 3600:.1f}", f"{hi_s / 3600:.1f}", "h"
    return f"{lo} {unit}" if lo == hi else f"{lo}-{hi} {unit}"
