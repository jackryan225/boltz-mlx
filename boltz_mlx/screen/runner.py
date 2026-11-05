"""The screen loop: chunks of ligands, each in TWO PHASES so only one checkpoint is resident:
  1. structure phase, conf model, every ligand of the chunk: features (per-record featurizer seed), noise
     record_noise(seed, record, "structure", 1), PredictArgs(recycling, steps, 1); writes the pose + pre_affinity npz;
  2. affinity phase, aff model: affinity features from this run's pose (crop), noise record_noise(seed, record, "affinity", 5),
     PredictArgs(5 recycles, affinity_steps, 5 samples), mw_correction off; value_mw_corrected computed from the ensemble
     value with upstream's constants.
Same flow, seeds and outputs as tools/validation/run_l3.py ligand batches, so a screen row and an L3 row of the same
(record id, seed, preset) are computed identically.

SIGTERM stops the run cleanly at the next ligand boundary (rows so far are saved; rerun the same command to resume); a
second SIGTERM stops immediately (as Ctrl-C). The GUI (`boltz-mlx gui`) uses this for its Stop button.

Per-record seeds (noise, featurizer) and the fixed RDKit seed of every SMILES ligand's reference conformer
(seeding.CONFORMER_SEED, installed by the engine) make results independent of input order, chunking and resume: a
ligand's row equals its solo-screen row. Rows are appended (fsynced) to
results.jsonl as each ligand finishes; results.csv is rebuilt (ranked) after every chunk. A finished structure pass is
reused after a kill (sidecar screen_struct.json with the run's config key).
"""

from __future__ import annotations

import gc
import json
import math
import shutil
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from boltz_mlx.screen import results as R
from boltz_mlx.screen.engine import LigandFailure, config_key
from boltz_mlx.screen.inputs import Ligand, Target
from boltz_mlx.screen.presets import (EST_LOAD_S, ESTIMATE_CHIP, Preset, affinity_estimate, est_overhead, est_point,
                                     estimate_measured_here, struct_estimate)

AFF_KEYS = ("affinity_pred_value", "affinity_probability_binary", "affinity_pred_value1", "affinity_probability_binary1",
            "affinity_pred_value2", "affinity_probability_binary2")


@dataclass
class ScreenConfig:
    out: Path
    preset: Preset
    steps: int
    recycling: int
    affinity_steps: int
    affinity_recycling: int = 5
    affinity_samples: int = 5
    seed: int = 0
    rank_by: str = "prob"
    chunk: int = 50
    save_structures: int = 10          # top-k poses copied to out/structures; -1 = all; 0 = none
    resume: bool = True
    retry_errors: bool = False
    prefetch: bool = True               # featurize the next ligand on a CPU thread (both phases; exact: seeded per record)
    keep_work: bool = False
    max_minutes: float | None = None
    max_msa_seqs: int = 8192
    output_format: str = "mmcif"
    device: str = "gpu"
    env: dict = field(default_factory=dict)   # effective numerics env (recorded in rows)
    hw: dict = field(default_factory=dict)    # kernels.device_tag(): chip, architecture, nax (recorded in rows)
    # Fast's tiny-ligand rule (boltz_mlx/tiny_ligand.py): a ligand with fewer than tiny_heavy heavy atoms runs tiny_steps
    # structure steps instead of `steps`; None = rule off (Full, or the user set --steps). Per ligand, not per run.
    tiny_steps: int | None = None
    tiny_heavy: int = 10

    def steps_for(self, lig) -> int:
        """Structure steps of this ligand's structure pass."""
        if self.tiny_steps is not None and lig.n_heavy is not None and lig.n_heavy < self.tiny_heavy:
            return max(self.steps, self.tiny_steps)
        return self.steps

    def steps_reason(self, lig) -> str | None:
        if self.steps_for(lig) == self.steps:
            return None
        return f"ligand {lig.ligand_id} has {lig.n_heavy} heavy atoms (< {self.tiny_heavy})"

    @property
    def mode(self) -> str:
        p = self.preset
        custom = [f"{k}={v}" for k, v, d in (("steps", self.steps, p.steps), ("recycling", self.recycling, p.recycling),
                                              ("affinity_steps", self.affinity_steps, p.affinity_steps)) if v != d]
        return p.name + (f"[{','.join(custom)}]" if custom else "")

    def result_config(self, target: Target) -> dict:
        """Everything that changes predictions (NOT chunk / rank_by / save_structures / prefetch: those must not)."""
        return {"target": target.key(), "ligand_chain": target.ligand_chain, "preset": self.preset.name, "env": self.preset.env,
                "steps": self.steps, "recycling": self.recycling, "affinity_steps": self.affinity_steps,
                "affinity_recycling": self.affinity_recycling, "affinity_samples": self.affinity_samples, "seed": self.seed,
                "max_msa_seqs": self.max_msa_seqs,
                # only when the rule is on, so a config without it (Full, explicit --steps) keeps its old key
                **({"tiny_ligand_steps": {"heavy_atoms_lt": self.tiny_heavy, "steps": self.tiny_steps}} if self.tiny_steps else {})}


def mw_correct(value: float, mw: float) -> float:
    """Upstream Boltz2.forward affinity_mw_correction on the ensemble value (as run_l3.mw_correct)."""
    from boltz_mlx.modules.affinity import MW_BIAS, MW_MODEL_COEF, MW_MW_COEF

    return float(MW_MODEL_COEF * float(value) + MW_MW_COEF * float(mw) ** 0.3 + MW_BIAS)


def _scalar(out: dict, key: str):
    import numpy as np

    if key not in out or out[key] is None:
        return None
    return float(np.array(out[key]).astype(np.float64).reshape(-1)[0])


def _finite(*vals) -> bool:
    return all(v is None or math.isfinite(v) for v in vals)


def _counts(gc_):
    import numpy as np

    if gc_ is None:
        return None
    if isinstance(gc_, dict):
        return {str(k): int(np.array(v).sum()) for k, v in gc_.items()}
    return int(np.array(gc_).sum())


def est_tokens(target: Target, lig: Ligand) -> int:
    return int(target.n_residues + (lig.n_atoms or 30))


def estimate_seconds(cfg: ScreenConfig, target: Target, lig: Ligand) -> float:
    chip = cfg.hw.get("chip") if cfg.hw and cfg.hw.get("path") == "nax" else None   # own rows: NAX path only
    steps = cfg.steps_for(lig)
    return (struct_estimate(cfg.preset, est_tokens(target, lig), steps, chip, cfg.affinity_steps)
            + affinity_estimate(cfg.preset, cfg.affinity_steps, chip, steps)
            + est_overhead(cfg.preset, chip, steps, cfg.affinity_steps))


class ScreenRunner:
    def __init__(self, cfg: ScreenConfig, target: Target, ligands: list[Ligand], engine, log=print, clock=time.time):
        self.cfg = cfg
        self.target = target
        self.ligands = ligands
        self.engine = engine
        self.log = log
        self.clock = clock
        self.out = Path(cfg.out)
        self.jsonl = self.out / "results.jsonl"
        self.csv = self.out / "results.csv"
        self.ckey = config_key(cfg.result_config(target))
        self.t0 = clock()
        self.n_rows = self.n_ok = self.n_err = 0
        self.load_s: dict[str, float] = {}
        self.stopped_early = False
        self.stop_requested = 0          # SIGTERM count (the handler only sets this; the log line comes from out_of_time)
        self._stop_logged = False
        self._mx = None

    # --------------------------------------------------------------------------------------------- bookkeeping
    @property
    def mx(self):
        if self._mx is None:
            import mlx.core as mx

            self._mx = mx
        return self._mx

    def check_config(self) -> None:
        """Refuse to mix results computed under a different configuration in one out dir (resume must be exact)."""
        p = self.out / "screen_config.json"
        cur = {"config_key": self.ckey, **self.cfg.result_config(self.target)}
        if p.exists():
            old = json.loads(p.read_text())
            if old.get("config_key") != self.ckey:
                diff = sorted(k for k in set(old) | set(cur) if old.get(k) != cur.get(k) and k != "config_key")
                raise SystemExit(f"screen: {self.out} holds results from a different configuration (differs in: {diff}). "
                                 f"Use a new --out directory.")
        self.out.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(cur, indent=1, default=str))

    def plan(self) -> dict:
        rows = R.read_rows(self.jsonl)
        latest = R.latest_rows(rows)
        done = R.done_ids(rows, self.cfg.retry_errors) if self.cfg.resume else set()
        conflicts, todo, invalid_new = [], [], []
        for lig in self.ligands:
            prev = latest.get(lig.ligand_id)
            if prev is not None and prev.get("smiles") != lig.smiles:
                if prev.get("status") != "invalid":
                    conflicts.append(lig)  # same id, different SMILES: never overwrite a prediction silently
                    continue
                done.discard(lig.ligand_id)  # a corrected SMILES for a previously invalid ligand: screen it (latest row wins)
            if lig.ligand_id in done:
                continue
            (invalid_new if lig.status == "invalid" else todo).append(lig)
        return {"done": done, "todo": todo, "invalid": invalid_new, "conflicts": conflicts, "rows": rows}

    def base_row(self, lig: Ligand) -> dict:
        c = self.cfg
        return {
            "ligand_id": lig.ligand_id, "record_id": lig.record_id, "index": lig.index, "smiles": lig.smiles,
            "smiles_used": lig.smiles_used, "n_atoms": lig.n_atoms, "n_heavy": lig.n_heavy, "mw": lig.mw,
            "mode": c.mode, "preset": c.preset.name, "preset_validated": c.preset.validated, "env": c.env,
            "steps": c.steps_for(lig), "steps_reason": c.steps_reason(lig), "recycling": c.recycling,
            "affinity_steps": c.affinity_steps, "affinity_recycling": c.affinity_recycling, "affinity_samples": c.affinity_samples, "seed": c.seed,
            "msa_source": self.target.msa_source, "target": self.target.source, "config_key": self.ckey,
            "warnings": list(lig.warnings), "device": c.device,
            "chip": c.hw.get("chip"), "gpu_arch": c.hw.get("architecture"), "kernel_path": c.hw.get("path"),
            "nax": c.hw.get("nax"),
        }

    def emit(self, row: dict) -> None:
        now = self.clock()
        row["t_done"] = round(now, 3)
        row["elapsed_s"] = round(now - self.t0, 3)
        R.append_row(self.jsonl, row)
        self.n_rows += 1
        if row.get("status") == "ok":
            self.n_ok += 1
        elif row.get("status") == "error":
            self.n_err += 1
        bits = [f"[{row['status']}] {row['ligand_id']}"]
        for k in ("affinity_pred_value", "affinity_probability_binary", "iptm"):
            if isinstance(row.get(k), float):
                bits.append(f"{k}={row[k]:.3f}")
        if row.get("seconds") is not None:
            bits.append(f"{row['seconds']:.1f}s")
        if row.get("error"):
            bits.append(f"error: {row['error']}")
        self.log(" ".join(bits))

    def error_row(self, lig: Ligand, phase: str, ex: BaseException, partial: dict | None = None) -> dict:
        row = self.base_row(lig)
        row.update(partial or {})
        msg = str(ex) if isinstance(ex, LigandFailure) else f"{type(ex).__name__}: {ex}"
        row.update(status="error", error_phase=phase, error=msg[:2000])
        return row

    def request_stop(self) -> None:
        """Ask for a clean stop at the next ligand boundary (SIGTERM handler; also callable from tests). The second request
        raises KeyboardInterrupt (stop now; rows so far are saved)."""
        self.stop_requested += 1
        if self.stop_requested > 1:
            raise KeyboardInterrupt

    def _install_term_handler(self):
        """SIGTERM -> request_stop() while run() is active (main thread only). Returns (installed, previous handler)."""
        if threading.current_thread() is not threading.main_thread():
            return False, None
        try:
            return True, signal.signal(signal.SIGTERM, lambda signum, frame: self.request_stop())
        except (ValueError, OSError):  # not supported here: keep the default action
            return False, None

    def out_of_time(self) -> bool:
        if self.stop_requested:
            if not self._stop_logged:
                self._stop_logged = True
                self.log("[stop] SIGTERM received: stopping at a ligand boundary; rows so far are saved, rerun the same command "
                         "to resume (a second SIGTERM stops immediately)")
            return True
        if self.cfg.max_minutes and (self.clock() - self.t0) / 60.0 >= self.cfg.max_minutes:
            if not self.stopped_early:
                self.log(f"[budget] --max-minutes {self.cfg.max_minutes} reached: stopping at a ligand boundary; rerun to resume")
            self.stopped_early = True
            return True
        return False

    # --------------------------------------------------------------------------------------------- passes
    def _load(self, name: str):
        t = time.perf_counter()
        m = self.engine.load_model(name)
        dt = time.perf_counter() - t
        self.load_s[name] = round(self.load_s.get(name, 0.0) + dt, 3)
        self.log(f"[load] {name} model {dt:.1f}s")
        return m

    def _unload(self, model) -> None:
        del model
        gc.collect()
        try:
            self.mx.clear_cache()
        except Exception:  # noqa: BLE001
            pass

    def _prep_features(self, lig: Ligand):
        t = time.perf_counter()
        self.engine.prepare(lig)
        batch = self.engine.featurize(lig, affinity=False)
        return batch, time.perf_counter() - t

    def struct_sidecar(self, lig: Ligand) -> Path:
        return self.engine.pred_dir(lig) / "screen_struct.json"

    def reusable_structure(self, lig: Ligand) -> dict | None:
        if not self.cfg.resume:
            return None
        p = self.struct_sidecar(lig)
        try:
            info = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        if info.get("config_key") != self.ckey:
            return None
        if not (self.engine.pose_path(lig).exists() and self.engine.pre_affinity_path(lig).exists()):
            return None
        return info

    def structure_pass(self, model, lig: Ligand, batch, prep_s: float) -> dict:
        import numpy as np

        from boltz_mlx.model import PredictArgs
        from boltz_mlx.sampler import record_noise

        mx = self.mx
        feats = self.engine.model_inputs(batch)
        noise = record_noise(self.cfg.seed, lig.record_id, "structure", 1)
        t = time.perf_counter()
        out = model(feats, PredictArgs(recycling_steps=self.cfg.recycling, sampling_steps=self.cfg.steps_for(lig), diffusion_samples=1), noise)
        mx.eval(out["sample_atom_coords"])
        model_s = time.perf_counter() - t
        lig_plddt = None
        asym = self.engine.ligand_asym(batch, self.target.ligand_chain)
        if "plddt" in out and asym is not None:
            pl = np.array(out["plddt"]).reshape(-1)
            a = np.array(feats["asym_id"]).reshape(-1)[: pl.shape[0]]
            m = np.array(feats["token_pad_mask"]).reshape(-1)[: pl.shape[0]] > 0
            sel = (a == asym) & m
            if sel.any():
                lig_plddt = float(pl[sel].mean())
        coords = np.array(out["sample_atom_coords"])
        t = time.perf_counter()
        self.engine.write_structure(out, batch)
        write_s = time.perf_counter() - t
        info = {
            "config_key": self.ckey, "record_id": lig.record_id,
            "n_tokens": int(np.array(feats["token_pad_mask"]).sum()), "n_atoms_complex": int(np.array(feats["atom_pad_mask"]).sum()),
            "ptm": _scalar(out, "ptm"), "iptm": _scalar(out, "iptm"), "ligand_iptm": _scalar(out, "ligand_iptm"),
            "complex_plddt": _scalar(out, "complex_plddt"), "ligand_plddt": lig_plddt,
            "structure_nonfinite": bool(not np.isfinite(coords).all()),
            "structure_guard_counts": _counts(out.get("guard_counts")),
            "seconds_prep": round(prep_s, 3), "seconds_structure": round(model_s, 3), "seconds_write": round(write_s, 3),
            "pose": str(self.engine.pose_path(lig)),
        }
        sc = self.struct_sidecar(lig)
        sc.parent.mkdir(parents=True, exist_ok=True)
        sc.write_text(json.dumps(info, indent=1))
        del out, feats
        return info

    def _aff_features(self, lig: Ligand):
        t = time.perf_counter()
        batch = self.engine.featurize(lig, affinity=True)
        return batch, time.perf_counter() - t

    def affinity_pass(self, model, lig: Ligand, pre: tuple | None = None) -> dict:
        """pre: (batch, seconds) from _aff_features (prefetched on the worker thread), else featurized here."""
        import numpy as np

        from boltz_mlx.model import PredictArgs
        from boltz_mlx.sampler import record_noise

        mx = self.mx
        batch, feat_s = pre if pre is not None else self._aff_features(lig)
        t = time.perf_counter()
        feats = self.engine.model_inputs(batch)
        feat_s += time.perf_counter() - t
        mw = self.engine.affinity_mw(batch)
        noise = record_noise(self.cfg.seed, lig.record_id, "affinity", self.cfg.affinity_samples)
        t = time.perf_counter()
        out = model(feats, PredictArgs(self.cfg.affinity_recycling, self.cfg.affinity_steps, self.cfg.affinity_samples), noise,
                    mw_correction=False)
        mx.eval(out["affinity_pred_value"], out["affinity_probability_binary"])
        model_s = time.perf_counter() - t
        vals = {k: _scalar(out, k) for k in AFF_KEYS}
        iptm = np.array(out["iptm"]).astype(np.float64).reshape(-1) if "iptm" in out else np.zeros(0)
        t = time.perf_counter()
        self.engine.write_affinity(out, batch)
        write_s = time.perf_counter() - t
        info = {
            **vals,
            "value_mw_corrected": None if (mw is None or vals["affinity_pred_value"] is None) else mw_correct(vals["affinity_pred_value"], mw),
            "affinity_mw": mw,
            "affinity_crop_tokens": int(np.array(feats["token_pad_mask"]).sum()),
            "affinity_best_sample": int(np.argmax(iptm)) if iptm.size else None,
            "affinity_iptm_samples": [float(x) for x in iptm],
            "affinity_guard_counts": _counts(out.get("guard_counts")),
            "seconds_affinity_features": round(feat_s, 3), "seconds_affinity": round(model_s, 3),
            "seconds_affinity_write": round(write_s, 3),
        }
        del out, feats
        return info

    # --------------------------------------------------------------------------------------------- driver
    def ligand_row(self, lig: Ligand, s: dict, a: dict) -> dict:
        row = self.base_row(lig)
        row.update({k: v for k, v in s.items() if k not in ("config_key", "record_id")})
        row.update(a)
        gcs, gca = s.get("structure_guard_counts"), a.get("affinity_guard_counts")
        if gcs is not None or gca is not None:
            tot = 0
            for g in (gcs, gca):
                if isinstance(g, dict):
                    tot += int(g.get("trunk_fallbacks", 0))
                elif g is not None:
                    tot += int(g)
            row["fp16_fallbacks"] = tot
        sec = [s.get("seconds_prep"), s.get("seconds_structure"), s.get("seconds_write"), a.get("seconds_affinity_features"),
               a.get("seconds_affinity"), a.get("seconds_affinity_write")]
        row["seconds"] = round(sum(x for x in sec if x is not None), 3)
        if s.get("reused_structure"):
            row["reused_structure"] = True
        keyvals = [row.get(k) for k in ("affinity_pred_value", "affinity_probability_binary", "iptm")]
        if not _finite(*keyvals) or s.get("structure_nonfinite"):
            row.update(status="error", error_phase="affinity", error="non-finite model output")
        else:
            row["status"] = "ok"
        return row

    def run_chunk(self, chunk: list[Ligand]) -> None:
        # ---- phase 1: structure (conf checkpoint)
        sinfo: dict[str, dict] = {}
        need = []
        for lig in chunk:
            info = self.reusable_structure(lig)
            if info is not None:
                sinfo[lig.record_id] = {**info, "reused_structure": True}
            else:
                need.append(lig)
        need.sort(key=lambda l: (est_tokens(self.target, l), l.index))  # same lengths adjacent (kernel/graph reuse)
        if need and not self.out_of_time():
            model = None
            pool = ThreadPoolExecutor(max_workers=1) if self.cfg.prefetch else None
            fut = pool.submit(self._prep_features, need[0]) if pool else None
            try:
                for i, lig in enumerate(need):
                    if self.out_of_time():
                        break
                    try:
                        if pool:
                            cur = fut
                            fut = pool.submit(self._prep_features, need[i + 1]) if i + 1 < len(need) else None
                            batch, prep_s = cur.result()
                        else:
                            batch, prep_s = self._prep_features(lig)
                        if model is None:
                            model = self._load("conf")
                        self.log(f"[start] structure {lig.ligand_id} {i + 1}/{len(need)}")
                        sinfo[lig.record_id] = self.structure_pass(model, lig, batch, prep_s)
                    except (KeyboardInterrupt, SystemExit):
                        raise
                    except Exception as ex:  # noqa: BLE001
                        self.emit(self.error_row(lig, "structure", ex))
            finally:
                if pool:
                    pool.shutdown(wait=True, cancel_futures=True)
                if model is not None:
                    self._unload(model)
                model = None
        # ---- phase 2: affinity (aff checkpoint)
        todo = [l for l in chunk if l.record_id in sinfo]
        todo.sort(key=lambda l: (est_tokens(self.target, l), l.index))
        if not todo:
            return
        model = None
        # prefetch: the next ligand's affinity features (crop around its phase-1 pose) are built on the worker thread
        # while this ligand's affinity pass runs on the GPU
        pool = ThreadPoolExecutor(max_workers=1) if self.cfg.prefetch else None
        fut = pool.submit(self._aff_features, todo[0]) if pool else None
        try:
            for i, lig in enumerate(todo):
                if self.out_of_time():
                    break
                s = sinfo[lig.record_id]
                try:
                    pre = None
                    if pool:
                        cur = fut
                        fut = pool.submit(self._aff_features, todo[i + 1]) if i + 1 < len(todo) else None
                        pre = cur.result()
                    if model is None:
                        model = self._load("aff")
                    self.log(f"[start] affinity {lig.ligand_id} {i + 1}/{len(todo)}")
                    a = self.affinity_pass(model, lig, pre)
                    row = self.ligand_row(lig, s, a)
                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception as ex:  # noqa: BLE001
                    partial = {k: v for k, v in s.items() if k not in ("config_key", "record_id")}
                    row = self.error_row(lig, "affinity", ex, partial)
                self.emit(row)
                if not self.cfg.keep_work and row["status"] == "ok":
                    self.engine.cleanup(lig)
        finally:
            if pool:
                pool.shutdown(wait=True, cancel_futures=True)
            if model is not None:
                self._unload(model)

    def save_structures(self, ranked: list[dict]) -> None:
        k = self.cfg.save_structures
        if not k:
            return
        d = self.out / "structures"
        d.mkdir(parents=True, exist_ok=True)
        for p in d.iterdir():
            if p.is_file():
                p.unlink()
        for r in ranked:
            if r.get("rank") is None or (k > 0 and r["rank"] > k):
                continue
            src = Path(r.get("pose") or "")
            if src.exists():
                shutil.copyfile(src, d / f"{r['rank']:04d}_{r['record_id']}{src.suffix}")

    def write_outputs(self) -> list[dict]:
        ranked = R.write_csv(self.csv, R.read_rows(self.jsonl), self.cfg.rank_by)
        self.save_structures(ranked)
        return ranked

    def run(self) -> dict:
        installed, prev = self._install_term_handler()
        try:
            return self._run()
        finally:
            if installed:
                try:
                    signal.signal(signal.SIGTERM, prev if prev is not None else signal.SIG_DFL)
                except (ValueError, OSError):
                    pass

    def _run(self) -> dict:
        self.check_config()
        p = self.plan()
        for lig in p["conflicts"]:
            self.log(f"[skip] {lig.ligand_id}: already in {self.jsonl.name} with a different SMILES; rename the id to screen it")
        for lig in p["invalid"]:
            row = self.base_row(lig)
            row.update(status="invalid", error=lig.error, error_phase="input")
            self.emit(row)
        todo = p["todo"]
        n_chunks = math.ceil(len(todo) / self.cfg.chunk) if todo else 0
        self.log(f"[plan] {len(self.ligands)} ligands: {len(p['done'])} already done, {len(p['invalid'])} invalid, "
                 f"{len(p['conflicts'])} id conflicts, {len(todo)} to run in {n_chunks} chunk(s) of <= {self.cfg.chunk}; "
                 f"mode {self.cfg.mode}{'' if self.cfg.preset.validated else ' (NOT a gated preset)'}")
        interrupted = False
        try:
            for ci in range(n_chunks):
                if self.out_of_time():
                    break
                chunk = todo[ci * self.cfg.chunk:(ci + 1) * self.cfg.chunk]
                self.log(f"[chunk {ci + 1}/{n_chunks}] {len(chunk)} ligands")
                self.run_chunk(chunk)
                self.write_outputs()
        except KeyboardInterrupt:
            interrupted = True
            self.log("[interrupted] rows so far are saved; rerun the same command to resume")
        ranked = self.write_outputs()
        latest = R.latest_rows(R.read_rows(self.jsonl))
        remaining = sum(1 for l in self.ligands if l.ligand_id not in latest)
        wall = self.clock() - self.t0
        status = {
            "complete": remaining == 0 and not interrupted, "interrupted": interrupted, "stopped_by_budget": self.stopped_early,
            "stopped_by_signal": self.stop_requested > 0,
            "ligands": len(self.ligands), "remaining": remaining, "rows_written_this_run": self.n_rows,
            "ok_this_run": self.n_ok, "errors_this_run": self.n_err,
            "ok_total": sum(1 for r in latest.values() if r.get("status") == "ok"),
            "invalid_total": sum(1 for r in latest.values() if r.get("status") == "invalid"),
            "error_total": sum(1 for r in latest.values() if r.get("status") == "error"),
            "wall_s": round(wall, 2), "s_per_ligand_this_run": round(wall / self.n_ok, 2) if self.n_ok else None,
            "load_s": self.load_s, "mode": self.cfg.mode, "config_key": self.ckey, "results_csv": str(self.csv),
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        (self.out / "screen_status.json").write_text(json.dumps(status, indent=1))
        top = [r for r in ranked if r.get("rank") is not None][:5]
        if top:
            key = R.RANK_BY[self.cfg.rank_by][0]
            self.log("[top] " + ", ".join(f"{r['rank']}. {r['ligand_id']} ({key}={r[key]:.3f})" for r in top))
        self.log("[status] " + json.dumps(status))
        if interrupted:
            raise KeyboardInterrupt
        return status


# ============================================================================================================ plan / estimate


def _dur(s: float) -> str:
    return f"{s / 3600:.1f} h" if s >= 3600 else f"{s / 60:.0f} min"


def _sec(s: float) -> str:
    return f"{s:.1f}" if s < 20 else f"{s:.0f}"


def plan_lines(cfg: ScreenConfig, target: Target, ligands: list[Ligand], done: set[str]) -> list[str]:
    import statistics

    valid = [l for l in ligands if l.status != "invalid"]
    todo = [l for l in valid if l.ligand_id not in done]
    invalid = [l for l in ligands if l.status == "invalid"]
    warned = [l for l in valid if l.warnings]
    p = cfg.preset
    lines = [
        f"target: {target.source}  ({target.n_residues} polymer residues; ligand chain {target.ligand_chain}; MSA {target.msa_source})",
        f"preset: {cfg.mode}  - {p.note}" + ("" if p.validated else "  [NOT a gated preset]"),
        f"  structure pass: {cfg.recycling} recycles, {cfg.steps} steps, 1 sample; affinity pass: {cfg.affinity_recycling} recycles, "
        f"{cfg.affinity_steps} steps, {cfg.affinity_samples} samples; seed {cfg.seed}; env {cfg.env or '{}'}",
    ]
    if cfg.tiny_steps:
        n_tiny = sum(1 for l in valid if cfg.steps_for(l) != cfg.steps)
        lines.append(f"  tiny ligands: {n_tiny} of {len(valid)} valid ligands have < {cfg.tiny_heavy} heavy atoms and run "
                     f"{max(cfg.steps, cfg.tiny_steps)} structure steps (the rest {cfg.steps}); --steps N turns this off")
    lines += [
        f"ligands: {len(ligands)} read, {len(valid)} valid, {len(invalid)} invalid, {len(warned)} with warnings, "
        f"{len(valid) - len(todo)} already done in {cfg.out}, {len(todo)} to run",
    ]
    for l in invalid[:20]:
        lines.append(f"  invalid: {l.ligand_id}: {l.error}")
    if len(invalid) > 20:
        lines.append(f"  ... {len(invalid) - 20} more invalid")
    for l in warned[:10]:
        lines.append(f"  warning: {l.ligand_id}: {'; '.join(l.warnings)}")
    for w in target.warnings:
        lines.append(f"  target warning: {w}")
    if todo:
        per = [estimate_seconds(cfg, target, l) for l in todo]
        n_chunks = math.ceil(len(todo) / cfg.chunk)
        total = sum(per) + n_chunks * 2 * EST_LOAD_S
        toks = [est_tokens(target, l) for l in todo]
        chip = cfg.hw.get("chip", "unknown")
        if cfg.hw and not estimate_measured_here(p, chip):
            src = (f"not a measurement; not measured on this chip ({chip}, kernel path {cfg.hw.get('path', '?')}, "
                   f"preset {p.name}): "
                   f"{ESTIMATE_CHIP} numbers with NAX kernels, expect different times here - use --dry-run --measure K")
        else:
            pt = est_point(p, chip if cfg.hw.get("path") == "nax" else None, cfg.steps, cfg.affinity_steps)
            src = (f"not a measurement; from {chip if chip != 'unknown' else ESTIMATE_CHIP} "
                   f"{pt.get('source', 'bench rows of the model passes')} + ~{pt['overhead']:.1f} s/ligand overhead; "
                   f"check with --dry-run --measure K")
        lines += [
            f"hardware: {chip} ({cfg.hw.get('architecture', '?')}), kernel path {cfg.hw.get('path', '?')}"
            if cfg.hw else "hardware: not detected",
            f"chunks: {n_chunks} x <= {cfg.chunk} ligands (conf then aff checkpoint per chunk; ranked results.csv after each)",
            f"ESTIMATE ({src}): "
            f"~{_sec(statistics.median(per))} s/ligand (range {_sec(min(per))}-{_sec(max(per))} s over {min(toks)}-{max(toks)} tokens), "
            f"total ~{_dur(total)} ({3600 * len(todo) / total:.0f} ligands/h) incl. {n_chunks * 2} checkpoint loads",
        ]
    return lines
