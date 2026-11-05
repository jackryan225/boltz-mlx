"""Run directories and the screen subprocess.

Layout of `<runs-dir>/<run-id>/` (created by the GUI):
  gui_run.json   what the user chose (input names, options), a summary of the validated inputs, and one record per launch
  inputs/        target.<ext>, msa.<ext>, ligands.<ext> (fixed names: an uploaded file name never decides where it is written)
  out/           the screen's --out directory (results.jsonl, results.csv, structures/, work/, screen_status.json, ...)
  screen.log     stdout + stderr of every launch (appended; each launch records its start offset)

A directory without gui_run.json that holds a screen output (results.jsonl in it or in its out/) is listed as an external
(CLI) run: results and poses are viewable, it cannot be started or resumed from the GUI.

The screen runs as `python -m boltz_mlx.cli screen ...` in its own session; the server never imports the model. Stop = SIGTERM
(the screen finishes the ligand in progress, writes its outputs and exits; resume = launch the same command again).

A run whose gui_run.json says "kind": "predict" is one structure job instead (boltz_mlx/gui/predict.py): inputs/job.yaml, a
`python -m boltz_mlx.cli predict ...` subprocess writing into out/boltz_results_job/, its output in predict.log. Its state comes
from the exit code and the files written (the predict CLI writes no status file); Stop = SIGTERM ends it, Resume reruns it
(upstream skips a finished structure pass, so a stopped affinity job continues with the affinity pass).
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path

from boltz_mlx.gui import presets as GP
from boltz_mlx.gui import progress as P
from boltz_mlx.gui.common import ApiError, is_within, iso, slug, valid_record_id, valid_run_id

import re

POSE_FILE = re.compile(r"\d{4}_(.+)\.(?:cif|pdb)")
STATES_ACTIVE = ("running", "stopping")
STATES_RESUMABLE = ("stopped", "failed", "interrupted", "new")
META = "gui_run.json"
LOG = "screen.log"
PREDICT_LOG = "predict.log"
ROW_CACHE = 8
# BOLTZ_MLX_* variables a predict job keeps from the server's environment (locations / memory, not numerics: those come
# from the job's preset flags), as boltz_mlx.screen.main.ENV_PASSTHROUGH
PREDICT_ENV_KEEP = {"BOLTZ_MLX_CACHE_GB", "BOLTZ_MLX_MEM_LIMIT_GB", "BOLTZ_MLX_WEIGHTS", "BOLTZ_MLX_CACHE", "BOLTZ_MLX_NAX",
                    "BOLTZ_MLX_KERNEL_PATH"}


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def pid_command(pid: int) -> str | None:
    try:
        r = subprocess.run(["ps", "-p", str(int(pid)), "-o", "command="], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    out = r.stdout.strip()
    return out or None


class RunStore:
    def __init__(self, runs_dir: str | os.PathLike, screen_cmd: list[str] | None = None, extra_args: list[str] | None = None,
                 clock=time.time, predict_cmd: list[str] | None = None, predict_args: list[str] | None = None,
                 predict_env: dict | None = None):
        self.root = Path(runs_dir).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.screen_cmd = list(screen_cmd) if screen_cmd else [sys.executable, "-m", "boltz_mlx.cli", "screen"]
        self.extra_args = list(extra_args or [])
        self.predict_cmd = list(predict_cmd) if predict_cmd else [sys.executable, "-m", "boltz_mlx.cli", "predict"]
        self.predict_args = list(predict_args or [])
        self.predict_env = dict(predict_env or {})
        self.clock = clock
        self.lock = threading.RLock()
        self.procs: dict[str, subprocess.Popen] = {}
        self._rows: "OrderedDict[str, P.RowStore]" = OrderedDict()
        self._scans: dict[str, P.StructScan] = {}
        self._pose_cache: "OrderedDict[tuple, dict]" = OrderedDict()
        self._cands: set[str] | None = None   # runs with a launch in flight (see active_run_id)
        self._scanned = 0.0

    # ------------------------------------------------------------------------------------------------ paths
    def run_dir(self, run_id: str) -> Path:
        if not valid_run_id(run_id):
            raise ApiError(404, "no such run")
        p = self.root / run_id
        if p.is_symlink() or not p.is_dir() or not is_within(p, self.root):
            raise ApiError(404, "no such run")
        return p

    def read_meta(self, run_id: str) -> dict | None:
        p = self.run_dir(run_id) / META
        try:
            m = json.loads(p.read_text())
        except (OSError, ValueError):
            return None
        return m if isinstance(m, dict) else None

    def write_meta(self, run_id: str, meta: dict) -> None:
        with self.lock:
            _atomic_write(self.run_dir(run_id) / META, json.dumps(meta, indent=1, default=str))

    def out_dir(self, run_id: str) -> Path:
        rd = self.run_dir(run_id)
        if (rd / META).exists() or (rd / "out").is_dir():
            return rd / "out"
        return rd

    def is_external(self, run_id: str) -> bool:
        return not (self.run_dir(run_id) / META).exists()

    def kind(self, run_id: str, meta: dict | None = None) -> str:
        meta = meta if meta is not None else self.read_meta(run_id)
        return "predict" if (meta or {}).get("kind") == "predict" else "screen"

    def log_path(self, run_id: str, meta: dict | None = None) -> Path:
        return self.run_dir(run_id) / (PREDICT_LOG if self.kind(run_id, meta) == "predict" else LOG)

    def _is_run_dir(self, p: Path) -> bool:
        if not p.is_dir() or p.is_symlink() or p.name.startswith(".") or not valid_run_id(p.name):
            return False
        return (p / META).exists() or (p / "results.jsonl").exists() or (p / "out" / "results.jsonl").exists()

    def allocate(self, name: str) -> tuple[str, Path]:
        """A fresh run directory (inputs/ and out/ inside): <YYYYmmdd-HHMMSS>-<slug of the name>."""
        base = time.strftime("%Y%m%d-%H%M%S", time.localtime(self.clock())) + "-" + slug(name)
        with self.lock:
            run_id, k = base, 1
            while (self.root / run_id).exists():
                k += 1
                run_id = f"{base}-{k}"
            rd = self.root / run_id
            (rd / "inputs").mkdir(parents=True)
            (rd / "out").mkdir()
        return run_id, rd

    def discard_new(self, run_id: str) -> None:
        """Remove a run directory created moments ago by allocate() whose creation then failed (never launched)."""
        import shutil

        with self.lock:
            rd = self.run_dir(run_id)
            if (rd / META).exists() and (self.read_meta(run_id) or {}).get("launches"):
                return
            shutil.rmtree(rd, ignore_errors=True)

    # ------------------------------------------------------------------------------------------ processes
    def _alive(self, run_id: str, launch: dict) -> bool:
        pid = launch.get("pid")
        if not pid:
            return False
        p = self.procs.get(run_id)
        if p is not None and p.pid == pid:
            rc = p.poll()
            if rc is None:
                return True
            self._mark_ended(run_id, launch, rc)
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return False  # someone else's process (recycled pid)
        cmd = pid_command(pid)
        # a recycled pid belongs to some other program: the screen's command line always names its run directory
        return bool(cmd) and str(self.root / run_id) in cmd and "defunct" not in cmd

    def _mark_ended(self, run_id: str, launch: dict, rc: int | None) -> None:
        with self.lock:
            if launch.get("ended_at") is not None:
                return
            launch["ended_at"] = self.clock()
            launch["exit_code"] = rc
            meta = self.read_meta(run_id)
            if meta and meta.get("launches"):
                for l in meta["launches"]:
                    if l.get("pid") == launch.get("pid") and l.get("started_at") == launch.get("started_at"):
                        l["ended_at"], l["exit_code"] = launch["ended_at"], rc
                        if launch.get("stop_requested_at"):
                            l["stop_requested_at"] = launch["stop_requested_at"]
                self.write_meta(run_id, meta)

    def _status_file(self, run_id: str, since: float | None = None) -> dict | None:
        p = self.out_dir(run_id) / "screen_status.json"
        try:
            if since is not None and p.stat().st_mtime < since - 2:
                return None  # left by an earlier launch
            return json.loads(p.read_text())
        except (OSError, ValueError):
            return None

    def state(self, run_id: str) -> str:
        """new | running | stopping | complete | stopped | failed | interrupted (| external states: complete | stopped)."""
        meta = self.read_meta(run_id)
        if meta is None:
            st = self._status_file(run_id)
            return "complete" if (st and st.get("complete")) else "stopped"
        launches = meta.get("launches") or []
        if not launches:
            return "new"
        last = launches[-1]
        if last.get("ended_at") is None:
            if self._alive(run_id, last):
                return "stopping" if last.get("stop_requested_at") else "running"
            if last.get("ended_at") is None:
                self._mark_ended(run_id, last, None)
        if meta.get("kind") == "predict":
            return self._predict_state(run_id, meta, last)
        st = self._status_file(run_id, last.get("started_at"))
        if st is not None:
            return "complete" if st.get("complete") else "stopped"
        rc = last.get("exit_code")
        return "failed" if (rc is not None and rc > 0 and rc != 130) else "interrupted"

    def _predict_state(self, run_id: str, meta: dict, last: dict) -> str:
        """An ended predict job: complete (its outputs are all there) | stopped (Stop asked) | failed | interrupted."""
        from boltz_mlx.gui import predict as PR

        out = self.run_dir(run_id) / "out"
        affinity = bool((meta.get("summary") or {}).get("affinity"))
        ph = PR.phase(out, affinity, last.get("started_at"))
        rc = last.get("exit_code")
        if ph == "done" and (rc in (0, None)):
            if not affinity or (PR.pred_dir(out) / f"affinity_{PR.RECORD}.json").exists():
                return "complete"
        if last.get("stop_requested_at"):
            return "stopped"
        return "failed" if (rc is not None and rc > 0 and rc not in (130, -15)) else "interrupted"

    def active_run_id(self, rescan: bool = False) -> str | None:
        """The run whose screen process is alive (one at a time: the GPU is shared). The page polls this every few seconds, so
        only runs known to have a launch in flight are checked; the whole folder is rescanned at start-up, before a launch,
        and at most once a minute (a screen started by another GUI process on the same folder shows up then)."""
        now = time.time()
        if rescan or self._cands is None or now - self._scanned > 60:
            self._cands = {c.name for c in self.root.iterdir()
                           if not c.name.startswith(".") and valid_run_id(c.name) and (c / META).exists()
                           and self._unfinished(c.name)}
            self._scanned = now
        for rid in sorted(self._cands):
            try:
                if self.state(rid) in STATES_ACTIVE:
                    return rid
            except ApiError:
                pass
            self._cands.discard(rid)
        return None

    def _unfinished(self, run_id: str) -> bool:
        meta = self.read_meta(run_id) or {}
        ls = meta.get("launches") or []
        return bool(ls) and ls[-1].get("ended_at") is None

    # -------------------------------------------------------------------------------------------- argv / launch
    def build_argv(self, run_id: str, meta: dict, retry_errors: bool = False) -> list[str]:
        rd = self.run_dir(run_id)
        if meta.get("kind") == "predict":
            from boltz_mlx.gui.predict import predict_argv

            return predict_argv(rd, meta["inputs"], meta["options"], self.predict_args)
        from boltz_mlx.gui.inputs import screen_argv

        return screen_argv(rd, meta["inputs"], meta["options"], rd / "out", self.extra_args, retry_errors=retry_errors)

    def launch(self, run_id: str, retry_errors: bool = False) -> dict:
        with self.lock:
            meta = self.read_meta(run_id)
            if meta is None:
                raise ApiError(400, "this run was not created by the GUI, so it cannot be started or resumed here")
            active = self.active_run_id(rescan=True)
            if active:
                raise ApiError(409, f"another screen is running ({active}): the GPU is shared, so only one screen runs at a time. "
                                    f"Wait for it or stop it first.")
            state = self.state(run_id)
            predict = meta.get("kind") == "predict"
            if state == "complete" and (predict or not retry_errors):
                raise ApiError(409, "this run is already complete" + ("" if predict else " (use 'retry errors' to rerun failed ligands)"))
            rd = self.run_dir(run_id)
            argv = self.build_argv(run_id, meta, retry_errors)
            cmd = (self.predict_cmd if predict else self.screen_cmd) + argv
            pkg_parent = str(Path(__file__).resolve().parents[2])
            env = dict(os.environ)
            if predict:  # numerics come from the job's flags only (as the screen's preset_environ does for a screen)
                env = {k: v for k, v in env.items()
                       if not ((k.startswith("BOLTZ_MLX_") and k not in PREDICT_ENV_KEEP) or k == "MLX_ENABLE_TF32")}
                env.update(self.predict_env)
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTHONPATH"] = os.pathsep.join([pkg_parent] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
            env.pop("BOLTZ_MLX_SCREEN_CHILD", None)  # the screen re-executes itself with its preset's numerics
            now = self.clock()
            log = self.log_path(run_id, meta)
            n = len(meta.get("launches") or []) + 1
            with open(log, "ab") as lf:
                lf.write(f"=== launch {n} at {iso(now)} ===\n".encode())
                lf.flush()
                start = lf.tell()
                counts = (self.rows(run_id).counts() if not predict and (rd / "out" / "results.jsonl").exists()
                          else {"ok": 0, "error": 0, "invalid": 0})
                proc = subprocess.Popen(cmd, cwd=str(rd), env=env, stdin=subprocess.DEVNULL, stdout=lf, stderr=subprocess.STDOUT,
                                        start_new_session=True)
            launch = {"started_at": now, "pid": proc.pid, "ended_at": None, "exit_code": None, "stop_requested_at": None,
                      "log_start": start, "computed_at_start": counts["ok"] + counts["error"], "retry_errors": bool(retry_errors),
                      "argv": argv}
            meta.setdefault("launches", []).append(launch)
            self.write_meta(run_id, meta)
            self.procs[run_id] = proc
            if self._cands is not None:
                self._cands.add(run_id)
            return launch

    def stop(self, run_id: str, force: bool = False) -> str:
        with self.lock:
            meta = self.read_meta(run_id)
            if meta is None or not meta.get("launches"):
                raise ApiError(409, "this run is not running")
            last = meta["launches"][-1]
            if last.get("ended_at") is not None or not self._alive(run_id, last):
                raise ApiError(409, "this run is not running")
            if not force and last.get("stop_requested_at"):
                return "stopping"
            try:
                os.kill(last["pid"], signal.SIGTERM)  # 1st: stop at a ligand boundary; 2nd (force): stop now, rows are kept
            except ProcessLookupError:
                raise ApiError(409, "this run is not running") from None
            last["stop_requested_at"] = last.get("stop_requested_at") or self.clock()
            last["stop_signals"] = int(last.get("stop_signals", 0)) + 1
            self.write_meta(run_id, meta)
            return "stopping"

    # ----------------------------------------------------------------------------------------------- data
    def rows(self, run_id: str) -> P.RowStore:
        with self.lock:
            rs = self._rows.get(run_id)
            if rs is None:
                rs = P.RowStore(self.out_dir(run_id) / "results.jsonl")
                self._rows[run_id] = rs
                while len(self._rows) > ROW_CACHE:
                    old, _ = self._rows.popitem(last=False)
                    self._scans.pop(old, None)
            else:
                self._rows.move_to_end(run_id)
        rs.refresh()
        return rs

    def results(self, run_id: str, rank_by: str, known_version: str | None = None) -> dict:
        from boltz_mlx.screen import results as R

        if rank_by not in R.RANK_BY:
            raise ApiError(400, f"rank_by must be one of {sorted(R.RANK_BY)}")
        rs = self.rows(run_id)
        version = f"{rs.version}:{rs.n_lines}:{rank_by}"
        if known_version == version:
            return {"version": version, "unchanged": True}
        ranked = R.rank_rows(rs.rows(), rank_by)
        have = self.pose_records(run_id)
        return {"version": version, "rank_by": rank_by, "n": len(ranked),
                "rows": [dict(P.slim(r), pose=r.get("status") == "ok" and r.get("record_id") in have) for r in ranked]}

    def pose_records(self, run_id: str) -> set[str]:
        """Record ids that still have a structure file: the top-N copies in structures/ and any kept work/predictions/<record>/
        model_0 (so a row whose pose was not retained can say so instead of failing when it is opened)."""
        out = self.out_dir(run_id)
        have: set[str] = set()
        try:
            with os.scandir(out / "structures") as it:
                for e in it:
                    m = POSE_FILE.fullmatch(e.name)
                    if m and e.is_file(follow_symlinks=False):
                        have.add(m.group(1))
        except OSError:
            pass
        try:
            with os.scandir(out / "work" / "predictions") as it:
                for e in it:
                    if e.is_dir(follow_symlinks=False) and any((Path(e.path) / f"{e.name}_model_0.{x}").is_file() for x in ("cif", "pdb")):
                        have.add(e.name)
        except OSError:
            pass
        return have

    def csv(self, run_id: str, rank_by: str, only: set[str] | None = None) -> str:
        from boltz_mlx.screen import results as R

        if rank_by not in R.RANK_BY:
            raise ApiError(400, f"rank_by must be one of {sorted(R.RANK_BY)}")
        return P.csv_text(self.rows(run_id).rows(), rank_by, only)

    def screen_config(self, run_id: str) -> dict:
        try:
            return json.loads((self.out_dir(run_id) / "screen_config.json").read_text())
        except (OSError, ValueError):
            return {}

    def row_by_record(self, run_id: str, record_id: str) -> dict:
        if not valid_record_id(record_id):
            raise ApiError(404, "no such ligand")
        for r in self.rows(run_id).rows():
            if r.get("record_id") == record_id:
                return r
        raise ApiError(404, "no such ligand")

    def pose_file(self, run_id: str, record_id: str) -> Path:
        """The pose of a ligand: structures/<rank>_<record>.<ext> (top-K copies) or work/predictions/<record>/<record>_model_0.<ext>.
        Only record ids that have a result row are served, and the resolved file must lie inside the run's out directory."""
        row = self.row_by_record(run_id, record_id)
        out = self.out_dir(run_id)
        cands: list[Path] = []
        sd = out / "structures"
        if sd.is_dir():
            cands += sorted(sd.glob(f"[0-9][0-9][0-9][0-9]_{record_id}.*"))
        pd = out / "work" / "predictions" / record_id
        for ext in ("cif", "pdb"):
            cands.append(pd / f"{record_id}_model_0.{ext}")
        for c in cands:
            if c.is_file() and is_within(c, out) and not c.is_symlink():
                return c
        if row.get("status") != "ok":
            raise ApiError(404, "no structure for this ligand (it did not finish)")
        raise ApiError(404, "no structure file found for this ligand (it may have been removed)")

    def ligand_detail(self, run_id: str, record_id: str) -> dict:
        from boltz_mlx.gui import pdbx

        row = dict(self.row_by_record(run_id, record_id))
        for k in ("env",):
            row.pop(k, None)
        out = self.out_dir(run_id)
        pd = out / "work" / "predictions" / record_id
        d: dict = {"row": row, "pose": None, "confidence": None, "affinity": None, "pocket": None,
                   "has_pae": (pd / f"pae_{record_id}_model_0.npz").is_file()}
        try:
            pose = self.pose_file(run_id, record_id)
            d["pose"] = {"format": "pdb" if pose.suffix == ".pdb" else "mmcif", "size": pose.stat().st_size}
            chain = (self.screen_config(run_id).get("ligand_chain")) or "L"
            key = (str(pose), pose.stat().st_mtime_ns, chain)
            with self.lock:
                hit = self._pose_cache.get(key)
            if hit is None and pose.suffix == ".cif":
                try:
                    hit = pdbx.pose_summary(pose, chain)
                except (OSError, ValueError):
                    hit = None
                if hit is not None:
                    with self.lock:
                        self._pose_cache[key] = hit
                        while len(self._pose_cache) > 64:
                            self._pose_cache.popitem(last=False)
            d["pocket"] = hit
        except ApiError:
            pass
        for name, key in ((f"confidence_{record_id}_model_0.json", "confidence"), (f"affinity_{record_id}.json", "affinity")):
            f = pd / name
            if f.is_file() and is_within(f, out):
                try:
                    d[key] = json.loads(f.read_text())
                except (OSError, ValueError):
                    pass
        return d

    # ------------------------------------------------------------------------------------------ summaries
    def _public_meta(self, meta: dict) -> dict:
        return {k: meta.get(k) for k in ("name", "created", "inputs", "options", "summary", "recheck_of", "recheck_of_preset")}

    def rechecks_of(self, run_id: str) -> list[str]:
        """Ids of the re-check runs made from this screen (newest last)."""
        out = []
        for d in sorted(self.root.iterdir()) if self.root.is_dir() else []:
            if d.is_dir() and not d.name.startswith(".") and d.name != run_id:
                m = self.read_meta(d.name)
                if m and m.get("recheck_of") == run_id:
                    out.append(d.name)
        return out

    def summary(self, run_id: str) -> dict:
        rd = self.run_dir(run_id)
        meta = self.read_meta(run_id)
        if meta is not None and meta.get("kind") == "predict":
            return self._predict_summary(run_id, meta)
        state = self.state(run_id)
        out = self.out_dir(run_id)
        st = self._status_file(run_id) if state not in STATES_ACTIVE else None
        s: dict = {"id": run_id, "kind": "screen", "state": state, "external": meta is None, "name": run_id, "created": None, "preset": None,
                   "rank_by": None, "target": None, "n_ligands": None, "n_ok": None, "n_error": None, "n_invalid": None,
                   "n_done": None, "wall_s": None, "path": str(rd)}
        if meta:
            o, sm = meta.get("options") or {}, meta.get("summary") or {}
            s.update(name=meta.get("name") or run_id, created=iso(meta.get("created")), preset=o.get("preset"),
                     rank_by=o.get("rank_by"), n_ligands=sm.get("total"),
                     target=((meta.get("inputs") or {}).get("target") or {}).get("name"))
            if s["target"] is None:
                s["target"] = (sm.get("target") or {}).get("source")
        else:
            try:
                s["created"] = iso(rd.stat().st_mtime)
            except OSError:
                pass
        if st is not None:
            s.update(n_ok=st.get("ok_total"), n_error=st.get("error_total"), n_invalid=st.get("invalid_total"),
                     wall_s=st.get("wall_s"), n_ligands=st.get("ligands", s["n_ligands"]))
            s["n_done"] = (st.get("ok_total") or 0) + (st.get("error_total") or 0)
            if s["preset"] is None:
                s["preset"] = str(st.get("mode", "")).split("[")[0] or None
        else:
            c = self.rows(run_id).counts() if (out / "results.jsonl").exists() else {"ok": 0, "error": 0, "invalid": 0}
            s.update(n_ok=c["ok"], n_error=c["error"], n_invalid=c["invalid"], n_done=c["ok"] + c["error"])
        if s["n_ligands"] is not None and s["n_invalid"]:  # one denominator everywhere: the ligands that can run (invalid ones are counted apart)
            s["n_ligands"] = max(0, s["n_ligands"] - s["n_invalid"])
        try:
            s["updated"] = iso(max(p.stat().st_mtime for p in (out / "results.jsonl", rd / (META if meta else "results.jsonl")) if p.exists()))
        except ValueError:
            s["updated"] = s["created"]
        return s

    def list_runs(self) -> list[dict]:
        out = []
        for child in self.root.iterdir():
            if self._is_run_dir(child):
                try:
                    out.append(self.summary(child.name))
                except ApiError:
                    continue
        out.sort(key=lambda s: (s.get("created") or "", s["id"]), reverse=True)
        return out

    def detail(self, run_id: str, log_lines: int = 40) -> dict:
        rd = self.run_dir(run_id)
        meta = self.read_meta(run_id)
        if meta is not None and meta.get("kind") == "predict":
            return self._predict_detail(run_id, meta, log_lines)
        state = self.state(run_id)
        alive = state in STATES_ACTIVE
        rs = self.rows(run_id)
        out = self.out_dir(run_id)
        launch = (meta or {}).get("launches", [None])[-1] if meta else None
        log_start = (launch or {}).get("log_start", 0)
        lines = P.read_log(rd / LOG, log_start) if meta else []
        info = P.parse_log(lines)
        summary = (meta or {}).get("summary") or {}
        rows = rs.rows()
        counts = rs.counts()
        terminal_records = {str(r.get("record_id")) for r in rows if r.get("status") in P.TERMINAL}
        pend = 0
        if alive:
            scan = self._scans.setdefault(run_id, P.StructScan())
            pend = scan.pending(out / "work" / "predictions", terminal_records)
        total = summary.get("total") or (info["plan"] or {}).get("ligands") or sum(counts.values())
        n_valid = summary.get("n_valid") if summary.get("n_valid") is not None else max(0, total - counts["invalid"])
        est = summary.get("estimate") or {}
        prog = P.build_progress(total=total, n_valid=n_valid, rows=rows, launch=launch, alive=alive, log=info,
                                struct_pending=pend, w_struct=est.get("w_struct", 0.4), est_total_s=est.get("total_s"),
                                now=self.clock())
        cfg = self.screen_config(run_id)
        status = self._status_file(run_id, (launch or {}).get("started_at")) if meta else self._status_file(run_id)
        d = {
            "id": run_id, "kind": "screen", "state": state, "external": meta is None, "name": (meta or {}).get("name") or run_id,
            "created": iso((meta or {}).get("created")), "progress": prog, "path": str(rd),
            "can_stop": state == "running", "can_force_stop": state == "stopping",
            "can_resume": meta is not None and state in STATES_RESUMABLE,
            "can_retry": meta is not None and state == "complete" and prog["error"] > 0,
            "keep_work": bool(((meta or {}).get("options") or {}).get("keep_work")),
            "launches": len((meta or {}).get("launches") or []), "pid": (launch or {}).get("pid") if alive else None,
            "ligand_chain": cfg.get("ligand_chain"), "config": {k: cfg.get(k) for k in ("preset", "steps", "recycling", "affinity_steps", "affinity_recycling", "affinity_samples", "seed", "env")},
            "preset_label": GP.run_label(cfg, ((meta or {}).get("options") or {}).get("preset")),
            "status": status, "plan": info["plan"], "stop_note": info["stop_note"],
            "target_warnings": info["target_warnings"] or (summary.get("target") or {}).get("warnings", []),
            "log_tail": info["tail"][-log_lines:], "meta": self._public_meta(meta) if meta else None,
            "results_version": f"{rs.version}:{rs.n_lines}",
            "recheck_of": (meta or {}).get("recheck_of"), "rechecks": self.rechecks_of(run_id) if meta else [],
        }
        if state in ("failed", "interrupted") and meta:
            d["run_error"] = "\n".join(info["tail"][-15:])
        return d

    def adopt_note(self) -> list[str]:
        """Run ids with a screen process still alive (for the shutdown / startup message)."""
        out = []
        for child in self.root.iterdir():
            if not child.name.startswith(".") and valid_run_id(child.name) and (child / META).exists():
                try:
                    if self.state(child.name) in STATES_ACTIVE:
                        out.append(child.name)
                except ApiError:
                    continue
        return out

    # ------------------------------------------------------------------------------------------ predict jobs
    def _json(self, p: Path, root: Path) -> dict | None:
        if not (p.is_file() and is_within(p, root)):
            return None
        try:
            v = json.loads(p.read_text())
        except (OSError, ValueError):
            return None
        return v if isinstance(v, dict) else None

    def _predict_metrics(self, run_id: str) -> dict:
        from boltz_mlx.gui import predict as PR

        out = self.run_dir(run_id) / "out"
        pd = PR.pred_dir(out)
        conf = self._json(pd / f"confidence_{PR.RECORD}_model_0.json", out) or {}
        aff = self._json(pd / f"affinity_{PR.RECORD}.json", out) or {}
        keys = ("ptm", "iptm", "ligand_iptm", "complex_plddt", "confidence_score")
        m = {k: conf.get(k) for k in keys if conf.get(k) is not None}
        m.update({k: aff.get(k) for k in ("affinity_pred_value", "affinity_probability_binary") if aff.get(k) is not None})
        return m

    def _predict_summary(self, run_id: str, meta: dict) -> dict:
        rd = self.run_dir(run_id)
        state = self.state(run_id)
        sm, o = meta.get("summary") or {}, meta.get("options") or {}
        last = (meta.get("launches") or [{}])[-1]
        wall = None
        if last.get("ended_at") and last.get("started_at"):
            wall = round(last["ended_at"] - last["started_at"], 1)
        s = {"id": run_id, "kind": "predict", "state": state, "external": False, "name": meta.get("name") or run_id,
             "created": iso(meta.get("created")), "preset": o.get("preset"), "rank_by": None,
             "target": sm.get("label"), "chains": sm.get("chains_short"), "n_chains": len(sm.get("chains") or []), "n_tokens": sm.get("n_tokens"),
             "affinity": bool(sm.get("affinity")), "n_ligands": None, "n_ok": None, "n_error": None, "n_invalid": None,
             "n_done": None, "wall_s": wall, "path": str(rd),
             "metrics": self._predict_metrics(run_id) if state == "complete" else None}
        if state == "complete":  # the CLI's own wall time when it wrote one (excludes the Python start-up of the subprocess)
            from boltz_mlx.gui import predict as PR

            t = self._json(PR.results_dir(rd / "out") / "boltz_mlx_timings.json", rd / "out") or {}
            if isinstance(t.get("total_wall_s"), (int, float)):
                s["wall_s"] = round(float(t["total_wall_s"]), 1)
        try:
            s["updated"] = iso((rd / META).stat().st_mtime)
        except OSError:
            s["updated"] = s["created"]
        return s

    def _predict_detail(self, run_id: str, meta: dict, log_lines: int) -> dict:
        from boltz_mlx.gui import predict as PR

        rd = self.run_dir(run_id)
        state = self.state(run_id)
        alive = state in STATES_ACTIVE
        sm = meta.get("summary") or {}
        launch = (meta.get("launches") or [None])[-1]
        lines = P.read_log(self.log_path(run_id, meta), (launch or {}).get("log_start", 0))
        out = rd / "out"
        affinity = bool(sm.get("affinity"))
        ph = PR.phase(out, affinity, (launch or {}).get("started_at")) if launch else "new"
        now = self.clock()
        started = (launch or {}).get("started_at")
        elapsed = (now - started) if (alive and started) else (
            (launch["ended_at"] - started) if launch and launch.get("ended_at") and started else None)
        est = sm.get("estimate") or {}
        total = est.get("total_s")
        eta = max(0.0, total - elapsed) if (alive and total and elapsed is not None) else None
        steps = ["starting", "structure"] + (["affinity"] if affinity else []) + ["done"]
        cur = "done" if state == "complete" else ("finishing" if ph == "finishing" else ph)
        prog = {"phase": cur if alive or state == "complete" else ph, "steps": steps,
                "elapsed_s": None if elapsed is None else round(elapsed, 1), "est_total_s": total,
                "eta_s": None if eta is None else round(eta), "over_estimate": bool(alive and total and elapsed and elapsed > total),
                "fraction": (1.0 if state == "complete" else
                             round(min(0.97, elapsed / total), 3) if (alive and total and elapsed is not None) else None)}
        timings = self._json(PR.results_dir(out) / "boltz_mlx_timings.json", out) if state == "complete" else None
        d = {
            "id": run_id, "kind": "predict", "state": state, "external": False, "name": meta.get("name") or run_id,
            "created": iso(meta.get("created")), "progress": prog, "path": str(rd),
            "can_stop": state == "running", "can_force_stop": state == "stopping",
            "can_resume": state in STATES_RESUMABLE, "can_retry": False,
            "launches": len(meta.get("launches") or []), "pid": (launch or {}).get("pid") if alive else None,
            "log_tail": [ln[:400] for ln in lines[-60:] if ln.strip()][-log_lines:], "meta": self._public_meta(meta),
            "timings": timings, "has_result": PR.model_file(out) is not None,
            "results_version": self._predict_version(out),
        }
        if state in ("failed", "interrupted"):
            d["run_error"] = "\n".join([ln for ln in lines if ln.strip()][-15:])
        return d

    @staticmethod
    def _predict_version(out: Path) -> str:
        from boltz_mlx.gui import predict as PR

        parts = []
        names = [f"{PR.RECORD}_model_{k}.cif" for k in (PR.model_indices(out) or [0])]
        for name in names + [f"confidence_{PR.RECORD}_model_0.json", f"affinity_{PR.RECORD}.json"]:
            try:
                parts.append(str((PR.pred_dir(out) / name).stat().st_mtime_ns))
            except OSError:
                parts.append("-")
        return ":".join(parts)

    def predict_out(self, run_id: str) -> Path:
        meta = self.read_meta(run_id)
        if self.kind(run_id, meta) != "predict":
            raise ApiError(404, "not a structure prediction run")
        return self.run_dir(run_id) / "out"

    def _model_index(self, out: Path, model) -> int:
        from boltz_mlx.gui import predict as PR

        have = PR.model_indices(out) or [0]
        try:
            k = int(model) if model not in (None, "") else have[0]
        except (TypeError, ValueError):
            raise ApiError(400, "model must be a whole number") from None
        if k not in have:
            raise ApiError(404, f"no model {k} (this job has {len(have)})")
        return k

    def prediction(self, run_id: str, model=None) -> dict:
        """Everything the results page shows for one model of a finished (or partly finished) predict job."""
        from boltz_mlx.gui import pdbx
        from boltz_mlx.gui import predict as PR

        out = self.predict_out(run_id)
        meta = self.read_meta(run_id) or {}
        sm = meta.get("summary") or {}
        pd = PR.pred_dir(out)
        k = self._model_index(out, model)
        pose = PR.model_file(out, k)
        if pose is None or not is_within(pose, out):
            raise ApiError(404, "no structure yet")
        models = []
        for i in PR.model_indices(out):
            c = self._json(pd / f"confidence_{PR.RECORD}_model_{i}.json", out) or {}
            models.append({"index": i, **{key: c.get(key) for key in ("confidence_score", "ptm", "iptm", "complex_plddt") if c.get(key) is not None}})
        d: dict = {"record": PR.RECORD, "model": k, "models": models,
                   "pose": {"format": "pdb" if pose.suffix == ".pdb" else "mmcif", "size": pose.stat().st_size},
                   "confidence": self._json(pd / f"confidence_{PR.RECORD}_model_{k}.json", out),
                   "affinity": self._json(pd / f"affinity_{PR.RECORD}.json", out),
                   "has_pae": (pd / f"pae_{PR.RECORD}_model_{k}.npz").is_file(),
                   "chains": sm.get("chains") or [], "binder": sm.get("binder"), "pocket": None,
                   "options": meta.get("options") or {}}
        if sm.get("binder") and pose.suffix == ".cif":
            key = (str(pose), pose.stat().st_mtime_ns, sm["binder"])
            with self.lock:
                hit = self._pose_cache.get(key)
            if hit is None:
                try:
                    hit = pdbx.pose_summary(pose, sm["binder"])
                except (OSError, ValueError):
                    hit = None
                if hit is not None:
                    with self.lock:
                        self._pose_cache[key] = hit
            d["pocket"] = hit
        return d

    def prediction_pose(self, run_id: str, model=None) -> Path:
        from boltz_mlx.gui import predict as PR

        out = self.predict_out(run_id)
        p = PR.model_file(out, self._model_index(out, model))
        if p is None or not is_within(p, out) or p.is_symlink():
            raise ApiError(404, "no structure yet")
        return p

    def prediction_pae(self, run_id: str, model=None) -> dict:
        from boltz_mlx.gui import predict as PR
        from boltz_mlx.gui.pae import pae_payload

        out = self.predict_out(run_id)
        k = self._model_index(out, model)
        f = PR.pred_dir(out) / f"pae_{PR.RECORD}_model_{k}.npz"
        if not is_within(f, out):
            raise ApiError(404, "no PAE matrix for this prediction")
        return pae_payload(f, PR.model_file(out, k))

    def prediction_pae_file(self, run_id: str, model=None) -> Path:
        """The original pae_*.npz of a model (full precision, as Boltz wrote it)."""
        from boltz_mlx.gui import predict as PR

        out = self.predict_out(run_id)
        f = PR.pred_dir(out) / f"pae_{PR.RECORD}_model_{self._model_index(out, model)}.npz"
        if not is_within(f, out) or f.is_symlink() or not f.is_file():
            raise ApiError(404, "no PAE matrix for this prediction")
        return f

    def ligand_pae_file(self, run_id: str, record_id: str) -> Path:
        """The original pae_*.npz kept for a screened ligand (full precision, as Boltz wrote it)."""
        self.row_by_record(run_id, record_id)
        out = self.out_dir(run_id)
        f = out / "work" / "predictions" / record_id / f"pae_{record_id}_model_0.npz"
        if not is_within(f, out) or f.is_symlink() or not f.is_file():
            raise ApiError(404, "no PAE matrix kept for this ligand (start the screen with 'Keep PAE')")
        return f

    def ligand_pae(self, run_id: str, record_id: str) -> dict:
        """A screened ligand's PAE: kept only by screens run with --keep-work (the GUI's 'Keep PAE' option)."""
        from boltz_mlx.gui.pae import pae_payload

        self.row_by_record(run_id, record_id)
        out = self.out_dir(run_id)
        pd = out / "work" / "predictions" / record_id
        f = pd / f"pae_{record_id}_model_0.npz"
        if not is_within(f, out) or not f.is_file():
            raise ApiError(404, "no PAE matrix kept for this ligand (start the screen with 'Keep PAE matrices')")
        try:
            pose = self.pose_file(run_id, record_id)
        except ApiError:
            pose = None
        return pae_payload(f, pose)
