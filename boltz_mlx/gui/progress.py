"""Progress of a screen, read from its files only (the GUI never talks to the screen process):

  * out/results.jsonl  - tailed incrementally (RowStore): one row per finished ligand, last row per ligand id wins
  * the subprocess log - parsed for the plan, chunk, current ligand ("[start] structure|affinity <id> k/n") and status
  * out/work/predictions/<record>/screen_struct.json - structure passes finished in the current chunk (the screen does
    two phases per chunk, so rows only appear when the affinity phase of a ligand ends)

The screen log lines come from boltz_mlx/screen/runner.py; if a line is absent (older screen) the matching field is None.
"""

from __future__ import annotations

import json
import os
import re
import statistics
import threading
from pathlib import Path

from boltz_mlx.screen import results as R

SLIM_KEYS = (
    "ligand_id", "record_id", "index", "smiles", "smiles_used", "status", "error", "error_phase", "warnings",
    "affinity_pred_value", "value_mw_corrected", "affinity_probability_binary", "iptm", "ligand_iptm", "complex_plddt",
    "ligand_plddt", "ptm", "n_atoms", "mw", "seconds", "fp16_fallbacks", "affinity_crop_tokens",
)
TERMINAL = ("ok", "error", "invalid")


def slim(row: dict) -> dict:
    d = {k: row.get(k) for k in SLIM_KEYS}
    d["rank"] = row.get("rank")
    return d


class RowStore:
    """Incremental reader of results.jsonl. Only complete lines are consumed (a torn last line of a killed run, or a line
    still being written, is picked up on a later refresh); a replaced or truncated file is re-read from the start."""

    MAX_READ = 64 * 1024 * 1024

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self.lock = threading.Lock()
        self._reset()

    def _reset(self) -> None:
        self.offset = 0
        self.ino = None
        self.latest: dict[str, dict] = {}
        self.n_lines = 0
        self.version = 0

    def refresh(self) -> bool:
        """Read what was appended since the last call. Returns True if rows changed."""
        with self.lock:
            try:
                st = os.stat(self.path)
            except OSError:
                if self.offset:
                    self._reset()
                    return True
                return False
            if self.ino is not None and (st.st_ino != self.ino or st.st_size < self.offset):
                self._reset()
            self.ino = st.st_ino
            if st.st_size <= self.offset:
                return False
            changed = False
            while True:
                with open(self.path, "rb") as f:
                    f.seek(self.offset)
                    data = f.read(min(st.st_size - self.offset, self.MAX_READ))
                end = data.rfind(b"\n")
                if end < 0:
                    break
                chunk = data[: end + 1]
                self.offset += len(chunk)
                for line in chunk.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue  # torn line
                    if isinstance(r, dict) and "ligand_id" in r:
                        self.latest[str(r["ligand_id"])] = r
                        self.n_lines += 1
                        changed = True
                if self.offset >= st.st_size:
                    break
            if changed:
                self.version += 1
            return changed

    def rows(self) -> list[dict]:
        with self.lock:
            return list(self.latest.values())

    def counts(self) -> dict:
        with self.lock:
            c = {"ok": 0, "error": 0, "invalid": 0}
            for r in self.latest.values():
                s = r.get("status")
                if s in c:
                    c[s] += 1
            return c


# ------------------------------------------------------------------------------------------------------------- log
RE_PLAN = re.compile(r"^\[plan\] (\d+) ligands: (\d+) already done, (\d+) invalid, (\d+) id conflicts, (\d+) to run in (\d+) "
                     r"chunk\(s\) of <= (\d+); mode (.*)$")
RE_CHUNK = re.compile(r"^\[chunk (\d+)/(\d+)\] (\d+) ligands")
RE_START = re.compile(r"^\[start\] (structure|affinity) (.*) (\d+)/(\d+)$")
RE_LOAD = re.compile(r"^\[load\] (\w+) model ([0-9.]+)s")
RE_STATUS = re.compile(r"^\[status\] (\{.*\})$")


def read_log(path: str | os.PathLike, start: int = 0, head: int = 64 * 1024, tail: int = 256 * 1024) -> list[str]:
    """Lines of the log from byte `start` (the beginning of the current launch): the first `head` and last `tail` bytes
    (the middle of a very long log is skipped; progress needs only the plan line and the recent markers)."""
    try:
        size = os.stat(path).st_size
    except OSError:
        return []
    start = min(max(0, int(start)), size)
    with open(path, "rb") as f:
        if size - start <= head + tail:
            f.seek(start)
            raw = f.read()
        else:
            f.seek(start)
            a = f.read(head)
            f.seek(size - tail)
            b = f.read(tail)
            nl = b.find(b"\n")
            raw = a + b"\n" + (b[nl + 1:] if nl >= 0 else b)
    text = raw.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    return [ln.rstrip() for ln in text.split("\n")]


def parse_log(lines: list[str]) -> dict:
    info: dict = {"plan": None, "chunk": None, "marker": None, "load": None, "status": None, "target_warnings": [],
                  "stop_note": None, "tail": []}
    for ln in lines:
        m = RE_PLAN.match(ln)
        if m and info["plan"] is None:
            n, done, inv, conf, todo, nch, ch, mode = m.groups()
            info["plan"] = {"ligands": int(n), "already_done": int(done), "invalid": int(inv), "conflicts": int(conf),
                            "to_run": int(todo), "chunks": int(nch), "chunk_size": int(ch), "mode": mode}
            continue
        m = RE_CHUNK.match(ln)
        if m:
            info["chunk"] = {"index": int(m.group(1)), "of": int(m.group(2)), "size": int(m.group(3))}
            continue
        m = RE_START.match(ln)
        if m:
            info["marker"] = {"phase": m.group(1), "ligand_id": m.group(2), "k": int(m.group(3)), "n": int(m.group(4))}
            continue
        m = RE_LOAD.match(ln)
        if m:
            info["load"] = {"model": m.group(1), "seconds": float(m.group(2))}
            continue
        m = RE_STATUS.match(ln)
        if m:
            try:
                info["status"] = json.loads(m.group(1))
            except ValueError:
                pass
            continue
        if ln.startswith("[target] warning:"):
            info["target_warnings"].append(ln[len("[target] warning:"):].strip())
        elif ln.startswith(("[stop]", "[budget]", "[interrupted]")):
            info["stop_note"] = ln
    info["tail"] = [ln[:400] for ln in lines[-60:] if ln.strip()]
    return info


def phase_of(info: dict, alive: bool) -> str:
    """starting | structure | affinity | finished | idle"""
    if not alive:
        return "idle" if info.get("status") is None else "finished"
    m = info.get("marker")
    if m:
        return m["phase"]
    return "starting"


# ---------------------------------------------------------------------------------------------------- struct passes
class StructScan:
    """Ligands whose structure pass finished (sidecar screen_struct.json) but that have no result row yet. Sidecars never
    disappear, so a confirmed name is remembered; only new directories are stat'ed."""

    def __init__(self) -> None:
        self.confirmed: set[str] = set()

    def pending(self, pred_root: Path, terminal_records: set[str]) -> int:
        try:
            names = os.listdir(pred_root)
        except OSError:
            return 0
        n = 0
        for nm in names:
            if nm in terminal_records:
                continue
            if nm in self.confirmed:
                n += 1
            elif os.path.exists(pred_root / nm / "screen_struct.json"):
                self.confirmed.add(nm)
                n += 1
        return n


# ------------------------------------------------------------------------------------------------------- progress
def build_progress(*, total: int, n_valid: int, rows: list[dict], launch: dict | None, alive: bool, log: dict,
                   struct_pending: int, w_struct: float, est_total_s: float | None, now: float, max_errors: int = 300) -> dict:
    ok = [r for r in rows if r.get("status") == "ok"]
    err = [r for r in rows if r.get("status") == "error"]
    inv = [r for r in rows if r.get("status") == "invalid"]
    computed = len(ok) + len(err)
    done = computed + len(inv)
    remaining = max(0, total - done)
    pend = max(0, struct_pending) if alive else 0
    w = min(0.9, max(0.1, w_struct))
    denom = max(1, n_valid)
    fraction = min(1.0, (computed + w * pend) / denom) if n_valid else 1.0
    started = launch.get("started_at") if launch else None
    elapsed = (now - started) if (alive and started) else None
    at_start = (launch or {}).get("computed_at_start", 0)
    this = max(0, computed - at_start)
    units = this + w * pend
    eta = eta_src = s_per = None
    if alive and elapsed is not None:
        left_units = max(0.0, (n_valid - computed) - w * pend)
        if units >= 1.0 and elapsed >= 5.0:
            s_per = elapsed / units
            eta, eta_src = left_units * s_per, "measured"
        elif est_total_s:
            eta, eta_src = est_total_s * max(0.0, 1 - fraction), "estimate"
    secs = [float(r["seconds"]) for r in ok if isinstance(r.get("seconds"), (int, float))]
    last_done = max((r.get("t_done") for r in rows if isinstance(r.get("t_done"), (int, float))), default=None)
    errors = sorted((r for r in err + inv), key=lambda r: (int(r.get("index") or 0), str(r.get("ligand_id"))))
    marker = log.get("marker")
    return {
        "total": total, "valid": n_valid, "done": done, "ok": len(ok), "error": len(err), "invalid": len(inv),
        "remaining": remaining, "remaining_valid": max(0, n_valid - computed), "structures_pending": pend,
        "fraction": round(fraction, 4), "fraction_done": round(computed / denom, 4) if n_valid else 1.0,
        "phase": phase_of(log, alive), "chunk": log.get("chunk"),
        "current": ({"ligand_id": marker["ligand_id"], "phase": marker["phase"], "k": marker["k"], "n": marker["n"]}
                    if (alive and marker) else None),
        "elapsed_s": None if elapsed is None else round(elapsed, 1),
        "s_per_ligand": None if s_per is None else round(s_per, 2),
        "median_model_s": round(statistics.median(secs), 2) if secs else None,
        "eta_s": None if eta is None else round(eta),
        "eta_source": eta_src,
        "last_row_age_s": None if last_done is None else round(max(0.0, now - last_done), 1),
        "errors": [{"ligand_id": r.get("ligand_id"), "status": r.get("status"), "phase": r.get("error_phase"),
                    "error": r.get("error")} for r in errors[:max_errors]],
        "errors_total": len(errors),
        "load": log.get("load"),
    }


def csv_text(rows: list[dict], rank_by: str, only: set[str] | None = None) -> str:
    """The ranked results.csv content for `rank_by` (same columns / formatting as the screen's results.csv). `only` (record ids)
    keeps those rows, with the ranks they have in the whole run."""
    import csv
    import io

    ranked = R.rank_rows(rows, rank_by)
    if only is not None:
        ranked = [r for r in ranked if r.get("record_id") in only]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(R.CSV_COLUMNS)
    for r in ranked:
        w.writerow([R._fmt(r.get(c)) for c in R.CSV_COLUMNS])
    return buf.getvalue()
