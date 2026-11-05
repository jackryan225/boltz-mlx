"""Machine state as JSON lines for the bench-r2 jobs (read-only: pmset, ioreg, sysctl; no power or system changes).

  python3 tools/bench_r2/state.py snap --label LABEL [--out FILE] [--extra '{"rc": 0}']   # one line (stdout + FILE)
  python3 tools/bench_r2/state.py sample --every 30 --out FILE                          # until SIGTERM / SIGINT

Fields: t (epoch s), utc, label, power_source ('AC Power' / 'Battery Power' / '?'), battery_pct, system_in_w,
system_load_w, battery_w (ioreg AppleSmartBattery PowerTelemetryData), gpu_util_pct (ioreg IOAccelerator Device
Utilization %), thermal (pmset -g therm, 'none recorded' when clean), loadavg1, swap_used_gb / swap_total_gb
(sysctl vm.swapusage), mem_level_pct (sysctl kern.memorystatus_level: % memory available as the kernel sees it).
Missing values are null. Read-only machine state.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import plistlib
import re
import signal
import subprocess
import sys
import time


def _run(cmd: list) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, timeout=10).stdout.decode(errors="replace")
    except (OSError, subprocess.SubprocessError):
        return ""


def _plist(cmd: list) -> list:
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=10).stdout
        d = plistlib.loads(out)
    except Exception:  # noqa: BLE001  (no ioreg data: nulls)
        return []
    return d if isinstance(d, list) else [d]


def swap_gb() -> tuple[float | None, float | None]:
    """(used, total) GB from sysctl vm.swapusage ('total = 3072.00M  used = 1799.88M ...')."""
    s = _run(["sysctl", "-n", "vm.swapusage"])
    def f(key):
        m = re.search(key + r"\s*=\s*([\d.]+)([MG])", s)
        return None if not m else round(float(m.group(1)) / (1024 if m.group(2) == "M" else 1), 3)
    return f("used"), f("total")


def power_source() -> str:
    """'AC Power' / 'Battery Power' / '?' (pmset -g batt first line; cheap enough for a 30 s watchdog tick)."""
    batt = _run(["pmset", "-g", "batt"]).splitlines()
    return batt[0].split("'")[1] if batt and "'" in batt[0] else "?"


def snapshot(label: str = "") -> dict:
    batt = _run(["pmset", "-g", "batt"]).splitlines()
    src = batt[0].split("'")[1] if batt and "'" in batt[0] else "?"
    pct = None
    for ln in batt[1:]:
        m = re.search(r"(\d+)%", ln)
        if m:
            pct = int(m.group(1))
            break
    tel = next((x.get("PowerTelemetryData", {}) for x in _plist(["ioreg", "-rn", "AppleSmartBattery", "-a"])), {})
    w = {k: (round(tel[k] / 1000, 1) if isinstance(tel.get(k), int) else None)
         for k in ("SystemPowerIn", "SystemLoad", "BatteryPower")}
    gpu = next((x.get("PerformanceStatistics", {}).get("Device Utilization %")
                for x in _plist(["ioreg", "-rc", "IOAccelerator", "-a"])), None)
    therm = " ".join(ln.strip() for ln in _run(["pmset", "-g", "therm"]).splitlines()[1:] if ln.strip())
    warn = "none recorded" if ("No " in therm and "warning level has been recorded" in therm) else (therm or None)
    used, total = swap_gb()
    lvl = _run(["sysctl", "-n", "kern.memorystatus_level"]).strip()
    now = time.time()
    return {
        "t": round(now, 3),
        "utc": dt.datetime.fromtimestamp(now, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "label": label,
        "power_source": src,
        "battery_pct": pct,
        "system_in_w": w["SystemPowerIn"],
        "system_load_w": w["SystemLoad"],
        "battery_w": w["BatteryPower"],
        "gpu_util_pct": gpu,
        "thermal": warn,
        "loadavg1": round(os.getloadavg()[0], 2),
        "swap_used_gb": used,
        "swap_total_gb": total,
        "mem_level_pct": int(lvl) if lvl.isdigit() else None,
    }


def _append(path: str | None, rec: dict) -> None:
    line = json.dumps(rec)
    print(line, flush=True)
    if path:
        with open(path, "a") as f:
            f.write(line + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s1 = sub.add_parser("snap")
    s1.add_argument("--label", default="")
    s1.add_argument("--out", default=None)
    s1.add_argument("--extra", default=None, help="JSON object merged into the record (step name, rc, wall_s, ...)")
    s2 = sub.add_parser("sample")
    s2.add_argument("--every", type=float, default=30.0)
    s2.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "snap":
        rec = snapshot(a.label)
        if a.extra:
            rec.update(json.loads(a.extra))
        _append(a.out, rec)
        return 0
    stop = {"flag": False}

    def _stop(*_):
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    while not stop["flag"]:
        rec = snapshot("sample")
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        t_next = time.time() + a.every
        while not stop["flag"] and time.time() < t_next:
            time.sleep(0.5)
    return 0


if __name__ == "__main__":
    sys.exit(main())
