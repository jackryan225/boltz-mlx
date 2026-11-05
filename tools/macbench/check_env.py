"""Environment pin check for the macbench jobs (stdlib only; exit 3 on mismatch, 0 ok).

  python tools/macbench/check_env.py [--need rbvi stock community msa]

rbvi : macbench-env/src/rbvi-boltz at branch chimerax_boltz22 HEAD 2be3519b..., venv-rbvi has boltz 2.2.0 + torch 2.7.1
stock: venv-stock has boltz 2.2.1 + torch 2.15.0
community: macbench-env/src/boltz-community at branch main HEAD 1431d573..., venv-community has boltz-community 2.13.0 + torch 2.15.0
msa  : sha256 of every csv in macbench-env/msa equals tools/macbench/msa_manifest.json
(macbench-env = $MACBENCH_ENV, default <checkout>/macbench-env; docs/macbench/ENV.md.) A freshly fetched MSA can differ from the
pinned one (the ColabFold databases change): the msa check then fails, and the benchmark is not the same benchmark.
"""
import argparse
import os
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENV = Path(os.environ.get("MACBENCH_ENV", HERE.parents[1] / "macbench-env")).expanduser()
RBVI_HEAD = "2be3519b5642f13b6519c408b5602a938aa8610d"
COMM_HEAD = "1431d573c3a02a2cb6b5c3b775e7128c7a98d539"
WANT = {"rbvi": ("2.2.0", "2.7.1"), "stock": ("2.2.1", "2.15.0"), "community": ("2.13.0", "2.15.0")}
DIST = {"rbvi": "boltz", "stock": "boltz", "community": "boltz-community"}


def versions(venv: str) -> tuple:
    out = subprocess.run([str(ENV / f"venv-{venv}" / "bin" / "python"), "-I", "-c",
                          f"import importlib.metadata as m; print(m.version('{DIST[venv]}'), m.version('torch'))"], capture_output=True, text=True)
    return tuple(out.stdout.split())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--need", nargs="*", default=[])
    a = ap.parse_args()
    bad = []
    for v in ("rbvi", "stock", "community"):
        if v in a.need:
            got = versions(v)
            if got != WANT[v]:
                bad.append(f"venv-{v}: boltz/torch {got} != {WANT[v]}")
    if "rbvi" in a.need:
        head = subprocess.run(["git", "-C", str(ENV / "src" / "rbvi-boltz"), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        if head != RBVI_HEAD:
            bad.append(f"rbvi-boltz HEAD {head} != {RBVI_HEAD}")
    if "community" in a.need:
        head = subprocess.run(["git", "-C", str(ENV / "src" / "boltz-community"), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        if head != COMM_HEAD:
            bad.append(f"boltz-community HEAD {head} != {COMM_HEAD}")
    if "msa" in a.need:
        man = json.loads((HERE / "msa_manifest.json").read_text())["files"]
        for name, d in man.items():
            p = ENV / "msa" / name
            h = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
            if h != d["sha256"]:
                bad.append(f"msa {name}: {h} != {d['sha256']}")
    if bad:
        print("ENV CHECK FAILED:\n  " + "\n  ".join(bad))
        return 3
    print("ENV CHECK OK:", ", ".join(a.need) or "(nothing requested)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
