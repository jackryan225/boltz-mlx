"""Run a predictor over the validation set and score every case against the experimental structure.

The predictor is given as a shell command template; placeholders:
  {id}        case id (e.g. 8j9r)             {yaml}     absolute path of validation/yaml/<id>.yaml
  {out_dir}   per-case output dir (created)   {feats}    data/<id>/feats.npz (upstream featurized batch)
  {data_dir}  data/<id>                       {root}     repo root
The predicted structure is then located as the newest ``*model_0.cif`` (override with --pred-glob, relative to
{out_dir}) and scored with tools/validation/score.py (CA-lDDT, TM-score, CA RMSD).

Examples
  # MLX CLI (GPU), all cases:
  python tools/validation/run_all.py --name mlx_gpu \
      --cmd 'boltz-mlx predict {yaml} --out_dir {out_dir} --override' --out validation/mlx_scores.json
  # score predictions that already exist (e.g. the torch CPU references) without running anything:
  python tools/validation/run_all.py --name torch_cpu --score-only \
      --pred-dir 'data/{id}/ref/predictions' --out validation/torch_cpu_scores.json
  # all 30 cases (10 original + 20 extra without torch reference):
  python tools/validation/run_all.py --manifest validation/manifest_all.json --name mlx_gpu_30 --cmd '...'
  # compare against the torch table and flag regressions:
  python tools/validation/run_all.py ... --compare validation/torch_cpu_scores.json
"""

from __future__ import annotations
import os

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("BOLTZ_MLX_DATA", Path(__file__).resolve().parents[2] / "data")).expanduser()
MANIFEST = ROOT / "validation" / "manifest.json"


def find_pred(out_dir: Path, pattern: str) -> Path | None:
    hits = sorted(out_dir.rglob(pattern), key=lambda p: p.stat().st_mtime)
    return hits[-1] if hits else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cmd", help="predictor command template (see placeholders above)")
    ap.add_argument("--score-only", action="store_true", help="do not run a command; score existing predictions in --pred-dir")
    ap.add_argument("--pred-dir", default=None, help="template for the prediction dir to search (default: {out_dir}); placeholders as above")
    ap.add_argument("--pred-glob", default="*model_0.cif")
    ap.add_argument("--manifest", type=Path, default=MANIFEST,
                    help="case manifest (default validation/manifest.json; validation/manifest_all.json = 30 cases)")
    ap.add_argument("--cases", nargs="*", default=None, help="subset of case ids (default: all in manifest)")
    ap.add_argument("--name", default="predictor", help="label stored in the output table")
    ap.add_argument("--work", type=Path, default=DATA / "validation" / "runs", help="parent of per-case output dirs ({out_dir}=<work>/<name>/<id>)")
    ap.add_argument("--out", type=Path, default=None, help="JSON table to write (default validation/<name>_scores.json)")
    ap.add_argument("--timeout", type=float, default=3600, help="per-case timeout, seconds")
    ap.add_argument("--compare", type=Path, default=None, help="reference table (e.g. validation/torch_cpu_scores.json) to print deltas against")
    ap.add_argument("--stop-on-error", action="store_true")
    a = ap.parse_args(argv)
    if not a.score_only and not a.cmd:
        ap.error("--cmd is required unless --score-only")

    manifest = json.loads(a.manifest.read_text())
    cases = [c for c in manifest["cases"] if a.cases is None or c["id"] in a.cases]
    if not cases:
        ap.error("no matching cases")
    out_path = a.out or ROOT / "validation" / f"{a.name}_scores.json"

    rows = []
    for c in cases:
        cid = c["id"]
        out_dir = a.work / a.name / cid
        out_dir.mkdir(parents=True, exist_ok=True)
        fmt = dict(
            id=cid,
            yaml=str((ROOT / c["yaml"]).resolve()),
            out_dir=str(out_dir),
            feats=str(DATA / cid / "feats.npz"),
            data_dir=str(DATA / cid),
            root=str(ROOT),
        )
        row = {"id": cid, "length_seqres": c["length_seqres"], "length_modeled": c["length_modeled"]}
        if not a.score_only:
            cmd = a.cmd.format(**fmt)
            row["command"] = cmd
            t0 = time.time()
            try:
                p = subprocess.run(cmd, shell=True, cwd=ROOT, timeout=a.timeout, capture_output=True, text=True)
                row["wall_s"] = round(time.time() - t0, 2)
                row["returncode"] = p.returncode
                (out_dir / "stdout.log").write_text(p.stdout)
                (out_dir / "stderr.log").write_text(p.stderr)
                if p.returncode != 0:
                    row["error"] = f"exit {p.returncode}: {p.stderr.strip().splitlines()[-1] if p.stderr.strip() else ''}"
            except subprocess.TimeoutExpired:
                row["wall_s"] = round(time.time() - t0, 2)
                row["error"] = f"timeout after {a.timeout}s"
        if "error" not in row:
            search = Path(a.pred_dir.format(**fmt)) if a.pred_dir else out_dir
            if not search.is_absolute():
                search = ROOT / search
            pred = find_pred(search, a.pred_glob)
            if pred is None:
                row["error"] = f"no {a.pred_glob} under {search}"
            else:
                row["pred"] = str(pred)
                try:
                    row.update(score.score_files(pred, DATA / "validation" / "cif" / f"{cid}.cif"))
                except Exception as ex:  # noqa: BLE001
                    row["error"] = f"scoring failed: {ex!r}"
        rows.append(row)
        if "error" in row:
            print(f"{cid}: ERROR {row['error']}", file=sys.stderr)
            if a.stop_on_error:
                break
        else:
            print(f"{cid}: lDDT {row['lddt_ca']:.4f} TM {row['tm_score']:.4f} RMSD {row['rmsd_ca']:.2f}"
                  + (f" ({row['wall_s']} s)" if "wall_s" in row else ""), flush=True)

    ok = [r for r in rows if "error" not in r]
    summary = {"n_cases": len(rows), "n_ok": len(ok)}
    for k in ("lddt_ca", "tm_score", "rmsd_ca"):
        if ok:
            summary[f"mean_{k}"] = sum(r[k] for r in ok) / len(ok)
    table = {"name": a.name, "command": a.cmd, "summary": summary, "cases": rows}

    if a.compare:
        ref = {r["id"]: r for r in json.loads(a.compare.read_text())["cases"] if "error" not in r}
        print(f"\n{'case':8s} {'dLDDT':>8s} {'dTM':>8s} {'dRMSD':>8s}  (this - {a.compare.name})")
        for r in ok:
            if r["id"] in ref:
                q = ref[r["id"]]
                r["delta_vs_ref"] = {k: r[k] - q[k] for k in ("lddt_ca", "tm_score", "rmsd_ca")}
                d = r["delta_vs_ref"]
                print(f"{r['id']:8s} {d['lddt_ca']:+8.4f} {d['tm_score']:+8.4f} {d['rmsd_ca']:+8.3f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(table, indent=1))
    print(f"\nmean over {len(ok)}/{len(rows)}: " + ", ".join(f"{k}={v:.4f}" for k, v in summary.items() if k.startswith("mean")))
    print(f"wrote {out_path}")
    return 0 if len(ok) == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
