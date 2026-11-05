"""Exports: a results CSV for any run and a zip of the results plus the structures, with mmCIF -> PDB conversion on request.

Screen runs: results.csv is the screen's own ranked table (RunStore.csv); the zip adds the poses the screen saved
(out/structures/, the top-K copies; "save structures: all" keeps every pose). Structure jobs: a one-row CSV with the
confidence and affinity outputs and the IC50-style conversions, and the model file(s) plus the confidence / affinity JSON.

Affinity conversions follow the upstream Boltz-2 docs (docs/prediction.md, affinity section): affinity_pred_value y is
log10(IC50) with IC50 in uM, so IC50 = 10^y uM; "(6 - y) * 1.364" is upstream's kcal/mol conversion, written as given; pIC50
= 6 - y is derived from the definition. These are model estimates, not measurements, and the CSV header says so.
"""

from __future__ import annotations

import csv
import io
import json
import re
import tempfile
import zipfile
from pathlib import Path

from boltz_mlx.gui.common import ApiError

EST_NOTE = "model estimates, not lab measurements"
POSE_NAME = re.compile(r"\d{4}_(.+)")  # a screen's saved pose: <rank>_<record>.<ext> (screen/runner.py)
AFF_KEYS = ("affinity_pred_value", "affinity_probability_binary", "affinity_pred_value1", "affinity_probability_binary1",
            "affinity_pred_value2", "affinity_probability_binary2")
CONF_KEYS = ("confidence_score", "ptm", "iptm", "ligand_iptm", "protein_iptm", "complex_plddt", "complex_iplddt", "complex_pde",
             "complex_ipde")


def conversions(y) -> dict:
    """IC50 (uM), pIC50 and upstream's (6 - y) * 1.364 kcal/mol for an affinity_pred_value (None-safe)."""
    if y is None:
        return {"ic50_um_est": None, "pic50_est": None, "upstream_6_minus_y_x1364_kcal_per_mol": None}
    y = float(y)
    return {"ic50_um_est": 10 ** y, "pic50_est": 6 - y, "upstream_6_minus_y_x1364_kcal_per_mol": (6 - y) * 1.364}


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.6g}"
    return "" if v is None else v


def predict_csv(store, run_id: str) -> str:
    """One row per model of a structure job (its record, settings, confidence, affinity and conversions)."""
    from boltz_mlx.gui import predict as PR

    meta = store.read_meta(run_id) or {}
    sm = meta.get("summary") or {}
    buf = io.StringIO()
    buf.write(f"# affinity conversions are {EST_NOTE}; IC50 = 10^affinity_pred_value uM (Boltz-2 docs)\n")
    rows = []
    for k in PR.model_indices(store.predict_out(run_id)) or [0]:
        pred = store.prediction(run_id, k)
        o = pred.get("options") or {}
        conf, aff = pred.get("confidence") or {}, pred.get("affinity") or {}
        row = {"run_id": run_id, "name": meta.get("name") or run_id, "model": k, "chains": sm.get("chains_short") or "",
               "n_tokens": sm.get("n_tokens"), "binder": pred.get("binder") or "", "preset": o.get("preset"), "steps": o.get("steps"),
               "recycling": o.get("recycling"), "affinity_steps": o.get("affinity_steps") if aff else None, "affinity_samples": o.get("affinity_samples") if aff else None,
               "samples": o.get("samples"),
               "seed": o.get("seed")}
        row.update({key: conf.get(key) for key in CONF_KEYS})
        row.update({key: aff.get(key) for key in AFF_KEYS})
        row["affinity_scope"] = "run-level (pose model_0)" if aff else ""
        row.update(conversions(aff.get("affinity_pred_value")) if aff else conversions(None))
        rows.append(row)
    w = csv.DictWriter(buf, fieldnames=list(rows[0]), lineterminator="\n")
    w.writeheader()
    for row in rows:
        w.writerow({k: _fmt(v) for k, v in row.items()})
    return buf.getvalue()


def to_pdb(path: Path) -> str:
    """mmCIF -> PDB text (gemmi). PDB files are returned as they are."""
    if path.suffix == ".pdb":
        return path.read_text()
    import gemmi

    try:
        st = gemmi.read_structure(str(path))
        st.setup_entities()
        if not len(st) or not st[0].count_atom_sites():
            raise ValueError("no atoms read")
        return st.make_pdb_string()
    except Exception as ex:  # noqa: BLE001
        raise ApiError(500, f"could not convert {path.name} to PDB: {str(ex)[:200]}") from None


def to_mmcif(path: Path) -> str:
    """PDB -> mmCIF text (gemmi). mmCIF files are returned as they are."""
    if path.suffix != ".pdb":
        return path.read_text()
    import gemmi

    try:
        st = gemmi.read_structure(str(path))
        st.setup_entities()
        if not len(st) or not st[0].count_atom_sites():
            raise ValueError("no atoms read")
        return st.make_mmcif_document().as_string()
    except Exception as ex:  # noqa: BLE001
        raise ApiError(500, f"could not convert {path.name} to mmCIF: {str(ex)[:200]}") from None


def pose_bytes(path: Path, fmt: str | None) -> tuple[bytes, str]:
    """(content, extension) of a pose in the requested format: 'pdb' / 'mmcif' convert when the file is the other one,
    no format serves the file as written. The extension is that of the bytes returned."""
    if fmt == "pdb":
        return to_pdb(path).encode(), "pdb"
    if fmt in ("mmcif", "cif"):
        return to_mmcif(path).encode(), "cif"
    if fmt not in (None, ""):
        raise ApiError(400, "format must be mmcif or pdb")
    return path.read_bytes(), path.suffix.lstrip(".") or "cif"


MAX_INPUT_FILE = 128 * 1024 * 1024


def _portable_yaml(text: str, root: Path, yaml_path: Path) -> str:
    """A staged YAML with its file references made relative to the YAML itself (they were absolute paths into the run folder), so
    the exported inputs/ folder runs from wherever it is extracted."""
    import os

    import yaml

    from boltz_mlx.gui import predict as PR
    from boltz_mlx.gui.common import is_within

    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError:
        return text
    if not isinstance(doc, dict):
        return text
    moved = False
    for box, key, val in PR.yaml_refs(doc):
        p = Path(val)
        if p.is_absolute() and is_within(p, root):
            box[key] = Path(os.path.relpath(p.resolve(), yaml_path.resolve().parent)).as_posix()
            moved = True
    return yaml.safe_dump(doc, sort_keys=False, width=10**9) if moved else text


def _add_inputs(z, store, run_id: str, base: str) -> list[str]:
    """The staged inputs of a run (YAML / FASTA, uploaded MSAs, templates, ligand tables) under inputs/. A YAML's file references
    are rewritten relative to the YAML."""
    from boltz_mlx.gui.common import is_within

    rd = store.run_dir(run_id)
    root = rd / "inputs"
    names = []
    for f in sorted(root.rglob("*")) if root.is_dir() else []:
        if f.is_file() and not f.is_symlink() and is_within(f, root) and f.stat().st_size <= MAX_INPUT_FILE:
            rel = f.relative_to(rd).as_posix()
            if f.suffix in (".yaml", ".yml") and f.stat().st_size <= 8 * 1024 * 1024:
                z.writestr(f"{base}/{rel}", _portable_yaml(f.read_text(encoding="utf-8", errors="replace"), root, f))
            else:
                z.write(f, f"{base}/{rel}")
            names.append(rel)
    return names


def _rerun(meta: dict) -> str | None:
    """The shell command that repeats a structure job from the exported inputs/ folder (None for screens)."""
    from boltz_mlx.gui import predict as PR

    o, yml = meta.get("options") or {}, ((meta.get("inputs") or {}).get("yaml") or {}).get("name")
    if meta.get("kind") != "predict" or not yml or o.get("preset") not in PR.PRESET_FLAGS:
        return None
    args = PR.predict_argv(Path("."), {"yaml": {"file": yml}}, o)
    return "cd inputs && boltz-mlx predict " + " ".join(args)


def _manifest(store, run_id: str, kind: str, inputs: list[str], models: dict, raw: list[str], n_poses: int, rank_by: str | None) -> dict:
    """Everything needed to rerun or re-read the export: the resolved settings, the inputs it was started from, the files."""
    meta = store.read_meta(run_id) or {}
    cfg = {}
    if kind != "predict":
        try:
            cfg = store.screen_config(run_id)
        except Exception:  # noqa: BLE001
            cfg = {}
    return {"run_id": run_id, "kind": kind, "name": meta.get("name"), "created": meta.get("created"),
            "options": meta.get("options"), "resolved": cfg or None, "inputs": meta.get("inputs"), "summary": meta.get("summary"),
            "files": {"inputs": inputs, "raw_arrays": raw, "poses": n_poses, "model_files": models or None},
            "rank_by": rank_by if kind != "predict" else None, "rerun": _rerun(meta), "note": EST_NOTE + " (affinity numbers)"}


def build_zip(store, run_id: str, fmt: str | None = None, rank_by: str | None = None) -> tuple[Path, str]:
    """A temporary zip (the caller deletes it) with results.csv and the structures; (path, download name)."""
    from boltz_mlx.gui.common import is_within

    if fmt not in (None, "", "mmcif", "cif", "pdb"):
        raise ApiError(400, "format must be mmcif or pdb")
    meta = store.read_meta(run_id) or {}
    kind = meta.get("kind") or "screen"
    base = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in run_id)[:60]
    fd = tempfile.NamedTemporaryFile(prefix=".export-", suffix=".zip", delete=False)
    fd.close()
    path = Path(fd.name)
    n = 0
    models: dict = {}
    raw: list[str] = []
    rb = rank_by
    try:
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            if kind == "predict":
                from boltz_mlx.gui import predict as PR

                z.writestr(f"{base}/results.csv", predict_csv(store, run_id))
                out = store.predict_out(run_id)
                pd = PR.pred_dir(out)
                for f in sorted(pd.glob(f"{PR.RECORD}_model_*.*")) if pd.is_dir() else []:
                    if f.suffix in (".cif", ".pdb") and f.is_file() and not f.is_symlink() and is_within(f, out):
                        data, ext = pose_bytes(f, fmt)
                        z.writestr(f"{base}/structures/{f.stem}.{ext}", data)
                        models[f.stem.rsplit("_", 1)[-1]] = {"structure": f"structures/{f.stem}.{ext}", "confidence": f"confidence_{f.stem}.json"}
                        n += 1
                for pat in ("pae_*.npz", "plddt_*.npz", "pde_*.npz"):  # full-precision arrays, as Boltz wrote them
                    for f in sorted(pd.glob(pat)) if pd.is_dir() else []:
                        if f.is_file() and not f.is_symlink() and is_within(f, out):
                            z.write(f, f"{base}/raw/{f.name}")
                            raw.append(f"raw/{f.name}")
                for k, m in models.items():
                    for key in ("pae", "plddt", "pde"):
                        if f"raw/{key}_{PR.RECORD}_model_{k}.npz" in raw:
                            m[key] = f"raw/{key}_{PR.RECORD}_model_{k}.npz"
                for pat in (f"confidence_{PR.RECORD}_model_*.json", f"affinity_{PR.RECORD}.json"):
                    for f in sorted(pd.glob(pat)) if pd.is_dir() else []:
                        if f.is_file() and not f.is_symlink() and is_within(f, out):
                            z.write(f, f"{base}/{f.name}")
            else:
                from boltz_mlx.screen import results as R

                rb = rank_by or (meta.get("options") or {}).get("rank_by") or "prob"
                z.writestr(f"{base}/results.csv", store.csv(run_id, rb))  # checks rank_by (400)
                # the screen saved its poses as <rank>_<record> under ITS ranking; name them by this CSV's ranking instead
                rank = {str(r.get("record_id")): r["rank"] for r in R.rank_rows(store.rows(run_id).rows(), rb) if r.get("rank")}
                out = store.out_dir(run_id)
                sd = out / "structures"
                for f in sorted(sd.iterdir()) if sd.is_dir() else []:
                    if f.suffix in (".cif", ".pdb") and f.is_file() and not f.is_symlink() and is_within(f, out):
                        data, ext = pose_bytes(f, fmt)
                        m = POSE_NAME.fullmatch(f.stem)
                        rk = rank.get(m.group(1)) if m else None
                        stem = f"{rk:04d}_{m.group(1)}" if rk else f.stem
                        z.writestr(f"{base}/structures/{stem}.{ext}", data)
                        n += 1
                pdir = out / "work" / "predictions"  # PAE kept by 'Keep PAE': the original arrays, one per ligand
                for d in sorted(pdir.iterdir()) if pdir.is_dir() else []:
                    for key in ("pae", "plddt", "pde"):   # whatever the screen kept for this ligand
                        f = d / f"{key}_{d.name}_model_0.npz"
                        if d.is_dir() and f.is_file() and not f.is_symlink() and is_within(f, out) and not d.is_symlink():
                            z.write(f, f"{base}/raw/{key}_{d.name}.npz")
                            raw.append(f"raw/{key}_{d.name}.npz")
                            models.setdefault(d.name, {})[key] = f"raw/{key}_{d.name}.npz"
            inputs = _add_inputs(z, store, run_id, base)
            z.writestr(f"{base}/manifest.json", json.dumps(_manifest(store, run_id, kind, inputs, models, raw, n, rb), indent=1, default=str))
            note = (f"Boltz-2 MLX export of run {run_id}.\nmanifest.json: resolved settings, input and output file list.\n"
                    f"inputs/: what the run started from (YAML, MSAs, ligand table); file references in the YAML are relative to it.\n"
                    + (f"Repeat the job from this folder: {_rerun(meta)}\n" if _rerun(meta) else "")
                    + ("raw/: full-precision PAE, pLDDT and PDE arrays (.npz) as kept.\n" if raw else "")
                    + f"results.csv: the scores; affinity numbers are {EST_NOTE}.\n"
                    f"structures/: {n} pose file(s)" + ("" if n else " (none were saved: set 'Save poses' higher to keep them)")
                    + (f"; each file name starts with the ligand's rank in results.csv (ranked by "
                       f"{'probability' if rb == 'prob' else 'affinity value'})" if n and kind != "predict" else "") + ".\n")
            z.writestr(f"{base}/README.txt", note)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path, f"{base}_export.zip"
