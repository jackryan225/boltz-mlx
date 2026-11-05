"""The browser's JSON payload -> files in a run directory, validated with the screen's own input code, and the screen CLI argv.

Payload (all text; the browser reads the chosen files itself):
  target: {"kind": "file"|"text", "filename": "t.yaml", "content": "..."}
  msa:    {"mode": "default"|"empty"|"file", "filename": "m.a3m", "content": "..."}      (default: as in the target YAML)
  ligands:{"kind": "file"|"text", "filename": "l.csv", "content": "..."}
  options:{"preset", "rank_by", "chunk", "save_structures", "seed", "max_minutes", "pocket", "ligand_chain", "id_col", "smiles_col",
           "steps", "recycling", "affinity_steps" (None = the preset's), "keep_work" (keep PAE matrices: --keep-work)}
Uploaded content is stored under fixed names (inputs/target.<ext>, ...); the user's file name is sanitised and kept for display.
Every option reaches the screen as `--flag=value` (never a separate token), so a value cannot be read as another flag.
"""

from __future__ import annotations

import argparse
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

from boltz_mlx.gui.common import ALLOWED_EXT, LIMITS, ApiError, safe_filename

# preset None: the GUI Screen default (Standard: the screen preset on M5-class GPUs, strict on M1-M4)
OPTION_DEFAULTS = {
    "preset": None, "rank_by": "prob", "chunk": 50, "save_structures": "10", "seed": 0, "max_minutes": None,
    "pocket": [], "ligand_chain": None, "id_col": None, "smiles_col": None,
    "steps": None, "recycling": None, "affinity_steps": None, "keep_work": False,
}
STEP_LIMITS = {"steps": (1, 1000), "recycling": (0, 20), "affinity_steps": (1, 1000)}
POCKET_RE = re.compile(r"^[A-Za-z0-9]{1,4}:\d{1,6}(,[A-Za-z0-9]{1,4}:\d{1,6})*$")
CHAIN_RE = re.compile(r"^[A-Za-z0-9]{1,4}$")
COL_RE = re.compile(r"^[^\x00-\x1f]{1,64}$")


@dataclass
class Staged:
    target: Path
    msa: Path | None
    ligands: Path
    msa_mode: str                   # default | empty | file
    names: dict                     # display names of the uploaded files


def _text(spec: object, what: str, limit: int) -> tuple[str, str, str]:
    """(content, display filename, kind) of a file-or-text payload entry."""
    if not isinstance(spec, dict):
        raise ApiError(400, f"{what} is required")
    content = spec.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ApiError(400, f"{what}: no content (choose a file or paste text)")
    if len(content) > limit:
        raise ApiError(413, f"{what} is too large ({len(content) / 1e6:.1f} MB; the limit is {limit / 1e6:.0f} MB)")
    if "\x00" in content[:4096]:
        raise ApiError(400, f"{what}: looks like a binary file; expected text")
    name, _ext = safe_filename(spec.get("filename"))
    return content, name, str(spec.get("kind") or "file")


def sniff_ligand_text(text: str) -> str:
    """Extension for pasted ligand text: .sdf (has an M  END record), .csv / .tsv (a delimited header naming a SMILES column),
    otherwise .smi ("SMILES [name]" per line)."""
    if "M  END" in text:
        return ".sdf"
    first = next((ln for ln in text.splitlines() if ln.strip()), "")
    low = first.lower()
    if "smiles" in low or "smi" in low.split():
        if "\t" in first:
            return ".tsv"
        if "," in first:
            return ".csv"
    return ".smi"


def sniff_target_text(text: str) -> str:
    s = text.lstrip()
    if s.startswith(">"):
        return ".fasta"
    if re.search(r"^\s*sequences\s*:", text, re.M):
        return ".yaml"
    return ".seq"


def clean_sequence(text: str) -> str:
    """A pasted amino-acid sequence: whitespace and digits (UniProt-style numbering) removed, upper case."""
    from boltz_mlx.screen.inputs import AA

    seq = re.sub(r"[\s\d]+", "", text).upper()
    bad = sorted(set(seq) - AA)
    if bad:
        raise ApiError(400, f"target: not an amino-acid sequence (characters {''.join(bad)!r}); paste a sequence, FASTA or Boltz YAML")
    if not seq:
        raise ApiError(400, "target: empty sequence")
    return seq


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def stage_target(spec: object, dest: Path, library=None) -> tuple[Path, str]:
    """Write the target (YAML, FASTA or a pasted sequence) as dest/target.<ext>, or copy a saved target from the library
    ({"kind": "saved", "id": ...}). Returns (path, display name)."""
    if isinstance(spec, dict) and spec.get("kind") == "saved":
        if library is None:
            raise ApiError(400, "target: saved targets are not available here")
        path, meta = library.stage_into(str(spec.get("id") or ""), dest)
        return path, f"saved target: {meta['name']}"
    content, name, kind = _text(spec, "target", LIMITS["target"])
    ext = safe_filename(name)[1]
    if kind == "text" or not ext:
        ext = sniff_target_text(content)
    elif ext not in ALLOWED_EXT["target"]:
        raise ApiError(400, f"target: unsupported file type {ext!r} (use .yaml, .yml, .fasta, .fa)")
    if ext == ".seq":
        content, ext = f">A\n{clean_sequence(content)}\n", ".fasta"
        shown = "pasted sequence"
    else:
        shown = name if kind != "text" else "pasted " + ("FASTA" if ext == ".fasta" else "YAML")
    dest.mkdir(parents=True, exist_ok=True)
    if ext in (".yaml", ".yml") and isinstance(spec, dict) and spec.get("files"):
        content = stage_companions(spec["files"], content, dest)
    path = dest / f"target{ext}"
    _write(path, content)
    return path, shown


def stage_companions(raw_files, content: str, dest: Path, prefix: str = "tf_") -> str:
    """A target YAML with the MSA / template files it names, attached by the page: written next to the YAML as <prefix><name>
    (after the same checks as a Predict YAML input) and the YAML's references pointed at them. Returns the YAML text."""
    import yaml

    from boltz_mlx.gui import predict as PR

    try:
        doc = yaml.safe_load(content)
    except yaml.YAMLError:
        return content    # the screen's own loader reports it
    if not isinstance(doc, dict) or not PR.yaml_refs(doc):
        return content
    files = PR.read_companions(raw_files, "target YAML")
    PR.validate_companions(doc, files, "target YAML")
    for name, text in files.items():
        _write(dest / f"{prefix}{name}", text)
    return yaml.safe_dump(PR.rewrite_refs(doc, "", prefix), sort_keys=False, width=10**9)


def stage_msa(spec: object, dest: Path) -> tuple[str, Path | None, str | None]:
    """(mode, path, display name): mode default (as in the target YAML) | empty | file."""
    spec = spec if isinstance(spec, dict) else {}
    mode = str(spec.get("mode") or "default")
    if mode not in ("default", "empty", "file"):
        raise ApiError(400, "msa: mode must be default, empty or file")
    if mode != "file":
        return mode, None, None
    content, name, _ = _text(spec, "MSA", LIMITS["msa"])
    ext = safe_filename(name)[1]
    if ext not in ALLOWED_EXT["msa"]:
        raise ApiError(400, f"MSA: unsupported file type {ext!r} (use .a3m or .csv)")
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"msa{ext}"
    _write(path, content)
    return mode, path, name


def stage_ligands(spec: object, dest: Path) -> tuple[Path, str]:
    content, name, kind = _text(spec, "ligands", LIMITS["ligands"])
    ext = safe_filename(name)[1]
    if kind == "text" or not ext:
        ext, shown = sniff_ligand_text(content), "pasted SMILES"
    elif ext not in ALLOWED_EXT["ligands"]:
        raise ApiError(400, f"ligands: unsupported file type {ext!r} (use .csv, .tsv, .smi or .sdf)")
    else:
        shown = name
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"ligands{ext}"
    _write(path, content)
    return path, shown


def is_saved(spec: object) -> bool:
    return isinstance(spec, dict) and spec.get("kind") == "saved"


def stage_rerun(parent_rd: Path, parent_inputs: dict, dest: Path) -> tuple[Path, str, str, Path | None, str | None]:
    """Copy a finished screen's staged target (and its MSA / template files) into dest, for a re-check run.
    Returns (target path, target name, msa mode, msa path, msa name). Ligand files are not copied."""
    import shutil

    src = parent_rd / "inputs"
    tfile = Path(str((parent_inputs.get("target") or {}).get("file") or "")).name
    if not tfile or not (src / tfile).is_file():
        raise ApiError(400, "re-check: the original run's target file is missing")
    lig = Path(str((parent_inputs.get("ligands") or {}).get("file") or "")).name
    dest.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file() and not f.is_symlink() and f.name != lig and not f.name.startswith("ligands"):
            shutil.copyfile(f, dest / f.name)
    m = parent_inputs.get("msa") or {}
    mode = str(m.get("mode") or "default")
    mpath = dest / Path(str(m["file"])).name if mode == "file" and m.get("file") else None
    return dest / tfile, str((parent_inputs.get("target") or {}).get("name") or tfile), mode, mpath, m.get("name")


def stage_inputs(payload: dict, dest: Path, library=None, rerun=None) -> Staged:
    """Write the payload's files into `dest` (inputs/) and return their paths. Raises ApiError(400/413).
    A saved target brings its own MSA (or none) in its YAML: the payload's msa is ignored for it. `rerun` =
    (parent run dir, parent inputs record): the target and MSA are copied from that run (server-side only)."""
    if not isinstance(payload, dict):
        raise ApiError(400, "expected a JSON object")
    if rerun is not None:
        tpath, tname, mode, mpath, mname = stage_rerun(rerun[0], rerun[1], dest)
        lpath, lname = stage_ligands(payload.get("ligands"), dest)
        names = {"target": tname, "ligands": lname}
        if mname:
            names["msa"] = mname
        saved = (rerun[1].get("target") or {}).get("saved")
        if saved:
            names["saved_target"] = saved
        return Staged(target=tpath, msa=mpath, ligands=lpath, msa_mode=mode, names=names)
    tpath, tname = stage_target(payload.get("target"), dest, library)
    if is_saved(payload.get("target")):
        mode, mpath, mname = "default", None, None
    else:
        mode, mpath, mname = stage_msa(payload.get("msa"), dest)
    lpath, lname = stage_ligands(payload.get("ligands"), dest)
    names = {"target": tname, "ligands": lname}
    if mname:
        names["msa"] = mname
    if is_saved(payload.get("target")):
        names["saved_target"] = str(payload["target"].get("id"))
    return Staged(target=tpath, msa=mpath, ligands=lpath, msa_mode=mode, names=names)


def inputs_meta(staged: Staged, rd: Path) -> dict:
    """The `inputs` record stored in gui_run.json: paths relative to the run directory + the names shown to the user."""
    rel = lambda p: str(Path(p).relative_to(rd))  # noqa: E731
    d = {"target": {"file": rel(staged.target), "name": staged.names["target"]},
         "ligands": {"file": rel(staged.ligands), "name": staged.names["ligands"]},
         "msa": {"mode": staged.msa_mode}}
    if staged.msa is not None:
        d["msa"].update(file=rel(staged.msa), name=staged.names.get("msa"))
    if staged.names.get("saved_target"):
        d["target"]["saved"] = staged.names["saved_target"]
    return d


def normalize_options(raw: object, presets: dict) -> dict:
    """Validated, typed options (defaults filled in). Raises ApiError(400)."""
    raw = raw if isinstance(raw, dict) else {}
    o = dict(OPTION_DEFAULTS)
    o.update({k: raw[k] for k in OPTION_DEFAULTS if k in raw and raw[k] not in (None, "")})
    from boltz_mlx.gui import presets as GP

    o["preset"] = GP.to_cli(o["preset"], GP.default_key())  # GUI keys (standard / reference) -> CLI preset names
    if o["preset"] not in presets:
        raise ApiError(400, f"unknown preset {o['preset']!r}; one of {sorted(presets)}")
    if o["rank_by"] not in ("prob", "value"):
        raise ApiError(400, "rank_by must be 'prob' or 'value'")
    try:
        o["chunk"] = int(o["chunk"])
        o["seed"] = int(o["seed"])
    except (TypeError, ValueError):
        raise ApiError(400, "chunk and seed must be integers") from None
    if not 1 <= o["chunk"] <= 10000:
        raise ApiError(400, "chunk must be between 1 and 10000")
    if not -2**31 <= o["seed"] < 2**31:
        raise ApiError(400, "seed out of range")
    ss = str(o["save_structures"]).strip().lower()
    if ss != "all":
        try:
            if int(ss) < 0:
                raise ValueError
        except ValueError:
            raise ApiError(400, "save_structures must be 'all' or a number >= 0") from None
        ss = str(int(ss))
    o["save_structures"] = ss
    if o["max_minutes"] is not None:
        try:
            o["max_minutes"] = float(o["max_minutes"])
        except (TypeError, ValueError):
            raise ApiError(400, "max_minutes must be a number") from None
        if not 0 < o["max_minutes"] <= 1_000_000:
            raise ApiError(400, "max_minutes must be positive")
    pocket = o["pocket"]
    if isinstance(pocket, str):
        pocket = [p for p in re.split(r"[\s;]+", pocket.strip()) if p]
    if not isinstance(pocket, list) or len(pocket) > LIMITS["pocket"]:
        raise ApiError(400, "pocket must be a list of CHAIN:RESIDUE[,CHAIN:RESIDUE...] items")
    for p in pocket:
        if not isinstance(p, str) or not POCKET_RE.match(p):
            raise ApiError(400, f"pocket {p!r}: expected CHAIN:RESIDUE_INDEX (e.g. A:45,A:67)")
    o["pocket"] = pocket
    if o["ligand_chain"] is not None and not CHAIN_RE.match(str(o["ligand_chain"])):
        raise ApiError(400, "ligand_chain must be 1-4 letters or digits")
    for k in ("id_col", "smiles_col"):
        if o[k] is not None and not COL_RE.match(str(o[k])):
            raise ApiError(400, f"{k}: bad column name")
    for k, (lo, hi) in STEP_LIMITS.items():
        if o[k] is not None:
            try:
                o[k] = int(o[k])
            except (TypeError, ValueError):
                raise ApiError(400, f"{k} must be a whole number") from None
            if not lo <= o[k] <= hi:
                raise ApiError(400, f"{k} must be between {lo} and {hi}")
    o["keep_work"] = o["keep_work"] in (True, "true", "1", 1)
    return o


def screen_argv(rd: Path, inputs: dict, options: dict, out: Path, extra: list[str] | None = None,
                retry_errors: bool = False) -> list[str]:
    """Arguments after `screen` for a run (paths absolute, every value attached with '=')."""
    a = [f"--target={rd / inputs['target']['file']}", f"--ligands={rd / inputs['ligands']['file']}", f"--out={out}",
         f"--preset={options['preset']}", f"--rank-by={options['rank_by']}", f"--save-structures={options['save_structures']}",
         f"--chunk={options['chunk']}", f"--seed={options['seed']}"]
    msa = inputs.get("msa") or {}
    if msa.get("mode") == "empty":
        a.append("--msa=empty")
    elif msa.get("mode") == "file":
        a.append(f"--msa={rd / msa['file']}")
    for p in options.get("pocket") or []:
        a.append(f"--pocket={p}")
    if options.get("ligand_chain"):
        a.append(f"--ligand-chain={options['ligand_chain']}")
    if options.get("max_minutes"):
        a.append(f"--max-minutes={options['max_minutes']}")
    for k, flag in (("id_col", "--id-col"), ("smiles_col", "--smiles-col")):
        if options.get(k):
            a.append(f"{flag}={options[k]}")
    for k, flag in (("steps", "--steps"), ("recycling", "--recycling"), ("affinity_steps", "--affinity-steps")):
        if options.get(k) is not None:
            a.append(f"{flag}={options[k]}")
    if options.get("keep_work"):
        a.append("--keep-work")
    if retry_errors:
        a.append("--retry-errors")
    return a + list(extra or [])


def parse_screen_argv(argv: list[str]) -> argparse.Namespace:
    """Parse with the screen's own parser (so defaults, choices and types are the CLI's). Raises ApiError(400)."""
    from boltz_mlx.screen.main import add_parser

    ap = argparse.ArgumentParser(prog="boltz-mlx", add_help=False)
    sp = add_parser(ap.add_subparsers(dest="cmd"))

    def fail(message):
        raise ApiError(400, f"invalid screen options: {message}")

    sp.error = fail
    return sp.parse_args(argv)


def load_target_checked(args: argparse.Namespace):
    from boltz_mlx.screen.inputs import InputError, load_target

    try:
        t = load_target(args.target, msa=args.msa, pocket=args.pocket, ligand_chain=args.ligand_chain)
    except InputError as ex:
        raise ApiError(400, f"target: {ex}") from None
    except Exception as ex:  # noqa: BLE001  (YAML syntax errors, odd documents)
        raise ApiError(400, f"target could not be read: {type(ex).__name__}: {str(ex)[:300]}") from None
    _check_msas(t)
    return t


def _check_msas(target) -> None:
    """Parse every alignment the target uses and check its query against the chain (before a run fails on it)."""
    from boltz_mlx.gui.predict import check_msa

    for ent in target.doc.get("sequences", []):
        body = ent.get("protein") if isinstance(ent, dict) else None
        msa = body.get("msa") if isinstance(body, dict) else None
        if not msa or msa == "empty" or not isinstance(body.get("sequence"), str):
            continue
        p = Path(str(msa))
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        who = "MSA" if len(target.doc.get("sequences", [])) == 1 else f"chain {body.get('id')}: MSA"
        check_msa(text, p.suffix.lower(), body["sequence"].upper(), who)


def load_ligands_checked(args: argparse.Namespace):
    from boltz_mlx.screen.inputs import InputError, read_ligands, validate_ligands

    try:
        return validate_ligands(read_ligands(args.ligands, args.id_col, args.smiles_col))
    except InputError as ex:
        raise ApiError(400, f"ligands: {ex}") from None
    except Exception as ex:  # noqa: BLE001
        raise ApiError(400, f"ligands could not be read: {type(ex).__name__}: {str(ex)[:300]}") from None


def load_inputs(args: argparse.Namespace):
    """(target, ligands) loaded and validated by the screen's own code, as `boltz-mlx screen` does."""
    return load_target_checked(args), load_ligands_checked(args)


def plain_warning(w: str) -> str:
    """A screen-target warning worded for the form: the command line's advice (--msa <a3m>) becomes the MSA choice on the page."""
    m = re.match(r"protein chain\(s\) \[(.*?)\] have no MSA", w)
    if m:
        ids = ", ".join(re.findall(r"'([^']*)'", m.group(1))) or "?"
        return f"chain {ids}: no MSA (single sequence): less accurate; an MSA file (.a3m / .csv) improves it"
    return w


def target_summary(target) -> dict:
    chains = []
    for s in target.doc.get("sequences", []):
        for kind, v in s.items():
            ids = v.get("id")
            ids = [ids] if isinstance(ids, str) else list(ids or [])
            seq = v.get("sequence") if kind != "ligand" else None
            chains.append({"type": kind, "ids": ids, "length": len(seq) if seq else None,
                           "msa": v.get("msa") if kind == "protein" else None})
    for c in chains:
        if c["msa"] not in (None, "empty"):
            c["msa"] = Path(str(c["msa"])).name
    return {"source": Path(target.source).name if target.source != "sequence" else "sequence", "n_residues": target.n_residues,
            "ligand_chain": target.ligand_chain, "msa_source": target.msa_source, "warnings": [plain_warning(w) for w in target.warnings],
            "chains": chains, "pocket": [c["pocket"] for c in target.doc.get("constraints", []) if "pocket" in c]}


def ligand_rows(ligs, max_rows: int) -> tuple[list[dict], dict]:
    from boltz_mlx.screen.inputs import MAX_ATOMS, WARN_ATOMS

    rows = []
    n_invalid = n_warn = n_large = n_dup = n_other = 0
    for l in ligs:
        size = None
        if l.n_atoms is not None:
            size = "too_large" if l.n_atoms > MAX_ATOMS else ("large" if l.n_atoms > WARN_ATOMS else "ok")
        if l.status == "invalid":
            n_invalid += 1
            if l.error and l.error.startswith("duplicate"):
                n_dup += 1
        elif l.warnings:
            n_warn += 1
            n_other += size != "large"     # a large ligand is counted once, as large
        n_large += size == "large"
        if len(rows) < max_rows:
            rows.append({"index": l.index, "id": l.ligand_id, "smiles": l.smiles, "smiles_used": l.smiles_used if l.status != "invalid" else None,
                         "status": "invalid" if l.status == "invalid" else ("warn" if l.warnings else "ok"), "error": l.error,
                         "warnings": list(l.warnings), "n_atoms": l.n_atoms, "n_heavy": l.n_heavy, "mw": None if l.mw is None else round(l.mw, 2), "size": size})
    sizes = [l.n_atoms for l in ligs if l.status != "invalid" and l.n_atoms]
    unique = {l.ligand_id for l in ligs}
    valid_unique = {l.ligand_id for l in ligs if l.status != "invalid"}
    counts = {"total": len(ligs), "unique": len(unique), "valid": len([l for l in ligs if l.status != "invalid"]),
              "valid_unique": len(valid_unique), "invalid": n_invalid, "duplicates": n_dup, "warn": n_warn, "warn_other": n_other, "large": n_large,
              "median_atoms": int(statistics.median(sizes)) if sizes else None, "max_atoms": max(sizes) if sizes else None,
              "truncated": len(ligs) > max_rows, "shown": len(rows)}
    return rows, counts
