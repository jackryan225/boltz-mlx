"""Single-structure jobs: the browser's chain list -> a Boltz YAML in the run directory -> a `boltz-mlx predict` subprocess.

Payload (all text; the browser reads any MSA file itself):
  {"kind": "predict", "name": "...",
   "chains": [{"type": "protein", "id": "A", "sequence": "MKT...", "cyclic": false,
               "msa": {"mode": "empty"|"file"|"server", "filename": "a.a3m", "content": "..."}},
              {"type": "dna"|"rna", "id": "C", "sequence": "ACGT..."},
              {"type": "ligand", "id": "B", "smiles": "CCO"} | {"type": "ligand", "id": "B", "ccd": "SAH"}],
   "affinity": {"binder": "B"} | null,       (any chain may also carry "copies": n, 2-26: one YAML entity `id: [A, C, D]`, the
                                                extra ids are the first free letters; not for the affinity ligand)
   "constraints": [{"pocket": {...}} | {"contact": {...}} | {"bond": {...}}],   (Boltz-2 YAML form; checked in gui/constraints.py)
   "templates": [{"filename": "t.pdb", "content": "...", "chain_id": "A", "template_id": "B" | null}],
                                                (a protein / DNA / RNA chain may carry "modifications": [{"position", "ccd"}])
   "options": {"preset": "strict", "steps": null, "recycling": null, "affinity_steps": null, "samples": 1, "seed": 0}}

The YAML is always written as inputs/job.yaml (record id "job"), protein MSAs as inputs/msa_<chain>.<ext>, templates (one chain
each, ligands and waters removed) as inputs/template_<k>.cif; proteins without an
uploaded MSA run `msa: empty`; the MSA server is contacted only for a chain whose mode is "server" (job runs with
--use_msa_server, which sends just those protein sequences to api.colabfold.com). Every option reaches `predict` as `--flag=value`.
Outputs (boltz-mlx predict layout): out/boltz_results_job/predictions/job/{job_model_0.cif, confidence_job_model_0.json,
pae_job_model_0.npz, plddt_job_model_0.npz, affinity_job.json}, then out/boltz_results_job/boltz_mlx_timings.json at the end.
"""

from __future__ import annotations

import re
from pathlib import Path

from boltz_mlx.gui import constraints as CX
from boltz_mlx.gui.common import ALLOWED_EXT, LIMITS, ApiError, safe_filename

RECORD = "job"
CHAIN_TYPES = ("protein", "dna", "rna", "ligand")
CHAIN_ID_RE = re.compile(r"^[A-Za-z0-9]{1,4}$")
CCD_RE = re.compile(r"^[A-Za-z0-9]{1,5}$")
NA = {"dna": set("ACGTN"), "rna": set("ACGUN")}
MAX_CHAINS = 26
MAX_TOKENS = 5000          # the M5 Pro 24 GB ran prot1000 at 16 GB peak; larger jobs are refused rather than swapped
WARN_TOKENS = 1500
PREDICT_OPTION_DEFAULTS = {"preset": None, "steps": None, "recycling": None, "affinity_steps": None, "affinity_samples": None,
                           "samples": 1, "seed": 0}
AFFINITY_RECYCLES = 5      # fixed by `predict` (boltz_mlx/cli.py): the affinity pass always runs 5 recycles
# preset -> numerics flags of `boltz-mlx predict` (the step counts are always passed explicitly)
PRESET_FLAGS = {"strict": ["--precision=fp32"], "turbo": ["--turbo"], "f16": ["--precision=f16"], "screen": ["--precision=f16"]}


def _int(v, what: str, lo: int, hi: int) -> int:
    try:
        x = int(v)
    except (TypeError, ValueError):
        raise ApiError(400, f"{what} must be a whole number") from None
    if not lo <= x <= hi:
        raise ApiError(400, f"{what} must be between {lo} and {hi}")
    return x


def normalize_predict_options(raw: object, presets: dict, default_preset: str) -> dict:
    raw = raw if isinstance(raw, dict) else {}
    o = dict(PREDICT_OPTION_DEFAULTS)
    o.update({k: raw[k] for k in PREDICT_OPTION_DEFAULTS if k in raw and raw[k] not in (None, "")})
    from boltz_mlx.gui import presets as GP

    o["preset"] = GP.to_cli(str(o["preset"] or default_preset), default_preset)  # GUI keys -> CLI preset names
    if o["preset"] not in presets or o["preset"] not in PRESET_FLAGS:
        raise ApiError(400, f"unknown preset {o['preset']!r}; one of {sorted(set(presets) & set(PRESET_FLAGS))}")
    p = presets[o["preset"]]
    o["steps_explicit"] = o["steps"] is not None   # an explicit step count always beats Fast's tiny-ligand rule
    o["steps"] = _int(o["steps"], "structure steps", 1, 1000) if o["steps"] is not None else p.steps
    o["recycling"] = _int(o["recycling"], "recycling steps", 0, 20) if o["recycling"] is not None else p.recycling
    o["affinity_steps"] = (_int(o["affinity_steps"], "affinity steps", 1, 1000) if o["affinity_steps"] is not None
                           else p.affinity_steps)
    o["affinity_samples"] = (_int(o["affinity_samples"], "affinity samples", 1, 25) if o["affinity_samples"] is not None
                             else p.affinity_samples)
    o["samples"] = _int(o["samples"], "structure samples", 1, 25)
    o["seed"] = _int(o["seed"], "seed", -2**31, 2**31 - 1)
    o["changed"] = [k for k, d in (("steps", p.steps), ("recycling", p.recycling), ("affinity_steps", p.affinity_steps),
                                 ("affinity_samples", p.affinity_samples))
                    if o[k] != d]
    return o


def apply_tiny_ligand_rule(options: dict, checked: dict) -> dict:
    """Fast (CLI "screen") with no explicit structure steps: a ligand with fewer than 10 heavy atoms makes the job run 200 structure
    steps (boltz_mlx/tiny_ligand.py). Sets options["steps"], ["steps_reason"] (None when the rule did not fire) and
    ["steps_auto"]; the step count stays out of options["changed"] (it is the preset's own rule, not a custom setting)."""
    from boltz_mlx import tiny_ligand as TL

    options["steps_reason"], options["steps_auto"] = None, False
    if options["preset"] != TL.RULE_PRESET or options.get("steps_explicit"):
        return options
    sizes = [TL.LigandSize(f"ligand {c['id']}", "ccd" if c.get("ccd") else "smiles", c.get("ccd") or c.get("smiles") or "", c.get("n_heavy"))
             for c in checked["chains"] if c["type"] == "ligand"]
    steps, why = TL.decide(sizes, options["steps"])
    if why:
        options.update(steps=steps, steps_reason=why, steps_auto=True)
        options["changed"] = [k for k in options["changed"] if k != "steps"]
    return options


def _auto_ids(chains: list[dict]) -> None:
    used = {str(c.get("id") or "").strip() for c in chains}
    letters = (chr(c) for c in range(ord("A"), ord("Z") + 1))
    for c in chains:
        if not str(c.get("id") or "").strip():
            c["id"] = next(l for l in letters if l not in used)
            used.add(c["id"])


def _spare_ids(chains: list[dict]):
    """Chain ids no chain uses (A-Z, then 0-9, then two-letter ones), for the extra copies of a chain."""
    import itertools
    import string

    used = {str(c.get("id") or "").strip().lower() for c in chains}
    for n in (1, 2):
        for t in itertools.product(string.ascii_uppercase + string.digits, repeat=n):
            i = "".join(t)
            if i.lower() not in used:
                used.add(i.lower())
                yield i


def _copies(v, where: str) -> int:
    try:
        n = 1 if v in (None, "") else int(v)
    except (TypeError, ValueError):
        raise ApiError(400, f"{where}: copies must be a whole number") from None
    if not 1 <= n <= MAX_CHAINS:
        raise ApiError(400, f"{where}: copies must be between 1 and {MAX_CHAINS}")
    return n


def check_msa(content: str, ext: str, sequence: str, where: str) -> int:
    """Parse an uploaded alignment (.a3m or Boltz .csv) and check it against the chain: the first row is the query and must be
    the chain's sequence, every row must align to the same length. Returns the number of rows. Raises ApiError(400)."""
    rows: list[str] = []
    if ext == ".csv":
        import csv
        import io

        rd = csv.reader(io.StringIO(content))
        head = next(rd, None)
        if not head or [h.strip().lower() for h in head][:2] != ["key", "sequence"]:
            raise ApiError(400, f"{where}: MSA csv needs the header 'key,sequence'")
        rows = [r[1].strip() for r in rd if len(r) >= 2 and r[1].strip()]
    else:
        cur: list[str] | None = None
        for line in content.splitlines():
            if line.startswith(">"):
                if cur is not None:
                    rows.append("".join(cur))
                cur = []
            elif line.startswith("#"):
                continue
            elif cur is not None:
                cur.append(line.strip())
            elif line.strip():
                raise ApiError(400, f"{where}: MSA is not a3m (sequence before the first '>' header)")
        if cur is not None:
            rows.append("".join(cur))
    rows = [r for r in rows if r]
    if not rows:
        raise ApiError(400, f"{where}: the MSA has no sequences")
    query = re.sub(r"[a-z\-]", "", rows[0]).upper()
    if query != sequence:
        raise ApiError(400, f"{where}: MSA query ({len(query)} residues) is not this chain's sequence ({len(sequence)})")
    width = len(re.sub(r"[a-z]", "", rows[0]))
    for i, r in enumerate(rows[1:], 2):
        w = len(re.sub(r"[a-z]", "", r))
        if w != width:
            raise ApiError(400, f"{where}: MSA row {i} aligns to {w} columns, the query to {width}")
    return len(rows)


def _mods(raw: dict, d: dict, where: str, cache_dir: Path | None, warnings: list[str] | None = None) -> None:
    mods = CX.check_modifications(raw.get("modifications"), d, where, cache_dir / "mols" if cache_dir is not None else None)
    if mods:
        d["modifications"] = mods
        if warnings is not None and d["type"] == "protein":
            warnings += CX.modification_warnings(d, where)


def check_chains(payload: dict, cache_dir: Path | None = None) -> dict:
    """Validate the chain list (+ affinity binder). Returns {"chains": [normalised chains with per-chain info], "n_tokens",
    "warnings", "affinity", "binder"}. Raises ApiError(400) on anything that would make `predict` fail."""
    from boltz_mlx.gui.inputs import clean_sequence
    from boltz_mlx.screen.inputs import Ligand, validate_ligand

    raw = payload.get("chains")
    if not isinstance(raw, list) or not raw:
        raise ApiError(400, "add at least one chain (a protein sequence, DNA / RNA, or a ligand)")
    if len(raw) > MAX_CHAINS:
        raise ApiError(400, f"at most {MAX_CHAINS} chains")
    chains = [dict(c) if isinstance(c, dict) else {} for c in raw]
    _auto_ids(chains)
    spare = _spare_ids(chains)
    aff = payload.get("affinity") if isinstance(payload.get("affinity"), dict) else None
    binder = str(aff.get("binder") or "").strip() if aff else ""
    out, warnings, seen = [], [], set()
    tokens = 0
    for k, c in enumerate(chains, 1):
        t = str(c.get("type") or "").lower()
        cid = str(c.get("id")).strip()
        where = f"chain {cid}"
        if t not in CHAIN_TYPES:
            raise ApiError(400, f"chain {k}: type must be one of {', '.join(CHAIN_TYPES)}")
        if not CHAIN_ID_RE.match(cid):
            raise ApiError(400, f"chain {k}: id {cid!r} must be 1-4 letters or digits")
        if cid.lower() in seen:
            raise ApiError(400, f"chain id {cid} is used twice")
        seen.add(cid.lower())
        d: dict = {"type": t, "id": cid}
        t0 = tokens
        copies = _copies(c.get("copies"), where)
        if copies > 1:   # one YAML entity with a list of ids: the first is this chain's, the others are free letters
            if cid == binder:
                raise ApiError(400, f"{where}: the affinity ligand is a single copy")
            d["copies"], d["ids"] = copies, [cid] + [next(spare) for _ in range(copies - 1)]
        if t == "protein":
            seq = str(c.get("sequence") or "")
            if seq.lstrip().startswith(">"):  # one FASTA record pasted into the box: its header is dropped; several are refused
                recs = [l for l in seq.splitlines() if l.startswith(">")]
                if len(recs) > 1:
                    raise ApiError(400, f"{where}: {len(recs)} FASTA records in one chain; use one chain per record")
                seq = "\n".join(l for l in seq.splitlines() if not l.startswith(">"))
            try:
                d["sequence"] = clean_sequence(seq)
            except ApiError as ex:
                msg = ex.message.replace("target: ", "").replace("; paste a sequence, FASTA or Boltz YAML", "")
                raise ApiError(400, f"{where}: {msg}") from None
            d["cyclic"] = bool(c.get("cyclic"))
            msa = c.get("msa") if isinstance(c.get("msa"), dict) else {}
            if (msa.get("mode") or "empty") == "file":
                content = msa.get("content")
                if not isinstance(content, str) or not content.strip():
                    raise ApiError(400, f"{where}: choose the MSA file, or run it without an MSA")
                if len(content) > LIMITS["msa"]:
                    raise ApiError(413, f"{where}: MSA too large (limit {LIMITS['msa'] // (1024 * 1024)} MB)")
                name, ext = safe_filename(msa.get("filename"))
                if ext not in ALLOWED_EXT["msa"]:
                    raise ApiError(400, f"{where}: MSA must be .a3m or .csv")
                check_msa(content, ext, d["sequence"], where)
                d["msa"] = {"mode": "file", "name": name, "ext": ext, "content": content}
            elif msa.get("mode") == "server":
                d["msa"] = {"mode": "server"}
                warnings.append(f"{where}: the sequence is sent to the public MSA server (api.colabfold.com)")
            else:
                d["msa"] = {"mode": "empty"}
                if len(d["sequence"]) > 30:
                    warnings.append(f"{where}: no MSA (single sequence)")
            d["length"] = len(d["sequence"])
            tokens += d["length"]
            if len(d["sequence"]) >= 10 and set(d["sequence"]) <= set("ACGTUN"):
                warnings.append(f"{where}: this looks like a DNA / RNA sequence (only A, C, G, T / U); it is read as protein: "
                                f"set the chain type to DNA or RNA if it is a nucleic acid")
            _mods(c, d, where, cache_dir, warnings)
        elif t in NA:
            seq = re.sub(r"\s+", "", str(c.get("sequence") or "")).upper()
            bad = sorted(set(seq) - NA[t])
            if not seq:
                raise ApiError(400, f"{where}: empty {t.upper()} sequence")
            if bad:
                raise ApiError(400, f"{where}: not a {t.upper()} sequence (characters {''.join(bad)!r})")
            d["sequence"], d["length"] = seq, len(seq)
            tokens += len(seq)
            _mods(c, d, where, cache_dir)
        else:
            smiles, ccd = str(c.get("smiles") or "").strip(), str(c.get("ccd") or "").strip().upper()
            if bool(smiles) == bool(ccd):
                raise ApiError(400, f"{where}: give a SMILES string or a CCD code (one of the two)")
            if ccd:
                if not CCD_RE.match(ccd):
                    raise ApiError(400, f"{where}: CCD codes are 1-5 letters or digits (e.g. SAH, ATP)")
                d["ccd"] = ccd
                from boltz_mlx.tiny_ligand import heavy_atoms_ccd

                d["n_heavy"] = heavy_atoms_ccd(ccd, cache_dir / "mols") if cache_dir is not None else None
                if cache_dir is not None and (cache_dir / "mols").is_dir() and not (cache_dir / "mols" / f"{ccd}.pkl").exists():
                    raise ApiError(400, f"{where}: CCD code {ccd} is not in the molecule dictionary ({cache_dir / 'mols'})")
                d["n_atoms"] = None
                tokens += 30  # unknown here; a typical cofactor
            else:
                is_binder = cid == binder
                lig = validate_ligand(Ligand(ligand_id=cid, smiles=smiles, index=k), affinity=is_binder, **({} if is_binder else {"max_atoms": 10**6}))
                if lig.status == "invalid":
                    raise ApiError(400, f"{where}: {lig.error}")
                d["smiles"] = lig.smiles_used or smiles
                d["n_atoms"], d["mw"] = lig.n_atoms, None if lig.mw is None else round(lig.mw, 2)
                d["n_heavy"] = lig.n_heavy
                warnings += [f"{where}: {w}" for w in lig.warnings if is_binder or "training maximum" not in w]
                tokens += lig.n_atoms or 0
        tokens += (tokens - t0) * (copies - 1)
        out.append(d)
    if aff is not None and binder:
        b = next((c for c in out if c["id"] == binder), None)
        if b is None or b["type"] != "ligand":
            raise ApiError(400, f"affinity: the binder must be one of the ligand chains (got {binder!r})")
        if not any(c["type"] == "protein" for c in out):
            raise ApiError(400, "affinity: needs a protein chain to bind to")
        if b.get("ccd"):
            warnings.append(f"affinity binder {binder} is a CCD code: its size is checked only when the job starts")
    elif aff is not None:
        raise ApiError(400, "affinity: choose which ligand chain is the binder")
    if sum(c.get("copies", 1) for c in out) > MAX_CHAINS:
        raise ApiError(400, f"at most {MAX_CHAINS} chains in all, counting copies")
    if tokens > MAX_TOKENS:
        raise ApiError(400, f"this complex has ~{tokens} tokens; the GUI accepts up to {MAX_TOKENS}")
    if tokens > WARN_TOKENS:
        warnings.append(f"~{tokens} tokens: a large job (memory grows with the square of the size)")
    return {"chains": out, "n_tokens": tokens, "warnings": warnings, "affinity": bool(binder), "binder": binder or None}


YAML_FILE_EXT = (".a3m", ".csv", ".cif", ".mmcif", ".pdb", ".ent")
MAX_YAML_FILES = 24


def yaml_refs(doc: dict) -> list[tuple[dict, str, str]]:
    """(mapping, key, path) of every file a Boltz YAML points at: protein `msa` paths and template `cif` / `pdb` paths."""
    out = []
    for ent in doc.get("sequences") or []:
        body = next(iter(ent.values()), None) if isinstance(ent, dict) and len(ent) == 1 else None
        if isinstance(body, dict) and isinstance(body.get("msa"), str) and body["msa"].strip() not in ("", "empty"):
            out.append((body, "msa", body["msa"]))
    for t in doc.get("templates") or []:
        if isinstance(t, dict):
            out += [(t, k, t[k]) for k in ("cif", "pdb") if isinstance(t.get(k), str)]
    return out


def _ref_name(path: str) -> str:
    return safe_filename(re.split(r"[\\/]", path.strip())[-1])[0]


def yaml_needs(content: str) -> list[str]:
    """Names of the companion files a YAML refers to (MSAs, templates), for the page to ask for."""
    import yaml

    try:
        doc = yaml.safe_load(content)
    except yaml.YAMLError:
        return []
    return list(dict.fromkeys(_ref_name(v) for _, _, v in yaml_refs(doc))) if isinstance(doc, dict) else []


def read_companions(raw_files, what: str = "YAML input") -> dict[str, str]:
    """The attached companion files ({"filename", "content"} each) as {sanitised name: text}."""
    files: dict[str, str] = {}
    raw_files = raw_files if isinstance(raw_files, list) else []
    if len(raw_files) > MAX_YAML_FILES:
        raise ApiError(400, f"{what}: at most {MAX_YAML_FILES} companion files")
    for f in raw_files:
        f = f if isinstance(f, dict) else {}
        name, ext = safe_filename(f.get("filename"))
        if ext not in YAML_FILE_EXT:
            raise ApiError(400, f"{what}: {name}: use {', '.join(YAML_FILE_EXT)}")
        text = f.get("content")
        if not isinstance(text, str) or not text:
            raise ApiError(400, f"{what}: {name} is empty")
        if len(text) > LIMITS["msa"]:
            raise ApiError(413, f"{what}: {name} is too large")
        files[name] = text
    return files


def _doc_chain_ids(doc: dict) -> set[str]:
    out: set[str] = set()
    for ent in doc.get("sequences") or []:
        body = next(iter(ent.values()), None) if isinstance(ent, dict) and len(ent) == 1 else None
        ids = body.get("id") if isinstance(body, dict) else None
        out.update(str(i) for i in (ids if isinstance(ids, list) else [ids]) if i is not None)
    return out


def _constraint_chains(c) -> list[str]:
    """Chain ids a constraint names: [chain, residue(, atom)] triples and the pocket binder."""
    out: list[str] = []

    def walk(v, key=""):
        if isinstance(v, dict):
            for k, x in v.items():
                walk(x, k)
        elif isinstance(v, list):
            if v and isinstance(v[0], str) and len(v) in (2, 3) and key in ("atom1", "atom2", "token1", "token2", "contacts"):
                out.append(v[0])
            else:
                for x in v:
                    walk(x, key)
        elif key == "binder" and isinstance(v, str):
            out.append(v)

    walk(c)
    return out


def validate_companions(doc: dict, files: dict[str, str], what: str = "YAML input") -> None:
    """Every file the YAML points at must be attached under a distinct name, each attached MSA must be this chain's alignment (the
    same checks as the form's upload), and templates / constraints must name chains the YAML defines. Raises ApiError(400)."""
    refs = yaml_refs(doc)
    paths: dict[str, set[str]] = {}
    for _, _, val in refs:
        paths.setdefault(_ref_name(val), set()).add(re.sub(r"/+", "/", val.strip().replace("\\", "/")))
    for n, ps in paths.items():
        if len(ps) > 1:
            raise ApiError(400, f"{what}: {n} is referenced from {len(ps)} different paths; give each file its own name")
    missing = [n for n in paths if n not in files]
    if missing:
        raise ApiError(400, f"{what}: add {', '.join(missing)}")
    for box, key, val in refs:
        if key != "msa":
            continue
        name = _ref_name(val)
        ext = safe_filename(name)[1]
        ids = box.get("id")
        cid = str((ids[0] if isinstance(ids, list) and ids else ids) or "?")
        if ext not in ALLOWED_EXT["msa"]:
            raise ApiError(400, f"chain {cid}: MSA must be .a3m or .csv")
        seq = re.sub(r"[\s\d]+", "", str(box.get("sequence") or "")).upper()
        if seq:
            check_msa(files[name], ext, seq, f"chain {cid}")
    have = _doc_chain_ids(doc)
    named: list[str] = []
    for t in doc.get("templates") or []:
        v = t.get("chain_id") if isinstance(t, dict) else None
        named += [str(x) for x in (v if isinstance(v, list) else [v] if v is not None else [])]
    for c in doc.get("constraints") or []:
        named += _constraint_chains(c)
    unknown = sorted({n for n in named if n not in have})
    if unknown:
        raise ApiError(400, f"{what}: chain {', '.join(unknown)} is not defined in the YAML")


def rewrite_refs(doc: dict, base: str, prefix: str = "") -> dict:
    """A copy of the YAML document with every file reference pointing at `base`/<prefix><name>."""
    import copy

    doc = copy.deepcopy(doc)
    for box, key, val in yaml_refs(doc):
        box[key] = f"{base}/{prefix}{_ref_name(val)}" if base else f"{prefix}{_ref_name(val)}"
    return doc


def check_yaml(spec: dict, cache_dir: Path | None = None) -> dict:
    """An advanced input: a Boltz YAML run exactly as written (templates, constraints, modifications, MSA paths ...) with its
    companion files. Chains are checked as for the form; every file the YAML refers to must be supplied by name and an MSA must be
    its chain's alignment. The result has the same shape as check_chains() plus "yaml_input": {"doc", "files"}. Raises ApiError(400)."""
    import yaml

    content = spec.get("content")
    if not isinstance(content, str) or not content.strip() or content.lstrip().startswith(">"):
        raise ApiError(400, "YAML input: choose a Boltz YAML file")
    if len(content) > LIMITS["target"]:
        raise ApiError(413, f"YAML input: too large (limit {LIMITS['target'] // (1024 * 1024)} MB)")
    imp = import_yaml(content)
    doc = yaml.safe_load(content)
    files = read_companions(spec.get("files"))
    payload = {"chains": [{k: v for k, v in c.items() if k != "msa"} for c in imp["chains"]],
               "affinity": {"binder": imp["binder"]} if imp["binder"] else None}
    checked = check_chains(payload, cache_dir)
    validate_companions(doc, files)
    CX.check_constraints(doc.get("constraints"), checked, cache_dir / "mols" if cache_dir is not None else None)   # residues, atoms
    withmsa = set()
    for box, key, val in yaml_refs(doc):
        if key == "msa":
            ids = box.get("id")
            for i in ids if isinstance(ids, list) else [ids]:
                withmsa.add(str(i))
                for c in checked["chains"]:
                    if c["id"] == str(i):
                        c["msa"] = {"mode": "file", "name": _ref_name(val), "ext": safe_filename(val)[1], "content": ""}
    checked["warnings"] = [w for w in checked["warnings"] if not any(w.startswith(f"chain {i}: no MSA") for i in withmsa)]
    checked["yaml_input"] = {"doc": doc, "files": files}
    return checked


def _yaml_text(checked: dict, base: str) -> str:
    """The advanced YAML with every file reference pointing at `base`/<name>."""
    import yaml

    yi = checked["yaml_input"]
    return yaml.safe_dump(rewrite_refs(yi["doc"], base), sort_keys=False, width=10**9)


def build_yaml(checked: dict, msa_paths: dict[str, Path] | None = None, template_dir: str = "inputs") -> str:
    """The Boltz YAML for checked chains (msa_paths: chain id -> absolute MSA path of the staged file; templates are
    <template_dir>/<name>)."""
    import yaml

    seqs = []
    for c in checked["chains"]:
        if c["type"] == "protein":
            v: dict = {"id": c.get("ids") or c["id"], "sequence": c["sequence"]}
            if c["msa"]["mode"] != "server":  # server: no msa key, so upstream asks the MSA server for this chain
                v["msa"] = str((msa_paths or {}).get(c["id"])) if c["msa"]["mode"] == "file" and (msa_paths or {}).get(c["id"]) else "empty"
            if c.get("modifications"):
                v["modifications"] = c["modifications"]
            if c.get("cyclic"):
                v["cyclic"] = True
            seqs.append({"protein": v})
        elif c["type"] in NA:
            v = {"id": c.get("ids") or c["id"], "sequence": c["sequence"]}
            if c.get("modifications"):
                v["modifications"] = c["modifications"]
            seqs.append({c["type"]: v})
        else:
            v = {"id": c.get("ids") or c["id"]}
            v.update({"ccd": c["ccd"]} if c.get("ccd") else {"smiles": c["smiles"]})
            seqs.append({"ligand": v})
    doc: dict = {"version": 1, "sequences": seqs}
    if checked.get("constraints"):
        doc["constraints"] = checked["constraints"]
    if checked.get("templates"):
        doc["templates"] = [{"cif": f"{template_dir}/{t['name']}", "chain_id": t["chain_id"] if len(t["chain_id"]) > 1 else t["chain_id"][0]}
                            for t in checked["templates"]]
    if checked.get("binder"):
        doc["properties"] = [{"affinity": {"binder": checked["binder"]}}]
    return yaml.safe_dump(doc, sort_keys=False, width=10**9, default_flow_style=None)


def msa_file(c: dict) -> str:
    """The name a chain's uploaded MSA is staged under, inside inputs/."""
    return f"msa_{c['id']}{c['msa']['ext']}"


def preview_yaml(checked: dict) -> str:
    """The YAML the job will run, with the staged MSA file names (relative to the run's folder): same document as stage()."""
    if checked.get("yaml_input"):
        return _yaml_text(checked, "inputs/files")
    paths = {c["id"]: Path("inputs") / msa_file(c) for c in checked["chains"] if c["type"] == "protein" and c["msa"]["mode"] == "file"}
    return build_yaml(checked, paths)


def stage(checked: dict, inputs: Path) -> dict:
    """Write the MSAs and the YAML into `inputs` (fixed names). Returns the `inputs` record for gui_run.json."""
    inputs.mkdir(parents=True, exist_ok=True)
    if checked.get("yaml_input"):
        fdir = inputs / "files"
        fdir.mkdir(exist_ok=True)
        for name, text in checked["yaml_input"]["files"].items():
            (fdir / name).write_text(text, encoding="utf-8")
        y = inputs / f"{RECORD}.yaml"
        y.write_text(_yaml_text(checked, str(fdir.resolve())), encoding="utf-8")
        return {"mode": "yaml", "yaml": {"file": f"inputs/{y.name}", "name": f"{RECORD}.yaml"}, "msa": {},
                "files": {n: f"inputs/files/{n}" for n in checked["yaml_input"]["files"]}}
    msa_paths: dict[str, Path] = {}
    msas = {}
    for c in checked["chains"]:
        if c["type"] == "protein" and c["msa"]["mode"] == "file":
            p = inputs / msa_file(c)
            p.write_text(c["msa"]["content"], encoding="utf-8")
            msa_paths[c["id"]] = p.resolve()
            msas[c["id"]] = {"file": f"inputs/{p.name}", "name": c["msa"]["name"]}
    tpl = {}
    for t in checked.get("templates") or []:
        (inputs / t["name"]).write_text(t["content"], encoding="utf-8")
        tpl[t["name"]] = {"file": f"inputs/{t['name']}", "from": t["from"], "chain_id": t["chain_id"][0], "template_chain": t["template_chain"]}
    y = inputs / f"{RECORD}.yaml"
    y.write_text(build_yaml(checked, msa_paths, str(inputs.resolve())), encoding="utf-8")
    rec = {"yaml": {"file": f"inputs/{y.name}", "name": f"{RECORD}.yaml"}, "msa": msas}
    if tpl:
        rec["templates"] = tpl
    if any(c["type"] == "protein" and c["msa"]["mode"] == "server" for c in checked["chains"]):
        rec["msa_server"] = True
    return rec


def public_templates(checked: dict) -> list[dict]:
    """Templates as the page shows them (no file content)."""
    return [{"name": t["name"], "from": t["from"], "chain_id": t["chain_id"][0], "template_chain": t["template_chain"], "length": t["length"]}
            for t in checked.get("templates") or []]


def public_chains(checked: dict) -> list[dict]:
    """Chains as stored in gui_run.json / shown on the page (no MSA content)."""
    out = []
    for c in checked["chains"]:
        d = {k: v for k, v in c.items() if k != "msa"}
        if c["type"] == "protein":
            d["msa"] = c["msa"]["name"] if c["msa"]["mode"] == "file" else c["msa"]["mode"]
        out.append(d)
    return out


def predict_argv(rd: Path, inputs: dict, options: dict, extra: list[str] | None = None) -> list[str]:
    """Arguments after `predict` for a run (the YAML path first; every option attached with '=')."""
    a = [str(rd / inputs["yaml"]["file"]), f"--out_dir={rd / 'out'}", "--output_format=mmcif"]
    a += PRESET_FLAGS[options["preset"]]
    if inputs.get("msa_server"):
        a.append("--use_msa_server")
    a += [f"--sampling_steps={options['steps']}", f"--recycling_steps={options['recycling']}",
          f"--sampling_steps_affinity={options['affinity_steps']}", f"--diffusion_samples={options['samples']}",
          f"--diffusion_samples_affinity={options.get('affinity_samples') or 5}", f"--seed={options['seed']}"]
    return a + list(extra or [])


# ------------------------------------------------------------------------------------------------------ outputs
def results_dir(out: Path) -> Path:
    return out / f"boltz_results_{RECORD}"


def pred_dir(out: Path) -> Path:
    return results_dir(out) / "predictions" / RECORD


def model_file(out: Path, k: int = 0) -> Path | None:
    for ext in ("cif", "pdb"):
        p = pred_dir(out) / f"{RECORD}_model_{k}.{ext}"
        if p.is_file():
            return p
    return None


def model_indices(out: Path) -> list[int]:
    """The diffusion samples that have a structure file, in order (model 0 is the best-ranked one)."""
    pd = pred_dir(out)
    found = set()
    if pd.is_dir():
        for f in pd.iterdir():
            m = re.fullmatch(rf"{RECORD}_model_(\d+)\.(cif|pdb)", f.name)
            if m and f.is_file():
                found.add(int(m.group(1)))
    return sorted(found)


# ------------------------------------------------------------------------------------------------------ import
IMPORT_KEYS = {"protein": {"id", "sequence", "msa", "cyclic", "modifications"}, "dna": {"id", "sequence", "modifications"},
               "rna": {"id", "sequence", "modifications"}, "ligand": {"id", "smiles", "ccd"}}


def import_yaml(text: str) -> dict:
    """A Boltz YAML (or FASTA) -> the form's chains: {"chains": [...], "binder": id | None, "constraints": [...], "templates":
    [{"file", "chain_id", "template_id"}], "skipped": [what was not carried over], "needs": [companion files]}.
    Carried over: protein / dna / rna / ligand entries (several ids become several chains), sequences, SMILES / CCD, cyclic,
    residue modifications, pocket / contact / bond constraints (without `force`), templates (the file is attached in the form) and
    the affinity binder. Everything else is listed in `skipped` (MSA file references, template force / threshold, other
    properties and unknown keys): the page refuses to start until those are dropped on purpose."""
    import yaml

    from boltz_mlx.gui.targets import parse_fasta

    text = str(text or "")
    if not text.strip():
        raise ApiError(400, "nothing to import")
    skipped: list[str] = []
    chains: list[dict] = []
    if text.lstrip().startswith(">"):
        for head, seq in parse_fasta(text):
            tok = re.split(r"[|\s]", head.strip())[0] if head.strip() else ""
            cid = tok if re.fullmatch(r"[A-Za-z0-9]", tok) and tok.lower() not in {c["id"].lower() for c in chains} else ""  # ">A|protein|..."
            chains.append({"type": "protein", "id": cid, "sequence": seq, "cyclic": False})
        _auto_ids(chains)
        return {"chains": chains, "binder": None, "skipped": skipped}
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError:
        raise ApiError(400, "not valid YAML or FASTA") from None
    if not isinstance(doc, dict) or not isinstance(doc.get("sequences"), list):
        raise ApiError(400, "no 'sequences' list in this YAML")
    for ent in doc["sequences"]:
        if not isinstance(ent, dict) or len(ent) != 1:
            skipped.append("an entry that is not a single chain")
            continue
        (kind, body), = ent.items()
        if kind not in CHAIN_TYPES or not isinstance(body, dict):
            skipped.append(f"{kind} entry")
            continue
        ids = body.get("id")
        ids = [str(i) for i in ids] if isinstance(ids, list) else [str(ids)] if ids is not None else [""]
        who = ",".join(i or "?" for i in ids)
        for key in sorted(set(body) - IMPORT_KEYS[kind]):
            skipped.append(f"{key} of chain {who}")  # modifications, templates, ...
        if kind == "protein" and body.get("msa") not in (None, "empty"):
            skipped.append(f"MSA file of chain {who}")
        for cid in ids:
            c = {"type": kind, "id": cid}
            if kind == "ligand":
                c.update({"smiles": str(body["smiles"])} if body.get("smiles") else {"ccd": str(body.get("ccd") or "")})
            else:
                c["sequence"] = str(body.get("sequence") or "")
            if kind == "protein":
                c["cyclic"] = bool(body.get("cyclic"))
            if kind != "ligand" and isinstance(body.get("modifications"), list) and body["modifications"]:
                c["modifications"] = [{"position": m.get("position"), "ccd": str(m.get("ccd") or "")} for m in body["modifications"] if isinstance(m, dict)]
            chains.append(c)
    _auto_ids(chains)
    constraints = []
    for c in doc.get("constraints") or []:
        if isinstance(c, dict) and len(c) == 1 and next(iter(c)) in ("pocket", "contact", "bond") and isinstance(next(iter(c.values())), dict):
            (kind, body), = c.items()
            if body.get("force"):
                skipped.append(f"force of a {kind} constraint")
            constraints.append({kind: {k: v for k, v in body.items() if k != "force"}})
        else:
            skipped.append("a constraint the form cannot hold")
    templates = []
    for t in doc.get("templates") or []:
        path = (t.get("cif") or t.get("pdb")) if isinstance(t, dict) else None
        if not isinstance(path, str):
            skipped.append("a template without a file")
            continue
        if t.get("force") or "threshold" in t:
            skipped.append("template force / threshold")
        cids = t.get("chain_id")
        cids = [str(x) for x in (cids if isinstance(cids, list) else [cids] if cids is not None else [])]
        tids = t.get("template_id")
        tids = [str(x) for x in (tids if isinstance(tids, list) else [tids] if tids is not None else [])]
        targets = cids or [c["id"] for c in chains if c["type"] == "protein"][:1]
        for k, cid in enumerate(targets):   # one form template per chain it serves
            templates.append({"file": _ref_name(path), "chain_id": cid, "template_id": tids[k] if k < len(tids) else None})
    binder = None
    for prop in doc.get("properties") or []:
        if isinstance(prop, dict) and isinstance(prop.get("affinity"), dict):
            binder = str(prop["affinity"].get("binder") or "") or None
            extra = set(prop["affinity"]) - {"binder"}
            if extra:
                skipped.append("affinity " + ", ".join(sorted(map(str, extra))))
        else:
            skipped.append("properties")
    for key in sorted(set(doc) - {"version", "sequences", "properties", "constraints", "templates"}):
        if doc.get(key):
            skipped.append(str(key))
    if not chains:
        raise ApiError(400, "no chains found in this YAML")
    return {"chains": chains, "binder": binder, "constraints": constraints, "templates": templates,
            "skipped": list(dict.fromkeys(skipped)), "needs": yaml_needs(text)}


def phase(out: Path, affinity: bool, since: float | None = None) -> str:
    """Where a predict job is, from its files: starting | structure | affinity | done."""
    rdir = results_dir(out)
    timings = rdir / "boltz_mlx_timings.json"

    def fresh(p: Path) -> bool:
        try:
            return p.exists() and (since is None or p.stat().st_mtime >= since - 2)
        except OSError:
            return False

    if fresh(timings) and model_file(out) is not None:
        return "done"
    if model_file(out) is not None:
        return "affinity" if affinity and not (pred_dir(out) / f"affinity_{RECORD}.json").exists() else "finishing"
    if (rdir / "processed").is_dir():
        return "structure"
    return "starting"
