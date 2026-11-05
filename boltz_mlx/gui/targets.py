"""The target library: prepare a receptor once, reuse it in every screen (and across sessions).

Preparing a target = check its sequence(s) + store an uploaded MSA (or none) + optionally store a PDB / mmCIF structure as
a template, with the sequence read from that file. There is no MSA server, ever: the app is offline, so a target without
an uploaded MSA runs single-sequence (`msa: empty`). Each prepared target is a folder

  <runs_dir>/.targets/<id>/
    target.json     what the page shows (name, chains, MSA, template, source), written last
    target.yaml     a Boltz input: protein chain(s) + `msa:` (msa.a3m / msa.csv / empty) + `templates:` (template.cif)
    msa.a3m|.csv    the uploaded MSA (one protein chain only; it is used as given)
    template.cif    the chosen chain of the uploaded structure, polymer only (ligands and waters removed), as mmCIF

The folder name starts with "." so the runs list never shows it. A screen started from a saved target copies these files
into its own run folder (inputs/), so the run stays reproducible if the target is deleted later. Every target.yaml is
checked with the screen's own loader (boltz_mlx.screen.inputs.load_target) before it is stored.

Templates: the trunk uses the template features; upstream's template guidance potential is not applied by the MLX port
(boltz_mlx/cli.py), so a template informs the structure but does not force it.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
import time
from pathlib import Path

import yaml

from boltz_mlx.gui.common import ApiError, safe_filename

MB = 1024 * 1024
STRUCT_LIMIT = 32 * MB
SEQ_LIMIT = 2 * MB
MSA_LIMIT = 64 * MB
NAME_LIMIT = 80
MAX_TARGETS = 500
ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")
STRUCT_EXT = (".pdb", ".ent", ".cif", ".mmcif")
SEQ_EXT = (".fasta", ".fa", ".fas", ".faa")
YAML_EXT = (".yaml", ".yml")


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40] or "target"
    return s


def _letters():
    for c in "ABCDEFGHIJKMNOPQRSTUVWXYZ":  # no L: the screen's default ligand chain
        yield c


def _clean_protein(seq: str, what: str) -> str:
    from boltz_mlx.screen.inputs import AA

    s = re.sub(r"[\s\d*]+", "", seq).upper()
    if not s:
        raise ApiError(400, f"{what}: empty sequence")
    bad = sorted(set(s) - AA)
    if bad:
        raise ApiError(400, f"{what}: not an amino-acid sequence (characters {''.join(bad)!r})")
    return s


def parse_fasta(text: str) -> list[tuple[str, str]]:
    """[(header, sequence)] of a FASTA text; a text without '>' is one bare sequence."""
    if not text.lstrip().startswith(">"):
        return [("sequence", text)]
    out, head, buf = [], None, []
    for ln in text.splitlines():
        if ln.startswith(">"):
            if head is not None:
                out.append((head, "".join(buf)))
            head, buf = ln[1:].strip() or f"chain {len(out) + 1}", []
        elif ln.strip() and not ln.startswith(";"):
            buf.append(ln.strip())
    if head is not None:
        out.append((head, "".join(buf)))
    return out


def read_structure(content: str, filename: str):
    """A gemmi Structure from PDB or mmCIF text (polymer chains kept; ligands and waters removed)."""
    import gemmi

    ext = Path(filename).suffix.lower()
    try:
        if ext in (".cif", ".mmcif") or content.lstrip().startswith("data_"):
            st = gemmi.make_structure_from_block(gemmi.cif.read_string(content).sole_block())
        else:
            st = gemmi.read_pdb_string(content)
    except Exception as ex:  # noqa: BLE001
        raise ApiError(400, f"structure: could not read {filename}: {str(ex)[:200]}") from None
    if len(st) == 0:
        raise ApiError(400, f"structure: {filename} has no models")
    st.setup_entities()
    st.remove_ligands_and_waters()
    st.remove_empty_chains()
    return st


def structure_chains(st) -> list[dict]:
    """Protein chains of model 1: [{"chain", "sequence", "n_observed", "source"}]. The sequence is the full one from SEQRES /
    entity_poly when the file has it, otherwise the observed residues (gaps then shorten it)."""
    import gemmi

    out = []
    model = st[0]
    for ch in model:
        pol = ch.get_polymer()
        if len(pol) == 0 or pol.check_polymer_type() not in (gemmi.PolymerType.PeptideL, gemmi.PolymerType.PeptideD):
            continue
        ent = st.get_entity_of(pol)
        names = list(ent.full_sequence) if ent is not None and ent.full_sequence else [r.name for r in pol]
        names = [gemmi.Entity.first_mon(n) for n in names]
        letters = []
        for n in names:
            info = gemmi.find_tabulated_residue(n)
            code = info.one_letter_code.upper() if info is not None else "?"
            letters.append(code if code.isalpha() and code != "?" else "X")
        out.append({"chain": ch.name, "sequence": "".join(letters), "n_observed": len(pol),
                    "source": "SEQRES" if ent is not None and ent.full_sequence else "observed residues"})
    return out


class TargetLibrary:
    def __init__(self, runs_root: Path):
        self.root = Path(runs_root) / ".targets"

    # ------------------------------------------------------------------------------------------ reading
    def _dir(self, tid: str) -> Path:
        if not isinstance(tid, str) or not ID_RE.match(tid):
            raise ApiError(404, "no such target")
        d = self.root / tid
        if not (d / "target.json").is_file() or d.is_symlink():
            raise ApiError(404, f"no such target: {tid}")
        return d

    def get(self, tid: str) -> dict:
        return json.loads((self._dir(tid) / "target.json").read_text())

    def list(self) -> list[dict]:
        out = []
        if self.root.is_dir():
            for d in self.root.iterdir():
                if ID_RE.match(d.name) and (d / "target.json").is_file() and not d.is_symlink():
                    try:
                        out.append(json.loads((d / "target.json").read_text()))
                    except (OSError, ValueError):
                        continue
        out.sort(key=lambda t: -float(t.get("created") or 0))
        return out

    def rename(self, tid: str, name) -> dict:
        d = self._dir(tid)
        name = re.sub(r"[\x00-\x1f]+", " ", str(name or "")).strip()[:80]
        if not name:
            raise ApiError(400, "give the target a name")
        meta = self.get(tid)
        meta["name"] = name
        (d / "target.json").write_text(json.dumps(meta, indent=1))
        return meta

    def delete(self, tid: str) -> dict:
        d = self._dir(tid)
        shutil.rmtree(d)
        return {"deleted": tid}

    def stage_into(self, tid: str, dest: Path) -> tuple[Path, dict]:
        """Copy a saved target's files into a run's inputs/ (target.yaml + its MSA / template under fixed names)."""
        d = self._dir(tid)
        meta = self.get(tid)
        dest.mkdir(parents=True, exist_ok=True)
        for f in d.iterdir():
            if f.name in ("target.yaml", "target.json") or f.is_symlink() or not f.is_file():
                continue
            shutil.copyfile(f, dest / f"target_{f.name}")
        doc = yaml.safe_load((d / "target.yaml").read_text()) or {}
        own = {f.name for f in d.iterdir() if f.is_file()}

        def ren(v):
            return f"target_{v}" if isinstance(v, str) and v in own else v

        for ent in doc.get("sequences") or []:
            for body in (ent or {}).values():
                if isinstance(body, dict) and "msa" in body:
                    body["msa"] = ren(body["msa"])
        for t in doc.get("templates") or []:
            for key in ("cif", "pdb"):
                if isinstance(t, dict) and key in t:
                    t[key] = ren(t[key])
        path = dest / "target.yaml"
        path.write_text(yaml.safe_dump(doc, sort_keys=False))
        return path, meta

    # ------------------------------------------------------------------------------------------ preparing
    def prepare(self, payload: dict, check) -> dict:
        """Prepare and store a target. `check(yaml_path)` loads it with the screen's loader (raises ApiError) and returns
        its summary. Payload: {"name", "source": {"kind": "text"|"file", "filename", "content"}, "chain" (structure files:
        a chain name, default the first protein chain), "template" (structure files: keep as template, default true),
        "msa": {"mode": "none"|"file", "filename", "content"}}."""
        if not isinstance(payload, dict):
            raise ApiError(400, "expected a JSON object")
        if len(self.list()) >= MAX_TARGETS:
            raise ApiError(400, f"the library holds at most {MAX_TARGETS} targets; delete some first")
        src = payload.get("source")
        if not isinstance(src, dict) or not isinstance(src.get("content"), str) or not src["content"].strip():
            raise ApiError(400, "target: no content (choose a file or paste a sequence)")
        content = src["content"]
        if "\x00" in content[:4096]:
            raise ApiError(400, "target: looks like a binary file; expected text")
        fname, ext = safe_filename(src.get("filename"))
        kind = str(src.get("kind") or "file")
        if kind == "text" or not ext:
            s = content.lstrip()
            ext = ".yaml" if re.search(r"^\s*sequences\s*:", content, re.M) else (
                ".cif" if s.startswith("data_") else ".pdb" if re.search(r"^(ATOM  |HETATM|HEADER|CRYST1)", content, re.M) else ".fasta")
        limit = STRUCT_LIMIT if ext in STRUCT_EXT else SEQ_LIMIT
        if len(content) > limit:
            raise ApiError(413, f"target is too large ({len(content) / MB:.1f} MB; the limit is {limit // MB} MB)")
        name = re.sub(r"[\x00-\x1f]+", " ", str(payload.get("name") or "")).strip()[:NAME_LIMIT]

        msa = payload.get("msa") if isinstance(payload.get("msa"), dict) else {"mode": "none"}
        mmode = str(msa.get("mode") or "none")
        if mmode not in ("none", "file"):
            raise ApiError(400, "msa: mode must be none or file (there is no MSA server)")

        chains: list[dict] = []          # [{"id", "sequence"}]
        template = None                  # (cif text, source chain)
        source = {"kind": "sequence", "filename": None}
        if ext in STRUCT_EXT:
            st = read_structure(content, fname or f"structure{ext}")
            found = structure_chains(st)
            if not found:
                raise ApiError(400, "structure: no protein chain found")
            want = str(payload.get("chain") or "").strip()
            pick = next((c for c in found if c["chain"] == want), None) if want else found[0]
            if pick is None:
                raise ApiError(400, f"structure: no protein chain {want!r} (protein chains: {', '.join(c['chain'] for c in found)})")
            seq = _clean_protein(pick["sequence"], f"chain {pick['chain']}")
            if set(seq) == {"X"}:
                raise ApiError(400, f"chain {pick['chain']}: no standard residues")
            chains = [{"id": "A", "sequence": seq}]
            source = {"kind": "structure", "filename": fname or f"structure{ext}", "chain": pick["chain"],
                      "sequence_from": pick["source"], "n_observed": pick["n_observed"],
                      "protein_chains": [c["chain"] for c in found]}
            if payload.get("template", True) not in (False, "false", 0, "0"):
                from boltz_mlx.gui.constraints import single_chain_cif

                template = (single_chain_cif(st, pick["chain"]), pick["chain"])
        elif ext in YAML_EXT:
            import yaml

            try:
                doc = yaml.safe_load(content)
            except yaml.YAMLError as ex:
                raise ApiError(400, f"target YAML: {str(ex)[:200]}") from None
            if not isinstance(doc, dict) or not isinstance(doc.get("sequences"), list):
                raise ApiError(400, "target YAML: expected a Boltz input with a 'sequences' list")
            from boltz_mlx.gui import predict as PR

            refs = PR.yaml_refs(doc)
            companions = PR.read_companions(payload.get("files"), "target YAML") if refs else {}
            if refs:
                PR.validate_companions(doc, companions, "target YAML")   # every MSA / template it names, attached and checked
            if mmode == "file":
                raise ApiError(400, "msa: add the MSA to a sequence or structure target (not to a YAML target)")
            source = {"kind": "yaml", "filename": fname or "pasted YAML"}
        else:
            recs = parse_fasta(content)
            if not recs:
                raise ApiError(400, "target: no sequence found")
            if len(recs) > 8:
                raise ApiError(400, "target: at most 8 protein chains")
            ids = _letters()
            chains = [{"id": next(ids), "sequence": _clean_protein(seq, f"sequence {h[:30]!r}")} for h, seq in recs]
            source = {"kind": "fasta" if content.lstrip().startswith(">") else "sequence",
                      "filename": fname if kind != "text" else None}
        if mmode == "file" and len(chains) != 1:
            raise ApiError(400, "msa: an uploaded MSA applies to a single protein chain; this target has "
                                f"{len(chains)} chains")

        tid = f"{_slug(name or (source.get('filename') or 'target'))}-{int(time.time() * 1000) % 10**8:08d}"
        work = Path(tempfile.mkdtemp(prefix=".prep-", dir=self._ensure_root()))
        try:
            msa_meta = {"mode": "none"}
            msa_ref = "empty"
            if mmode == "file":
                mtext = msa.get("content")
                if not isinstance(mtext, str) or not mtext.strip():
                    raise ApiError(400, "MSA: no content")
                if len(mtext) > MSA_LIMIT:
                    raise ApiError(413, f"MSA is too large (limit {MSA_LIMIT // MB} MB)")
                mname, mext = safe_filename(msa.get("filename"))
                if mext not in (".a3m", ".csv"):
                    raise ApiError(400, f"MSA: unsupported file type {mext!r} (use .a3m or .csv)")
                from boltz_mlx.gui.predict import check_msa

                check_msa(mtext, mext, chains[0]["sequence"], "MSA")  # parsed here, not when the first screen fails on it
                (work / f"msa{mext}").write_text(mtext, encoding="utf-8")
                msa_ref = f"msa{mext}"
                rows = sum(1 for ln in mtext.splitlines() if ln.startswith(">")) if mext == ".a3m" else max(0, len(mtext.splitlines()) - 1)
                msa_meta = {"mode": "file", "filename": mname, "rows": rows}
            if ext in YAML_EXT:
                ytext = content
                if companions:
                    for cname, ctext in companions.items():
                        (work / f"tf_{cname}").write_text(ctext, encoding="utf-8")
                    ytext = yaml.safe_dump(PR.rewrite_refs(doc, "", "tf_"), sort_keys=False, width=10**9)
                    names = list(companions)
                    msas = [n for n in names if Path(n).suffix.lower() in (".a3m", ".csv")]
                    msa_meta = {"mode": "file", "filename": ", ".join(msas), "rows": None} if msas else msa_meta
                    tpl = [t for t in doc.get("templates") or [] if isinstance(t, dict)]
                    if tpl:
                        c0 = tpl[0].get("chain_id")
                        template = (None, str(c0[0] if isinstance(c0, list) and c0 else c0 or "?"))
            else:
                lines = ["version: 1", "sequences:"]
                for c in chains:
                    lines += ["  - protein:", f"      id: {c['id']}", f"      sequence: {c['sequence']}",
                              f"      msa: {msa_ref}"]
                if template is not None:
                    (work / "template.cif").write_text(template[0], encoding="utf-8")
                    lines += ["templates:", "  - cif: template.cif", f"    chain_id: {chains[0]['id']}"]
                ytext = "\n".join(lines) + "\n"
            (work / "target.yaml").write_text(ytext, encoding="utf-8")
            summary = check(work / "target.yaml")
            seqs = "|".join(c["sequence"] for c in chains) or ytext
            meta = {
                "version": 1, "id": tid, "name": name or source.get("filename") or "target", "created": time.time(),
                "source": source, "chains": summary["chains"],
                "n_residues": summary["n_residues"], "msa": msa_meta,
                "template": None if template is None else {"file": "template.cif" if template[0] else "per YAML", "chain": template[1],
                                                           "from": source.get("filename")},
                "warnings": summary.get("warnings", []),
                "sequence_sha": hashlib.sha256(seqs.encode()).hexdigest()[:16],
            }
            (work / "target.json").write_text(json.dumps(meta, indent=1))
            work.rename(self.root / tid)
        except BaseException:
            shutil.rmtree(work, ignore_errors=True)
            raise
        return meta

    def _ensure_root(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root
