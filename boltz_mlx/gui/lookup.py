"""Protein search and sequence lookup for Predict and the Screen target form.

The only code of the GUI that contacts a third party, and only for what the user types into a search box:
  * search (POST /api/search, as the user types; >= 2 characters, debounced by the page): a gene / protein name or keyword
    -> https://rest.uniprot.org/uniprotkb/search?query=... (two queries: human Swiss-Prot first, then everything); a PDB id
    -> https://www.rcsb.org/fasta/entry/<id> (the entry's chains). Results are cached in memory for CACHE_TTL seconds.
  * lookup (POST /api/lookup, when a result is chosen without its sequence, or an id is entered):
    UniProt accession (P00533, P00533-2):  https://rest.uniprot.org/uniprotkb/<acc>.fasta
    PDB id (8CPK) or PDB id + chain (8CPK_A): https://www.rcsb.org/fasta/entry/<id>, and for the entry's bound small molecules
    https://data.rcsb.org/rest/v1/core/{entry,nonpolymer_entity,chemcomp}/...
Nothing else is sent (the typed text or identifier is in the URL; no cookies, no user data). All network access goes through
_fetch(), the one function the tests replace.
"""

from __future__ import annotations

import json
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from boltz_mlx.gui.common import ApiError

TIMEOUT = 15
MAX_BYTES = 4 * 1024 * 1024
MAX_LIGANDS = 8
UNIPROT_RE = re.compile(r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})(?:-\d{1,3})?$")
PDB_RE = re.compile(r"^([1-9][A-Za-z0-9]{3})(?:[_:.\s]+([A-Za-z0-9]{1,4}))?$")
# crystallisation additives, buffers and ions: not what anyone wants as a ligand suggestion
SKIP_COMP = {"GOL", "EDO", "PEG", "PGE", "PG4", "P6G", "1PE", "2PE", "MPD", "DMS", "ACT", "ACY", "FMT", "TRS", "EPE", "MES", "BME", "SO4", "PO4",
             "NO3", "CL", "NA", "K", "MG", "ZN", "CA", "MN", "FE", "FE2", "CU", "CO", "NI", "CD", "IOD", "BR", "SCN", "IMD", "DTT", "BOG", "OGA"}
NA_LETTERS = set("ACGTUN")
SEARCH_MIN, SEARCH_MAX, SEARCH_N = 2, 60, 10
SEARCH_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.,:/()'+-]*$")
CACHE_TTL, CACHE_MAX = 600, 256
OFFLINE = "Search unavailable offline \u2014 paste a sequence or choose a file"
HUMAN = 9606
UNIPROT_FIELDS = "accession,protein_name,gene_names,organism_name,organism_id,length,reviewed,sequence"
_cache: dict = {}
_cache_lock = threading.Lock()


def _fetch(url: str) -> str:
    """GET one URL as text (UrlError / HTTP errors -> ApiError). The only network call of the GUI's lookup; tests replace it."""
    req = urllib.request.Request(url, headers={"User-Agent": "boltz-mlx-gui (sequence lookup)", "Accept": "text/plain, application/json"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:  # noqa: S310 (https URLs built in this module only)
            data = r.read(MAX_BYTES + 1)
    except urllib.error.HTTPError as ex:
        if ex.code in (400, 404, 410):
            raise ApiError(404, "not found") from None
        raise ApiError(502, f"server answered {ex.code}") from None
    except (urllib.error.URLError, TimeoutError, OSError) as ex:
        why = getattr(ex, "reason", None) or ex
        raise ApiError(502, f"could not reach the server ({why})") from None
    if len(data) > MAX_BYTES:
        raise ApiError(502, "reply too large")
    return data.decode("utf-8", errors="replace")


def _records(fasta: str) -> list[tuple[str, str]]:
    out, head, buf = [], None, []
    for line in fasta.splitlines():
        if line.startswith(">"):
            if head is not None:
                out.append((head, "".join(buf)))
            head, buf = line[1:].strip(), []
        elif head is not None:
            buf.append(re.sub(r"\s+", "", line))
    if head is not None:
        out.append((head, "".join(buf)))
    return [(h, s.upper()) for h, s in out if s]


def _kind(seq: str) -> str:
    return "dna" if set(seq) <= NA_LETTERS and "U" not in seq else "rna" if set(seq) <= NA_LETTERS else "protein"


def _chains(head: str) -> list[str]:
    """Chain ids of an RCSB FASTA header (">8CPK_1|Chains A, B[auth C]|name|organism"): label ids, then the author ids."""
    m = re.search(r"\|Chains? ([^|]+)\|", head + "|")
    out: list[str] = []
    for tok in (m.group(1) if m else "").split(","):
        tok = tok.strip()
        a = re.fullmatch(r"(\S+)\[auth (\S+)\]", tok)
        out += [a.group(1), a.group(2)] if a else [tok] if tok else []
    return out


def _n_chains(head: str) -> int:
    """How many chains of the entry are this entity (">1HSG_1|Chains A, B|..." -> 2; an "[auth X]" id is the same chain)."""
    m = re.search(r"\|Chains? ([^|]+)\|", head + "|")
    return max(1, len([t for t in (m.group(1) if m else "").split(",") if t.strip()]))


def _ok(seq: str, want: str) -> bool:
    k = _kind(seq)
    return k == "protein" if want == "protein" else k != "protein"   # DNA vs RNA is not told apart reliably (poly-G ...): both fit


def _entity_no(head: str) -> int:
    m = re.match(r"^\w+_(\d+)\|", head)
    return int(m.group(1)) if m else 0


def _name(head: str) -> str:
    parts = head.split("|")
    return parts[2].strip() if len(parts) > 2 else ""


def parse_query(q: object) -> tuple[str, str, str | None]:
    """("uniprot", accession, None) or ("pdb", id, chain | None); ApiError(400) for anything else."""
    s = str(q or "").strip()
    if not s:
        raise ApiError(400, "enter a UniProt accession (P00533) or a PDB id (8CPK, 8CPK_A)")
    up = s.upper()
    if UNIPROT_RE.match(up):
        return "uniprot", up, None
    m = PDB_RE.match(s)
    if m:
        return "pdb", m.group(1).upper(), (m.group(2) if m.group(2) else None)
    raise ApiError(400, f"{s[:30]!r} is not a UniProt accession (P00533) or a PDB id (8CPK, 8CPK_A)")


def _uniprot(acc: str) -> dict:
    recs = _records(_fetch(f"https://rest.uniprot.org/uniprotkb/{urllib.parse.quote(acc)}.fasta"))
    if not recs:
        raise ApiError(404, f"UniProt has no sequence for {acc}")
    head, seq = recs[0]
    title = re.sub(r"^\w+\|[^|]+\|\S+\s*", "", head)
    title = re.split(r"\s+OS=", title)[0]
    return {"source": "UniProt", "id": acc, "title": title or acc, "sequence": seq, "length": len(seq), "type": "protein", "ligands": []}


def _pdb(pid: str, chain: str | None, want: str) -> dict:
    recs = sorted(_records(_fetch(f"https://www.rcsb.org/fasta/entry/{pid}")), key=lambda r: _entity_no(r[0]))
    if not recs:
        raise ApiError(404, f"RCSB has no entry {pid}")
    if chain:
        hit = [r for r in recs if chain in _chains(r[0]) or chain.upper() in (c.upper() for c in _chains(r[0]))]
        if not hit:
            raise ApiError(404, f"{pid} has no chain {chain}")
        head, seq = hit[0]
        if not _ok(seq, want):
            raise ApiError(400, f"chain {chain} of {pid} is a {_kind(seq)} chain, not {want}")
    else:
        hit = [r for r in recs if _ok(r[1], want)]
        if not hit:
            raise ApiError(404, f"{pid} has no {want} chain")
        head, seq = hit[0]
    title = _name(head)
    parts = head.split("|")
    note = f"entity {_entity_no(head)} of {len(recs)}" + (f" ({parts[1]})" if len(parts) > 1 else "")
    try:
        ligands = _pdb_ligands(pid)
    except ApiError:
        ligands = []   # the suggestions are a bonus: the sequence is what was asked for
    return {"source": "RCSB", "id": pid, "title": title or pid, "sequence": seq, "length": len(seq), "type": _kind(seq),
            "chain": chain, "note": note, "ligands": ligands, "copies": _n_chains(head)}


def _json(url: str) -> dict:
    try:
        d = json.loads(_fetch(url))
    except ValueError:
        raise ApiError(502, "unexpected reply") from None
    return d if isinstance(d, dict) else {}


def _pdb_ligands(pid: str) -> list[dict]:
    """The entry's bound small molecules as [{id (CCD code), name, smiles}] (additives and ions left out), at most MAX_LIGANDS."""
    base = "https://data.rcsb.org/rest/v1/core"
    ents = _json(f"{base}/entry/{pid}").get("rcsb_entry_container_identifiers", {}).get("non_polymer_entity_ids") or []
    out: list[dict] = []
    seen: set[str] = set()
    for e in ents:
        comp = str(((_json(f"{base}/nonpolymer_entity/{pid}/{urllib.parse.quote(str(e))}").get("pdbx_entity_nonpoly")) or {}).get("comp_id") or "").upper()
        if not comp or comp in seen or comp in SKIP_COMP or not re.fullmatch(r"[A-Z0-9]{1,5}", comp):
            continue
        seen.add(comp)
        cc = _json(f"{base}/chemcomp/{comp}")
        d = cc.get("rcsb_chem_comp_descriptor") or {}
        smi = d.get("SMILES_stereo") or d.get("SMILES")
        if smi:
            out.append({"id": comp, "name": (cc.get("chem_comp") or {}).get("name") or comp, "smiles": smi})
        if len(out) >= MAX_LIGANDS:
            break
    return out


def lookup(payload: object) -> dict:
    """{"query": "P00533" | "8CPK" | "8CPK_A", "type": "protein" | "dna" | "rna"} -> {"source", "id", "title", "sequence", "length",
    "type", "ligands": [{"id", "name", "smiles"}], ("chain", "note")}. Raises ApiError (400 bad query, 404 not found, 502 network)."""
    payload = payload if isinstance(payload, dict) else {}
    want = str(payload.get("type") or "protein").lower()
    if want not in ("protein", "dna", "rna"):
        raise ApiError(400, "lookup is for protein, DNA or RNA chains")
    kind, ident, chain = parse_query(payload.get("query"))
    if kind == "uniprot":
        if want != "protein":
            raise ApiError(400, "a UniProt accession is a protein: use a PDB id for DNA / RNA")
        return _uniprot(ident)
    return _pdb(ident, chain, want)


# ------------------------------------------------------------------------------------------------------ search
def clean_search(q: object) -> str:
    """The typed text, whitespace collapsed; ApiError(400) when too short, too long or with characters a search never needs."""
    s = re.sub(r"\s+", " ", str(q or "")).strip()
    if len(s) < SEARCH_MIN:
        raise ApiError(400, f"type at least {SEARCH_MIN} characters")
    if len(s) > SEARCH_MAX:
        raise ApiError(400, f"at most {SEARCH_MAX} characters")
    if not SEARCH_RE.match(s):
        raise ApiError(400, "use letters, digits, spaces and - _ . , : / ( ) ' +")
    return s


def _uniprot_query(s: str) -> str:
    """Free text -> a UniProt query: every word quoted (so '-', ':' or '(' never become query syntax), all required."""
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9_.'/+-]*", s)
    if not words:
        raise ApiError(400, "type a gene or protein name, a UniProt accession or a PDB id")
    return " AND ".join(f'"{w}"' for w in words[:8])


def _hit(r: dict) -> dict | None:
    """One UniProt search result -> {source, id, name, gene, organism, organism_id, length, reviewed, human, sequence}."""
    acc = r.get("primaryAccession")
    if not isinstance(acc, str):
        return None
    pd = r.get("proteinDescription") or {}
    name = ((pd.get("recommendedName") or {}).get("fullName") or {}).get("value")
    if not name:
        sub = pd.get("submissionNames") or [{}]
        name = ((sub[0] or {}).get("fullName") or {}).get("value")
    genes = r.get("genes") or [{}]
    gene = ((genes[0] or {}).get("geneName") or {}).get("value") or ""
    org = r.get("organism") or {}
    seq = (r.get("sequence") or {}).get("value") or ""
    tax = org.get("taxonId")
    return {"source": "UniProt", "id": acc, "name": name or acc, "gene": gene, "organism": org.get("scientificName") or "",
            "organism_id": tax, "length": (r.get("sequence") or {}).get("length") or len(seq),
            "reviewed": "unreviewed" not in str(r.get("entryType") or "").lower() and bool(r.get("entryType")),
            "human": tax == HUMAN, "sequence": re.sub(r"\s+", "", seq).upper()}


def _uniprot_search(query: str, size: int) -> list[dict]:
    url = "https://rest.uniprot.org/uniprotkb/search?" + urllib.parse.urlencode(
        {"query": query, "fields": UNIPROT_FIELDS, "size": size, "format": "json"})
    res = _json(url).get("results") or []
    return [h for h in (_hit(r) for r in res if isinstance(r, dict)) if h]


def _rank(hits: list[dict], s: str) -> list[dict]:
    """Human Swiss-Prot entries first, then other reviewed ones, then human, then the rest; an exact gene match first in each."""
    seen, out = set(), []
    for x in hits:
        if x["id"] not in seen:
            seen.add(x["id"])
            out.append(x)
    low = s.lower()
    return sorted(out, key=lambda x: (not (x["human"] and x["reviewed"]), not x["reviewed"], not x["human"],
                                      x["gene"].lower() != low))[:SEARCH_N]


def _pdb_entities(pid: str, chain: str | None, want: str) -> list[dict]:
    out = []
    for head, seq in sorted(_records(_fetch(f"https://www.rcsb.org/fasta/entry/{pid}")), key=lambda r: _entity_no(r[0])):
        chains = _chains(head)
        if chain and chain.upper() not in (c.upper() for c in chains):
            continue
        if not _ok(seq, want):
            continue
        parts = head.split("|")
        org = re.sub(r"\s*\(\d+\)\s*$", "", parts[3].strip()) if len(parts) > 3 else ""
        cid = chain or (chains[0] if chains else "")
        out.append({"source": "RCSB", "id": f"{pid}_{cid}" if cid else pid, "entry": pid, "chains": chains, "name": _name(head) or pid,
                    "organism": org, "length": len(seq), "type": _kind(seq), "entity": _entity_no(head)})
    return out[:SEARCH_N]


def search(payload: object) -> dict:
    """{"query": "EGFR" | "thrombin" | "P00533" | "8CPK" | "8CPK_A", "type": "protein" | "dna" | "rna"} -> {"query", "results": [...],
    "offline": bool, "message"}. UniProt results carry their sequence (an isoform accession does not: the page then asks
    /api/lookup); RCSB results are an entry's chains of the wanted type (the page asks /api/lookup for the sequence and the
    entry's small molecules). Network failures are not errors: {"results": [], "offline": true, "message": OFFLINE}."""
    payload = payload if isinstance(payload, dict) else {}
    want = str(payload.get("type") or "protein").lower()
    if want not in ("protein", "dna", "rna"):
        raise ApiError(400, "search is for protein, DNA or RNA chains")
    s = clean_search(payload.get("query"))
    key = (s.lower(), want)
    now = time.time()
    with _cache_lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < CACHE_TTL:
            return dict(hit[1], cached=True)
    up, pdb = s.upper(), PDB_RE.match(s)
    jobs: list = []
    if want == "protein" and UNIPROT_RE.match(up):
        jobs.append(("acc", lambda: _uniprot_search(f"accession:{up.split('-')[0]}", 1)))
    elif want == "protein" and not pdb:   # 8CPK, 8CPK_A: the entry's chains only
        q = _uniprot_query(s)
        jobs.append(("human", lambda: _uniprot_search(f"({q}) AND (organism_id:{HUMAN}) AND (reviewed:true)", 6)))
        jobs.append(("all", lambda: _uniprot_search(q, SEARCH_N)))
    if pdb:
        jobs.append(("pdb", lambda: _pdb_entities(pdb.group(1).upper(), pdb.group(2), want)))
    got: dict = {}
    errs: list[ApiError] = []

    def run(name, fn):
        try:
            got[name] = fn()
        except ApiError as ex:
            errs.append(ex)

    threads = [threading.Thread(target=run, args=j, daemon=True) for j in jobs]
    for t in threads:
        t.start()
    for t in threads:
        t.join(TIMEOUT + 5)
    if not jobs:
        return {"query": s, "results": [], "offline": False, "message": "for DNA / RNA, type a PDB id (8CPK, 8CPK_B)"}
    offline = not got and any(e.status == 502 for e in errs)
    if offline:
        return {"query": s, "results": [], "offline": True, "message": OFFLINE}
    results = list(got.get("pdb") or [])
    if "acc" in got:
        acc = [dict(x, id=up, sequence="" if "-" in up else x["sequence"]) for x in got["acc"]]
        results += acc
    else:
        results += _rank((got.get("human") or []) + (got.get("all") or []), s)
    out = {"query": s, "results": results, "offline": False, "message": "" if results else "no matches"}
    with _cache_lock:
        _cache[key] = (now, out)
        while len(_cache) > CACHE_MAX:
            _cache.pop(next(iter(_cache)))
    return out
