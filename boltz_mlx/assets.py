"""First-run setup: download the Boltz-2 checkpoints + molecule dictionary and convert them to MLX (`boltz-mlx setup`).

Sources are the ones upstream `boltz predict` uses (boltz/main.py download_boltz2, boltz 2.2.1): the Boltz model gateway
(which redirects to Hugging Face) and the boltz-community/boltz-2 Hugging Face repository. The cache layout is upstream's
(`boltz2_conf.ckpt`, `boltz2_aff.ckpt`, `mols.tar`, `mols/`), so a cache filled by upstream boltz is reused as is, and the
converted MLX weights go next to it in `<cache>/mlx_weights` (boltz_mlx/paths.py).

  ensure(cache, weights)   do whatever is missing: download (resumable, size + sha256 checked), extract mols/,
                           convert boltz2_{conf,aff}.ckpt -> .safetensors + .json, write the mixed fp16 checkpoints.
  plan(cache, weights)     the steps ensure() would take (no network, nothing written); `boltz-mlx check` shows it.

Every step is idempotent: a finished file is never redone, a download continues from its `.part` file, extraction and
conversion write to temporary names and rename at the end. The network is only touched by download steps (fetch_url,
replaceable for tests). Conversion runs on the CPU device (no GPU work).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import struct
import sys
import tarfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

GB = 1e9
# boltz 2.2.1 boltz/main.py: MOL_URL, BOLTZ2_URL_WITH_FALLBACK, BOLTZ2_AFFINITY_URL_WITH_FALLBACK (same order).
_HF = "https://huggingface.co/boltz-community/boltz-2/resolve/main/"
_GATEWAY = "https://model-gateway.boltz.bio/"


@dataclass(frozen=True)
class Asset:
    name: str
    urls: tuple
    size: int
    sha256: str  # = the Hugging Face LFS oid (x-linked-etag), checked 2026-11-03 against the files boltz-mlx was validated with
    what: str


ASSETS = {
    "mols.tar": Asset("mols.tar", (_HF + "mols.tar",), 1_855_662_080,
                      "39e076d96dbec6b4e86982bbda16f3a53a2a60c9bdc17828d88f6f9a0c7d1fd7", "molecule dictionary (CCD)"),
    "boltz2_conf.ckpt": Asset("boltz2_conf.ckpt", (_GATEWAY + "boltz2_conf.ckpt", _HF + "boltz2_conf.ckpt"), 2_286_561_469,
                              "090e82ac8c92f5e943fa1b39e7410a44027bea7243c0bbb3caa67a77fc1428e1", "Boltz-2 structure model"),
    "boltz2_aff.ckpt": Asset("boltz2_aff.ckpt", (_GATEWAY + "boltz2_aff.ckpt", _HF + "boltz2_aff.ckpt"), 2_062_139_170,
                             "dcc5cd3722b1c9eaa34267e4ae32f55cbbf1963f4c19319381ccfa30fdd2ca9e", "Boltz-2 affinity model"),
}
MODELS = ("conf", "aff")
# disk needed by the non-download steps (measured on the validated files): extracted mols/, fp32 + mixed safetensors
EXTRACTED_MOLS_BYTES = 1_900_000_000
CONVERTED_BYTES = {"conf": 2_085_000_000, "aff": 2_119_000_000}
MIXED_BYTES = {"conf": 1_272_000_000, "aff": 1_291_000_000}
DROP_PREFIXES = ("validators.", "train_confidence_loss", "val_", "train_")
USER_AGENT = "boltz-mlx-setup"


class SetupError(RuntimeError):
    """A setup step failed; .fix says what the user can do."""

    def __init__(self, message: str, fix: str | None = None):
        super().__init__(message)
        self.message, self.fix = message, fix


# ------------------------------------------------------------------------------------------------ locations
def locations(cache=None, weights=None) -> tuple[Path, Path]:
    """(cache dir, weights dir). Weights: --weights > $BOLTZ_MLX_WEIGHTS > an existing converted dir > <cache>/mlx_weights."""
    from boltz_mlx.paths import default_cache, weights_dir

    c = default_cache(cache)
    if weights:
        return c, Path(weights).expanduser()
    return c, weights_dir(cache)


def weight_files(name: str) -> tuple[str, str]:
    return f"boltz2_{name}.safetensors", f"boltz2_{name}.json"


def mixed_file(name: str) -> str:
    return f"boltz2_{name}_mixed.safetensors"


def has_mols(cache: Path) -> bool:
    m = cache / "mols"
    try:
        return m.is_dir() and any(True for _ in os.scandir(m))
    except OSError:
        return False


def _safetensors_metadata(path: Path) -> dict:
    """The __metadata__ of a safetensors file, read from its header only (no tensor data)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if n > 100_000_000:
            raise ValueError("bad safetensors header")
        return json.loads(f.read(n)).get("__metadata__") or {}


def mixed_is_current(path: Path, name: str) -> bool:
    """The mixed checkpoint exists and was written for this build's fp16 manifest (else it must be rebuilt)."""
    from boltz_mlx import mixed as MX  # imports mlx; only called when the file exists

    try:
        meta = _safetensors_metadata(path)
    except (OSError, ValueError, struct.error):
        return False
    return meta.get("format") == MX.FORMAT and meta.get("manifest_sha") == MX.entry_sha(MX.load_manifest()[name])


# ------------------------------------------------------------------------------------------------ plan
def plan(cache=None, weights=None) -> dict:
    """What is missing and the steps that fix it. Never touches the network; reads only file names, sizes and headers."""
    c, w = locations(cache, weights)
    steps = []
    need_ckpt = []
    for n in MODELS:
        if not all((w / f).is_file() for f in weight_files(n)):
            need_ckpt.append(n)
            steps.append({"step": "convert", "model": n, "bytes_written": CONVERTED_BYTES[n] + MIXED_BYTES[n],
                          "detail": f"boltz2_{n}.ckpt -> {w}/boltz2_{n}.safetensors + mixed fp16 copy"})
        elif not (w / mixed_file(n)).is_file() or not mixed_is_current(w / mixed_file(n), n):
            steps.append({"step": "mixed", "model": n, "bytes_written": MIXED_BYTES[n],
                          "detail": f"{w}/{mixed_file(n)} (fp16 weights for the fast preset)"})
    downloads = []
    if not has_mols(c):
        steps.insert(0, {"step": "extract", "bytes_written": EXTRACTED_MOLS_BYTES, "detail": f"mols.tar -> {c}/mols"})
        downloads.append("mols.tar")
    downloads += [f"boltz2_{n}.ckpt" for n in need_ckpt]
    dl = []
    for a in (ASSETS[d] for d in downloads):
        f = c / a.name
        if f.is_file() and f.stat().st_size == a.size:
            continue
        part = c / (a.name + ".part")
        have = part.stat().st_size if part.is_file() and part.stat().st_size <= a.size else 0
        dl.append({"step": "download", "file": a.name, "bytes": a.size - have, "size": a.size, "resume_from": have,
                   "detail": f"{a.what} from {a.urls[-1].split('/resolve/')[0]}"})
    steps = dl + steps
    return {"cache": str(c), "weights": str(w), "steps": steps, "ready": not steps,
            "download_bytes": sum(s["bytes"] for s in dl),
            "disk_bytes": sum(s.get("bytes", 0) + s.get("bytes_written", 0) for s in steps)}


def describe(p: dict) -> str:
    if p["ready"]:
        return "nothing to do: weights and molecule dictionary are in place."
    dl = p["download_bytes"]
    return (f"{len(p['steps'])} step(s): " + (f"download {dl / GB:.1f} GB, " if dl else "nothing to download, ") +
            f"about {p['disk_bytes'] / GB:.0f} GB of disk "
            f"(cache {p['cache']}, weights {p['weights']}). Run `boltz-mlx setup`.")


# ------------------------------------------------------------------------------------------------ download
def fetch_url(url: str, start: int = 0, timeout: float = 60.0):
    """Open url (following redirects) from byte `start`. Returns (response, resumed: bool). Replaced in tests."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **({"Range": f"bytes={start}-"} if start else {})})
    r = urllib.request.urlopen(req, timeout=timeout)  # noqa: S310 (fixed https URLs above)
    return r, (start > 0 and getattr(r, "status", 200) == 206)


def sha256_file(path: Path, h=None, chunk: int = 1 << 24):
    h = h or hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h


class _Progress:
    def __init__(self, label: str, total: int, done: int, stream=None):
        self.label, self.total, self.t0, self.start, self.last = label, total, time.time(), done, 0.0
        self.stream = stream or sys.stderr
        self.tty = hasattr(self.stream, "isatty") and self.stream.isatty()
        self.next_pct = (done * 100 // max(total, 1) // 10 + 1) * 10

    def update(self, done: int, final: bool = False) -> None:
        now = time.time()
        pct = done * 100 / max(self.total, 1)
        rate = (done - self.start) / max(now - self.t0, 1e-6)
        eta = (self.total - done) / rate if rate > 0 else 0
        msg = (f"  {self.label:<18} {done / GB:5.2f} / {self.total / GB:.2f} GB {pct:5.1f}%  {rate / 1e6:6.1f} MB/s"
               + (f"  {int(eta // 60)}:{int(eta % 60):02d} left" if not final and rate > 0 else ""))
        if self.tty:
            if final or now - self.last > 0.25:
                self.stream.write("\r" + msg + ("\n" if final else ""))
                self.stream.flush()
                self.last = now
        elif final or pct >= self.next_pct:
            print(msg, file=self.stream, flush=True)
            while self.next_pct <= pct:
                self.next_pct += 10


def download(asset: Asset, cache: Path, log=print, fetch=None, stream=None) -> Path:
    """Download one asset into cache (resumable via <name>.part; size + sha256 verified before the final rename)."""
    fetch = fetch or fetch_url
    dst, part = cache / asset.name, cache / (asset.name + ".part")
    if dst.is_file() and dst.stat().st_size == asset.size:
        return dst
    cache.mkdir(parents=True, exist_ok=True)
    errors = []
    for attempt in range(2):  # a corrupt partial file is discarded and the download restarted once
        for url in asset.urls:
            have = part.stat().st_size if part.is_file() else 0
            if have > asset.size:
                part.unlink()
                have = 0
            try:
                r, resumed = (None, True) if have == asset.size else fetch(url, have)
                if r is not None:
                    if have and not resumed:
                        have = 0  # server ignored the Range request: start over
                    log(f"  {asset.name}: {'resuming at %.2f GB' % (have / GB) if have else 'downloading'} "
                        f"{asset.size / GB:.2f} GB ({asset.what})")
                    prog = _Progress(asset.name, asset.size, have, stream)
                    with r, open(part, "ab" if have else "wb") as f:
                        done = have
                        while True:
                            b = r.read(1 << 20)
                            if not b:
                                break
                            f.write(b)
                            done += len(b)
                            prog.update(done)
                    prog.update(done, final=True)
                break
            except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError, OSError) as ex:
                reason = getattr(ex, "reason", None) or ex
                errors.append(f"{url.split('/')[2]}: {reason}")
                continue
        else:
            raise SetupError(f"could not download {asset.name} ({'; '.join(errors[-len(asset.urls):])})",
                             "check the internet connection (or proxy / firewall) and run `boltz-mlx setup` again; the "
                             "download resumes where it stopped. On a machine without internet, copy boltz2_conf.ckpt, "
                             f"boltz2_aff.ckpt and mols/ (or mols.tar) from another computer into {cache}.")
        size = part.stat().st_size
        if size == asset.size:
            log(f"  {asset.name}: checking sha256")
            if sha256_file(part).hexdigest() == asset.sha256:
                os.replace(part, dst)
                return dst
            errors.append("sha256 mismatch after download")
        else:
            errors.append(f"size {size} != {asset.size} after download")
            if size < asset.size and attempt == 0:
                continue  # interrupted stream: resume from what is there
        part.unlink(missing_ok=True)
        if attempt == 0:
            log(f"  {asset.name}: {errors[-1]}; downloading again")
    raise SetupError(f"{asset.name} failed verification twice ({errors[-1]})",
                     "run `boltz-mlx setup` again later; if it keeps failing, the upstream file may have changed.")


def verify_existing(asset: Asset, cache: Path, log=print) -> None:
    """A checkpoint already in the cache (e.g. from upstream `boltz predict`) must be the validated one before converting."""
    f = cache / asset.name
    if f.stat().st_size != asset.size or sha256_file(f).hexdigest() != asset.sha256:
        raise SetupError(f"{f} is not the Boltz-2 file boltz-mlx was validated with (size or sha256 differ)",
                         f"move it away (e.g. `mv {f} {f}.old`) and run `boltz-mlx setup` again to download it.")


# ------------------------------------------------------------------------------------------------ extract
def extract_mols(cache: Path, log=print) -> None:
    tar = cache / "mols.tar"
    tmp = cache / ".mols.extracting"
    log(f"  extracting {tar.name} -> {cache / 'mols'} (45k small files, about a minute)")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    with tarfile.open(tar, "r") as t:
        t.extractall(tmp, filter="data")
    src = tmp / "mols"
    if not src.is_dir():
        raise SetupError(f"{tar} has no mols/ directory", f"delete {tar} and run `boltz-mlx setup` again.")
    dst = cache / "mols"
    if dst.exists():  # an empty / partial mols/ (has_mols was false)
        shutil.rmtree(dst)
    os.replace(src, dst)
    shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------------------------------------ convert
def _jsonable(x):
    from collections.abc import Mapping, Sequence

    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, Mapping):  # dict, omegaconf DictConfig
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, Sequence):  # list, tuple, omegaconf ListConfig
        return [_jsonable(v) for v in x]
    return str(x)


def convert_checkpoint(name: str, cache: Path, out: Path, log=print) -> dict:
    """boltz2_<name>.ckpt (Lightning) -> <out>/boltz2_<name>.safetensors + .json (hyper-parameters). Parameter names are
    kept verbatim (the MLX module tree mirrors the PyTorch names); training-only state and non-float buffers are dropped.
    The .json is written last: it marks a finished conversion."""
    import mlx.core as mx
    import numpy as np
    import torch

    ckpt = torch.load(cache / f"boltz2_{name}.ckpt", map_location="cpu", weights_only=False, mmap=True)
    sd = ckpt["state_dict"]
    tensors, dropped = {}, []
    for k, v in sd.items():
        if k.startswith(DROP_PREFIXES) or not torch.is_floating_point(v):
            dropped.append(k)
            continue
        tensors[k] = mx.array(v.detach().float().numpy())
    out.mkdir(parents=True, exist_ok=True)
    st, js = (out / f for f in weight_files(name))
    tmp = out / f"boltz2_{name}.tmp.safetensors"
    mx.save_safetensors(str(tmp), tensors)
    os.replace(tmp, st)
    tmpj = out / f"boltz2_{name}.json.tmp"
    tmpj.write_text(json.dumps(_jsonable(ckpt["hyper_parameters"]), indent=1))
    os.replace(tmpj, js)
    res = {"name": name, "path": str(st), "tensors": len(tensors), "dropped": len(dropped),
           "params": sum(int(np.prod(v.shape)) for v in tensors.values())}
    del tensors, sd, ckpt
    mx.clear_cache()
    return res


def convert_mixed(name: str, src_dir: Path, out_dir: Path, entry: dict) -> dict:
    """<src_dir>/boltz2_<name>.safetensors -> <out_dir>/boltz2_<name>_mixed.safetensors (boltz_mlx/mixed.py): fp16 copies of
    the manifest's parameters, fp32 for the rest. Fails on any fp16 overflow. One tensor at a time."""
    import mlx.core as mx

    from boltz_mlx import mixed as MX

    src = src_dir / f"boltz2_{name}.safetensors"
    dst = out_dir / mixed_file(name)
    t0 = time.time()
    src_sha = sha256_file(src).hexdigest()
    w = mx.load(str(src))
    params = set(entry["params"])
    missing = params - set(w)
    if missing:
        raise KeyError(f"{name}: manifest parameters missing from {src}: {sorted(missing)[:5]}")
    out, stats = {}, {}
    for k in sorted(w):
        v = w.pop(k)
        if k in params:  # the fp32 source is dropped once its fp16 copy exists
            if v.dtype != mx.float32:
                raise TypeError(f"{k}: expected an fp32 source, got {v.dtype}")
            stats[k] = MX.fp16_range(v)
            v = v.astype(mx.float16)
        mx.eval(v)
        out[k] = v
    bad = {k: s["overflow"] for k, s in stats.items() if s["overflow"]}
    if bad:
        raise OverflowError(f"{name}: fp16 overflow (|w| > {MX.FP16_MAX}) in {len(bad)} tensors: {bad}")
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"format": MX.FORMAT, "manifest_sha": MX.entry_sha(entry), "source": src.name, "source_sha256": src_sha}
    tmp = dst.with_suffix(".tmp.safetensors")
    mx.save_safetensors(str(tmp), out, metadata=meta)
    os.replace(tmp, dst)
    n16 = sum(out[k].size for k in params)
    n32 = sum(v.size for k, v in out.items() if k not in params)
    res = {"name": name, "path": str(dst), "bytes": dst.stat().st_size, "sha256": sha256_file(dst).hexdigest(),
           "source": str(src), "source_bytes": src.stat().st_size, "source_sha256": src_sha,
           "manifest_sha": meta["manifest_sha"], "n_tensors_fp16": len(params), "n_tensors_fp32": len(out) - len(params),
           "params_fp16": int(n16), "params_fp32": int(n32), "stats": stats, "seconds": round(time.time() - t0, 1)}
    del out, w
    mx.clear_cache()
    return res


# ------------------------------------------------------------------------------------------------ ensure
def _free_bytes(path: Path) -> int:
    p = path
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free


def ensure(cache=None, weights=None, log=print, fetch=None, prune: bool = False, stream=None) -> dict:
    """Run every missing setup step (see plan()). Returns the plan that was executed. Raises SetupError on failure."""
    p = plan(cache, weights)
    c, w = Path(p["cache"]), Path(p["weights"])
    if p["ready"]:
        if prune:
            _prune(c, log)
        return p
    log(f"boltz-mlx setup: {describe(p).replace(' Run `boltz-mlx setup`.', '')}")
    need = p["disk_bytes"] + 1_000_000_000
    free = min(_free_bytes(c), _free_bytes(w))
    if free < need:
        raise SetupError(f"not enough free disk space: {need / GB:.1f} GB needed, {free / GB:.1f} GB free",
                         "free some space, or point BOLTZ_MLX_CACHE (and BOLTZ_MLX_WEIGHTS) at a disk with more room.")
    t0 = time.time()
    for s in p["steps"]:
        if s["step"] == "download":
            download(ASSETS[s["file"]], c, log=log, fetch=fetch, stream=stream)
    import mlx.core as mx

    for s in p["steps"]:
        if s["step"] == "extract":
            extract_mols(c, log)
        elif s["step"] in ("convert", "mixed"):
            n = s["model"]
            from boltz_mlx import mixed as MX

            with mx.stream(mx.cpu):  # conversion never uses the GPU
                if s["step"] == "convert":
                    verify_existing(ASSETS[f"boltz2_{n}.ckpt"], c, log)
                    log(f"  converting boltz2_{n}.ckpt -> {w / weight_files(n)[0]}")
                    convert_checkpoint(n, c, w, log)
                log(f"  writing {w / mixed_file(n)} (fp16 weights for the fast preset)")
                convert_mixed(n, w, w, MX.load_manifest()[n])
    if prune:
        _prune(c, log)
    log(f"boltz-mlx setup: done in {(time.time() - t0) / 60:.1f} min.")
    return p


def _prune(cache: Path, log=print) -> None:
    """Delete the downloads that are no longer needed once the MLX weights and mols/ exist (frees ~6 GB). Upstream
    `boltz predict` with the same cache would download them again."""
    for n in ("mols.tar", "boltz2_conf.ckpt", "boltz2_aff.ckpt"):
        f = cache / n
        if f.is_file() or f.is_symlink():
            log(f"  removing {f} ({f.stat().st_size / GB:.1f} GB; no longer needed)")
            f.unlink()
