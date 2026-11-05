"""First-run setup (boltz_mlx/assets.py, `boltz-mlx setup`, `boltz-mlx check`, the GUI's first start).

No test touches the network: downloads go through a stubbed fetch (fake files with their own sizes and sha256), and the
conversion steps are checked on tiny synthetic checkpoints. The real URLs are checked by tools/check_asset_urls.py (HEAD).
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import urllib.error
from pathlib import Path

import pytest

from boltz_mlx import assets as A


def _asset(name, data, urls=("https://a.example/f", "https://b.example/f")):
    return A.Asset(name, tuple(u + "/" + name for u in urls), len(data), hashlib.sha256(data).hexdigest(), "test file")


class _Resp(io.BytesIO):
    def __init__(self, data, status):
        super().__init__(data)
        self.status = status


def _fetcher(blobs, fail_hosts=(), ignore_range=False, truncate=None, calls=None):
    """fetch(url, start) over in-memory blobs keyed by file name. fail_hosts raise URLError."""

    def fetch(url, start=0, timeout=60):
        if calls is not None:
            calls.append((url, start))
        host = url.split("/")[2]
        if host in fail_hosts:
            raise urllib.error.URLError(OSError(8, "nodename nor servname provided, or not known"))
        data = blobs[url.rsplit("/", 1)[1]]
        if ignore_range or not start:
            body, status = data, 200
        else:
            body, status = data[start:], 206
        if truncate is not None:
            body = body[:truncate]
        return _Resp(body, status), (start > 0 and status == 206)

    return fetch


def _quiet(*a, **k):
    pass


# ------------------------------------------------------------------------------------------------ download
def test_download_fresh_verifies_and_renames(tmp_path):
    data = b"x" * 3_000_000
    a = _asset("m.ckpt", data)
    calls = []
    p = A.download(a, tmp_path, log=_quiet, fetch=_fetcher({"m.ckpt": data}, calls=calls), stream=io.StringIO())
    assert p.read_bytes() == data and not (tmp_path / "m.ckpt.part").exists()
    assert calls == [(a.urls[0], 0)]
    calls.clear()
    A.download(a, tmp_path, log=_quiet, fetch=_fetcher({"m.ckpt": data}, calls=calls))  # idempotent: no network
    assert calls == []


def test_download_resumes_from_part_file(tmp_path):
    data = bytes(range(256)) * 10_000
    a = _asset("m.ckpt", data)
    (tmp_path / "m.ckpt.part").write_bytes(data[:1_000_000])
    calls = []
    A.download(a, tmp_path, log=_quiet, fetch=_fetcher({"m.ckpt": data}, calls=calls), stream=io.StringIO())
    assert calls == [(a.urls[0], 1_000_000)] and (tmp_path / "m.ckpt").read_bytes() == data


def test_download_restarts_when_server_ignores_range(tmp_path):
    data = bytes(range(256)) * 4000
    a = _asset("m.ckpt", data)
    (tmp_path / "m.ckpt.part").write_bytes(data[:5000])
    A.download(a, tmp_path, log=_quiet, fetch=_fetcher({"m.ckpt": data}, ignore_range=True), stream=io.StringIO())
    assert (tmp_path / "m.ckpt").read_bytes() == data


def test_download_falls_back_to_second_source(tmp_path):
    data = b"y" * 1000
    a = _asset("m.ckpt", data)
    calls = []
    A.download(a, tmp_path, log=_quiet, fetch=_fetcher({"m.ckpt": data}, fail_hosts=("a.example",), calls=calls),
               stream=io.StringIO())
    assert [c[0].split("/")[2] for c in calls] == ["a.example", "b.example"] and (tmp_path / "m.ckpt").exists()


def test_download_offline_is_a_clear_setup_error(tmp_path):
    a = _asset("m.ckpt", b"z" * 10)
    with pytest.raises(A.SetupError) as ex:
        A.download(a, tmp_path, log=_quiet, fetch=_fetcher({}, fail_hosts=("a.example", "b.example")))
    assert "could not download m.ckpt" in ex.value.message and "a.example" in ex.value.message
    assert "internet connection" in ex.value.fix and "resumes" in ex.value.fix and str(tmp_path) in ex.value.fix
    assert not (tmp_path / "m.ckpt").exists()


def test_download_interrupted_stream_resumes_then_completes(tmp_path):
    data = bytes(range(256)) * 100
    a = _asset("m.ckpt", data)
    state = {"n": 0}
    full = _fetcher({"m.ckpt": data})

    def flaky(url, start=0, timeout=60):  # first response stops half way, the retry continues from the .part file
        state["n"] += 1
        r, resumed = full(url, start)
        if state["n"] == 1:
            return _Resp(r.read()[: len(data) // 2], r.status), resumed
        return r, resumed

    A.download(a, tmp_path, log=_quiet, fetch=flaky, stream=io.StringIO())
    assert (tmp_path / "m.ckpt").read_bytes() == data and state["n"] == 2


def test_download_checksum_mismatch_retries_then_fails(tmp_path):
    good = b"g" * 100
    a = _asset("m.ckpt", good)
    bad = b"b" * 100  # right size, wrong content
    with pytest.raises(A.SetupError) as ex:
        A.download(a, tmp_path, log=_quiet, fetch=_fetcher({"m.ckpt": bad}), stream=io.StringIO())
    assert "verification" in ex.value.message and not (tmp_path / "m.ckpt").exists()
    assert not (tmp_path / "m.ckpt.part").exists()


def test_progress_lines_without_a_tty():
    s = io.StringIO()
    p = A._Progress("f", 1000, 0, s)
    for d in range(0, 1001, 50):
        p.update(d)
    p.update(1000, final=True)
    lines = s.getvalue().splitlines()
    assert 10 <= len(lines) <= 12 and "100.0%" in lines[-1] and "GB" in lines[0]


# ------------------------------------------------------------------------------------------------ plan / ensure
def _mols_tar(n=3) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as t:
        info = tarfile.TarInfo("mols")
        info.type = tarfile.DIRTYPE
        t.addfile(info)
        for i in range(n):
            b = f"mol{i}".encode()
            info = tarfile.TarInfo(f"mols/M{i}.pkl")
            info.size = len(b)
            t.addfile(info, io.BytesIO(b))
    return buf.getvalue()


@pytest.fixture
def fake_assets(monkeypatch, tmp_path):
    blobs = {"mols.tar": _mols_tar(), "boltz2_conf.ckpt": b"c" * 5000, "boltz2_aff.ckpt": b"a" * 4000}
    monkeypatch.setattr(A, "ASSETS", {k: _asset(k, v) for k, v in blobs.items()})
    monkeypatch.delenv("BOLTZ_MLX_WEIGHTS", raising=False)
    converted = []

    def fake_convert(name, cache, out, log=print):
        assert (cache / f"boltz2_{name}.ckpt").read_bytes() == blobs[f"boltz2_{name}.ckpt"]
        out.mkdir(parents=True, exist_ok=True)
        for f in A.weight_files(name):
            (out / f).write_text("{}")
        converted.append(("convert", name))

    def fake_mixed(name, src, out, entry):
        (out / A.mixed_file(name)).write_bytes(b"m")
        converted.append(("mixed", name))

    monkeypatch.setattr(A, "convert_checkpoint", fake_convert)
    monkeypatch.setattr(A, "convert_mixed", fake_mixed)
    monkeypatch.setattr(A, "mixed_is_current", lambda path, name: path.read_bytes() == b"m")
    return blobs, converted


def test_plan_empty_cache_lists_everything(tmp_path, fake_assets):
    blobs, _ = fake_assets
    p = A.plan(tmp_path / "cache", tmp_path / "w")
    assert not p["ready"]
    assert [s["step"] for s in p["steps"]] == ["download"] * 3 + ["extract", "convert", "convert"]
    assert p["download_bytes"] == sum(len(b) for b in blobs.values())
    assert "Run `boltz-mlx setup`" in A.describe(p)


def test_ensure_first_run_then_idempotent(tmp_path, fake_assets):
    blobs, converted = fake_assets
    c, w = tmp_path / "cache", tmp_path / "w"
    calls, logs = [], []
    A.ensure(c, w, log=logs.append, fetch=_fetcher(blobs, calls=calls), stream=io.StringIO())
    assert sorted(p.name for p in (c / "mols").iterdir()) == ["M0.pkl", "M1.pkl", "M2.pkl"]
    assert not (c / ".mols.extracting").exists()
    assert converted == [("convert", "conf"), ("mixed", "conf"), ("convert", "aff"), ("mixed", "aff")]
    assert len(calls) == 3 and A.plan(c, w)["ready"]
    assert any("setup: done" in m for m in logs)
    calls.clear(), converted.clear()
    p = A.ensure(c, w, log=_quiet, fetch=_fetcher(blobs, calls=calls))
    assert p["ready"] and calls == [] and converted == []


def test_ensure_reuses_an_upstream_cache(tmp_path, fake_assets):
    """A cache filled by upstream `boltz predict` (checkpoints + mols/) is converted without downloading anything."""
    blobs, converted = fake_assets
    c = tmp_path / "cache"
    (c / "mols").mkdir(parents=True)
    (c / "mols" / "ALA.pkl").write_bytes(b"x")
    for n in ("boltz2_conf.ckpt", "boltz2_aff.ckpt"):
        (c / n).write_bytes(blobs[n])
    p = A.plan(c, tmp_path / "w")
    assert [s["step"] for s in p["steps"]] == ["convert", "convert"] and p["download_bytes"] == 0
    A.ensure(c, tmp_path / "w", log=_quiet, fetch=_fetcher({}, fail_hosts=("a.example", "b.example")))
    assert [x for x, _ in converted] == ["convert", "mixed", "convert", "mixed"]


def test_ensure_refuses_a_different_checkpoint(tmp_path, fake_assets):
    blobs, _ = fake_assets
    c = tmp_path / "cache"
    (c / "mols").mkdir(parents=True)
    (c / "mols" / "ALA.pkl").write_bytes(b"x")
    (c / "boltz2_conf.ckpt").write_bytes(b"C" * 5000)  # right size, other content
    (c / "boltz2_aff.ckpt").write_bytes(blobs["boltz2_aff.ckpt"])
    with pytest.raises(A.SetupError) as ex:
        A.ensure(c, tmp_path / "w", log=_quiet)
    assert "not the Boltz-2 file" in ex.value.message and "mv " in ex.value.fix


def test_ensure_only_rebuilds_stale_mixed(tmp_path, fake_assets):
    blobs, converted = fake_assets
    c, w = tmp_path / "cache", tmp_path / "w"
    A.ensure(c, w, log=_quiet, fetch=_fetcher(blobs), stream=io.StringIO())
    (w / A.mixed_file("aff")).write_bytes(b"old build")
    converted.clear()
    p = A.plan(c, w)
    assert [(s["step"], s.get("model")) for s in p["steps"]] == [("mixed", "aff")] and p["download_bytes"] == 0
    A.ensure(c, w, log=_quiet, fetch=_fetcher({}))
    assert converted == [("mixed", "aff")]


def test_ensure_checks_disk_space(tmp_path, fake_assets, monkeypatch):
    blobs, _ = fake_assets
    monkeypatch.setattr(A, "_free_bytes", lambda p: 10)
    with pytest.raises(A.SetupError) as ex:
        A.ensure(tmp_path / "c", tmp_path / "w", log=_quiet, fetch=_fetcher(blobs))
    assert "disk space" in ex.value.message and "BOLTZ_MLX_CACHE" in ex.value.fix


def test_prune_removes_downloads_only(tmp_path, fake_assets):
    blobs, _ = fake_assets
    c, w = tmp_path / "cache", tmp_path / "w"
    A.ensure(c, w, log=_quiet, fetch=_fetcher(blobs), stream=io.StringIO(), prune=True)
    assert not any((c / n).exists() for n in blobs) and (c / "mols").is_dir() and A.plan(c, w)["ready"]


# ------------------------------------------------------------------------------------------------ conversion (tiny)
def test_convert_checkpoint_and_mixed_on_a_tiny_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    import mlx.core as mx

    from boltz_mlx import mixed as MX

    sd = {"a.weight": torch.randn(4, 3), "b.bias": torch.randn(3).half(), "val_lddt.total": torch.tensor(1.0),
          "steps": torch.tensor(3)}
    torch.save({"state_dict": sd, "hyper_parameters": {"x": 1, "p": Path("/tmp")}}, tmp_path / "boltz2_conf.ckpt")
    with mx.stream(mx.cpu):
        r = A.convert_checkpoint("conf", tmp_path, tmp_path / "w")
        assert r["tensors"] == 2 and r["dropped"] == 2
        w = mx.load(str(tmp_path / "w" / "boltz2_conf.safetensors"))
        assert set(w) == {"a.weight", "b.bias"} and all(v.dtype == mx.float32 for v in w.values())
        assert json.loads((tmp_path / "w" / "boltz2_conf.json").read_text()) == {"x": 1, "p": "/tmp"}
        entry = {"params": ["a.weight"], "groups": []}
        m = A.convert_mixed("conf", tmp_path / "w", tmp_path / "w", entry)
        wm, meta = mx.load(m["path"], return_metadata=True)
    assert wm["a.weight"].dtype == mx.float16 and wm["b.bias"].dtype == mx.float32
    assert A._safetensors_metadata(Path(m["path"])) == meta and meta["manifest_sha"] == MX.entry_sha(entry)


def test_mixed_is_current_reads_the_header(tmp_path, monkeypatch):
    import mlx.core as mx

    from boltz_mlx import mixed as MX

    monkeypatch.setattr(MX, "load_manifest", lambda *a: {"conf": {"params": ["a"], "groups": []}})
    p = tmp_path / "m.safetensors"
    mx.save_safetensors(str(p), {"a": mx.zeros(2)}, metadata={"format": MX.FORMAT,
                                                             "manifest_sha": MX.entry_sha({"params": ["a"], "groups": []})})
    assert A.mixed_is_current(p, "conf")
    mx.save_safetensors(str(p), {"a": mx.zeros(2)}, metadata={"format": MX.FORMAT, "manifest_sha": "old"})
    assert not A.mixed_is_current(p, "conf")
    p.write_bytes(b"garbage")
    assert not A.mixed_is_current(p, "conf")


def test_pinned_assets_match_upstream_names():
    """Sources are the ones upstream boltz uses; the checksums are the Hugging Face LFS oids of the validated files."""
    assert set(A.ASSETS) == {"mols.tar", "boltz2_conf.ckpt", "boltz2_aff.ckpt"}
    for a in A.ASSETS.values():
        assert a.urls[-1] == f"https://huggingface.co/boltz-community/boltz-2/resolve/main/{a.name}"
        assert len(a.sha256) == 64 and a.size > 1e9
    try:
        import boltz.main as up
    except Exception:  # noqa: BLE001
        return
    assert A.ASSETS["mols.tar"].urls == (up.MOL_URL,)
    assert list(A.ASSETS["boltz2_conf.ckpt"].urls) == up.BOLTZ2_URL_WITH_FALLBACK
    assert list(A.ASSETS["boltz2_aff.ckpt"].urls) == up.BOLTZ2_AFFINITY_URL_WITH_FALLBACK


# ------------------------------------------------------------------------------------------------ paths / check / CLI / GUI
def test_default_weights_dir_is_in_the_cache(tmp_path, monkeypatch):
    from boltz_mlx import paths

    monkeypatch.delenv("BOLTZ_MLX_WEIGHTS", raising=False)
    monkeypatch.setenv("BOLTZ_MLX_CACHE", str(tmp_path / "c"))
    monkeypatch.setattr(paths, "_PKG_ROOT", tmp_path / "pkg")
    assert paths.weights_dir() == tmp_path / "c" / "mlx_weights"
    monkeypatch.delenv("BOLTZ_MLX_CACHE")
    monkeypatch.delenv("BOLTZ_CACHE", raising=False)
    assert paths.default_cache() == Path.home() / ".boltz"


def test_check_reports_setup_steps(tmp_path, monkeypatch):
    from boltz_mlx import check

    monkeypatch.delenv("BOLTZ_MLX_WEIGHTS", raising=False)
    r = check.run_checks(tmp_path / "cache", tmp_path / "w")
    bad = {c["name"]: c for c in r["checks"] if not c["ok"]}
    assert {"weights", "cache_mols"} <= set(bad) and "boltz-mlx setup" in bad["weights"]["fix"]
    assert "6.2 GB" in bad["weights"]["fix"]
    assert [s["step"] for s in r["setup"]["steps"]][:3] == ["download"] * 3
    assert "`boltz-mlx setup` would:" in check.format_text(r)


def test_cli_setup_dry_run_json(tmp_path, monkeypatch, capsys):
    from boltz_mlx import cli

    monkeypatch.delenv("BOLTZ_MLX_WEIGHTS", raising=False)
    with pytest.raises(SystemExit) as ex:
        cli.main(["setup", "--dry-run", "--json", "--cache", str(tmp_path / "c"), "--weights", str(tmp_path / "w")])
    assert ex.value.code == 0
    obj = json.loads(capsys.readouterr().out)
    assert obj["command"] == "setup" and obj["dry_run"] and obj["download_bytes"] == sum(a.size for a in A.ASSETS.values())
    assert not (tmp_path / "c").exists()  # a dry run writes nothing


def test_cli_setup_offline_exit_3(tmp_path, monkeypatch, capsys):
    from boltz_mlx import cli

    monkeypatch.delenv("BOLTZ_MLX_WEIGHTS", raising=False)
    monkeypatch.setattr(A, "fetch_url", _fetcher({}, fail_hosts=("model-gateway.boltz.bio", "huggingface.co")))
    with pytest.raises(SystemExit) as ex:
        cli.main(["setup", "--json", "--cache", str(tmp_path / "c"), "--weights", str(tmp_path / "w")])
    assert ex.value.code == 3
    out, err = capsys.readouterr()
    obj = json.loads(out)
    assert obj["error"]["kind"] == "setup" and "could not download mols.tar" in obj["error"]["message"]
    assert "internet connection" in err


def test_gui_first_start_runs_setup_and_survives_offline(tmp_path, monkeypatch):
    import argparse

    from boltz_mlx.gui import main as G

    calls, logs = [], []
    monkeypatch.setattr(A, "plan", lambda c, w: {"ready": False})

    def fail(c, w, log=print):
        calls.append((c, w))
        raise A.SetupError("could not download mols.tar (offline)", "check the internet connection")

    monkeypatch.setattr(A, "ensure", fail)
    G.first_run_setup(argparse.Namespace(cache="C", weights="W"), log=logs.append)
    assert calls == [("C", "W")] and "first start" in logs[0] and "Fix: check the internet" in logs[-1]
    calls.clear()
    monkeypatch.setattr(A, "plan", lambda c, w: {"ready": True})
    G.first_run_setup(argparse.Namespace(cache=None, weights=None), log=logs.append)
    assert calls == []


def test_real_urllib_download_with_redirect_and_range(tmp_path):
    """fetch_url (the real urllib path) against a local HTTP server: a redirect like the gateway -> Hugging Face hop,
    and a Range request resuming a .part file (206). Loopback only, nothing leaves the machine."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    data = bytes(range(256)) * 2000
    seen = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            seen.append((self.path, self.headers.get("Range")))
            if self.path == "/gw/m.ckpt":
                self.send_response(307)
                self.send_header("Location", "/hf/m.ckpt")
                self.end_headers()
                return
            start = int(self.headers["Range"].split("=")[1].rstrip("-")) if self.headers.get("Range") else 0
            body = data[start:]
            self.send_response(206 if start else 200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        a = A.Asset("m.ckpt", (base + "/gw/m.ckpt",), len(data), hashlib.sha256(data).hexdigest(), "test")
        (tmp_path / "m.ckpt.part").write_bytes(data[:100_000])
        A.download(a, tmp_path, log=_quiet, stream=io.StringIO())
    finally:
        srv.shutdown()
    assert (tmp_path / "m.ckpt").read_bytes() == data
    assert seen == [("/gw/m.ckpt", "bytes=100000-"), ("/hf/m.ckpt", "bytes=100000-")]
