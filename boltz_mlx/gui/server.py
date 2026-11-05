"""HTTP layer of the GUI: http.server only, 127.0.0.1 only.

Protections (a local web server that starts GPU jobs and reads the user's files must not be reachable from other machines or
from other web pages the user happens to have open):
  * binds 127.0.0.1 (anything else is refused);
  * Host must be 127.0.0.1:<port> or localhost:<port> (DNS-rebinding), Origin (when sent) the same origin;
  * every /api/ call carries a per-server random token (X-Boltz-Token) that only the page we serve contains, so another
    web page cannot call the API (cross-origin pages cannot read the token); POST bodies must be application/json;
  * request bodies are size-limited and read by Content-Length; static files and run files are served only from inside
    their roots (symlinks resolved), run files only by run id + result-row record id, never by a client-chosen path.
"""

from __future__ import annotations

import hmac
import json
import mimetypes
import posixpath
import re
import sys
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from boltz_mlx.gui.api import GuiApp
from boltz_mlx.gui.common import DEFAULT_PORT, LIMITS, LOCAL_HOSTS, PORT_TRIES, ApiError, is_within

STATIC = Path(__file__).resolve().parent / "static"
CSP_APP = ("default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; font-src 'self' data:; "
           "connect-src 'self'; frame-src {origins}; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
# The pose viewer page (Mol*): its bundle evaluates generated code (Emscripten glue), so this page - and only this page - gets
# 'unsafe-eval'. It is framed from the other loopback host name (http://localhost:PORT while the app is on 127.0.0.1:PORT), i.e. a
# different origin: it cannot read the app page, its token or the API, and gets poses only by postMessage.
CSP_VIEWER = ("default-src 'none'; script-src 'self' 'unsafe-eval'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; "
              "font-src 'self' data:; worker-src 'self' blob:; connect-src 'self' blob: data:; base-uri 'none'; form-action 'none'; "
              "frame-ancestors {origins}")
FRAMED = {"viewer.html"}
MIME = {".js": "text/javascript; charset=utf-8", ".mjs": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
        ".html": "text/html; charset=utf-8", ".json": "application/json", ".svg": "image/svg+xml", ".png": "image/png",
        ".ico": "image/x-icon", ".md": "text/plain; charset=utf-8", ".txt": "text/plain; charset=utf-8",
        ".woff2": "font/woff2", ".map": "application/json"}
SEG = r"([^/?#]+)"
ROUTES: list[tuple[str, "re.Pattern", str]] = []


def route(method: str, pattern: str):
    def deco(fn):
        ROUTES.append((method, re.compile("^" + pattern + "$"), fn.__name__))
        return fn
    return deco


class GuiServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 64

    def __init__(self, address, handler, app: GuiApp, static_dir: Path = STATIC, verbose: bool = False):
        super().__init__(address, handler)
        self.app = app
        self.static_dir = Path(static_dir).resolve()
        self.verbose = verbose
        self.port = self.server_address[1]
        self.allowed_hosts = {f"127.0.0.1:{self.port}", f"localhost:{self.port}"}
        self.allowed_origins = {f"http://{h}" for h in self.allowed_hosts}
        self.origins_csp = " ".join(sorted(self.allowed_origins))

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "boltz-mlx-gui"
    sys_version = ""
    timeout = 120

    # ------------------------------------------------------------------------------------------ plumbing
    def log_message(self, fmt, *args):  # noqa: D401
        if self.server.verbose:
            sys.stderr.write("[gui] %s - %s\n" % (self.address_string(), fmt % args))

    def _headers(self, status: int, ctype: str, length: int, extra: dict | None = None, csp: str | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(length))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        extra = dict(extra or {})
        self.send_header("Cross-Origin-Resource-Policy", extra.pop("Cross-Origin-Resource-Policy", "same-origin"))
        xfo = extra.pop("X-Frame-Options", "DENY")
        if xfo:  # None: the framed viewer page relies on its CSP frame-ancestors (X-Frame-Options cannot name two origins)
            self.send_header("X-Frame-Options", xfo)
        if csp:
            self.send_header("Content-Security-Policy", csp)
        for k, v in extra.items():
            self.send_header(k, v)
        self.end_headers()

    def send_bytes(self, status: int, body: bytes, ctype: str, extra: dict | None = None, csp: str | None = None) -> None:
        self._headers(status, ctype, len(body), {"Cache-Control": "no-store", **(extra or {})}, csp)
        if self.command != "HEAD":
            self.wfile.write(body)

    def send_json(self, status: int, obj) -> None:
        self.send_bytes(status, self.server.app.json(obj), "application/json")

    def send_error_json(self, status: int, message: str, close: bool = False) -> None:
        if close:
            self.close_connection = True
        self.send_bytes(status, self.server.app.json({"error": message}), "application/json",
                        {"Connection": "close"} if close else None)

    def send_file(self, path: Path, ctype: str, extra: dict | None = None, csp: str | None = None) -> None:
        st = path.stat()
        etag = f'"{st.st_mtime_ns:x}-{st.st_size:x}"'
        if self.headers.get("If-None-Match") == etag:
            self._headers(304, ctype, 0, {"ETag": etag, "Cache-Control": "no-cache"})
            return
        self._headers(200, ctype, st.st_size, {"ETag": etag, "Cache-Control": "no-cache", **(extra or {})}, csp=csp)
        if self.command == "HEAD":
            return
        with open(path, "rb") as f:
            while True:
                b = f.read(256 * 1024)
                if not b:
                    break
                self.wfile.write(b)

    # ------------------------------------------------------------------------------------------ guards
    def _guard(self, api: bool) -> bool:
        """Host / Origin / token checks. Sends the 403 and returns False when a request must not be served."""
        host = (self.headers.get("Host") or "").strip().lower()
        if host not in self.server.allowed_hosts:
            self.send_error_json(403, "forbidden host (this server answers only on 127.0.0.1 / localhost)", close=True)
            return False
        origin = self.headers.get("Origin")
        if origin is not None and origin.strip().lower() not in self.server.allowed_origins:
            self.send_error_json(403, "forbidden origin", close=True)
            return False
        if api and self.headers.get("Sec-Fetch-Site", "same-origin") not in ("same-origin", "none"):
            self.send_error_json(403, "cross-site request refused", close=True)
            return False
        if api:
            tok = self.headers.get("X-Boltz-Token") or ""
            if not hmac.compare_digest(tok.encode(), self.server.app.token.encode()):
                self.send_error_json(403, "missing or wrong session token: reload the page", close=True)
                return False
        return True

    def _read_json(self) -> dict:
        ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if ctype != "application/json":
            self.close_connection = True
            raise ApiError(415, "POST bodies must be application/json")
        if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
            self.close_connection = True
            raise ApiError(411, "chunked bodies are not supported (send Content-Length)")
        try:
            n = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self.close_connection = True
            raise ApiError(411, "Content-Length required") from None
        if n < 0 or n > LIMITS["body"]:
            self.close_connection = True
            raise ApiError(413, f"request too large (limit {LIMITS['body'] // (1024 * 1024)} MB)")
        raw = self.rfile.read(n) if n else b""
        if len(raw) != n:
            self.close_connection = True
            raise ApiError(400, "incomplete request body")
        try:
            obj = json.loads(raw.decode("utf-8")) if raw else {}
        except (ValueError, UnicodeDecodeError):
            raise ApiError(400, "body is not valid JSON") from None
        if not isinstance(obj, dict):
            raise ApiError(400, "expected a JSON object")
        return obj

    # ------------------------------------------------------------------------------------------ dispatch
    def _dispatch(self, method: str) -> None:
        try:
            sp = urlsplit(self.path)
            path = sp.path
            if not path.startswith("/") or "\x00" in path:
                self.send_error_json(400, "bad request", close=True)
                return
            query = {k: v[-1] for k, v in parse_qs(sp.query, keep_blank_values=True).items()}
            api = path.startswith("/api/")
            if not self._guard(api):
                return
            if method == "GET" and path == "/":
                return self.page_index()
            if method == "GET" and path.startswith("/static/"):
                return self.page_static(path[len("/static/"):])
            if method == "GET" and path == "/favicon.ico":
                return self.page_static("icon.svg")
            for m, rx, name in ROUTES:
                if m != method:
                    continue
                mt = rx.match(path)
                if mt:
                    segs = [unquote(g) for g in mt.groups()]
                    body = self._read_json() if method == "POST" else {}
                    return getattr(self, name)(*segs, query=query, body=body)
            if api and any(rx.match(path) for _, rx, _ in ROUTES):
                self.send_error_json(405, "method not allowed", close=True)
            else:
                self.send_error_json(404, "not found", close=(method == "POST"))
        except ApiError as ex:
            self.send_error_json(ex.status, ex.message, close=ex.status in (411, 413, 415))
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception as ex:  # noqa: BLE001
            if self.server.verbose:
                traceback.print_exc()
            try:
                self.send_error_json(500, f"internal error: {type(ex).__name__}: {str(ex)[:200]}")
            except Exception:  # noqa: BLE001
                self.close_connection = True

    def do_GET(self):
        self._dispatch("GET")

    def do_HEAD(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def do_PUT(self):
        self.send_error_json(405, "method not allowed", close=True)

    do_DELETE = do_PATCH = do_OPTIONS = do_PUT

    # ------------------------------------------------------------------------------------------ pages
    def page_index(self) -> None:
        html = (self.server.static_dir / "index.html").read_text(encoding="utf-8")
        html = html.replace("__BOLTZ_TOKEN__", self.server.app.token)
        self.send_bytes(200, html.encode("utf-8"), "text/html; charset=utf-8", csp=CSP_APP.format(origins=self.server.origins_csp))

    def page_static(self, rel: str) -> None:
        rel = unquote(rel)
        if "\x00" in rel or "\\" in rel or rel.startswith("/"):
            raise ApiError(404, "not found")
        norm = posixpath.normpath(rel)
        if norm.startswith("..") or norm == ".":
            raise ApiError(404, "not found")
        p = self.server.static_dir / norm
        if not is_within(p, self.server.static_dir) or not p.is_file():
            raise ApiError(404, "not found")
        ext = p.suffix.lower()
        ctype = MIME.get(ext) or mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        if ctype.startswith("text/html"):
            o = self.server.origins_csp
            if norm in FRAMED:
                self.send_file(p, ctype, {"X-Frame-Options": None, "Cross-Origin-Resource-Policy": "cross-origin"}, csp=CSP_VIEWER.format(origins=o))
            else:
                self.send_file(p, ctype, csp=CSP_APP.format(origins=o))
        else:
            self.send_file(p, ctype)

    # ------------------------------------------------------------------------------------------ API
    @route("GET", "/api/info")
    def api_info(self, query, body):
        self.send_json(200, self.server.app.info())

    @route("GET", "/api/presets")
    def api_presets(self, query, body):
        from boltz_mlx.screen.presets import EST_LOAD_S, EST_OVERHEAD_S

        from boltz_mlx.gui import estimates as EST
        from boltz_mlx.gui import presets as GP

        self.send_json(200, {"presets": self.server.app.presets_payload(), "overhead_s": EST_OVERHEAD_S, "load_s": EST_LOAD_S,
                             "gui": GP.payload(EST.chip_name())})

    # ------------------------------------------------------------------------------------------ target library
    @route("GET", "/api/targets")
    def api_targets(self, query, body):
        self.send_json(200, {"targets": self.server.app.list_targets()})

    @route("POST", "/api/targets")
    def api_target_prepare(self, query, body):
        self.send_json(201, self.server.app.prepare_target(body))

    @route("GET", f"/api/targets/{SEG}")
    def api_target(self, tid, query, body):
        self.send_json(200, self.server.app.get_target(tid))

    @route("POST", f"/api/targets/{SEG}/rename")
    def api_target_rename(self, tid, query, body):
        self.send_json(200, self.server.app.targets.rename(tid, (body or {}).get("name") if isinstance(body, dict) else None))

    @route("POST", f"/api/targets/{SEG}/delete")
    def api_target_delete(self, tid, query, body):
        self.send_json(200, self.server.app.delete_target(tid))

    @route("POST", "/api/validate")
    def api_validate(self, query, body):
        self.send_json(200, self.server.app.validate(body))

    @route("POST", "/api/estimate")
    def api_estimate(self, query, body):
        self.send_json(200, self.server.app.estimate(body))

    @route("POST", "/api/lookup")
    def api_lookup(self, query, body):
        from boltz_mlx.gui import lookup

        self.send_json(200, lookup.lookup(body))

    @route("POST", "/api/search")
    def api_search(self, query, body):
        from boltz_mlx.gui import lookup

        self.send_json(200, lookup.search(body))

    @route("POST", "/api/depict")
    def api_depict(self, query, body):
        svg = self.server.app.depict(str(body.get("smiles") or ""), bool(body.get("dark")), int(body.get("w") or 320), int(body.get("h") or 240))
        self.send_bytes(200, svg.encode(), "image/svg+xml")

    @route("GET", "/api/runs")
    def api_runs(self, query, body):
        self.send_json(200, {"runs": self.server.app.runs.list_runs(), "runs_dir": str(self.server.app.runs.root),
                             "active_run": self.server.app.runs.active_run_id()})

    @route("POST", "/api/runs")
    def api_create(self, query, body):
        self.send_json(201, self.server.app.create_run(body))

    @route("GET", f"/api/runs/{SEG}")
    def api_run(self, run_id, query, body):
        self.send_json(200, self.server.app.runs.detail(run_id))

    @route("POST", f"/api/runs/{SEG}/stop")
    def api_stop(self, run_id, query, body):
        self.send_json(200, self.server.app.stop(run_id, force=bool(body.get("force"))))

    @route("POST", f"/api/runs/{SEG}/resume")
    def api_resume(self, run_id, query, body):
        self.send_json(200, self.server.app.start_existing(run_id, retry_errors=bool(body.get("retry_errors"))))

    @route("GET", f"/api/runs/{SEG}/results")
    def api_results(self, run_id, query, body):
        runs = self.server.app.runs
        runs.run_dir(run_id)
        rank_by = query.get("rank_by") or (runs.read_meta(run_id) or {}).get("options", {}).get("rank_by") or "prob"
        self.send_json(200, runs.results(run_id, rank_by, query.get("v")))

    @route("GET", f"/api/runs/{SEG}/results\\.csv")
    def api_csv(self, run_id, query, body):
        runs = self.server.app.runs
        runs.run_dir(run_id)
        rank_by = query.get("rank_by") or (runs.read_meta(run_id) or {}).get("options", {}).get("rank_by") or "prob"
        if ((runs.read_meta(run_id) or {}).get("kind")) == "predict":
            from boltz_mlx.gui.exports import predict_csv

            text = predict_csv(runs, run_id)
        else:   # ?ids=a,b: only those ligands (record ids), same columns and ranks as the whole table
            ids = {x for x in (query.get("ids") or "").split(",") if x} if "ids" in query else None
            text = runs.csv(run_id, rank_by, ids)
        self.send_bytes(200, text.encode("utf-8"), "text/csv; charset=utf-8",
                        {"Content-Disposition": f'attachment; filename="{run_id[:60]}_results.csv"'})

    @route("POST", f"/api/runs/{SEG}/results\\.csv")
    def api_csv_ids(self, run_id, query, body):
        """The results.csv rows of some ligands only (record ids in the body), same columns and ranks as the whole table."""
        runs = self.server.app.runs
        runs.run_dir(run_id)
        b = body if isinstance(body, dict) else {}
        ids = b.get("ids")
        if not isinstance(ids, list) or not all(isinstance(x, str) for x in ids):
            raise ApiError(400, "ids must be a list of record ids")
        rank_by = b.get("rank_by") or (runs.read_meta(run_id) or {}).get("options", {}).get("rank_by") or "prob"
        self.send_bytes(200, runs.csv(run_id, rank_by, set(ids)).encode("utf-8"), "text/csv; charset=utf-8")

    @route("GET", f"/api/runs/{SEG}/export\\.zip")
    def api_export(self, run_id, query, body):
        from boltz_mlx.gui.exports import build_zip

        runs = self.server.app.runs
        runs.run_dir(run_id)
        path, name = build_zip(runs, run_id, query.get("format"), query.get("rank_by"))
        try:
            self.send_file(path, "application/zip", {"Content-Disposition": f'attachment; filename="{name}"', "Cache-Control": "no-store"})
        finally:
            path.unlink(missing_ok=True)

    def _send_pose(self, p: Path, query: dict, name: str) -> None:
        fmt = query.get("format")
        if not fmt:
            self.send_file(p, "text/plain; charset=utf-8", {"X-Pose-Format": "pdb" if p.suffix == ".pdb" else "mmcif"})
            return
        from boltz_mlx.gui.exports import pose_bytes

        data, ext = pose_bytes(p, fmt)
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)[:80]
        self.send_bytes(200, data, "chemical/x-pdb" if ext == "pdb" else "chemical/x-mmcif",
                        {"Content-Disposition": f'attachment; filename="{safe}.{ext}"'})

    @route("POST", f"/api/runs/{SEG}/recheck")
    def api_recheck(self, run_id, query, body):
        self.send_json(201, self.server.app.recheck(run_id, body))

    @route("GET", f"/api/runs/{SEG}/inputs")
    def api_run_inputs(self, run_id, query, body):
        self.server.app.runs.run_dir(run_id)
        self.send_json(200, self.server.app.run_inputs(run_id))

    @route("GET", f"/api/runs/{SEG}/compare")
    def api_compare(self, run_id, query, body):
        self.send_json(200, self.server.app.compare(run_id))

    @route("GET", f"/api/runs/{SEG}/log")
    def api_log(self, run_id, query, body):
        from boltz_mlx.gui import progress as P

        q = str(query.get("lines") or "200")
        n = max(10, min(int(q), 2000)) if q.isdigit() else 200
        lines = P.read_log(self.server.app.runs.log_path(run_id), 0)
        self.send_bytes(200, ("\n".join(lines[-n:]) + "\n").encode("utf-8", "replace"), "text/plain; charset=utf-8")

    @route("GET", f"/api/runs/{SEG}/ligand/{SEG}")
    def api_ligand(self, run_id, record_id, query, body):
        self.send_json(200, self.server.app.runs.ligand_detail(run_id, record_id))

    @route("GET", f"/api/runs/{SEG}/ligand/{SEG}/pose")
    def api_pose(self, run_id, record_id, query, body):
        ref = query.get("align_to")
        if ref and ref != record_id:  # lead optimisation: in the frame of another pose of this run
            text, fmt, rmsd, n = self.server.app.aligned_pose(run_id, record_id, ref)
            self.send_bytes(200, text.encode(), "text/plain; charset=utf-8",
                            {"X-Pose-Format": fmt, "X-Align-RMSD": f"{rmsd:.3f}", "X-Align-N": str(n)})
            return
        p = self.server.app.runs.pose_file(run_id, record_id)
        self._send_pose(p, query, f"{run_id}_{record_id}")

    @route("GET", f"/api/runs/{SEG}/ligand/{SEG}/pae")
    def api_ligand_pae(self, run_id, record_id, query, body):
        self.send_json(200, self.server.app.runs.ligand_pae(run_id, record_id))

    @route("GET", f"/api/runs/{SEG}/prediction")
    def api_prediction(self, run_id, query, body):
        self.send_json(200, self.server.app.runs.prediction(run_id, query.get("model")))

    @route("GET", f"/api/runs/{SEG}/ligand/{SEG}/pae\\.npz")
    def api_ligand_pae_npz(self, run_id, record_id, query, body):
        f = self.server.app.runs.ligand_pae_file(run_id, record_id)
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in f"{run_id}_{record_id}_pae")[:80]
        self.send_file(f, "application/octet-stream", {"Content-Disposition": f'attachment; filename="{safe}.npz"'})

    @route("GET", f"/api/runs/{SEG}/prediction/pose")
    def api_prediction_pose(self, run_id, query, body):
        p = self.server.app.runs.prediction_pose(run_id, query.get("model"))
        self._send_pose(p, query, f"{run_id}_model_{query.get('model') or 0}")

    @route("GET", f"/api/runs/{SEG}/prediction/pae")
    def api_prediction_pae(self, run_id, query, body):
        self.send_json(200, self.server.app.runs.prediction_pae(run_id, query.get("model")))

    @route("GET", f"/api/runs/{SEG}/prediction/pae\\.npz")
    def api_prediction_pae_npz(self, run_id, query, body):
        f = self.server.app.runs.prediction_pae_file(run_id, query.get("model"))
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in f"{run_id}_model_{query.get('model') or 0}_pae")[:80]
        self.send_file(f, "application/octet-stream", {"Content-Disposition": f'attachment; filename="{safe}.npz"'})

    @route("POST", "/api/predict/import")
    def api_predict_import(self, query, body):
        from boltz_mlx.gui import predict as PR

        self.send_json(200, PR.import_yaml((body or {}).get("content") if isinstance(body, dict) else None))

    @route("POST", "/api/predict/validate")
    def api_predict_validate(self, query, body):
        self.send_json(200, self.server.app.validate_predict(body))

    @route("GET", f"/api/runs/{SEG}/ligand/{SEG}/2d\\.svg")
    def api_2d(self, run_id, record_id, query, body):
        dark = query.get("theme") == "dark"
        if query.get("ref"):
            svg, info = self.server.app.depict_vs(run_id, record_id, query["ref"], dark=dark)
            self.send_bytes(200, svg.encode(), "image/svg+xml", {"X-Changed-Atoms": str(info["changed_atoms"]),
                                                                "X-MCS-Atoms": str(info["mcs_atoms"]),
                                                                "X-MCS-Incomplete": "1" if info.get("mcs_timed_out") else "0"})
            return
        svg = self.server.app.depict_ligand(run_id, record_id, dark=dark)
        self.send_bytes(200, svg.encode(), "image/svg+xml")


# ====================================================================================================== start-up
def make_server(app: GuiApp, host: str = "127.0.0.1", port: int | None = None, static_dir: Path = STATIC,
                verbose: bool = False) -> GuiServer:
    """Bind 127.0.0.1 only. port None: try 8765, 8766, ... then any free port; an explicit port (0 = any) must be free."""
    if host not in LOCAL_HOSTS:
        raise ValueError(f"refusing to bind to {host!r}: the GUI reads your files and starts jobs, so it only listens on 127.0.0.1 "
                         f"(use an SSH tunnel for remote access)")
    tries = [int(port)] if port is not None else list(range(DEFAULT_PORT, DEFAULT_PORT + PORT_TRIES)) + [0]
    last: OSError | None = None
    for p in tries:
        try:
            return GuiServer(("127.0.0.1", p), Handler, app, static_dir=static_dir, verbose=verbose)
        except OSError as ex:
            last = ex
            if port is not None:
                break
    raise OSError(f"could not bind 127.0.0.1:{DEFAULT_PORT if port is None else port}: {last}")


def serve(server: GuiServer, open_browser: bool = True, log=print) -> int:
    app = server.app
    log(f"Boltz-2 MLX GUI: {server.url}   (runs in {app.runs.root})")
    log("  listening on 127.0.0.1 only; Ctrl-C to quit. A running screen keeps going in the background if you quit "
        "(reopen the GUI to follow or stop it).")
    for rid in app.runs.adopt_note():
        log(f"  a screen from an earlier session is still running: {rid}")
    for problem in app.info()["problems"]:
        log(f"  warning: {problem}")
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(server.url)).start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    running = app.runs.adopt_note()
    for rid in running:
        launch = (app.runs.read_meta(rid) or {}).get("launches", [{}])[-1]
        log(f"screen {rid} is still running (pid {launch.get('pid')}); it continues in the background. "
            f"Stop it with the GUI, or `kill -TERM {launch.get('pid')}` (it stops at a ligand boundary; resume later).")
    return 0
