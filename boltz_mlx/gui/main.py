"""`boltz-mlx gui`: a local web interface for `boltz-mlx screen` (offline; listens on 127.0.0.1 only).

  boltz-mlx gui                      # http://127.0.0.1:8765 (next free port if busy), opens your browser
  boltz-mlx gui --no-browser --port 9000 --runs-dir ~/screens

Everything runs on this computer: uploads become files in <runs-dir>/<run-id>/inputs, a screen is a `boltz-mlx screen`
subprocess (the GUI never runs the model itself), and the pages are served from the package (Mol* is bundled; no CDN).
Stop = SIGTERM (the screen stops at a ligand boundary; Resume continues from results.jsonl).
First start: anything missing (Boltz-2 weights, molecule dictionary) is downloaded and converted first, as
`boltz-mlx setup` does (--no-setup skips that).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from boltz_mlx.gui.common import DEFAULT_PORT, DEFAULT_RUNS_DIR


def add_arguments(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR, help=f"where runs are kept (default {DEFAULT_RUNS_DIR}); a directory of "
                   f"`boltz-mlx screen` outputs also works (listed read-only)")
    p.add_argument("--port", type=int, default=None, help=f"port (default {DEFAULT_PORT}; the next free one if busy)")
    p.add_argument("--host", default="127.0.0.1", help="only 127.0.0.1 / localhost are accepted (anything else is refused)")
    p.add_argument("--no-browser", action="store_true", help="do not open the page in the default browser")
    p.add_argument("--cache", default=None, help="dir holding the upstream mols/ dictionary (as `screen --cache`)")
    p.add_argument("--weights", default=None, help="dir with boltz2_{conf,aff}.safetensors (as `screen --weights`)")
    p.add_argument("--no-setup", action="store_true", help="do not download / convert missing weights before starting")
    p.add_argument("--verbose", action="store_true", help="log requests and tracebacks to stderr")
    return p


def first_run_setup(args, log=print) -> None:
    """Download + convert whatever is missing (boltz_mlx/assets.py) before the server starts. A failure (e.g. offline) is
    reported and the GUI starts anyway: past runs stay browsable and the page names what is missing."""
    from boltz_mlx import assets

    try:
        if assets.plan(args.cache, args.weights)["ready"]:
            return
        log("boltz-mlx gui: first start, fetching what is missing (this happens once).")
        assets.ensure(args.cache, args.weights, log=log)
    except assets.SetupError as ex:
        log(f"boltz-mlx gui: setup did not finish: {ex.message}. Fix: {ex.fix}")


def add_parser(sub) -> argparse.ArgumentParser:
    p = sub.add_parser("gui", help="local web interface for `screen` (offline, 127.0.0.1 only)", description=__doc__,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    return add_arguments(p)


def run(args) -> int:
    from boltz_mlx.gui.api import GuiApp
    from boltz_mlx.gui.server import make_server, serve

    runs = Path(args.runs_dir).expanduser()
    if not getattr(args, "no_setup", False):
        first_run_setup(args, log=lambda m: print(m, file=sys.stderr, flush=True))
    try:
        app = GuiApp(runs, cache=args.cache, weights=args.weights)
        server = make_server(app, host=args.host, port=args.port, verbose=args.verbose)
    except (ValueError, OSError) as ex:
        print(f"boltz-mlx gui: {ex}", file=sys.stderr)
        return 2
    return serve(server, open_browser=not args.no_browser)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="boltz-mlx gui", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    return run(add_arguments(ap).parse_args(argv))
