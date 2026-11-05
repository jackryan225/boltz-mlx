"""`boltz-mlx gui`: an offline, local web interface for `boltz-mlx screen`.

Python standard library only (plus the RDKit / PyYAML the screen already needs); the pages are plain HTML/CSS/JS in
`static/` (Mol* 4.5.0 is vendored in `static/vendor/molstar/`, nothing is loaded from the network). The server binds to
127.0.0.1 only and never runs the model itself: a screen is a `python -m boltz_mlx.cli screen ...` subprocess that writes
into one run directory (`<runs-dir>/<run-id>/out`); the GUI reads results.jsonl and the subprocess log from there.
"""
