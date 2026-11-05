# Mol* 4.5.0 viewer (vendored)

Two files, downloaded once (owner-approved, 2026-10-28) so the GUI works offline. Nothing in this directory is fetched at
run time, and nothing else is downloaded at install or build time.

| File | Source | Size (bytes) | sha256 |
|---|---|---|---|
| `molstar.js`  | https://cdn.jsdelivr.net/npm/molstar@4.5.0/build/viewer/molstar.js  | 4897515 | `24ccbabdc5e96849581f691206535650ea9db74d61e14a44f6e7bb45f64d07a6` |
| `molstar.css` | https://cdn.jsdelivr.net/npm/molstar@4.5.0/build/viewer/molstar.css | 69936   | `274c2a57d21c6a3e5f7f5d5e51ca0a58c2fef6bcd44b6b588dbe2f622c197e64` |

Mol* ("molstar" on npm; https://molstar.org, https://github.com/molstar/molstar) is MIT licensed: see `LICENSE`.
If you use the viewer in a publication, cite: Sehnal D, et al. Mol* Viewer: modern web app for 3D visualization and
analysis of large biomolecular structures. Nucleic Acids Res. 2021;49(W1):W431-W437.

Verification done when the files were added (details in DATA.md): HTTP 200 with `x-jsd-version: 4.5.0`; the bundle contains
the version string `"4.5.0"` and exports `Viewer` (`molstar.Viewer`); `molstar.css` contains no remote URL; the tests
(`tests/test_gui.py::test_vendored_molstar_matches_data_md`) pin both hashes.

The pages never call the Mol* data providers (RCSB, PDBe, AlphaFold DB, ...): structures are loaded from this
computer, and the Content-Security-Policy the server sends (`connect-src 'self'`) would block any other request.
