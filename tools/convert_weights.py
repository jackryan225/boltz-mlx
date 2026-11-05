"""Convert Boltz-2 Lightning checkpoints to safetensors + JSON config for MLX (developer tool).

Users do not need this: `boltz-mlx setup` (or the first `boltz-mlx gui`) downloads and converts everything. This script
runs only the conversion, from checkpoints already in a cache, into a chosen directory.

Usage:
  python tools/convert_weights.py [--cache DIR] [--out DIR] [--no-mixed]   # converts conf and aff checkpoints
Reads <cache>/boltz2_{conf,aff}.ckpt (cache: --cache > $BOLTZ_MLX_CACHE > $BOLTZ_CACHE > ~/.boltz, boltz_mlx/paths.py) and writes
<out>/boltz2_{conf,aff}.safetensors + .json (out default: <cache>/mlx_weights, where boltz_mlx.paths.weights_dir looks),
then the mixed fp16/fp32 checkpoints <out>/boltz2_{conf,aff}_mixed.safetensors used by --precision f16 (skipped with
--no-mixed). The conversion code is boltz_mlx/assets.py (convert_checkpoint, convert_mixed).
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None, help="dir with boltz2_{conf,aff}.ckpt")
    ap.add_argument("--out", default=None, help="output dir (default <cache>/mlx_weights)")
    ap.add_argument("--no-mixed", action="store_true", help="skip the mixed fp16 checkpoints (--precision f16)")
    a = ap.parse_args()
    import boltz_mlx  # noqa: F401
    import mlx.core as mx

    from boltz_mlx import assets, mixed
    from boltz_mlx.paths import default_cache

    mx.set_default_device(mx.cpu)
    cache = default_cache(a.cache)
    out = Path(a.out).expanduser() if a.out else cache / "mlx_weights"
    for n in assets.MODELS:
        r = assets.convert_checkpoint(n, cache, out)
        print(f"{n}: kept {r['tensors']} tensors ({r['params'] / 1e6:.1f} M params), dropped {r['dropped']} -> {r['path']}")
    if not a.no_mixed:
        manifest = mixed.load_manifest()
        for n in assets.MODELS:
            r = assets.convert_mixed(n, out, out, manifest[n])
            st = r["stats"].values()
            print(f"{n}: mixed {r['path']} {r['bytes'] / 1e9:.2f} GB, {r['n_tensors_fp16']} fp16 tensors, overflow "
                  f"{sum(s['overflow'] for s in st)}, subnormal {sum(s['subnormal'] for s in st)}, sha256 {r['sha256']}")
