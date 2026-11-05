"""Diffusion-step profile in the current mode (set BOLTZ_MLX_PRECISION / BOLTZ_MLX_DIFF_DTYPE before running).

  python tools/prof/diff_step.py lig_aff --affinity      # aff checkpoint, 5 samples (the affinity pass)
  python tools/prof/diff_step.py prot400                 # conf checkpoint, 1 sample

Prints the compiled full-step time (what the sampler runs) and an eager per-component split (each component synced;
sums exceed the compiled step because of the syncs and lost fusion). Rows -> logs/diff_step.jsonl.
"""

from __future__ import annotations
import os

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # this checkout's boltz_mlx first: the editable install maps boltz_mlx to the main checkout
import boltz_mlx  # noqa: F401
import mlx.core as mx
import numpy as np

from boltz_mlx import kernels as K
from boltz_mlx.model import load_model

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("BOLTZ_MLX_DATA", Path(__file__).resolve().parents[2] / "data")).expanduser()


def _arrs(x):
    """mx.eval-able form of a conditioning object (RawCond / RawCond16 hold arrays as attributes)."""
    if isinstance(x, tuple):
        return tuple(_arrs(y) for y in x)
    return [x.h1, x.h2] if hasattr(x, "h1") else x


def timeit(f, n=10):
    mx.eval(f())
    t = time.perf_counter()
    for _ in range(n):
        mx.eval(f())
    return (time.perf_counter() - t) / n * 1e3


def main():
    case = sys.argv[1]
    aff = "--affinity" in sys.argv
    mult = 5 if aff else 1
    model = load_model("aff" if aff else "conf")
    z = np.load(DATA / case / ("feats_affinity.npz" if aff else "feats.npz"))
    feats = {k: mx.array(z[k]) for k in z.files}
    s_inputs, s, zz, rpe = model.trunk(feats, 1)
    q, c, to_keys, eb, db, tb = model.diffusion_conditioning(s, zz, rpe, feats)
    cond = dict(q=q, c=c, to_keys=to_keys, atom_enc_bias=eb, atom_dec_bias=db, token_trans_bias=tb)
    sm = model.structure_module
    sc = sm.score_model
    cond["prepared"] = sc.prepare(feats, cond, mult)
    cond["shared_times"] = True
    mx.eval(q, c, eb, db, tb, cond["prepared"])
    n_atoms = feats["atom_pad_mask"].shape[1]
    x = mx.random.normal((mult, n_atoms, 3)) * 10
    sig = mx.full((mult,), 5.0)

    def den(x, sig):
        K.GUARD_SUSPEND += 1
        try:
            return sm.preconditioned_network_forward(x, sig, dict(s_inputs=s_inputs, s_trunk=s, feats=feats,
                                                                  diffusion_conditioning=cond, multiplicity=mult))
        finally:
            K.GUARD_SUSPEND -= 1

    comp = mx.compile(den)
    rows = {"case": case, "affinity": aff, "multiplicity": mult, "n_tokens": int(z["token_pad_mask"].sum()),
            "n_atoms": int(z["atom_pad_mask"].sum()), "precision": K.PRECISION, "diff_dtype": str(K.DIFF_DTYPE),
            "nax": {k: K.nax(k) for k in K.NAX_FLAGS}}
    rows["compiled_step_ms"] = round(timeit(lambda: comp(x, sig), 20), 3)
    rows["eager_step_ms"] = round(timeit(lambda: den(x, sig), 5), 3)

    # eager components (same code path as DiffusionModule.__call__ with shared times)
    prep = cond["prepared"]
    times = sig[:1] * 0 + 0.1
    s1, _ = sc.single_conditioner(times, s, s_inputs)
    tok_cond = sc.token_transformer.precompute_cond_shared(s1, mult) if prep.get("tok") is not None else None
    sr = mx.repeat(s1, mult, axis=0)
    split = {}
    split["single_cond+tok_cond"] = timeit(lambda: _arrs((sc.single_conditioner(times, s, s_inputs)[0],
                                                          sc.token_transformer.precompute_cond_shared(s1, mult)
                                                          if prep.get("tok") is not None else s1)))
    enc_out = sc.atom_attention_encoder(feats=feats, q=q, c=c, atom_enc_bias=eb, to_keys=to_keys, r=x,
                                        multiplicity=mult, prepared=prep.get("enc"))
    mx.eval(enc_out)
    split["atom_encoder"] = timeit(lambda: sc.atom_attention_encoder(feats=feats, q=q, c=c, atom_enc_bias=eb, to_keys=to_keys,
                                                                     r=x, multiplicity=mult, prepared=prep.get("enc"))[0])
    a, q_skip, c_skip, tk = enc_out
    mask = mx.repeat(feats["token_pad_mask"], mult, axis=0).astype(mx.float32)
    tprep = prep.get("tok") if tok_cond is None else {**prep["tok"], "cond": tok_cond}
    split["token_transformer"] = timeit(lambda: sc.token_transformer(a, mask=mask, s=sr, bias=tb, multiplicity=mult,
                                                                     prepared=tprep))
    a2 = sc.a_norm(sc.token_transformer(a, mask=mask, s=sr, bias=tb, multiplicity=mult, prepared=tprep))
    mx.eval(a2)
    split["atom_decoder"] = timeit(lambda: sc.atom_attention_decoder(a=a2, q=q_skip, c=c_skip, atom_dec_bias=db, feats=feats,
                                                                     multiplicity=mult, to_keys=tk, prepared=prep.get("dec")))
    rows["eager_split_ms"] = {k: round(v, 3) for k, v in split.items()}
    rows["git"] = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], text=True).strip()
    rows["date"] = time.strftime("%Y-%m-%d %H:%M:%S")
    print(json.dumps(rows, indent=1))
    with open(ROOT / "logs" / "diff_step.jsonl", "a") as f:
        f.write(json.dumps(rows) + "\n")


if __name__ == "__main__":
    main()
