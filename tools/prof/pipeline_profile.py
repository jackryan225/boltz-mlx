"""Whole-pipeline profile (R6-prof): where one T400 screen ligand (preset screen: f16, 50-step structure pass, 200 x 5
affinity pass) and prot400 @50 spend their time, with the costs OUTSIDE the pair-stack kernels broken out. GPU job.

  python tools/prof/pipeline_profile.py t400      # ~3 min: real screen engine (featurization, writers) + both passes
  python tools/prof/pipeline_profile.py prot400   # ~1 min: cached features, structure pass @50
  python tools/prof/pipeline_profile.py prot117   # the same views for prot117 (small-size host overhead, O3)
  python tools/prof/pipeline_profile.py report    # CPU: ranked tables from logs/pipeline_profile.jsonl

Views (rows -> logs/pipeline_profile.jsonl, one row per measurement, `exp` names the view):
  plain     the shipped code path (Boltz2.__call__, the screen's engine calls) timed only at its own sync points:
            model timings (trunk / diffusion / confidence / affinity), per-ligand CPU work, writers; plus a host
            timeline: time blocked in mx.eval vs busy building graphs, per model phase (kernels.PHASE_HOOK).
  pipe      the R6 flags in the same process on warm shapes, ABAB: kernels.PIPE_LAYERS + sampler.PIPELINE off vs on.
  synced    the same call with every listed submodule wrapped: its inputs are evaluated first ("pre": the glue before
            it), then Python builds its graph ("build": host only, GPU idle), then it is evaluated ("eval": GPU + the
            encode that does not overlap). The diffusion sampler is replaced by an op-for-op copy that evaluates after
            each sub-step (noise, augmentation, network, alignment, update). Pipelining is off in this view so that
            "build" is pure graph construction. Outputs are checked bitwise against the plain run.
  cprofile  one plain run (pipelining off) under cProfile: host->device syncs (mx.eval / .item() / array exports)
            with counts and blocked time, and Python time per boltz_mlx function (inflated ~1.5x by the profiler).
  shapes    the first prediction at a shape vs a warm repeat: in-process first call (compile retrace, MLX compiled
            helpers, kernel libraries from the system Metal cache), a new N seen before on this machine, and a
            never-seen N (cold Metal JIT of every N-templated kernel).
"""

from __future__ import annotations

import argparse
import cProfile
import gc
import json
import os
import pstats
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # this checkout's boltz_mlx first: the editable install maps boltz_mlx to the main checkout
DATA = Path(os.environ.get("BOLTZ_MLX_DATA", Path(__file__).resolve().parents[2] / "data")).expanduser()
OUT = ROOT / "logs" / "pipeline_profile.jsonl"
T400_TARGET = ROOT / "cases" / "lig_aff.yaml"
T400_MSA = Path(__file__).resolve().parents[2] / "cases" / "msa" / "seq1.a3m"
T400_LIGANDS = Path(os.environ.get("BOLTZ_MLX_EXT_DATA", Path(__file__).resolve().parents[2] / "data" / "external")).expanduser() / "fep" / "t400_ligands.csv"
CACHE = Path(os.environ.get("BOLTZ_MLX_CACHE", Path.home() / ".boltz")).expanduser()
# bace_cat-13a/b/c: N 406/407/408, run by every round check (system Metal cache warm); bace_cat-13g: 27 atoms -> N 411,
# never run on this machine before R6 (cold Metal JIT for every N-templated kernel)
WARM_LIGS = ["bace_cat-13a", "bace_cat-13b"]
COLD_LIG = "bace_cat-13g"


def sh(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


GIT = sh(f"git -C {ROOT} rev-parse --short HEAD")
PKG = sh(f"git -C {ROOT} rev-parse --short HEAD:boltz_mlx")


def emit(row: dict) -> None:
    row = {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "git": GIT, "pkg_tree": PKG, **row}
    print(json.dumps(row), flush=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(row) + "\n")


# ============================================================================================================ helpers
def _mx():
    import mlx.core as mx

    return mx


def arrays_in(x, acc: list, depth: int = 0) -> list:
    """Every mx.array reachable through tuples / lists / dicts / simple objects (RawCond, ToKeys) — to evaluate them."""
    mx = _mx()
    if isinstance(x, mx.array):
        acc.append(x)
    elif isinstance(x, (list, tuple)):
        for y in x:
            arrays_in(y, acc, depth + 1)
    elif isinstance(x, dict):
        for y in x.values():
            arrays_in(y, acc, depth + 1)
    elif depth < 3 and hasattr(x, "__dict__") and not callable(x):
        for y in vars(x).values():
            if isinstance(y, (mx.array, list, tuple, dict)):
                arrays_in(y, acc, depth + 1)
    return acc


class Rec:
    """name -> [pre_s, build_s, eval_s, calls]."""

    def __init__(self):
        self.d = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
        self.order: list[str] = []

    def add(self, name, pre, build, ev):
        if name not in self.d:
            self.order.append(name)
        r = self.d[name]
        r[0] += pre
        r[1] += build
        r[2] += ev
        r[3] += 1

    def rows(self) -> list[dict]:
        return [{"name": k, "pre_ms": round(self.d[k][0] * 1e3, 3), "build_ms": round(self.d[k][1] * 1e3, 3),
                 "eval_ms": round(self.d[k][2] * 1e3, 3), "calls": self.d[k][3]} for k in self.order]


class Timed:
    """Synced wrapper: eval inputs (pre), call (build), eval outputs (eval). Attribute access passes through."""

    def __init__(self, fn, name, rec: Rec, label=None):
        object.__setattr__(self, "_fn", fn)
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_rec", rec)
        object.__setattr__(self, "_label", label)

    def __call__(self, *a, **kw):
        mx = _mx()
        t0 = time.perf_counter()
        ins = arrays_in((a, kw), [])
        if ins:
            mx.eval(ins)
        t1 = time.perf_counter()
        out = self._fn(*a, **kw)
        t2 = time.perf_counter()
        outs = arrays_in(out, [])
        if outs:
            mx.eval(outs)
        t3 = time.perf_counter()
        name = self._label(a, kw) if self._label else self._name
        self._rec.add(name, t1 - t0, t2 - t1, t3 - t2)
        return out

    def __getattr__(self, k):
        return getattr(self._fn, k)


class Wrapping:
    def __init__(self, rec: Rec):
        self.rec = rec
        self.undo = []

    def wrap(self, owner, attr: str, name: str, label=None):
        if owner is None or not hasattr(owner, attr):
            return
        object.__setattr__(owner, attr, Timed(getattr(owner, attr), name, self.rec, label))
        self.undo.append((owner, attr))

    def restore(self):
        for owner, attr in reversed(self.undo):
            object.__delattr__(owner, attr)
        self.undo = []


def install_wrappers(model, w: Wrapping) -> None:
    """The submodules timed in the synced view (whole layer stacks only: per-layer wrappers would break the PAIRLNX
    carry and distort the per-layer pipeline)."""
    w.wrap(model, "input_embedder", "embed.input_embedder",
           label=lambda a, kw: "affinity.input_embedder(affinity=True)" if kw.get("affinity") else "embed.input_embedder")
    w.wrap(model, "rel_pos", "embed.rel_pos")
    w.wrap(model, "contact_conditioning", "embed.contact_conditioning")
    if getattr(model, "use_templates", False):
        w.wrap(model, "template_module", "trunk.template_module")
    mm = model.msa_module
    for layer in mm.layers:
        w.wrap(layer, "pair_weighted_averaging", "trunk.msa.pwa")
        w.wrap(layer, "msa_transition", "trunk.msa.transition")
        w.wrap(layer, "outer_product_mean", "trunk.msa.opm")
        w.wrap(layer, "pairformer_layer", "trunk.msa.pairformer_layer")
    w.wrap(model, "msa_module", "trunk.msa_module(total)")
    w.wrap(model, "pairformer_module", "trunk.pairformer(64 layers)")
    w.wrap(model, "distogram_module", "distogram")
    dc = model.diffusion_conditioning
    w.wrap(dc, "pairwise_conditioner", "diff.cond.pairwise_conditioner")
    w.wrap(dc, "atom_encoder", "diff.cond.atom_encoder")
    w.wrap(model, "diffusion_conditioning", "diff.conditioning(total)")
    sc = model.structure_module.score_model
    w.wrap(sc, "prepare", "diff.prepare")
    if getattr(model, "confidence", False):
        cm = model.confidence_module
        w.wrap(cm, "rel_pos", "confidence.rel_pos")
        w.wrap(cm, "contact_conditioning", "confidence.contact_conditioning")
        w.wrap(cm, "pairformer_stack", "confidence.pairformer(8 layers)")
        w.wrap(cm, "confidence_heads", "confidence.heads+ptm(host frames)")
        w.wrap(model, "confidence_module", "confidence(total)")
    if getattr(model, "affinity", False):
        for k in ("affinity_module1", "affinity_module2"):
            am = getattr(model, k)
            w.wrap(am, "pairwise_conditioner", f"affinity.{k}.pairwise_conditioner")
            w.wrap(am, "pairformer_stack", f"affinity.{k}.pairformer(8 layers)")
            w.wrap(am, "affinity_heads", f"affinity.{k}.heads")
            w.wrap(model, k, f"affinity.{k}(total)")


def instrumented_sample(rec: Rec):
    """Op-for-op copy of sampler.sample (Euler) that evaluates after each sub-step."""
    from math import sqrt

    import numpy as np

    from boltz_mlx import sampler as S

    mx = _mx()

    def sample(denoise, atom_mask, num_steps, multiplicity, noise, align, p=S.DiffusionParams(), eval_every=1):
        assert p.solver == "euler"
        atom_mask = mx.repeat(atom_mask, multiplicity, axis=0)
        shape = (*atom_mask.shape, 3)
        sigmas = S.sample_schedule(p, num_steps)
        gammas = np.where(sigmas > p.gamma_min, np.float32(p.gamma_0), np.float32(0.0)).astype(np.float32)
        x = float(sigmas[0]) * noise.randn(shape)
        mx.eval(x)

        def ph(name, fn):
            t0 = time.perf_counter()
            out = fn()
            t1 = time.perf_counter()
            mx.eval(arrays_in(out, []))
            rec.add(name, 0.0, t1 - t0, time.perf_counter() - t1)
            return out

        for step in range(num_steps):
            sigma_tm, sigma_t, gamma = float(sigmas[step]), float(sigmas[step + 1]), float(gammas[step + 1])
            t_hat = sigma_tm * (1 + gamma)
            noise_var = p.noise_scale**2 * (t_hat**2 - sigma_tm**2)
            q, tr, eps = ph("diff.step.noise_draws", lambda: (noise.randn((multiplicity, 4)), noise.randn((multiplicity, 1, 3)),
                                                              noise.randn(shape)))
            x_noisy = ph("diff.step.augment(pre_step)",
                         lambda: S._pre_step(x, q, tr, eps, mx.array(sqrt(noise_var), mx.float32)))
            name = "diff.step.network[first call: trace+compile]" if step == 0 else "diff.step.network"
            x_den = ph(name, lambda: denoise(x_noisy, t_hat))
            if p.alignment_reverse_diff and align is not None:
                x_noisy = ph("diff.step.align(CPU SVD)", lambda: align(x_noisy, x_den, atom_mask, atom_mask))
            x = ph("diff.step.update(post_step)",
                   lambda: S._post_step(x_noisy, x_den, mx.array(t_hat, mx.float32),
                                        mx.array(p.step_scale * (sigma_t - t_hat), mx.float32)))
        return x

    return sample


class Timeline:
    """Unsynced host timeline: phase changes (kernels.PHASE_HOOK) and the time blocked in mx.eval / array.item()
    (host<->device syncs), per phase. np.array exports of device arrays use the buffer protocol and are not counted."""

    def __init__(self):
        self.events = []

    def __enter__(self):
        from boltz_mlx import kernels as K

        mx = _mx()
        self.K, self.mx = K, mx
        self.real_eval, self.real_async, self.real_item = mx.eval, mx.async_eval, mx.array.item
        self.phase = "start"
        self.t0 = time.perf_counter()
        self.blocked = defaultdict(float)
        self.n_eval = defaultdict(int)
        self.n_item = defaultdict(int)
        self.n_async = defaultdict(int)
        self.async_s = defaultdict(float)
        self.phase_t = defaultdict(float)
        self.last = self.t0
        tl = self

        def item(arr):
            t = time.perf_counter()
            r = tl.real_item(arr)
            tl.blocked[tl.phase] += time.perf_counter() - t
            tl.n_item[tl.phase] += 1
            return r

        def hook(name):
            now = time.perf_counter()
            tl.phase_t[tl.phase] += now - tl.last
            tl.phase, tl.last = name, now

        def ev(*a):
            t = time.perf_counter()
            r = tl.real_eval(*a)
            tl.blocked[tl.phase] += time.perf_counter() - t
            tl.n_eval[tl.phase] += 1
            return r

        def aev(*a):
            t = time.perf_counter()
            r = tl.real_async(*a)
            tl.async_s[tl.phase] += time.perf_counter() - t
            tl.n_async[tl.phase] += 1
            return r

        K.PHASE_HOOK = hook
        mx.eval, mx.async_eval, mx.array.item = ev, aev, item
        return self

    def __exit__(self, *exc):
        now = time.perf_counter()
        self.phase_t[self.phase] += now - self.last
        self.K.PHASE_HOOK = None
        self.mx.eval, self.mx.async_eval, self.mx.array.item = self.real_eval, self.real_async, self.real_item
        return False

    def summary(self) -> dict:
        return {p: {"wall_ms": round(self.phase_t[p] * 1e3, 1), "blocked_in_eval_ms": round(self.blocked[p] * 1e3, 1),
                    "host_busy_ms": round((self.phase_t[p] - self.blocked[p] - self.async_s[p]) * 1e3, 1),
                    "in_async_eval_ms": round(self.async_s[p] * 1e3, 1),
                    "n_eval": self.n_eval[p], "n_item": self.n_item[p], "n_async_eval": self.n_async[p]}
                for p in self.phase_t}


def set_pipe(on: bool) -> None:
    from boltz_mlx import kernels as K
    from boltz_mlx import sampler as S

    K.PIPE_LAYERS = on
    S.PIPELINE = on


def fingerprint(out: dict, keys) -> str:
    import hashlib

    import numpy as np

    h = hashlib.sha256()
    for k in keys:
        if k in out:
            h.update(np.ascontiguousarray(np.array(out[k]).astype(np.float32)).tobytes())
    return h.hexdigest()[:12]


STRUCT_KEYS = ("sample_atom_coords", "plddt", "pae", "iptm", "complex_plddt")
AFF_KEYS = ("sample_atom_coords", "iptm", "affinity_pred_value", "affinity_probability_binary")


def run_plain(model, feats, args, noise_fn, mw=False, timeline=False):
    """The shipped call. Returns (out, wall_s, model timings, timeline summary or None)."""
    mx = _mx()
    tm = {}
    tl = Timeline() if timeline else None
    t = time.perf_counter()
    if tl:
        with tl:
            out = model(feats, args, noise_fn(), tm, mw_correction=mw)
            mx.eval(out["sample_atom_coords"])
    else:
        out = model(feats, args, noise_fn(), tm, mw_correction=mw)
        mx.eval(out["sample_atom_coords"])
    wall = time.perf_counter() - t
    return out, wall, {k: round(v, 4) for k, v in tm.items()}, (tl.summary() if tl else None)


def run_synced(model, feats, args, noise_fn, mw=False):
    """All wrappers + instrumented sampler, pipelining off."""
    import boltz_mlx.model as BM

    mx = _mx()
    rec = Rec()
    w = Wrapping(rec)
    install_wrappers(model, w)
    real_sample = BM.sample
    BM.sample = instrumented_sample(rec)
    set_pipe(False)
    tm = {}
    try:
        t = time.perf_counter()
        out = model(feats, args, noise_fn(), tm, mw_correction=mw)
        mx.eval(out["sample_atom_coords"])
        wall = time.perf_counter() - t
    finally:
        BM.sample = real_sample
        w.restore()
        set_pipe(True)
    return out, wall, {k: round(v, 4) for k, v in tm.items()}, rec.rows()


def run_cprofile(model, feats, args, noise_fn, mw=False, top=45):
    mx = _mx()
    set_pipe(False)
    pr = cProfile.Profile()
    try:
        t = time.perf_counter()
        pr.enable()
        out = model(feats, args, noise_fn(), None, mw_correction=mw)
        mx.eval(out["sample_atom_coords"])
        pr.disable()
        wall = time.perf_counter() - t
    finally:
        set_pipe(True)
    st = pstats.Stats(pr)
    syncs, funcs = [], []
    for (fn, line, name), (cc, nc, tt, ct, _callers) in st.stats.items():
        if fn == "~" and any(s in name for s in ("eval", "item", "tolist", "__array__", "synchronize", "numpy.array",
                                                 "metal_kernel", "compile")):
            syncs.append({"fn": name, "calls": nc, "tottime_ms": round(tt * 1e3, 1), "cumtime_ms": round(ct * 1e3, 1)})
        elif "boltz_mlx" in fn or "/sampler" in fn:
            funcs.append({"fn": f"{Path(fn).parent.name}/{Path(fn).name}:{line}({name})", "calls": nc,
                          "tottime_ms": round(tt * 1e3, 1), "cumtime_ms": round(ct * 1e3, 1)})
    syncs.sort(key=lambda r: -r["tottime_ms"])
    funcs.sort(key=lambda r: -r["cumtime_ms"])
    return out, wall, syncs[:20], funcs[:top]


# ============================================================================================================ T400
def t400(a):
    import numpy as np

    from boltz_mlx.cli import _structure_pred_dict
    from boltz_mlx.model import PredictArgs, load_model
    from boltz_mlx.sampler import record_noise
    from boltz_mlx.screen.engine import UpstreamEngine
    from boltz_mlx.screen.inputs import load_target, read_ligands, validate_ligand

    mx = _mx()
    case = "t400"
    t = time.perf_counter()
    target = load_target(str(T400_TARGET), msa=str(T400_MSA))
    all_ligs = {l.ligand_id: l for l in (validate_ligand(x) for x in read_ligands(str(T400_LIGANDS), None, None))}
    ligs = [all_ligs[k] for k in WARM_LIGS + [COLD_LIG]]
    import tempfile

    work = Path(tempfile.gettempdir()) / "boltz_pipeline_profile" / time.strftime("%Y%m%d_%H%M%S")
    eng = UpstreamEngine(work, CACHE, target, 8192, "mmcif", 0, keep_extras=True)
    emit({"exp": "plain", "case": case, "part": "process: target + ligand list + engine init (once per screen)",
          "s": round(time.perf_counter() - t, 3)})

    # ---- CPU side per ligand (structure phase)
    batches = {}
    for lig in ligs:
        t0 = time.perf_counter()
        eng.prepare(lig)
        t1 = time.perf_counter()
        b = eng.featurize(lig, affinity=False)
        t2 = time.perf_counter()
        feats = eng.model_inputs(b)
        mx.eval(list(feats.values()))
        t3 = time.perf_counter()
        batches[lig.ligand_id] = (b, feats)
        emit({"exp": "plain", "case": case, "ligand": lig.ligand_id, "part": "cpu.structure_features",
              "prepare_s": round(t1 - t0, 3), "featurize_s": round(t2 - t1, 3), "model_inputs_s": round(t3 - t2, 3),
              "n_tokens": int(np.array(feats["token_pad_mask"]).sum()), "n_atoms": int(feats["atom_pad_mask"].shape[1]),
              "msa_rows": int(feats["msa"].shape[1])})

    # ---- structure phase
    t = time.perf_counter()
    model = load_model("conf")
    emit({"exp": "plain", "case": case, "part": "load conf checkpoint", "s": round(time.perf_counter() - t, 3),
          "weights": getattr(model, "weights_kind", "?")})
    sargs = PredictArgs(recycling_steps=3, sampling_steps=50, diffusion_samples=1)

    def snoise(lig):
        return lambda: record_noise(0, lig.record_id, "structure", 1)

    outs = {}
    l0 = ligs[0]
    b0, f0 = batches[l0.ligand_id]
    order = [("first-in-process", True), ("warm", False), ("warm", True), ("warm", False), ("warm", True)]
    fps = set()
    for i, (kind, pipe) in enumerate(order):
        set_pipe(pipe)
        out, wall, tm, tl = run_plain(model, f0, sargs, snoise(l0), timeline=(i >= 3))
        fps.add(fingerprint(out, STRUCT_KEYS))
        emit({"exp": "pipe" if i else "shapes", "case": case, "pass": "structure", "ligand": l0.ligand_id, "run": i,
              "kind": kind, "pipe": pipe, "wall_s": round(wall, 3), "timings": tm, "timeline": tl,
              "fingerprint": fingerprint(out, STRUCT_KEYS)})
        outs[l0.ligand_id] = out
    set_pipe(True)
    emit({"exp": "pipe_bitwise", "case": case, "pass": "structure", "fingerprints": sorted(fps), "identical": len(fps) == 1})

    # synced view + cProfile on the warm shape
    out_s, wall, tm, rows = run_synced(model, f0, sargs, snoise(l0))
    emit({"exp": "synced", "case": case, "pass": "structure", "ligand": l0.ligand_id, "wall_s": round(wall, 3),
          "timings": tm, "rows": rows, "fingerprint": fingerprint(out_s, STRUCT_KEYS),
          "equals_plain": fingerprint(out_s, STRUCT_KEYS) in fps})
    del out_s
    out_p, wall, syncs, funcs = run_cprofile(model, f0, sargs, snoise(l0))
    emit({"exp": "cprofile", "case": case, "pass": "structure", "ligand": l0.ligand_id, "wall_s": round(wall, 3),
          "syncs": syncs, "funcs": funcs})
    del out_p

    # runner extras + writers (ligand 0)
    out = outs[l0.ligand_id]
    t0 = time.perf_counter()
    pl = np.array(out["plddt"]).reshape(-1)
    asym = eng.ligand_asym(b0, target.ligand_chain)
    a_ = np.array(f0["asym_id"]).reshape(-1)[: pl.shape[0]]
    m_ = np.array(f0["token_pad_mask"]).reshape(-1)[: pl.shape[0]] > 0
    _ = float(pl[(a_ == asym) & m_].mean())
    _ = np.array(out["sample_atom_coords"])
    t1 = time.perf_counter()
    _ = {k: (np.array(v) if isinstance(v, mx.array) else v) for k, v in out.items()  # pre-R6 cost: every output exported
         if k not in ("s_inputs", "pdistogram")}
    t2 = time.perf_counter()
    _ = _structure_pred_dict(out, b0, embeddings=False, pae_pde=False)
    t3 = time.perf_counter()
    eng.keep_extras = True
    eng.write_structure(out, b0)
    t4 = time.perf_counter()
    eng.keep_extras = False
    eng.write_structure(out, b0)
    t5 = time.perf_counter()
    emit({"exp": "plain", "case": case, "ligand": l0.ligand_id, "part": "runner extras + structure writer",
          "ligand_plddt_and_coords_np_s": round(t1 - t0, 4), "pred_dict_full_s": round(t2 - t1, 4),
          "pred_dict_lean_s": round(t3 - t2, 4), "write_structure_full_s": round(t4 - t3, 4),
          "write_structure_lean_s": round(t5 - t4, 4)})

    # shapes: a new N seen before on this machine (13b) and a never-seen N (13g); write their poses for phase 2
    for lig, kind in ((ligs[1], "new N in process (system Metal cache warm)"), (ligs[2], "never-seen N (cold Metal JIT)")):
        b, f = batches[lig.ligand_id]
        out, wall, tm, _ = run_plain(model, f, sargs, snoise(lig))
        out2, wall2, tm2, _ = run_plain(model, f, sargs, snoise(lig))
        emit({"exp": "shapes", "case": case, "pass": "structure", "ligand": lig.ligand_id, "kind": kind,
              "n_tokens": int(np.array(f["token_pad_mask"]).sum()), "first_wall_s": round(wall, 3), "first_timings": tm,
              "repeat_wall_s": round(wall2, 3), "repeat_timings": tm2})
        eng.write_structure(out2, b)
        del out, out2
    del outs
    t = time.perf_counter()
    del model
    gc.collect()
    mx.clear_cache()
    emit({"exp": "plain", "case": case, "part": "unload conf checkpoint (del + gc + clear_cache)",
          "s": round(time.perf_counter() - t, 3)})

    # ---- affinity phase
    t = time.perf_counter()
    model = load_model("aff")
    emit({"exp": "plain", "case": case, "part": "load aff checkpoint", "s": round(time.perf_counter() - t, 3)})
    aargs = PredictArgs(5, 200, 5)
    abatches = {}
    for lig in ligs:
        t0 = time.perf_counter()
        b = eng.featurize(lig, affinity=True)
        t1 = time.perf_counter()
        f = eng.model_inputs(b)
        mx.eval(list(f.values()))
        t2 = time.perf_counter()
        abatches[lig.ligand_id] = (b, f)
        emit({"exp": "plain", "case": case, "ligand": lig.ligand_id, "part": "cpu.affinity_features",
              "featurize_s": round(t1 - t0, 3), "model_inputs_s": round(t2 - t1, 3),
              "crop_tokens": int(np.array(f["token_pad_mask"]).sum()), "n_atoms": int(f["atom_pad_mask"].shape[1]),
              "msa_rows": int(f["msa"].shape[1])})

    def anoise(lig):
        return lambda: record_noise(0, lig.record_id, "affinity", 5)

    b0, f0 = abatches[l0.ligand_id]
    fps = set()
    out = None
    for i, (kind, pipe) in enumerate(order):
        set_pipe(pipe)
        out, wall, tm, tl = run_plain(model, f0, aargs, anoise(l0), timeline=(i >= 3))
        fps.add(fingerprint(out, AFF_KEYS))
        emit({"exp": "pipe" if i else "shapes", "case": case, "pass": "affinity", "ligand": l0.ligand_id, "run": i,
              "kind": kind, "pipe": pipe, "wall_s": round(wall, 3), "timings": tm, "timeline": tl,
              "fingerprint": fingerprint(out, AFF_KEYS)})
    set_pipe(True)
    emit({"exp": "pipe_bitwise", "case": case, "pass": "affinity", "fingerprints": sorted(fps), "identical": len(fps) == 1})
    t0 = time.perf_counter()
    eng.write_affinity(out, b0)
    emit({"exp": "plain", "case": case, "ligand": l0.ligand_id, "part": "affinity writer", "s": round(time.perf_counter() - t0, 4)})
    del out
    out_s, wall, tm, rows = run_synced(model, f0, aargs, anoise(l0))
    emit({"exp": "synced", "case": case, "pass": "affinity", "ligand": l0.ligand_id, "wall_s": round(wall, 3),
          "timings": tm, "rows": rows, "fingerprint": fingerprint(out_s, AFF_KEYS),
          "equals_plain": fingerprint(out_s, AFF_KEYS) in fps})
    del out_s
    out_p, wall, syncs, funcs = run_cprofile(model, f0, aargs, anoise(l0))
    emit({"exp": "cprofile", "case": case, "pass": "affinity", "ligand": l0.ligand_id, "wall_s": round(wall, 3),
          "syncs": syncs, "funcs": funcs})
    del out_p
    for lig, kind in ((ligs[1], "next ligand (new crop size?)"), (ligs[2], "never-seen structure N (crop size unknown)")):
        b, f = abatches[lig.ligand_id]
        out, wall, tm, _ = run_plain(model, f, aargs, anoise(lig))
        out2, wall2, tm2, _ = run_plain(model, f, aargs, anoise(lig))
        emit({"exp": "shapes", "case": case, "pass": "affinity", "ligand": lig.ligand_id, "kind": kind,
              "crop_tokens": int(np.array(f["token_pad_mask"]).sum()), "first_wall_s": round(wall, 3),
              "first_timings": tm, "repeat_wall_s": round(wall2, 3), "repeat_timings": tm2})
        del out, out2
    t = time.perf_counter()
    eng.cleanup(l0)
    emit({"exp": "plain", "case": case, "part": "cleanup per ligand", "s": round(time.perf_counter() - t, 4)})
    emit({"exp": "done", "case": case, "work": str(work)})


# ============================================================================================================ prot400
def prot400(a, case: str = "prot400"):
    import numpy as np

    from boltz_mlx.model import PredictArgs, load_model
    from boltz_mlx.sampler import record_noise

    mx = _mx()
    z = np.load(DATA / case / "feats.npz")
    feats = {k: mx.array(z[k]) for k in z.files}
    model = load_model("conf")
    args = PredictArgs(3, 50, 1)

    def noise():
        return record_noise(0, case, "structure", 1)

    fps = set()
    for i, (kind, pipe) in enumerate([("first-in-process", True), ("warm", False), ("warm", True), ("warm", False),
                                      ("warm", True)]):
        set_pipe(pipe)
        out, wall, tm, tl = run_plain(model, feats, args, noise, timeline=(i >= 3))
        fps.add(fingerprint(out, STRUCT_KEYS))
        emit({"exp": "pipe" if i else "shapes", "case": case, "pass": "structure", "run": i, "kind": kind, "pipe": pipe,
              "wall_s": round(wall, 3), "timings": tm, "timeline": tl, "fingerprint": fingerprint(out, STRUCT_KEYS)})
        del out
    set_pipe(True)
    emit({"exp": "pipe_bitwise", "case": case, "pass": "structure", "fingerprints": sorted(fps), "identical": len(fps) == 1})
    out_s, wall, tm, rows = run_synced(model, feats, args, noise)
    emit({"exp": "synced", "case": case, "pass": "structure", "wall_s": round(wall, 3), "timings": tm, "rows": rows,
          "fingerprint": fingerprint(out_s, STRUCT_KEYS), "equals_plain": fingerprint(out_s, STRUCT_KEYS) in fps})
    del out_s
    out_p, wall, syncs, funcs = run_cprofile(model, feats, args, noise)
    emit({"exp": "cprofile", "case": case, "pass": "structure", "wall_s": round(wall, 3), "syncs": syncs, "funcs": funcs})
    emit({"exp": "done", "case": case})


# ============================================================================================================ report (CPU)
PARENTS = {"trunk.msa_module(total)": "trunk.msa.", "diff.conditioning(total)": "diff.cond.",
           "confidence(total)": "confidence.", "affinity.affinity_module1(total)": "affinity.affinity_module1.",
           "affinity.affinity_module2(total)": "affinity.affinity_module2."}


def ranked_table(s: dict) -> list[str]:
    """Synced rows ranked by total; a '(total)' wrapper is shown as its self time (glue around its timed children);
    'unattributed' = synced wall - the top-level rows (trunk recycle projections + guard syncs, sampler setup, ...)."""
    rows = {r["name"]: dict(r, total=r["pre_ms"] + r["build_ms"] + r["eval_ms"]) for r in s["rows"]}
    child_of = {n: p for n in rows for p, pre in PARENTS.items() if p in rows and n.startswith(pre)}
    top = sum(r["total"] for n, r in rows.items() if n not in child_of)
    shown = []
    for n, r in rows.items():
        if n in PARENTS:
            kids = [rows[k] for k, p in child_of.items() if p == n]
            r = dict(r, name=n.replace("(total)", " self (glue)"), total=r["total"] - sum(k["total"] for k in kids),
                     build_ms=r["build_ms"] - sum(k["pre_ms"] + k["build_ms"] + k["eval_ms"] for k in kids))
        shown.append(r)
    wall = s["wall_s"] * 1e3
    shown.append({"name": "unattributed (between wrappers: recycle guard syncs, glue, sampler setup)", "calls": "",
                  "pre_ms": 0.0, "build_ms": 0.0, "eval_ms": 0.0, "total": wall - top})
    out = ["| sub-phase (synced view) | calls | pre ms | build ms (host; GPU idle) | eval ms | total ms | % of pass |",
           "|---|---|---|---|---|---|---|"]
    for r in sorted(shown, key=lambda r: -r["total"]):
        out.append(f"| {r['name']} | {r['calls']} | {r['pre_ms']:.1f} | {r['build_ms']:.1f} | {r['eval_ms']:.1f} | "
                   f"{r['total']:.1f} | {100 * r['total'] / wall:.1f} |")
    out.append(f"| **pass (synced wall)** | | | | | {wall:.1f} | 100 |")
    return out


def report(a):
    rows = [json.loads(l) for l in open(OUT)] if OUT.exists() else []
    if a.git:
        rows = [r for r in rows if r.get("git") == a.git]
    for case in ("t400", "prot400", "prot117"):
        rc = [r for r in rows if r.get("case") == case]
        if not rc:
            continue
        print(f"\n## {case} (git {rc[-1].get('git')}, {rc[-1].get('date')})")
        for ps in ("structure", "affinity"):
            sy = [r for r in rc if r["exp"] == "synced" and r["pass"] == ps]
            pp = [r for r in rc if r["exp"] == "pipe" and r["pass"] == ps]
            if not sy:
                continue
            s = sy[-1]
            on = [r["wall_s"] for r in pp if r["pipe"]]
            off = [r["wall_s"] for r in pp if not r["pipe"]]
            print(f"\n### {ps} pass: plain (unsynced) pipe off {statistics.median(off) if off else '-'} s, "
                  f"pipe on {statistics.median(on) if on else '-'} s; synced view {s['wall_s']} s "
                  f"(timings {s['timings']}; bitwise vs plain: {s['equals_plain']})")
            for line in ranked_table(s):
                print(line)
            tl = [r for r in pp if r.get("timeline")]
            for r in tl:
                print(f"timeline (pipe {'on' if r['pipe'] else 'off'}): " + json.dumps(r["timeline"]))
        for r in rc:
            if r["exp"] in ("plain", "shapes", "pipe_bitwise"):
                print(json.dumps({k: v for k, v in r.items() if k not in ("date", "git", "pkg_tree", "timeline")}))
        for r in rc:
            if r["exp"] == "cprofile":
                print(f"\ncprofile {r['pass']}: wall {r['wall_s']} s; syncs: " + json.dumps(r["syncs"][:8]))
                for f in r["funcs"][:25]:
                    print(f"   {f['cumtime_ms']:9.1f} ms cum {f['tottime_ms']:8.1f} tot {f['calls']:6d} x {f['fn']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["t400", "prot400", "prot117", "report"])
    ap.add_argument("--git", default=None, help="report: rows of this git hash only")
    ap.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.what == "report":
        return report(a)
    if not a._child:  # numerics env before MLX initialises (as bench_mlx): f16 = the screen preset's precision
        keep = ("BOLTZ_MLX_CACHE", "BOLTZ_MLX_CACHE_GB", "BOLTZ_MLX_MEM_LIMIT_GB", "BOLTZ_MLX_WEIGHTS")  # as the screen
        env = {k: v for k, v in os.environ.items() if not k.startswith("BOLTZ_MLX_") or k in keep}
        env.update(BOLTZ_MLX_PRECISION="f16", PYTHONPATH=str(ROOT))
        os.execve(sys.executable, [sys.executable, __file__, *sys.argv[1:], "--_child"], env)
    sys.path.insert(0, str(ROOT))
    import boltz_mlx  # noqa: F401
    import mlx.core as mx

    mx.set_cache_limit(int(4e9))
    {"t400": t400, "prot400": prot400, "prot117": lambda a_: prot400(a_, "prot117")}[a.what](a)


if __name__ == "__main__":
    main()
