"""MSA-module kernels (boltz_mlx/kernels_msa.py) vs the shipped f16 MLX path, on the recorded L1 captures.

  python tools/prof/msa_kernels.py              # GPU, ~2 min; rows -> logs/msa_kernels.jsonl
  python tools/prof/msa_kernels.py --quick      # msa384 call 0 only, default configs
  python tools/prof/msa_kernels.py --cpu-smoke  # harness logic on CPU (kernels -> MLX references, mixed.cpu_f16)
  python tools/prof/msa_kernels.py suggest [--write]   # fastest gated config per kernel from this Mac's sweep rows ->
                                                        # its entry of boltz_mlx/msa_cfg.json (kernels.write_device_cfg)

Captures: data/cache/l1/<case>/trunk.msa__{OuterProductMean,PairWeightedAveraging,MSALayer}__{0,3,7}.pkl (strict fp32
inputs + outputs). Cases: msa384 (N=384, S=249: the T400 screen receptor), ex_ligand (N=423, S=249: ragged tiles),
prot117 (S=1: PWA only — the OPM kernel is not used below S*32 = OPM_REASSOC_MAX_K).
Numerics, every capture ("numerics" rows): module output in f16 with the kernel flags off (shipped MLX f16 path,
`mlx`), on (kernel path, `kern`) and, for OPM / MSALayer, on with the experimental proj_o-in-kernel NAX variant
(`kern_nax`, BOLTZ_MLX_MSA_OPM_NAX) vs the captured strict output: rel_fro, cosine, max_abs (L1 gate rel_fro <= 3e-3,
cosine >= 0.99999; acceptance: kern no worse than mlx), plus kern vs mlx.
Timing, msa384 call 0 ("timing" rows): module ms per variant (min of 3 rounds x 10 pipelined calls) and peak MB; kernel
config sweeps ("sweep" rows: every OPM_CFGS / OPM_FUSED_CFGS / PWA_CFGS entry: core-kernel ms + TFLOP/s, module ms,
numerics); parts of the kernel paths ("parts" rows); MSALayer ms per variant.
"""

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import triatt_common as tc  # noqa: E402
from triatt_common import L1, ROOT, bench, emit, metrics, mx, peak_mb  # noqa: E402

from boltz_mlx import kernels as K  # noqa: E402
from boltz_mlx import kernels_msa as KM  # noqa: E402
from boltz_mlx.layers.outer_product_mean import OuterProductMean  # noqa: E402
from boltz_mlx.layers.pair_averaging import PairWeightedAveraging  # noqa: E402
from boltz_mlx.modules.trunkv2 import MSALayer  # noqa: E402
from boltz_mlx.weights import load_module  # noqa: E402

tc.OUT = ROOT / "logs" / "msa_kernels.jsonl"
EXP = "msa_kernels"
# the OPM proj_o-in-kernel variant (opm_outer_proj) is an MPP kernel: swept only on the nax / tensorops kernel paths
NAX_OK = "suggest" not in sys.argv and not tc.CPU_SMOKE and K.mpp_enabled()
QUICK = "--quick" in sys.argv
CPU = tc.CPU_SMOKE
REPS = 10
CLASSES = ["OuterProductMean", "PairWeightedAveraging", "MSALayer"]
CASES = {"msa384": (0,) if QUICK else (0, 3, 7), "ex_ligand": () if QUICK else (0, 7), "prot117": () if QUICK else (0,)}
TIMED = ("msa384", 0)
OPM_CFGS = [KM.OPM_DEFAULT] if QUICK else KM.OPM_CFGS
FUSED_CFGS = [KM.OPM_FUSED_DEFAULT] if QUICK else KM.OPM_FUSED_CFGS
PWA_CFGS = [KM.PWA_DEFAULT] if QUICK else KM.PWA_CFGS

CROP = None
if CPU:  # logic only: f16 paths on the CPU (mixed.cpu_f16), kernels replaced by their MLX references, inputs cropped
    import contextlib

    from boltz_mlx import mixed as MX

    f16_ctx = MX.cpu_f16
    KM.opm_outer = lambda ab, mask, num, i_base=0, n_i=None, **cfg: KM.opm_outer_ref(ab, mask, num, i_base, n_i)
    KM.pwa_weights = lambda z, pm, lw, lb, eps, wz, inf, **kw: KM.pwa_weights_ref(z, pm, lw, lb, eps, wz, inf)
    KM.pwa_avg = lambda w, v, g, **cfg: KM.pwa_avg_ref(w, v, g)
    KM.opm_outer_proj = lambda ab, mask, num, wo, bias, i_base=0, n_i=None, **cfg: KM.opm_outer_proj_ref(
        ab, mask, num, wo, bias, i_base, n_i)
    K.msa_kernel = lambda kind: K.MSA_FLAGS.get(kind) == "1"
    CASES, OPM_CFGS, PWA_CFGS = {"msa384": (0,)}, [KM.OPM_DEFAULT], [KM.PWA_DEFAULT]
    FUSED_CFGS = [KM.OPM_FUSED_DEFAULT]
    CROP = (48, 40)  # S rows, N tokens (refs recomputed with the strict fp32 module on the CPU)
    REPS = 1
else:
    import contextlib

    f16_ctx = contextlib.nullcontext
    K.PRECISION = "f16"  # K.f16() is read at call time: modules run their shipped f16 paths

_mods: dict = {}


def module_for(rec):
    p = rec["path"]
    if p not in _mods:
        cls = rec["cls"]
        if cls == "OuterProductMean":
            m = OuterProductMean(64, 32, 128)
        elif cls == "PairWeightedAveraging":
            m = PairWeightedAveraging(64, 128, 32, 8)
        else:
            m = MSALayer(64, 128, 0.0, 0.0)
        _mods[p] = load_module(m, p)
    return _mods[p]


def outputs(t):
    """Captured output tree -> list of arrays (module output order)."""
    return [t[1]] if t[0] == "arr" else [x[1] for x in t[1]]


def set_flags(on: bool, kinds=("OPM", "PWA")):
    for k in kinds:
        K.MSA_FLAGS[k] = "1" if on else "0"


VARIANTS = {"mlx": (False, False), "kern": (True, False), "kern_nax": (True, True)}  # (flags on, OPM NAX proj_o)


def call(mod, args, kwargs, on: bool, ctx=None, nax: bool = False):
    set_flags(on)
    K.MSA_OPM_NAX = nax
    with (ctx or f16_ctx)():
        o = type(mod).__call__(mod, *args, **kwargs)
        o = tuple(K.unscale(x) for x in o) if isinstance(o, tuple) else K.unscale(o)
    return list(o) if isinstance(o, tuple) else [o]


def crop(cls, args):
    """CPU smoke: first S rows / N tokens of every input (by role)."""
    S, N = CROP
    if cls == "OuterProductMean":  # m (1, S, N, c), msa mask (1, S, N)
        return [args[0][:, :S, :N], args[1][:, :S, :N]]
    if cls == "PairWeightedAveraging":  # m, z (1, N, N, d), pair mask (1, N, N)
        return [args[0][:, :S, :N], args[1][:, :N, :N], args[2][:, :N, :N]]
    return [args[0][:, :N, :N], args[1][:, :S, :N], args[2][:, :N, :N], args[3][:, :S, :N]]  # z, m, token, msa mask


def numerics_row(base, outs, refs, tag):
    row = dict(base, variant=tag)
    for n, (o, r) in enumerate(zip(outs, refs)):
        sfx = "" if len(outs) == 1 else f"_o{n}"
        for k, v in metrics(o, r).items():
            row[k + sfx] = v
    return row


def sweep_opm(mod, args, refs, base):
    """Core-kernel and module time for every OPM config (msa384 call 0)."""
    m, mask = args[0], args[1]
    _, S, N, _ = m.shape
    maskf = mask.astype(mx.float32).reshape(S, N)
    wab = mx.concatenate([mod.proj_a.weight, mod.proj_b.weight], 0)
    ab = mod.norm(m) @ wab.T
    num = mx.maximum(maskf.T @ maskf, 1.0)
    z16 = KM.opm_outer(ab, maskf, num, **KM.OPM_DEFAULT)
    mx.eval(ab, num, z16)
    flops = 2.0 * N * N * S * 1024
    saved = dict(K.MSA_OPM_CFG)
    for cfg in OPM_CFGS:
        row = dict(base, kind="sweep", kernel="opm_outer", **cfg)
        try:
            core = bench(lambda c=cfg: KM.opm_outer(ab, maskf, num, **c), REPS)
            K.MSA_OPM_CFG.clear()
            K.MSA_OPM_CFG.update(cfg)
            out = call(mod, args, {}, True)
            row.update(core_ms=round(core, 3), core_tflops=round(flops / core / 1e9, 2),
                       module_ms=round(bench(lambda: call(mod, args, {}, True)[0], REPS), 3),
                       **metrics(out[0], refs[0]))
        except Exception as e:  # a config that fails to build / run is reported, the sweep goes on
            row["error"] = repr(e)[:300]
        emit(row)
    K.MSA_OPM_CFG.clear()
    K.MSA_OPM_CFG.update(saved)
    wo = K.w16(mod.proj_o, "_w16", mod.proj_o.weight)
    fused_cfgs = FUSED_CFGS if NAX_OK else []  # the proj_o-in-kernel variant is an MPP kernel (portable path: none)
    flops_f = flops + 2.0 * N * N * 1024 * 128
    saved = dict(K.MSA_OPM_NAX_CFG)
    for cfg in fused_cfgs:
        row = dict(base, kind="sweep", kernel="opm_outer_proj", **cfg)
        try:
            core = bench(lambda c=cfg: KM.opm_outer_proj(ab, maskf, num, wo, mod.proj_o.bias, **c), REPS)
            K.MSA_OPM_NAX_CFG.clear()
            K.MSA_OPM_NAX_CFG.update(cfg)
            out = call(mod, args, {}, True, nax=True)
            row.update(core_ms=round(core, 3), core_tflops=round(flops_f / core / 1e9, 2),
                       module_ms=round(bench(lambda: call(mod, args, {}, True, nax=True)[0], REPS), 3),
                       **metrics(out[0], refs[0]))
        except Exception as e:
            row["error"] = repr(e)[:300]
        emit(row)
    K.MSA_OPM_NAX_CFG.clear()
    K.MSA_OPM_NAX_CFG.update(saved)
    # parts of the kernel path, and the MLX fp32 GEMM the kernel replaces (same 75 GFLOP at msa384)
    a2 = (ab[..., :32] * maskf[None, ..., None]).transpose(0, 2, 3, 1).reshape(1, N * 32, S)
    b2 = (ab[..., 32:] * maskf[None, ..., None]).reshape(1, S, N * 32)
    mx.eval(a2, b2)
    parts = {
        "ln_proj_ab": lambda: mod.norm(m) @ wab.T,
        "num_mask": lambda: mx.maximum(maskf.T @ maskf, 1.0),
        "opm_outer": lambda: KM.opm_outer(ab, maskf, num, **K.MSA_OPM_CFG),
        "proj_o_lin16": lambda: K.lin16(z16.reshape(1, N, N, 1024), mod.proj_o),
        "mlx_fp32_outer_gemm": lambda: a2 @ b2,
        "mlx_pm_num_mask": lambda: mx.maximum((mask.astype(mx.float32)[..., None][:, :, None] *
                                              mask.astype(mx.float32)[..., None][:, :, :, None]).sum(axis=1), 1),
    }
    if NAX_OK:
        parts["opm_outer_proj_nax"] = lambda: KM.opm_outer_proj(ab, maskf, num, wo, mod.proj_o.bias, **K.MSA_OPM_NAX_CFG)
    emit(dict(base, kind="parts", module="OuterProductMean",
              **{k + "_ms": round(bench(f, REPS), 3) for k, f in parts.items()}))


def sweep_pwa(mod, args, refs, base):
    m, z, pm = args[0], args[1], args[2]
    _, S, N, _ = m.shape
    mn = mod.norm_m(m)
    v, g = mod.proj_m(mn), mod.proj_g(mn)
    wz = K.w16(mod.proj_z, "_w16", mod.proj_z.weight)
    w = KM.pwa_weights(z, pm, mod.norm_z.weight, mod.norm_z.bias, mod.norm_z.eps, wz, mod.inf)
    mx.eval(v, g, w)
    flops = 2.0 * 8 * N * KM.pwa_np(N) * S * 32
    saved = dict(K.MSA_PWA_CFG)
    for cfg in PWA_CFGS:
        row = dict(base, kind="sweep", kernel="pwa_avg", **cfg)
        try:
            core = bench(lambda c=cfg: KM.pwa_avg(w, v, g, **c), REPS)
            K.MSA_PWA_CFG.clear()
            K.MSA_PWA_CFG.update(cfg)
            out = call(mod, args, {}, True)
            row.update(core_ms=round(core, 3), core_tflops=round(flops / core / 1e9, 2),
                       module_ms=round(bench(lambda: call(mod, args, {}, True)[0], REPS), 3),
                       **metrics(out[0], refs[0]))
        except Exception as e:
            row["error"] = repr(e)[:300]
        emit(row)
    K.MSA_PWA_CFG.clear()
    K.MSA_PWA_CFG.update(saved)
    og = KM.pwa_avg(w, v, g, **K.MSA_PWA_CFG)
    mx.eval(og)
    parts = {
        "ln_m": lambda: mod.norm_m(m),
        "proj_m_g": lambda: [mod.proj_m(mn), mod.proj_g(mn)],
        "pwa_weights": lambda: KM.pwa_weights(z, pm, mod.norm_z.weight, mod.norm_z.bias, mod.norm_z.eps, wz, mod.inf),
        "pwa_avg": lambda: KM.pwa_avg(w, v, g, **K.MSA_PWA_CFG),
        "proj_o": lambda: mod.proj_o(og),
    }
    emit(dict(base, kind="parts", module="PairWeightedAveraging",
              **{k + "_ms": round(bench(f, REPS), 3) for k, f in parts.items()}))


def main():
    for case, calls in CASES.items():
        for cls in CLASSES:
            for c in calls:
                f = L1 / case / f"trunk.msa__{cls}__{c}.pkl"
                if not f.exists():
                    continue
                rec = pickle.load(open(f, "rb"))
                mod = module_for(rec)
                args = [mx.array(a[1]) for a in rec["args"][1]]
                kwargs = {k: mx.array(v[1]) for k, v in rec["kwargs"][1].items()}
                refs = outputs(rec["out"])
                del rec
                if CROP is not None:
                    args = crop(cls, args)
                    refs = [r.astype(mx.float32) for r in call(mod, args, kwargs, False, contextlib.nullcontext)]
                mx.eval(args, kwargs)
                base = {"exp": EXP, "case": case, "cls": cls, "call": c, "shape_m": list(args[0].shape)}
                if cls == "OuterProductMean" and args[0].shape[1] * 32 <= K.OPM_REASSOC_MAX_K:
                    continue  # S small: re-associated path, the kernel is not used
                variants = [v for v in VARIANTS if v != "kern_nax" or (cls != "PairWeightedAveraging" and NAX_OK)]
                outs = {}
                for tag in variants:
                    o = call(mod, args, kwargs, VARIANTS[tag][0], nax=VARIANTS[tag][1])
                    mx.eval(o)
                    outs[tag] = o
                    emit(numerics_row(dict(base, kind="numerics"), o, refs, tag))
                emit(numerics_row(dict(base, kind="numerics"), outs["kern"],
                                  [x.astype(mx.float32) for x in outs["mlx"]], "kern_vs_mlx"))
                del outs
                if (case, c) == TIMED:
                    row = dict(base, kind="timing")
                    for tag in variants:
                        fl = VARIANTS[tag]
                        row[f"{tag}_ms"] = round(bench(lambda f=fl: call(mod, args, kwargs, f[0], nax=f[1]), REPS), 3)
                        row[f"{tag}_peak_mb"] = round(peak_mb(lambda f=fl: call(mod, args, kwargs, f[0], nax=f[1])), 1)
                    emit(row)
                    if cls == "OuterProductMean":
                        sweep_opm(mod, args, refs, base)
                    elif cls == "PairWeightedAveraging":
                        sweep_pwa(mod, args, refs, base)
                del args, kwargs, refs
                mx.clear_cache()
    set_flags(True)
    K.MSA_OPM_NAX = False


GATE = {"rel_fro": 3e-3, "cosine": 0.99999}


def suggest() -> dict:
    """Per kernel the config with the least module ms among this Mac's sweep rows of the newest build that ran them,
    passing the L1 gate (rel_fro <= 3e-3, cosine >= 0.99999, no error) -> {"opm": .., "opm_nax": .., "pwa": ..}."""
    import json

    rows = K.rows_for_chip([json.loads(x) for x in open(tc.OUT)] if tc.OUT.exists() else [])
    rows = [r for r in rows if r.get("exp") == EXP and r.get("kind") == "sweep"]
    out = {}
    for key, kernel, default in (("opm", "opm_outer", KM.OPM_DEFAULT), ("opm_nax", "opm_outer_proj", KM.OPM_FUSED_DEFAULT),
                                 ("pwa", "pwa_avg", KM.PWA_DEFAULT)):
        rk = [r for r in rows if r.get("kernel") == kernel]
        if not rk:
            continue
        rk = [r for r in rk if r["git"] == rk[-1]["git"] and "error" not in r and "module_ms" in r
              and r.get("rel_fro", 1.0) <= GATE["rel_fro"] and r.get("cosine", 0.0) >= GATE["cosine"]]
        if rk:
            best = min(rk, key=lambda r: r["module_ms"])
            out[key] = {k: best[k] for k in default}
            out[key + "_module_ms"] = best["module_ms"]
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "suggest":
        import json
        import time

        sg = suggest()
        e = K.device_cfg_entry(K.MSA_CFG_FILE)  # this chip's entry, else default: kernels without new rows keep theirs
        e.update({k: v for k, v in sg.items() if not k.endswith("_module_ms")})
        e["source"] = (f"tools/prof/msa_kernels.py suggest (logs/msa_kernels.jsonl, git {tc.git_rev()}, "
                       f"{time.strftime('%Y-%m-%d %H:%M')}; module ms {({k: v for k, v in sg.items() if k.endswith('_ms')})})")
        print(json.dumps(e))
        if "--write" in sys.argv and sg:
            K.write_device_cfg(K.MSA_CFG_FILE, e)
            print(f"wrote the {K.chip_name()!r} entry of {K.MSA_CFG_FILE}")
        sys.exit(0)
    main()
