"""Compile-only check of mx.fast.metal_kernel sources (CPU side; nothing is dispatched to the GPU).

MLX generates the kernel source when a metal_kernel is *called* (graph construction), not when it is evaluated. This
tool builds every kernel of a callable with the CPU as default device, captures the generated source (verbose=True),
prepends a minimal prelude + the kernel's header and compiles it with the Metal runtime compiler
(MTLDevice.makeLibrary(source:), language 4.0) through a small Swift binary (tools/prof/metal_check.swift, built once
into $TMPDIR). No command buffer is created and no array is evaluated, so no GPU work runs.

  from metal_check import check
  check(lambda: KN.transition_nax(x16, w12h, w3h))   # -> list of (kernel name, "OK" | error text)

  python tools/prof/metal_check.py            # checks the kernels of boltz_mlx/kernels_nax*.py at a few shapes
  python tools/prof/metal_check.py atom       # only kernels_atom (+ the kernels_nax3 instantiations it uses)
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import mlx.core as mx  # noqa: E402

PRELUDE = """#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;
typedef half float16_t;
typedef bfloat bfloat16_t;
"""
_BIN = Path(tempfile.gettempdir()) / "boltz_metal_check"


def _binary(kind: str = "check") -> Path:
    """kind 'check': compile the library only; 'pipeline': also build a compute pipeline state per function (the GPU
    backend compile; still nothing is dispatched) and print maxTotalThreadsPerThreadgroup (register-pressure proxy)."""
    name = "metal_check" if kind == "check" else "metal_pipeline"
    src = Path(__file__).with_name(f"{name}.swift")
    out = _BIN if kind == "check" else _BIN.with_name("boltz_metal_pipeline")
    if not out.exists() or out.stat().st_mtime < src.stat().st_mtime:
        subprocess.run(["swiftc", "-O", str(src), "-o", str(out)], check=True)
    return out


@contextlib.contextmanager
def _capture_fd1():
    """Capture C++ std::cout (MLX prints the generated source there)."""
    sys.stdout.flush()
    fd, path = tempfile.mkstemp()
    saved = os.dup(1)
    os.dup2(fd, 1)
    out = {"text": ""}
    try:
        yield out
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)
        os.close(fd)
        out["text"] = Path(path).read_text()
        os.unlink(path)


def check(build, label: str = "", kind: str = "check") -> list:
    """Run `build()` (which constructs metal_kernel calls) with kernels patched to verbose; compile every generated
    source. Returns [(label/kernel, 'OK' or the compiler error)]."""
    prev_dev = mx.default_device()
    mx.set_default_device(mx.cpu)
    orig = mx.fast.metal_kernel
    made = []

    def patched(*a, **kw):
        k = orig(*a, **kw)
        header = kw.get("header", a[4] if len(a) > 4 else "")
        name = kw.get("name", a[0] if a else "?")

        def call(*ca, **ckw):
            ckw["verbose"] = True
            made.append((name, header))
            return k(*ca, **ckw)

        return call

    import importlib

    mods = [importlib.import_module(m) for m in ("boltz_mlx.kernels", "boltz_mlx.kernels_nax")]
    for name in ("boltz_mlx.kernels_nax2", "boltz_mlx.kernels_nax3", "boltz_mlx.kernels_msa", "boltz_mlx.kernels_atom"):
        try:
            mods.append(importlib.import_module(name))
        except ImportError:
            pass
    saved = [(m, dict(m._kernels)) for m in mods]
    for m, _ in saved:
        m._kernels.clear()
    mx.fast.metal_kernel = patched
    try:
        with _capture_fd1() as cap:
            build()
    finally:
        mx.fast.metal_kernel = orig
        for m, d in saved:
            m._kernels.clear()
            m._kernels.update(d)
        mx.set_default_device(prev_dev)
    # split the captured text into generated sources (one per call, in call order)
    srcs = []
    for block in cap["text"].split("Generated source code for `")[1:]:
        body = block.split("```", 2)[1]
        srcs.append(body)
    res = []
    for (name, header), body in zip(made, srcs):
        full = PRELUDE + body  # the generated source already starts with the kernel's header
        with tempfile.NamedTemporaryFile("w", suffix=".metal", delete=False) as f:
            f.write(full)
            p = f.name
        r = subprocess.run([str(_binary(kind)), p], capture_output=True, text=True)
        msg = (r.stdout + r.stderr).strip()
        if kind == "pipeline" and r.returncode == 0:
            msg = "OK " + " | ".join(x.split(" ", 2)[2] for x in msg.splitlines() if x.startswith("PSO "))
        res.append((f"{label}{name}", (msg if kind == "pipeline" else "OK") if r.returncode == 0 else msg[:4000]))
        if r.returncode == 0:
            os.unlink(p)
        else:
            res[-1] = (res[-1][0], res[-1][1] + f"\n[source kept: {p}]")
    return res


def nax3_checks() -> list:
    """Every kernels_nax3 kernel / config (+ the shared-bias NAX diffusion attention)."""
    try:
        from boltz_mlx import kernels_nax2 as KN2
        from boltz_mlx import kernels_nax3 as N3
    except ImportError:
        return []
    from types import SimpleNamespace

    res = []
    D = 128
    z = mx.zeros((1, 40, 40, D), mx.float32)
    x16 = mx.zeros((1, 40, 40, D), mx.float16)
    w16 = mx.zeros((D, D), mx.float16)
    ln = SimpleNamespace(eps=1e-5, get=lambda k, d=None: mx.ones((D,)))
    ln_nob = SimpleNamespace(eps=1e-5, get=lambda k, d=None: mx.ones((D,)) if k == "weight" else None)
    for cfg in N3.PAIR_CFGS:
        for ty, to in ((0, 1), (1, 0)):
            res += check(lambda: N3.linear_addln(x16, w16, z, ln, trans_y=bool(ty), trans_out=bool(to), **cfg),
                         f"nax3 lin_addln ty{ty} to{to} {cfg} ")
    res += check(lambda: N3.linear_addln(x16, w16, z, ln_nob, bias=mx.zeros((D,))), "nax3 lin_addln bias/no-ln-bias ")
    o = mx.zeros((D, 1600), mx.float16)
    for cfg in N3.TRIMUL_TAIL_CFGS:
        res += check(lambda: N3.trimul_tail_addln(o, ln, 2.0 ** -16, w16, x16, w16, z, ln, **cfg), f"nax3 tail {cfg} ")
    L, Dd, N = 24, 768, 50
    rc = N3.RawCond16(mx.zeros((N, L * 4 * Dd), mx.float16), mx.zeros((L * 4 * Dd,)), mx.zeros((N, L * 2 * Dd), mx.float16),
                      mx.zeros((L * 2 * Dd,)), ntok=N, L=L, D=Dd, rep=3)
    a2 = mx.zeros((3 * N, Dd))
    res += check(lambda: N3.adaln16(a2, rc, 5, 2), "nax3 adaln16 ")
    t16 = mx.zeros((3 * N, Dd), mx.float16)
    for cfg in N3.GEMM_CFGS:
        res += check(lambda: N3.gemm16(t16, mx.zeros((3072, Dd), mx.float16), **cfg), f"nax3 gemm store {cfg} ")
        res += check(lambda: N3.gemm16(t16, mx.zeros((Dd, Dd), mx.float16), epi=N3.EPI_GRESID, gate=rc.h2,
                                       gate_bias=rc.b2, gate_off=7 * Dd, resid=a2, ntok=N, rep=3, **cfg),
                     f"nax3 gemm gresid {cfg} ")
    res += check(lambda: N3.gemm16(mx.zeros((3 * N, 2 * Dd), mx.float16), mx.zeros((Dd, 2 * Dd), mx.float16),
                                   epi=N3.EPI_GRESID, gate=rc.h2, gate_bias=rc.b2, gate_off=Dd, resid=a2, ntok=N, rep=3),
                 "nax3 gemm gresid K1536 ")
    res += check(lambda: N3.gemm16(t16, mx.zeros((Dd, Dd), mx.float16), out_dtype=mx.float32, bias=mx.zeros((Dd,))),
                 "nax3 gemm store f32 bias ")
    for cfg in N3.SWIGLU_CFGS:
        res += check(lambda: N3.gemm16(t16, mx.zeros((3 * 1536, Dd), mx.float16), epi=N3.EPI_SWIGLU, ngrp=3, **cfg),
                     f"nax3 gemm swiglu {cfg} ")
    # R3-pair: tri-bias epilogue variants, transition + LN / pair-bias epilogue, tri-att projection on gemm16
    wtb = mx.zeros((4, D), mx.float16)
    for ty, to in ((0, 1), (1, 0)):
        res += check(lambda: N3.linear_addln(x16, w16, z, ln, trans_y=bool(ty), trans_out=bool(to), w_tb=wtb),
                     f"r3 lin_addln_tb ty{ty} to{to} ")
    res += check(lambda: N3.trimul_tail_addln(o, ln, 2.0 ** -16, w16, x16, w16, z, ln, w_tb=wtb), "r3 tail_tb ")
    w12 = mx.zeros((1024, D), mx.float16)
    w3 = mx.zeros((D, 512), mx.float16)
    wz = mx.zeros((16, D), mx.float16)
    for cfg in (dict(tm=16, hb=32, nsg=8, tgw=1), dict(tm=16, hb=32, nsg=4, tgw=0)):
        res += check(lambda: N3.transition_lnx(x16, w12, w3, z, next_ln=ln, ln_s=ln, w_s=wz, **cfg), f"r3 tlnx both {cfg} ")
        res += check(lambda: N3.transition_lnx(x16, w12, w3, z, ln_s=ln_nob, w_s=wz, **cfg), f"r3 tlnx bias {cfg} ")
        res += check(lambda: N3.transition_lnx(x16, w12, w3, z, next_ln=ln, **cfg), f"r3 tlnx xn {cfg} ")
    res += check(lambda: N3.pair_proj16(x16, mx.zeros((512, D), mx.float16)), "r3 pair_proj16 ")
    # R4-pair: split tri-mult tail, tail sub-step ablation, transition with double-buffered staging / butterfly bias
    for cfg in N3.TLN_CFGS:
        res += check(lambda: N3.transpose_ln16(o, ln, 2.0 ** -16, **cfg), f"r4 tln {cfg} ")
    on = mx.zeros((1600, D), mx.float16)
    for cfg in N3.TAIL2_CFGS:
        res += check(lambda: N3.gemm2_gate_addln(on, w16, on, w16, z, ln, **cfg), f"r4 tail2 {cfg} ")
    res += check(lambda: N3.gemm2_gate_addln(on, w16, on, w16, z, ln, w_tb=wtb), "r4 tail2_tb ")
    for st in (1, 2, 3, 4):
        res += check(lambda: N3.trimul_tail_stage(o, ln, 2.0 ** -16, w16, x16, w16, z, ln, st), f"r4 tail stage{st} ")
    # R5-tail: low-register fused tail (configs, stage cuts, tri bias), diagnostics, thread-bound variants
    for name, fn in N3.R5_PROBES(o, x16, w16, z, ln, on):
        res += check(fn, f"r5 {name} ")
    res += check(lambda: N3.linear_addln_maxt(x16, w16, z, ln, trans_out=True), "r5 lin_addln maxt ")
    res += check(lambda: N3.transition_maxt(x16, w12, w3, z), "r5 transition maxt ")
    # R6-trans: register-fragment transition (+ in-register LN / pair bias), standalone LN / bias rows
    for cfg in N3.TRANS3_CFGS:
        if cfg["impl"] != "reg":
            continue
        c = {k: v for k, v in cfg.items() if k != "impl"}
        res += check(lambda c=c: N3.transition_reg(x16, w12, w3, z, **c), f"r6 rtrans plain {c} ")
        res += check(lambda c=c: N3.transition_reg(x16, w12, w3, z, next_ln=ln, ln_s=ln, w_s=wz, **c), f"r6 rtrans lnx {c} ")
    res += check(lambda: N3.transition_reg(x16, w12, w3, z, next_ln=ln), "r6 rtrans xn only ")
    res += check(lambda: N3.lnx_rows(z, next_ln=ln, ln_s=ln, w_s=wz), "r6 lnx_rows ")
    res += check(lambda: N3.transition3(x16, w12, w3, z, next_ln=ln, ln_s=ln, w_s=wz, cfg=N3.TRANS3_CFGS[6]), "r6 mpp hb64 lnx ")
    hq = mx.zeros((3 * N, 4 * Dd), mx.float16)
    res += check(lambda: KN2.attention_nax(hq, mx.zeros((1, 16, N, N)), B=3, N=N, heads=16, c=48, qbias=mx.zeros((Dd,)),
                                           brep=3), "attn48 brep3 ")
    return res


def dynn_builds(N: int) -> list:
    """(label, build) of every R8-coldN DYNN kernel (run-time sizes) at token count N, in the shipped configurations
    (+ the tri-bias / transposed / chunk-row variants). Shared by dynn_checks and tests/test_dynn_cpu.py."""
    from types import SimpleNamespace

    from boltz_mlx import kernels as K
    from boltz_mlx import kernels_nax as KN
    from boltz_mlx import kernels_nax2 as KN2
    from boltz_mlx import kernels_nax3 as N3

    D, H, C = 128, 4, 32
    z = mx.zeros((1, N, N, D), mx.float32)
    x16 = mx.zeros((1, N, N, D), mx.float16)
    w16 = mx.zeros((D, D), mx.float16)
    o = mx.zeros((D, N * N), mx.float16)
    wtb = mx.zeros((H, D), mx.float16)
    ln = SimpleNamespace(eps=1e-5, get=lambda k, d=None: mx.ones((D,)))
    h = mx.zeros((1, N, N, 4 * H * C), mx.float16)
    tb = mx.zeros((1, H, N, N), mx.float16)
    m = mx.ones((1, N, N))
    rows = min(N, 24)  # a row chunk (I != J), as the chunked tri-att path
    w12, w3 = mx.zeros((1024, D), mx.float16), mx.zeros((D, 512), mx.float16)
    wz = mx.zeros((16, D), mx.float16)
    t3 = {k: v for k, v in (N3.PAIR_SITE_CFG.get("trans3") or {}).items() if k != "impl"}
    if (N3.PAIR_SITE_CFG.get("trans3") or {}).get("impl", "reg") != "reg":
        t3 = {}
    tail3 = dict(N3.PAIR_SITE_CFG.get("tail3") or {})
    Dd, B = 768, 3
    hq = mx.zeros((B * N, 4 * Dd), mx.float16)
    out = [
        ("tri_nax", lambda: KN.tri_attention_nax(h, H, C, m, 1e9, bias=tb, **K.TRIATT_NAX_CFG)),
        ("tri_nax trans", lambda: KN.tri_attention_nax(h, H, C, m, 1e9, bias=tb, trans=True, **K.TRIATT_NAX_CFG)),
        ("tri_nax rows", lambda: KN.tri_attention_nax(h[:, :rows], H, C, m[:, :rows], 1e9, bias=tb, **K.TRIATT_NAX_CFG)),
        ("lin_addln ty", lambda: N3.linear_addln(x16, w16, z, ln, trans_y=True, **N3.PAIR_SITE_CFG["lin"])),
        ("lin_addln to tb", lambda: N3.linear_addln(x16, w16, z, ln, trans_out=True, w_tb=wtb, **N3.PAIR_SITE_CFG["lin"])),
        ("tail", lambda: N3.trimul_tail_addln(o, ln, 2.0 ** -16, w16, x16, w16, z, ln, **N3.PAIR_SITE_CFG["tail"])),
        ("tail tb", lambda: N3.trimul_tail_addln(o, ln, 2.0 ** -16, w16, x16, w16, z, ln, w_tb=wtb)),
        ("tail3", lambda: N3.trimul_tail3_addln(o, ln, 2.0 ** -16, w16, x16, w16, z, ln, **tail3)),
        ("tail3 tb", lambda: N3.trimul_tail3_addln(o, ln, 2.0 ** -16, w16, x16, w16, z, ln, w_tb=wtb, **tail3)),
        ("tail2 tb", lambda: N3.gemm2_gate_addln(x16, w16, x16, w16, z, ln, w_tb=wtb)),
        ("rtrans lnx", lambda: N3.transition_reg(x16, w12, w3, z, next_ln=ln, ln_s=ln, w_s=wz, **t3)),
        ("rtrans", lambda: N3.transition_reg(x16, w12, w3, z, **t3)),
        ("attn48", lambda: KN2.attention_nax(hq, mx.zeros((B, 16, N, N)), B=B, N=N, heads=16, c=48,
                                             qbias=mx.zeros((Dd,)))),
        ("attn48 brep3", lambda: KN2.attention_nax(hq, mx.zeros((1, 16, N, N)), B=B, N=N, heads=16, c=48,
                                                   qbias=mx.zeros((Dd,)), brep=3)),
        ("attn48 prepared", lambda: KN2.attention_nax(hq, None, B=B, N=N, heads=16, c=48,
                                                      prepared=KN2.attention_nax_bias(mx.zeros((B, 16, N, N))))),
        ("add_ln", lambda: K.add_ln(z, x16, ln, trans_y=True)),
        ("add_ln to", lambda: K.add_ln(z, z, ln, trans_out=True)),
    ]
    return out


@contextlib.contextmanager
def dynn(on: bool = True):
    """Force kernels.nax('DYNN') (CPU-device builds: nax() is otherwise off without the GPU device)."""
    from boltz_mlx import kernels as K

    orig = K.nax
    K.nax = lambda kind: on if kind == "DYNN" else orig(kind)
    try:
        yield
    finally:
        K.nax = orig


def dynn_checks(kind: str = "check", on: bool = True, ns=(40, 72)) -> list:
    """R8-coldN: compile every DYNN kernel source (two token counts; the sources must not depend on N). on=False:
    the same builds with the shipped static-size kernels (kind='pipeline' A/B of maxTotalThreadsPerThreadgroup)."""
    res = []
    with dynn(on):
        for N in ns:
            for label, build in dynn_builds(N):
                res += check(build, f"{'dynn' if on else 'static'} N{N} {label} ", kind)
    return res


def atom_checks() -> list:
    """kernels_atom: windowed NAX attention (every listed config, shared bias, no q bias, P half), segment mean, and
    the kernels_nax3 instantiations of the atom path (adaln16 D 128, gemm16 at the atom sites incl. 'ba' K 256)."""
    try:
        from boltz_mlx import kernels_atom as KA
        from boltz_mlx import kernels_nax3 as N3
    except ImportError:
        return []
    res = []
    NA, S = 96, 2
    h = mx.zeros((S * NA, 512), mx.float16)
    for cfg in KA.ATT_CFGS:
        res += check(lambda: KA.atom_attention(h, mx.zeros((NA // 32, 4, 32, 128)), NA=NA, qbias=mx.zeros((128,)),
                                               brep=2, **cfg), f"atom att {cfg} ")
    res += check(lambda: KA.atom_attention(h, mx.zeros((S * NA // 32, 4, 32, 128)), NA=NA), "atom att no-qb ")
    res += check(lambda: KA.segment_mean_relu(mx.zeros((S, NA, 768)), (mx.array([0, 3, 7], mx.int32),
                                                                       mx.zeros((7,), mx.int32), mx.ones((2,)))),
                 "atom seg ")
    L, D = 3, 128
    rc = N3.RawCond16(mx.zeros((NA, L * 4 * D), mx.float16), mx.zeros((L * 4 * D,)), mx.zeros((NA, L * 2 * D), mx.float16),
                      mx.zeros((L * 2 * D,)), ntok=NA, L=L, D=D, rep=S)
    a2 = mx.zeros((S * NA, D))
    t16 = mx.zeros((S * NA, D), mx.float16)
    res += check(lambda: N3.adaln16(a2, rc, 2, 2), "atom adaln16 ")
    for i, cfg in enumerate(N3.GEMM_CFGS):
        res += check(lambda: N3.gemm16(t16, mx.zeros((512, D), mx.float16), **cfg), f"atom qkvg {i} ")
        res += check(lambda: N3.gemm16(t16, mx.zeros((D, D), mx.float16), epi=N3.EPI_GRESID, gate=rc.h2, gate_bias=rc.b2,
                                       gate_off=2 * D, resid=a2, ntok=NA, rep=S, **cfg), f"atom o {i} ")
        res += check(lambda: N3.gemm16(mx.zeros((S * NA, 256), mx.float16), mx.zeros((D, 256), mx.float16),
                                       epi=N3.EPI_GRESID, gate=rc.h2, gate_bias=rc.b2, gate_off=3 * D, resid=a2, ntok=NA,
                                       rep=S, **cfg), f"atom ba {i} ")
    for i, cfg in enumerate(N3.SWIGLU_CFGS):
        res += check(lambda: N3.gemm16(t16, mx.zeros((768, D), mx.float16), epi=N3.EPI_SWIGLU, ngrp=3, **cfg),
                     f"atom ff {i} ")
    for i, cfg in enumerate(KA.TRANS_CFGS):
        res += check(lambda: KA.atom_transition(a2, rc, 1, mx.zeros((768, D), mx.float16), mx.zeros((D, 256), mx.float16),
                                                **cfg), f"atom trans {cfg} ")
    for i, cfg in enumerate(KA.ADALN_GEMM_CFGS):
        res += check(lambda: KA.adaln_gemm(a2, rc, 1, 0, mx.zeros((512, D), mx.float16), **cfg), f"atom adaln_gemm {cfg} ")
    return res


def check_msa() -> list:
    """MSA-module kernels (kernels_msa.py): every listed config, ragged shapes, fp16 / fp32 proj_z weight."""
    try:
        from boltz_mlx import kernels_msa as KM
    except ImportError:
        return []
    res = []
    S, N = 37, 13
    ab = mx.zeros((1, S, N, 64), mx.float32)
    mk = mx.ones((S, N), mx.float32)
    num = mx.ones((N, N), mx.float32)
    for cfg in KM.OPM_CFGS:
        res += check(lambda: KM.opm_outer(ab, mk, num, 3, 7, **cfg), f"opm {cfg} ")
    wo = mx.zeros((128, 1024), mx.float16)
    bo = mx.zeros((128,), mx.float32)
    for cfg in KM.OPM_FUSED_CFGS:
        res += check(lambda: KM.opm_outer_proj(ab, mk, num, wo, bo, 3, 7, **cfg), f"opm_fused {cfg} ")
    z = mx.zeros((1, N, N, 128), mx.float32)
    pm = mx.ones((1, N, N), mx.float32)
    lw = mx.ones((128,), mx.float32)
    for wdt in (mx.float16, mx.float32):
        res += check(lambda: KM.pwa_weights(z, pm, lw, lw, 1e-5, mx.zeros((8, 128), wdt), 1e6), f"pwa_w {wdt} ")
    w = mx.zeros((8, N, KM.pwa_np(N)), mx.float32)
    v = mx.zeros((1, S, N, 256), mx.float32)
    for cfg in KM.PWA_CFGS:
        res += check(lambda: KM.pwa_avg(w, v, v, **cfg), f"pwa_avg {cfg} ")
    return res


def _main():
    import numpy as np

    from boltz_mlx import kernels as K
    from boltz_mlx import kernels_nax as KN

    results = []
    H, C = 4, 32
    for N in (1, 5, 40):
        h = mx.zeros((1, N, N, 4 * H * C + H), mx.float16)
        m = mx.ones((1, N, N))
        results += check(lambda: K.tri_attention_fused(h.astype(mx.float32), H, C, m, 1e9, impl="mma"), f"N{N} mma32 ")
        results += check(lambda: K.tri_attention_fused(h, H, C, m, 1e9, impl="mma", out_dtype=mx.float16),
                         f"N{N} mma16 ")
        results += check(lambda: K.tri_attention_fused(h.astype(mx.float32), H, C, m, 1e9, impl="scalar"),
                         f"N{N} scalar ")
        results += check(lambda: KN.tri_attention_nax(h, H, C, m, 1e9, order=1, wq=5), f"N{N} nax ")
    try:
        from boltz_mlx import kernels_nax2 as KN2
    except ImportError:
        KN2 = None
    results += check(lambda: KN.nax_layout(True, False, True, mx.float16, mx.float16, mx.float32), "layout ")
    results += check(lambda: KN.nax_mma_generic(mx.zeros((16, 16), mx.float16), mx.zeros((32, 16), mx.float16), True,
                                                False), "mma_generic ")
    if KN2 is not None:
        x = mx.zeros((1, 50, 50, 128), mx.float16)
        w12 = mx.zeros((1024, 128), mx.float16)
        w3 = mx.zeros((128, 512), mx.float16)
        for cfg in KN2.TRANS_CFGS:
            results += check(lambda: KN2.transition_nax(x, w12, w3, **cfg), f"trans {cfg} ")
        z = mx.zeros((1, 50, 50, 128), mx.float32)
        results += check(lambda: KN2.transition_nax(x, w12, w3, resid=z), "trans resid ")
        x64 = mx.zeros((1, 7, 50, 64), mx.float16)
        results += check(lambda: KN2.transition_nax(x64, mx.zeros((512, 64), mx.float16), mx.zeros((64, 256), mx.float16),
                                                    hidden_scale=0.25), "trans64 ")
        for n in (100, 64):  # K tail / no tail
            a = mx.zeros((1, 128, n, n), mx.float16)
            for cfg in KN2.TRIMUL_CFGS:
                for og in (True, False):
                    results += check(lambda: KN2.trimul_contract_nax(a, a, og, **cfg), f"trimul N{n} og={og} {cfg} ")
        xf = mx.zeros((2500, 128), mx.float16)
        wpg = mx.zeros((512, 128), mx.float16)
        for cfg in KN2.TRIMUL_PROJ_CFGS:
            results += check(lambda: KN2.trimul_proj_nax(xf, wpg, mx.ones((2500,)), 1 / 16, **cfg), f"proj {cfg} ")
        results += check(lambda: KN2.trimul_proj_nax(xf, wpg, mx.ones((2500,)), 1 / 16, mx.zeros((128, 128), mx.float16)),
                         "proj+gout ")
        for B, N in ((1, 50), (3, 117)):
            hq = mx.zeros((B * N, 4 * 768), mx.float16)
            bias = mx.zeros((B, 16, N, N))
            results += check(lambda: KN2.attention_nax(hq, bias, B=B, N=N, heads=16, c=48, qbias=mx.zeros((768,))),
                             f"attn48 B{B} N{N} ")
    results += nax3_checks()
    results += check_msa()
    results += atom_checks()
    bad = 0
    for name, msg in results:
        print(f"{name}: {msg}")
        bad += msg != "OK"
    print(f"{len(results) - bad}/{len(results)} kernels compile")
    assert np  # noqa
    return bad


if __name__ == "__main__":
    if sys.argv[1:2] == ["dynn"]:  # python tools/prof/metal_check.py dynn [pipeline]
        r = dynn_checks(*(sys.argv[2:3] or ["check"]))
        for name, msg in r:
            print(f"{name}: {msg}")
        bad = sum(not msg.startswith("OK") for _, msg in r)
        print(f"{len(r) - bad}/{len(r)} kernels compile")
        sys.exit(1 if bad else 0)
    if sys.argv[1:] == ["atom"]:
        r = atom_checks()
        for name, msg in r:
            print(f"{name}: {msg}")
        bad = sum(msg != "OK" for _, msg in r)
        print(f"{len(r) - bad}/{len(r)} kernels compile")
        sys.exit(1 if bad else 0)
    sys.exit(1 if _main() else 0)
