"""Per-device kernel config entries are bitwise neutral: for every chip entry of nax3_cfg.json ("gemm", "pair",
"kern") and atom_cfg.json ("atom"), each kernel site's config at each tested size (the L1 case sizes plus both sides of
every size-bucket edge) gives bitwise the same outputs as the "default" entry's config (the M5 Pro tuning every
untuned chip runs). GPU:

  BOLTZ_MLX_DEVICE=gpu python -m pytest -q tests/test_cfg_bitwise.py

The resolver tests at the end run on the CPU too. End to end (whole f16 predictions of the L1 cases, chip entry vs
default): tools/prof/cfg_e2e_bitwise.py.
"""

import json
import os

import mlx.core as mx
import numpy as np
import pytest

gpu = pytest.mark.skipif(mx.default_device() != mx.gpu, reason="custom kernels need the GPU device")
PKG = os.path.join(os.path.dirname(__file__), "..", "boltz_mlx")


def _entries(fname: str) -> tuple:
    with open(os.path.join(PKG, fname)) as f:
        d = json.load(f)
    return [(c, d[c]) for c in d if c not in ("default", "_doc")], d["default"]


def _sizes(base: list, buckets, align: int = 1) -> list:
    """base sizes plus both sides of every bucket edge (aligned: atom rows come in windows of 32)."""
    s = set(base)
    for mx_, _ in buckets:
        lo = mx_ // align * align
        s.update((lo, lo + align))
    return sorted(x for x in s if x > 0)


def _h(rng, shape, s=1.0):
    return mx.array((rng.normal(size=shape) * s).astype(np.float16))


def _f32(rng, shape):
    return mx.array(rng.normal(size=shape).astype(np.float32))


def _same(a, b) -> bool:
    a = a if isinstance(a, (tuple, list)) else (a,)
    b = b if isinstance(b, (tuple, list)) else (b,)
    return len(a) == len(b) and all((x is None and y is None) or bool(mx.array_equal(x, y).item())
                                    for x, y in zip(a, b))


class _LN:
    def __init__(self, D, rng):
        self.w = mx.array(rng.normal(1.0, 0.1, D).astype(np.float32))
        self.b = mx.array(rng.normal(0.0, 0.1, D).astype(np.float32))
        self.eps = 1e-5

    def get(self, k, d=None):
        return {"weight": self.w, "bias": self.b}.get(k, d)


def _check(run, pick_c, pick_d, site, sizes, done: list):
    """Every size where the two entries' configs differ: outputs bitwise equal (run(size, cfg) -> array(s))."""
    for n in sizes:
        cc, cd = pick_c(site, n), pick_d(site, n)
        if cc == cd:
            continue
        assert "mlx" not in (cc, cd), f"{site}@{n}: 'mlx' vs a NAX config is not bitwise ({cc} vs {cd})"
        a, b = run(n, cc), run(n, cd)
        assert _same(a, b), f"{site} size {n}: {cc} differs from the default entry's {cd}"
        done.append((site, n))
        mx.clear_cache()


# --------------------------------------------------------------------------------------- token GEMM sites (gemm)
TOK_M = [37, 117, 213, 323, 384, 397, 400, 423, 586, 1065]


def _gemm_runner(N3, rng, site, D=768, gate_mult=48, gate_off=5, atom=False):
    from_k = {"o": (D, D, N3.EPI_GRESID, 1), "ff": (D, 2 * D, N3.EPI_SWIGLU, 3), "ba": (2 * D, D, N3.EPI_GRESID, 1),
              "qkvg": (D, 4 * D, N3.EPI_STORE, 1)}
    Kd, NOUT, epi, ngrp = from_k[site]
    cache = {}

    def run(M, cfg):
        if M not in cache:
            ntok, rep = (M // 5, 5) if M % 5 == 0 and M > 500 else (M, 1)
            x, w = _h(rng, (M, Kd)), _h(rng, (ngrp * NOUT, Kd), Kd ** -0.5)
            kw = {}
            if epi == N3.EPI_GRESID:
                nc = -(-M // (ntok * rep)) * ntok
                kw = dict(gate=_h(rng, (nc, gate_mult * D)), gate_bias=_f32(rng, (gate_mult * D,)),
                          gate_off=gate_off * D, resid=_f32(rng, (M, NOUT)), ntok=ntok, rep=rep)
            cache.clear()
            cache[M] = (x, w, kw)
        x, w, kw = cache[M]
        if Kd % (16 * cfg["ku"]) or NOUT % (cfg["wn"] * cfg["sn"]):
            pytest.fail(f"{site}: config {cfg} does not tile K={Kd} N={NOUT}")
        return N3.gemm16(x, w, epi=epi, ngrp=ngrp, **kw, **cfg)

    return run


@gpu
@pytest.mark.parametrize("site", ["o", "ff", "ba", "qkvg", "cond", "pproj"])
def test_nax3_gemm_sites(site):
    from boltz_mlx import kernels_nax3 as N3

    chips, dflt = _entries("nax3_cfg.json")
    if not chips:
        pytest.skip("no per-chip entries")
    rd = N3.resolve_entry(dflt)
    rng = np.random.default_rng(1)
    done = []
    for chip, e in chips:
        rc = N3.resolve_entry(e)
        tmp = {}
        N3._apply_cfg(e.get("gemm", "") + "," + e.get("pair", ""), {}, {**N3._GEMM_LISTS, **N3._PAIR_LISTS}, tmp)
        if site == "cond":
            cache = {}

            def run(M, cfg):
                if M not in cache:
                    cache.clear()
                    cache[M] = (_h(rng, (M, 768)), _h(rng, (24 * 2 * 768, 768), 768 ** -0.5))
                x, w = cache[M]
                return N3.gemm16(x, w, **cfg)

            _check(run, rc, rd, "cond", _sizes([117, 213, 323, 397, 400, 586], tmp.get("cond", [])), done)
        elif site == "pproj":
            def run(N, cfg):
                x, w = _h(np.random.default_rng(N), (N * N, 128)), _h(np.random.default_rng(N + 1), (512, 128), 128 ** -0.5)
                return N3.gemm16(x, w, **cfg)

            _check(run, rc, rd, "pproj", _sizes([40, 117, 213, 384, 400], tmp.get("pproj", [])), done)
        else:
            _check(_gemm_runner(N3, rng, site), rc, rd, site, _sizes(TOK_M, tmp.get(site, [])), done)
    print(f"{site}: {len(done)} (size, config) pairs differ from the default entry, all bitwise equal: {done}")


# --------------------------------------------------------------------------------------- pair sites (pair)
PAIR_N = [6, 37, 40, 117, 213, 323, 384, 400]


def _tail_inputs(N, D=128):
    rng = np.random.default_rng(N)
    o = _h(rng, (D, N * N), 0.05)
    xn, wpo, wgo = _h(rng, (N * N, D)), _h(rng, (D, D), D ** -0.5), _h(rng, (D, D), D ** -0.5)
    z = _f32(rng, (1, N, N, D))
    return o, _LN(D, rng), 2.0 ** -16, wpo, xn, wgo, z, _LN(D, rng), _h(rng, (4, D), 0.1)


@gpu
@pytest.mark.parametrize("site", ["tail3", "lin", "lin_end", "trans3"])
def test_nax3_pair_sites(site):
    from boltz_mlx import kernels_nax3 as N3

    chips, dflt = _entries("nax3_cfg.json")
    if not chips:
        pytest.skip("no per-chip entries")
    if site not in N3._PAIR_LISTS:
        pytest.skip(f"no site {site} in this build")
    rd = N3.resolve_entry(dflt)
    done = []

    def run(N, cfg):
        rng = np.random.default_rng(N + 5)
        D = 128
        if site == "tail3":
            o, lo, es, wpo, xn, wgo, z, ln, wtb = _tail_inputs(N)
            return (N3.trimul_tail3_addln(o, lo, es, wpo, xn, wgo, z, ln, **cfg)
                    + N3.trimul_tail3_addln(o, lo, es, wpo, xn, wgo, z, ln, w_tb=wtb, **cfg))
        if site in ("lin", "lin_end"):
            x, w, wtb = _h(rng, (1, N, N, D)), _h(rng, (D, D), D ** -0.5), _h(rng, (4, D), D ** -0.5)
            z, ln = _f32(rng, (1, N, N, D)), _LN(D, rng)
            b = _f32(rng, (D,))
            if site == "lin":   # starting node: transposed output + the ending node's tri bias
                return N3.linear_addln(x, w, z, ln, bias=b, trans_y=False, trans_out=True, w_tb=wtb, **cfg)
            return N3.linear_addln(x, w, z, ln, bias=b, trans_y=True, trans_out=False, **cfg)
        H = 512
        xn = _h(rng, (1, N, N, D))
        w12, w3 = _h(rng, (2 * H, D), D ** -0.5), _h(rng, (D, H), H ** -0.5)
        z = _f32(rng, (1, N, N, D))
        return N3.transition3(xn, w12, w3, z, next_ln=_LN(D, rng), ln_s=_LN(D, rng), w_s=_h(rng, (16, D), D ** -0.5),
                              cfg=cfg)

    for chip, e in chips:
        rc = N3.resolve_entry(e)
        tmp = {}
        N3._apply_cfg(e.get("pair", ""), {}, N3._PAIR_LISTS, tmp)
        edges = tmp.get(site, []) + (tmp.get("lin", []) if site == "lin_end" else [])   # lin_end falls back to lin
        _check(run, rc, rd, site, _sizes(PAIR_N, edges), done)
    print(f"{site}: {len(done)} (size, config) pairs differ from the default entry, all bitwise equal: {done}")


# --------------------------------------------------------------------------------------- attention launch shapes (kern)
@gpu
@pytest.mark.parametrize("site", ["triatt", "diffattn"])
def test_kern_sites(site):
    from boltz_mlx import kernels as K
    from boltz_mlx import kernels_nax as KN
    from boltz_mlx import kernels_nax2 as KN2

    chips, dflt = _entries("nax3_cfg.json")
    if not chips:
        pytest.skip("no per-chip entries")
    rd = K.resolve_kern(dflt.get("kern", ""))
    done = []

    def run(n, cfg):
        rng = np.random.default_rng(n + 11)
        if site == "triatt":
            hh = _h(rng, (1, n, n, 512), 0.5)
            mask = mx.array((rng.random(size=(1, n, n)) > 0.02).astype(np.float32))
            tb = _h(rng, (1, 4, n, n))
            prep = KN.nax_tri_inputs(hh, 4, 32, mask, 1e9, tb, kp_mult=64, bias_dtype=mx.float16)
            return KN.tri_attention_nax(hh, 4, 32, mask, 1e9, bias=tb, prepared=prep, **cfg)
        B = 5 if n % 5 == 0 and n > 600 else 1
        N = n // B
        hh = _h(rng, (B * N, 4 * 768), 0.5)
        bias = _f32(rng, (1, 16, N, N))
        qb = mx.array(rng.normal(size=768).astype(np.float32) * 0.1)
        return KN2.attention_nax(hh, bias, B=B, N=N, heads=16, c=48, qbias=qb, brep=B, **cfg)

    # triatt: J >= 32 (the model's NAX tri-att condition); diffattn: rows B * N
    base = [37, 117, 213, 323, 384, 400, 586] if site == "triatt" else [37, 117, 323, 384, 400, 586, 1065]
    for chip, e in chips:
        rc = K.resolve_kern(e.get("kern", ""))
        kb: dict = {}
        K.apply_kern_cfg(e.get("kern", ""), {k: dict(v) for k, v in K._KERN_BASE.items()}, kb)
        _check(run, rc, rd, site, [n for n in _sizes(base, kb.get(site, [])) if site != "triatt" or n >= 32], done)
    print(f"{site}: {len(done)} (size, config) pairs differ from the default entry, all bitwise equal: {done}")


# --------------------------------------------------------------------------------------- atom transformer (atom)
ATOM_M = [32, 928, 1536, 2976, 3072, 4544, 7680]


@gpu
@pytest.mark.parametrize("site", ["qkvg", "o", "ff", "ba", "aqkvg", "trans", "att"])
def test_atom_sites(site):
    from boltz_mlx import kernels_atom as KA
    from boltz_mlx import kernels_nax3 as N3

    chips, dflt = _entries("atom_cfg.json")
    if not chips:
        pytest.skip("no per-chip entries")
    rd = KA.resolve_entry(dflt.get("atom", ""))
    D = 128
    done = []

    def rc_for(M, S):
        rng = np.random.default_rng(M + 3)
        NA = M // S
        return N3.RawCond16(_h(rng, (NA, 3 * 4 * D), 0.5), _f32(rng, (3 * 4 * D,)), _h(rng, (NA, 3 * 2 * D), 0.5),
                            _f32(rng, (768,)), ntok=NA, L=3, D=D, rep=S), rng

    def run(M, cfg):
        S = 5 if M == 7680 else 1
        NA = M // S
        if site == "att":
            rng = np.random.default_rng(M + 7)
            hh = _h(rng, (M, 4 * D), 0.5)
            bias = _f32(rng, (NA // 32, 4, 32, 128))
            qb = mx.array(rng.normal(size=D).astype(np.float32) * 0.1)
            return KA.atom_attention(hh, bias, NA=NA, qbias=qb, brep=S, **cfg)
        if site == "aqkvg":
            rc, rng = rc_for(M, S)
            a2, wq = _f32(rng, (M, D)), _h(rng, (4 * D, D), D ** -0.5)
            return KA.adaln_gemm(a2, rc, 1, 0, wq, **cfg) if cfg != "split" else KA.adaln_gemm_ref(a2, rc, 1, 0, wq)
        if site == "trans":
            rc, rng = rc_for(M, S)
            a2 = _f32(rng, (M, D))
            w_ff, w_ba = _h(rng, (3 * 2 * D, D), D ** -0.5), _h(rng, (D, 2 * D), (2 * D) ** -0.5)
            assert cfg != "split", "trans: split vs a fused config is not a launch-shape choice"
            return KA.atom_transition(a2, rc, 1, w_ff, w_ba, **cfg)
        return _gemm_runner(N3, np.random.default_rng(M), site, D=D, gate_mult=6, gate_off=1)(M, cfg)

    for chip, e in chips:
        spec = e.get("atom", "")
        rc_ = KA.resolve_entry(spec)
        tmp: dict = {}
        KA.apply_cfg(spec, dict(KA._BASE_GEMM), dict(KA._BASE_ATT), tmp)
        sizes = _sizes(ATOM_M, tmp.get(site, []), 32)
        if site == "att":   # key blocking (nkf) and the P precision change the softmax arithmetic: never per-chip
            for M in sizes:
                for k, dv in (("nkf", None), ("pt", "float")):
                    assert rc_("att", M).get(k, dv) == rd("att", M).get(k, dv), (chip, M, k)
        if site in ("aqkvg", "trans"):   # 'split' vs a fused kernel is not a launch-shape choice: must agree
            for M in sizes:
                assert (rc_(site, M) == "split") == (rd(site, M) == "split"), (chip, site, M)
        _check(run, rc_, rd, site, sizes, done)
    print(f"atom {site}: {len(done)} (size, config) pairs differ from the default entry, all bitwise equal: {done}")


# --------------------------------------------------------------------------------------- resolvers (CPU)
def test_resolve_matches_live_tables_under_default(monkeypatch):
    """resolve_entry of the entry this process loaded reproduces the live tables (no hidden state in the resolvers)."""
    from boltz_mlx import kernels as K
    from boltz_mlx import kernels_atom as KA
    from boltz_mlx import kernels_nax3 as N3

    d = json.load(open(os.path.join(PKG, "nax3_cfg.json")))
    e = d.get(K.cfg_chip(), d["default"])
    if any(os.environ.get(v) for v in ("BOLTZ_MLX_NAX_GEMM_CFG", "BOLTZ_MLX_NAX_PAIR_CFG", "BOLTZ_MLX_NAX_KERN_CFG",
                                       "BOLTZ_MLX_ATOM_CFG")):
        pytest.skip("environment overrides active")
    r = N3.resolve_entry(e)
    for site in ("qkvg", "o", "ff", "ba", "cond", "pproj", "tail3", "lin", "lin_end", "trans3"):
        for n in (1, 37, 117, 213, 400, 1065, 4096):
            assert r(site, n) == N3.site_cfg(site, n), (site, n)
    rk = K.resolve_kern(e.get("kern", ""))
    for n in (1, 37, 117, 213, 400, 1065):
        assert rk("triatt", n) == K.triatt_cfg(n) and rk("diffattn", n) == K.diffattn_cfg(n)
    a = json.load(open(os.path.join(PKG, "atom_cfg.json")))
    ra = KA.resolve_entry(a.get(K.cfg_chip(), a["default"]).get("atom", ""))
    for site in ("qkvg", "o", "ff", "ba", "aqkvg", "trans", "att"):
        for M in ATOM_M:
            assert ra(site, M) == KA.site_cfg(site, M), (site, M)


def test_buckets_and_plain_reset():
    from boltz_mlx import kernels as K
    from boltz_mlx import kernels_nax3 as N3

    r = N3.resolve_entry({"gemm": "o=5,o@128=0,o@512=1", "pair": ""})
    assert r("o", 117) == N3.GEMM_CFGS[0] and r("o", 128) == N3.GEMM_CFGS[0]
    assert r("o", 129) == N3.GEMM_CFGS[1] and r("o", 512) == N3.GEMM_CFGS[1] and r("o", 513) == N3.GEMM_CFGS[5]
    r = N3.resolve_entry({"gemm": "o@128=0,o=5"})   # a later plain item clears the site's buckets
    assert r("o", 117) == N3.GEMM_CFGS[5]
    r = N3.resolve_entry({"pair": "lin=1,lin@128=2"})   # lin_end without its own entry follows lin (per size)
    assert r("lin_end", 117) == N3.PAIR_CFGS[2] and r("lin_end", 400) == N3.PAIR_CFGS[1]
    r = N3.resolve_entry({"pair": "lin=1,lin_end=0,lin_end@128=3"})
    assert r("lin", 117) == N3.PAIR_CFGS[1] and r("lin_end", 117) == N3.PAIR_CFGS[3] and r("lin_end", 400) == N3.PAIR_CFGS[0]
    live = {k: dict(v) for k, v in K._kern_tables().items()}
    live_b = {k: list(v) for k, v in K.KERN_BUCKETS.items()}
    rk = K.resolve_kern("triatt.wq=4,triatt@128.wq=8,triatt@128.hpt=2,diffattn.nkf=1")
    assert rk("triatt", 117)["wq"] == 8 and rk("triatt", 117)["hpt"] == 2 and rk("triatt", 129)["wq"] == 4
    assert rk("diffattn", 117)["nkf"] == K._KERN_BASE["diffattn"]["nkf"]   # nkf is not a launch-shape key: ignored
    assert K._kern_tables() == live and K.KERN_BUCKETS == live_b   # the live tables are untouched
