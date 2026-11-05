#!/usr/bin/env python
"""Round-2 statistical gates.

Reads jsonl rows (one per (case, seed, mode); schema in tools/validation/GATES.md) and prints every gate's statistic,
CI and PASS/FAIL. Thresholds come from validation/gates_config.json (PREREG-A defaults = the SPEC values; PREREG-B may
change them only through the calibration rule, without code changes).

  python tools/validation/gates.py rows.jsonl [more.jsonl] --candidate fp16 --strict strict
         [--candidate-type precision|sampler] [--cand-steps N --strict-steps N] [--config CFG.json]
         [--n-boot 10000] [--seed 0] [--json out.json] [--a4]
  python tools/validation/gates.py rows.jsonl --calibrate --mode strict
         --protein-cand-seeds 5-9 --protein-ref-seeds 0-4 --ligand-cand-seeds 3-5 --ligand-ref-seeds 0-2
         [--design-ligands 87] [--mc-reps 1000] [--out calibration.json]

Gate ids: L3P.lddt L3P.tm | L3L.median L3L.p90 L3L.chirality | A1.value A1.prob A2.value A2.prob (precision)
          A2p.value (sampler) | A3.pearson A3.kendall.   L2 and A4 are reporting helpers, not gates.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import sys
import warnings
import zlib
from pathlib import Path

import numpy as np
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "validation" / "gates_config.json"

# --------------------------------------------------------------------------- configuration (PREREG-A = SPEC values)

DEFAULTS: dict = {
    "version": "PREREG-A",
    "ci_level": 0.95,
    "l3p": {
        "lddt_ci_lower": -0.010,  # PASS: CI lower bound >= this ...
        "lddt_mean": -0.005,  # ... and mean >= this
        "tm_ci_lower": -0.015,
        "outlier_abs": 0.05,
        "large_chain_min_residues": 380,  # the 6 headline-size chains (380-500 residues)
    },
    "l3l": {
        "median_ratio_max": 1.25,  # one-sided bootstrap 95% upper bound of the ratio of medians
        "p90_ratio_max": 1.5,  # ... and of P90s
        "chirality_excess_max": 0,  # candidate flips - strict flips (same seeds) <= this
        "success_cutoff": 2.0,
        "pocket_radius": 10.0,
        "n_boot": 10000,
    },
    "l3a": {
        "a1_value": 0.05,  # 95% CI of mean seed-averaged delta(value) within +-this (log10)
        "a1_prob": 0.02,
        "a2_ratio_max": 0.5,  # RMS(seed-matched delta) <= this x RMS(strict i - strict j)   (precision)
        "a2p_ratio_max": 1.25,  # RMS(seed-averaged delta) <= this x sigma_seed x sqrt(1/n_c+1/n_r) (sampler)
        "a3_pearson_delta_lower": -0.05,  # paired-bootstrap CI lower bound of delta(weighted mean Pearson)
        "a3_tau_margin": 0.05,  # tau(cand, strict) CI lower bound >= tau_null - this
        "tau_null": None,  # strict-vs-strict value from calibration (PREREG-B); None = gate cannot be evaluated
        "n_boot": 10000,
        "optional_targets": {"chymase": 15},  # target -> min ligands before it is pooled into the gates
    },
    "a4": {
        "exclude_targets": ["bace1"],
        "published_pearson": 0.66,
        "published_source_url": "TODO: source URL of the published 0.66 (reported next to it)",
        "primary": "unweighted/mw_on",
        "primary_label": "aggregation assumed (paper setting not recorded by the owner)",
    },
    "calibration": {
        "null_pass_min": 0.9,
        "conjunction_min": 0.8,
        "power_min": 0.8,
        "respec_cap": 2.0,  # margins may be widened to at most this multiple of the PREREG-A value
        "respec_grid": [round(1.0 + 0.05 * k, 2) for k in range(21)],  # 1.00 ... 2.00
        # bad candidates for the power analysis: SPEC-defined ones marked (SPEC); the others are defined here and must
        # be confirmed in PREREG-A.
        "bad": {
            "affinity_value_bias": 0.10,  # (SPEC) +0.10 log10 bias in affinity_pred_value
            "weighted_pearson_delta": -0.10,  # (SPEC) weighted Pearson -0.10
            "ca_lddt_shift": -0.02,  # (SPEC) CA-lDDT -0.02
            "tm_shift": -0.02,  # (defined here) same size as the lDDT shift
            "affinity_prob_bias": 0.04,  # (defined here) twice the A1 prob window
            "rmsd_scale": 1.5,  # (defined here) candidate-vs-strict RMSDs inflated x1.5
            "chirality_extra_flip_rate": 0.05,  # (defined here) extra flips per pose
        },
        "effect_multiples": [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
        "a2_kappa_null": 0.25,  # null model for A2: RMS(matched delta)/RMS(strict i - j) = 0.25 (round 1: 0.02-0.05)
    },
}


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path=None) -> dict:
    """DEFAULTS (PREREG-A) overlaid with a JSON/YAML file (default validation/gates_config.json if it exists)."""
    p = Path(path) if path else (DEFAULT_CONFIG if DEFAULT_CONFIG.exists() else None)
    if p is None:
        return copy.deepcopy(DEFAULTS)
    txt = p.read_text()
    if p.suffix in (".yaml", ".yml"):
        import yaml

        over = yaml.safe_load(txt)
    else:
        over = json.loads(txt)
    return deep_merge(DEFAULTS, over)


def make_rng(seed: int, *tags) -> np.random.Generator:
    """Deterministic, independent stream per (seed, tag...)."""
    return np.random.default_rng([int(seed)] + [zlib.crc32(str(t).encode()) for t in tags])


# --------------------------------------------------------------------------- rows


def read_rows(paths) -> list[dict]:
    rows = []
    for p in paths:
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#"):
                    rows.append(json.loads(line))
    return rows


def select(rows, mode, steps=None):
    return [r for r in rows if r.get("mode") == mode and (steps is None or r.get("steps") == steps)]


def _f(r, k):
    v = r.get(k)
    return float("nan") if v is None else float(v)


def guard_total(r) -> int:
    g = r.get("guard_counts", r.get("guard_count", 0))
    if isinstance(g, dict):
        return int(sum(int(v) for v in g.values()))
    return int(g or 0)


def parse_seeds(s: str) -> list[int]:
    out = []
    for part in str(s).split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


# --------------------------------------------------------------------------- aligned data matrices


def build_matrix(rows_c, rows_r, key, pairs=None):
    """Per-case matrices of ``key`` for the candidate and reference arms with seed columns aligned.

    ``pairs`` = [(cand_seed, ref_seed), ...] (default: every seed present in both arms, matched on equal seed).
    Returns dict(cases, target, cand (C,S), ref (C,S), pairs); entries where either arm is missing are NaN in both.
    """
    vc, vr, tgt, order = {}, {}, {}, []
    for rows, d in ((rows_c, vc), (rows_r, vr)):
        for r in rows:
            if r.get(key) is None:
                continue
            c = r["case"]
            d.setdefault(c, {})[int(r["seed"])] = float(r[key])
            if c not in tgt:
                tgt[c] = r.get("target", c)
                order.append(c)
    if pairs is None:
        seeds = sorted({s for d in vc.values() for s in d} & {s for d in vr.values() for s in d})
        pairs = [(s, s) for s in seeds]
    cases = sorted(c for c in order if c in vc and c in vr)
    C, S = len(cases), len(pairs)
    a, b = np.full((C, S), np.nan), np.full((C, S), np.nan)
    for i, c in enumerate(cases):
        for j, (sc, sr) in enumerate(pairs):
            if sc in vc[c] and sr in vr[c]:
                a[i, j], b[i, j] = vc[c][sc], vr[c][sr]
    keep = ~np.all(np.isnan(a), axis=1)
    cases = [c for c, k in zip(cases, keep) if k]
    return {"cases": cases, "target": [tgt[c] for c in cases], "cand": a[keep], "ref": b[keep], "pairs": list(pairs)}


def build_all_seed_matrix(rows, key, cases=None, seeds=None):
    """(cases, targets, seeds, X (C, S_all)) of one arm over all its seeds (for estimating seed SDs)."""
    d, tgt = {}, {}
    for r in rows:
        if r.get(key) is not None:
            d.setdefault(r["case"], {})[int(r["seed"])] = float(r[key])
        tgt[r["case"]] = r.get("target", r["case"])
    cases = sorted(d) if cases is None else cases
    seeds = sorted({s for v in d.values() for s in v}) if seeds is None else seeds
    X = np.full((len(cases), len(seeds)), np.nan)
    for i, c in enumerate(cases):
        for j, s in enumerate(seeds):
            X[i, j] = d.get(c, {}).get(s, np.nan)
    return cases, [tgt[c] for c in cases], seeds, X


# --------------------------------------------------------------------------- statistics (vectorised over leading dims)


def tci(delta, level=0.95):
    """t-based CI of the mean over the last axis (NaN-aware; df = n - 1). Returns mean, lo, hi, n."""
    delta = np.asarray(delta, float)
    n = np.isfinite(delta).sum(-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m = np.nanmean(delta, -1)
        sd = np.nanstd(delta, -1, ddof=1)
        q = sps.t.ppf(0.5 + level / 2, np.maximum(n - 1, 1))
    half = q * sd / np.sqrt(np.maximum(n, 1))
    return m, m - half, m + half, n


def _nanmean(x, axis):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(x, axis)


def protein_stats(lddt_c, lddt_r, tm_c, tm_r, level=0.95) -> dict:
    """L3-P: per-case delta averaged over seeds; mean and t-based CI over cases. Arrays (..., C, S)."""
    out = {}
    for name, c, r in (("lddt", lddt_c, lddt_r), ("tm", tm_c, tm_r)):
        both = np.isfinite(c) & np.isfinite(r)
        d = _nanmean(np.where(both, c, np.nan), -1) - _nanmean(np.where(both, r, np.nan), -1)
        m, lo, hi, n = tci(d, level)
        out.update({f"{name}_mean": m, f"{name}_lo": lo, f"{name}_hi": hi, f"{name}_n": n, f"{name}_delta": d})
    return out


def _rms(x, axes):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.sqrt(np.nanmean(x * x, axes))


def affinity_ab_stats(vc, vr, pc, pr, level=0.95) -> dict:
    """L3-A A1 / A2 / A2' statistics. Arrays (..., L, S): candidate and reference, seed columns aligned."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _affinity_ab_stats(vc, vr, pc, pr, level)


def _affinity_ab_stats(vc, vr, pc, pr, level):
    out = {}
    for name, c, r in (("v", vc, vr), ("p", pc, pr)):
        if c is None:
            continue
        d = _nanmean(c, -1) - _nanmean(r, -1)  # seed-averaged per-ligand delta
        m, lo, hi, n = tci(d, level)
        out.update({f"a1{name}_mean": m, f"a1{name}_lo": lo, f"a1{name}_hi": hi, f"a1{name}_n": n})
        S = r.shape[-1]
        rms_matched = _rms(c - r, (-2, -1))  # seed-matched per-run delta
        if S >= 2:
            ii, jj = np.triu_indices(S, 1)
            rms_strict = _rms(r[..., ii] - r[..., jj], (-2, -1))
            sig = np.sqrt(np.nanmean(np.nanvar(r, -1, ddof=1), -1)) if S >= 2 else np.nan  # pooled per-ligand sigma
        else:
            rms_strict, sig = np.full(rms_matched.shape, np.nan), np.full(rms_matched.shape, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out[f"a2{name}_ratio"] = rms_matched / rms_strict
            out[f"a2{name}_rms_matched"] = rms_matched
            out[f"a2{name}_rms_strict_pair"] = rms_strict
            Sc = c.shape[-1]
            out[f"a2p{name}_ratio"] = _rms(d, -1) / (sig * np.sqrt(1.0 / Sc + 1.0 / S))
            out[f"a2p{name}_rms_avg"] = _rms(d, -1)
            out[f"sigma_seed_{name}"] = sig
    return out


def a3_stats(cand_avg, ref_avg, exp, groups, n_boot, rng, level=0.95) -> dict:
    """L3-A A3: paired within-target bootstrap of delta(ligand-weighted mean Pearson vs experiment) and of the
    ligand-weighted pooled Kendall tau(candidate, strict). Inputs are seed-averaged per-ligand predictions."""
    groups = np.asarray(groups)
    boots = score.grouped_bootstrap_indices(groups, n_boot, rng)
    pe = score.boot_weighted_pearson({"c": cand_avg, "r": ref_avg}, exp, boots, "ligands")
    d = pe["c"] - pe["r"]
    tau = score.boot_pooled_kendall(cand_avg, ref_avg, boots)
    a = (1 - level) / 2
    fin = lambda x: x[np.isfinite(x)]  # noqa: E731
    return {
        "a3_pear_delta": score.weighted_pearson(cand_avg, exp, groups) - score.weighted_pearson(ref_avg, exp, groups),
        "a3_pear_lo": float(np.quantile(fin(d), a)) if np.isfinite(d).any() else np.nan,
        "a3_pear_hi": float(np.quantile(fin(d), 1 - a)) if np.isfinite(d).any() else np.nan,
        "a3_tau": score.pooled_kendall(cand_avg, ref_avg, groups),
        "a3_tau_lo": float(np.quantile(fin(tau), a)) if np.isfinite(tau).any() else np.nan,
        "a3_tau_hi": float(np.quantile(fin(tau), 1 - a)) if np.isfinite(tau).any() else np.nan,
        "a3_pear_cand": score.weighted_pearson(cand_avg, exp, groups),
        "a3_pear_ref": score.weighted_pearson(ref_avg, exp, groups),
    }


def pose_stats(cs, ss, groups, n_boot, rng, ub_level=0.95) -> dict:
    """L3-L: ratio of medians and of P90s of RMSD(candidate, strict same seed) (cs, (L, kc)) over
    RMSD(strict i, strict j) (ss, (L, ks)); one-sided bootstrap upper bounds (ligands resampled within target)."""
    cs, ss = np.asarray(cs, float), np.asarray(ss, float)

    def ratios(c, s):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            c, s = c.reshape(c.shape[0], -1), s.reshape(s.shape[0], -1)
            mc, ms = np.nanmedian(c, 1), np.nanmedian(s, 1)
            p9c, p9s = np.nanpercentile(c, 90, 1), np.nanpercentile(s, 90, 1)
            return mc / ms, p9c / p9s

    pm, pp = ratios(cs.reshape(1, -1), ss.reshape(1, -1))
    boots = score.grouped_bootstrap_indices(groups, n_boot, rng)
    idx = np.concatenate([bi for _, bi in boots], 1)  # (B, L) resampled ligand indices (stratified)
    rm, rp = ratios(cs[idx].reshape(n_boot, -1), ss[idx].reshape(n_boot, -1))
    return {
        "l3l_med": float(pm[0]),
        "l3l_p90": float(pp[0]),
        "l3l_med_ub": float(np.nanquantile(rm, ub_level)),
        "l3l_p90_ub": float(np.nanquantile(rp, ub_level)),
        "l3l_med_cand": float(np.nanmedian(cs)),
        "l3l_med_strict_pair": float(np.nanmedian(ss)),
        "l3l_p90_cand": float(np.nanpercentile(cs, 90)),
        "l3l_p90_strict_pair": float(np.nanpercentile(ss, 90)),
    }


# --------------------------------------------------------------------------- gate definitions

GATE_ORDER = [
    "L3P.lddt",
    "L3P.tm",
    "L3L.median",
    "L3L.p90",
    "L3L.chirality",
    "A1.value",
    "A1.prob",
    "A2.value",
    "A2.prob",
    "A2p.value",
    "A3.pearson",
    "A3.kendall",
]
GATE_GROUP = {
    "L3P.lddt": "protein",
    "L3P.tm": "protein",
    "L3L.median": "pose",
    "L3L.p90": "pose",
    "L3L.chirality": "pose",
    "A1.value": "affinity",
    "A1.prob": "affinity",
    "A2.value": "affinity",
    "A2.prob": "affinity",
    "A2p.value": "affinity",
    "A3.pearson": "affinity",
    "A3.kendall": "affinity",
}
GATE_SETS = {
    "precision": [g for g in GATE_ORDER if g != "A2p.value"],
    "sampler": [g for g in GATE_ORDER if g not in ("A2.value", "A2.prob")],
}
# SPEC scenario each gate is stress-tested against for power / MDE
GATE_SCENARIO = {
    "L3P.lddt": "lddt_shift",
    "L3P.tm": "tm_shift",
    "L3L.median": "rmsd_scale",
    "L3L.p90": "rmsd_scale",
    "L3L.chirality": "chir_rate",
    "A1.value": "value_bias",
    "A1.prob": "prob_bias",
    "A2.value": "value_bias",
    "A2.prob": "prob_bias",
    "A2p.value": "value_bias",
    "A3.pearson": "pearson_drop",
    "A3.kendall": "pearson_drop",
}


def gate_pass(gid, s, cfg, f=1.0, tau_null=None):
    """Pass rule of one gate at margin factor ``f`` (f = 1: the configured threshold). ``s`` holds statistics (floats
    or arrays). Margins are scaled as the distance of the threshold from its null reference (0 for differences and CI
    bounds, 1 for ratios, tau_null for the tau gate)."""
    p, l, a = cfg["l3p"], cfg["l3l"], cfg["l3a"]
    eps = 1e-12
    with np.errstate(invalid="ignore"):
        if gid == "L3P.lddt":
            return (s["lddt_lo"] >= p["lddt_ci_lower"] * f - eps) & (s["lddt_mean"] >= p["lddt_mean"] * f - eps)
        if gid == "L3P.tm":
            return s["tm_lo"] >= p["tm_ci_lower"] * f - eps
        if gid == "L3L.median":
            return s["l3l_med_ub"] <= 1 + (l["median_ratio_max"] - 1) * f + eps
        if gid == "L3L.p90":
            return s["l3l_p90_ub"] <= 1 + (l["p90_ratio_max"] - 1) * f + eps
        if gid == "L3L.chirality":
            return s["chir_excess"] <= l["chirality_excess_max"] * f + eps
        if gid == "A1.value":
            w = a["a1_value"] * f
            return (s["a1v_lo"] >= -w - eps) & (s["a1v_hi"] <= w + eps)
        if gid == "A1.prob":
            w = a["a1_prob"] * f
            return (s["a1p_lo"] >= -w - eps) & (s["a1p_hi"] <= w + eps)
        if gid == "A2.value":
            return s["a2v_ratio"] <= a["a2_ratio_max"] * f + eps
        if gid == "A2.prob":
            return s["a2p_ratio"] <= a["a2_ratio_max"] * f + eps
        if gid == "A2p.value":
            return s["a2pv_ratio"] <= 1 + (a["a2p_ratio_max"] - 1) * f + eps
        if gid == "A3.pearson":
            return s["a3_pear_lo"] >= a["a3_pearson_delta_lower"] * f - eps
        if gid == "A3.kendall":
            if tau_null is None:
                tau_null = a.get("tau_null")
            if tau_null is None:
                return np.zeros_like(np.asarray(s["a3_tau_lo"], float), dtype=bool)
            return s["a3_tau_lo"] >= tau_null - a["a3_tau_margin"] * f - eps
    raise KeyError(gid)


def gate_threshold_text(gid, cfg, f=1.0, tau_null=None):
    p, l, a = cfg["l3p"], cfg["l3l"], cfg["l3a"]
    g = lambda x: f"{x:.4g}"  # noqa: E731
    return {
        "L3P.lddt": f"CI lower >= {g(p['lddt_ci_lower'] * f)} and mean >= {g(p['lddt_mean'] * f)}",
        "L3P.tm": f"CI lower >= {g(p['tm_ci_lower'] * f)}",
        "L3L.median": f"ratio of medians, 95% upper bound <= {g(1 + (l['median_ratio_max'] - 1) * f)}",
        "L3L.p90": f"ratio of P90s, 95% upper bound <= {g(1 + (l['p90_ratio_max'] - 1) * f)}",
        "L3L.chirality": f"candidate flips - strict flips <= {g(l['chirality_excess_max'] * f)}",
        "A1.value": f"95% CI within +-{g(a['a1_value'] * f)} log10",
        "A1.prob": f"95% CI within +-{g(a['a1_prob'] * f)}",
        "A2.value": f"RMS(matched delta)/RMS(strict i-j) <= {g(a['a2_ratio_max'] * f)}",
        "A2.prob": f"RMS(matched delta)/RMS(strict i-j) <= {g(a['a2_ratio_max'] * f)}",
        "A2p.value": f"RMS(avg delta)/(sigma_seed*sqrt(1/nc+1/nr)) <= {g(1 + (a['a2p_ratio_max'] - 1) * f)}",
        "A3.pearson": f"delta wPearson CI lower >= {g(a['a3_pearson_delta_lower'] * f)}",
        "A3.kendall": "tau(cand,strict) CI lower >= "
        + (
            "tau_null - %s (tau_null NOT SET)" % g(a["a3_tau_margin"] * f)
            if (tau_null if tau_null is not None else a.get("tau_null")) is None
            else g((tau_null if tau_null is not None else a["tau_null"]) - a["a3_tau_margin"] * f)
        ),
    }[gid]


def gate_stat_text(gid, s):
    """(statistic text) for a scalar-statistics dict ``s``."""
    t = lambda x: "nan" if x is None or not np.isfinite(x) else f"{x:+.4f}"  # noqa: E731
    return {
        "L3P.lddt": lambda: f"mean {t(s['lddt_mean'])} CI [{t(s['lddt_lo'])}, {t(s['lddt_hi'])}] n={int(s['lddt_n'])}",
        "L3P.tm": lambda: f"mean {t(s['tm_mean'])} CI [{t(s['tm_lo'])}, {t(s['tm_hi'])}] n={int(s['tm_n'])}",
        "L3L.median": lambda: f"ratio {s['l3l_med']:.3f} ub95 {s['l3l_med_ub']:.3f}",
        "L3L.p90": lambda: f"ratio {s['l3l_p90']:.3f} ub95 {s['l3l_p90_ub']:.3f}",
        "L3L.chirality": lambda: f"cand {int(s['chir_cand'])} strict {int(s['chir_strict'])} (excess {int(s['chir_excess'])})",
        "A1.value": lambda: f"mean {t(s['a1v_mean'])} CI [{t(s['a1v_lo'])}, {t(s['a1v_hi'])}] n={int(s['a1v_n'])}",
        "A1.prob": lambda: f"mean {t(s['a1p_mean'])} CI [{t(s['a1p_lo'])}, {t(s['a1p_hi'])}] n={int(s['a1p_n'])}",
        "A2.value": lambda: f"ratio {s['a2v_ratio']:.3f} (RMS {s['a2v_rms_matched']:.4f} / {s['a2v_rms_strict_pair']:.4f})",
        "A2.prob": lambda: f"ratio {s['a2p_ratio']:.3f} (RMS {s['a2p_rms_matched']:.4f} / {s['a2p_rms_strict_pair']:.4f})",
        "A2p.value": lambda: f"ratio {s['a2pv_ratio']:.3f} (RMS {s['a2pv_rms_avg']:.4f}, sigma_seed {s['sigma_seed_v']:.4f})",
        "A3.pearson": lambda: f"delta {t(s['a3_pear_delta'])} CI [{t(s['a3_pear_lo'])}, {t(s['a3_pear_hi'])}]",
        "A3.kendall": lambda: f"tau {t(s['a3_tau'])} CI [{t(s['a3_tau_lo'])}, {t(s['a3_tau_hi'])}]",
    }[gid]()


def _scalar(d):
    return {k: (float(v) if np.ndim(v) == 0 else v) for k, v in d.items()}


# --------------------------------------------------------------------------- pose source (ligand RMSD / chirality)


class PoseSource:
    """Pairwise ligand RMSDs and chirality flips from rows. Precomputed ``type: "ligand_rmsd_pair"`` rows take
    precedence; otherwise RMSDs are computed from the ``pose_cif`` paths with score.ligand_pose_rmsd (cached)."""

    def __init__(self, rows, cfg):
        self.cfg = cfg
        self.rows = {}
        self.pairs = {}
        for r in rows:
            if r.get("type") == "ligand_rmsd_pair":
                self.pairs[(r["case"], r["a_mode"], int(r["a_seed"]), r["b_mode"], int(r["b_seed"]))] = float(r["rmsd"])
            elif r.get("pose_cif") or r.get("chirality_flips") is not None:
                self.rows[(r["case"], r["mode"], int(r["seed"]), r.get("steps"))] = r
        self.cache = {}

    def row(self, case, mode, seed, steps=None):
        r = self.rows.get((case, mode, seed, steps))
        if r is None:
            for (c, m, s, _), v in self.rows.items():
                if (c, m, s) == (case, mode, seed):
                    return v
        return r

    def rmsd(self, case, a, b):
        """RMSD of pose a = (mode, seed, steps) against reference pose b (pocket frame), or NaN if unavailable."""
        key = (case, a, b)
        if key in self.cache:
            return self.cache[key]
        v = self.pairs.get((case, a[0], a[1], b[0], b[1]))
        if v is None:
            v = self.pairs.get((case, b[0], b[1], a[0], a[1]))  # RMSD is symmetric up to the pocket definition
        if v is None:
            ra, rb = self.row(case, *a), self.row(case, *b)
            if ra is None or rb is None or not ra.get("pose_cif") or not rb.get("pose_cif"):
                v = float("nan")
            else:
                v = score.ligand_pose_rmsd(
                    ra["pose_cif"],
                    rb["pose_cif"],
                    ra.get("ligand_chain", "B"),
                    ra["smiles"],
                    ref_ligand_chain=rb.get("ligand_chain", ra.get("ligand_chain", "B")),
                    radius=self.cfg["l3l"]["pocket_radius"],
                )["ligand_rmsd"]
        self.cache[key] = v
        return v

    def flips(self, case, mode, seed, steps=None):
        r = self.row(case, mode, seed, steps)
        if r is None:
            return float("nan")
        if r.get("chirality_flips") is not None:
            return float(r["chirality_flips"])
        if not r.get("pose_cif") or not r.get("smiles"):
            return float("nan")
        key = ("chir", r["pose_cif"], r.get("ligand_chain", "B"))
        if key not in self.cache:
            c = score.load_complex(r["pose_cif"], r.get("ligand_chain", "B"), receptor_chains=[])
            self.cache[key] = float(score.chirality_flips(r["smiles"], c.lig_symbols, c.lig_xyz)["n_flips"])
        return self.cache[key]


def build_pose(src: PoseSource, cases, targets, cand_mode, cand_steps, ref_mode, ref_steps, pairs):
    """cs[l, j] = RMSD(cand seed s_c, ref seed s_r) for pair j; ss[l, k] = RMSD(ref seed j, ref seed i) over ref-seed
    pairs; chirality flip totals over the matched poses of each arm."""
    ref_seeds = sorted({sr for _, sr in pairs})
    sp = list(itertools.combinations(ref_seeds, 2))
    L = len(cases)
    cs, ss = np.full((L, len(pairs)), np.nan), np.full((L, max(len(sp), 1)), np.nan)
    chc, chs = np.zeros(L), np.zeros(L)
    for i, c in enumerate(cases):
        for j, (sc, sr) in enumerate(pairs):
            cs[i, j] = src.rmsd(c, (cand_mode, sc, cand_steps), (ref_mode, sr, ref_steps))
            chc[i] += np.nan_to_num(src.flips(c, cand_mode, sc, cand_steps))
            chs[i] += np.nan_to_num(src.flips(c, ref_mode, sr, ref_steps))
        for k, (a, b) in enumerate(sp):
            ss[i, k] = src.rmsd(c, (ref_mode, b, ref_steps), (ref_mode, a, ref_steps))
    ok = np.isfinite(cs).any(1) & np.isfinite(ss).any(1)
    return {
        "cases": [c for c, k in zip(cases, ok) if k],
        "target": [t for t, k in zip(targets, ok) if k],
        "cs": cs[ok],
        "ss": ss[ok],
        "chir_c": chc[ok],
        "chir_s": chs[ok],
    }


# --------------------------------------------------------------------------- dataset assembly


def case_targets(rows):
    return {r["case"]: r.get("target", r["case"]) for r in rows}


def case_attr(rows, key):
    out = {}
    for r in rows:
        if r.get(key) is not None:
            out.setdefault(r["case"], r[key])
    return out


def build_dataset(rows, cand_mode, ref_mode, cfg, cand_steps=None, ref_steps=None, pairs=None, pose_source=None):
    """Assemble protein / affinity / pose datasets for one candidate-vs-reference comparison."""
    rc, rr = select(rows, cand_mode, cand_steps), select(rows, ref_mode, ref_steps)
    data = {"cand_mode": cand_mode, "ref_mode": ref_mode, "pairs": pairs}
    if any(r.get("ca_lddt") is not None for r in rc):
        ml = build_matrix(rc, rr, "ca_lddt", pairs)
        mt = build_matrix(rc, rr, "tm", pairs)
        tm_c = np.full_like(ml["cand"], np.nan)
        tm_r = np.full_like(ml["ref"], np.nan)
        idx = {c: i for i, c in enumerate(mt["cases"])}
        for i, c in enumerate(ml["cases"]):
            if c in idx and mt["cand"].shape[1] == ml["cand"].shape[1]:
                tm_c[i], tm_r[i] = mt["cand"][idx[c]], mt["ref"][idx[c]]
        nres = case_attr(rc + rr, "n_res")
        large = case_attr(rc + rr, "large_chain")
        thr = cfg["l3p"]["large_chain_min_residues"]
        data["protein"] = {
            "cases": ml["cases"],
            "target": ml["target"],
            "lddt_c": ml["cand"],
            "lddt_r": ml["ref"],
            "tm_c": tm_c,
            "tm_r": tm_r,
            "large": np.array([bool(large.get(c, nres.get(c, 0) >= thr)) for c in ml["cases"]]),
        }
    if any(r.get("affinity_pred_value") is not None for r in rc):
        mv = build_matrix(rc, rr, "affinity_pred_value", pairs)
        mp = build_matrix(rc, rr, "affinity_probability_binary", pairs)
        cases = mv["cases"]
        pc = np.full_like(mv["cand"], np.nan)
        pr = np.full_like(mv["ref"], np.nan)
        idx = {c: i for i, c in enumerate(mp["cases"])}
        for i, c in enumerate(cases):
            if c in idx:
                pc[i], pr[i] = mp["cand"][idx[c]], mp["ref"][idx[c]]
        exp = case_attr(rr + rc, "exp_value")
        e = np.array([exp.get(c, np.nan) for c in cases], float)
        tg = np.array(mv["target"])
        keep = np.ones(len(cases), bool)
        for tname, nmin in cfg["l3a"]["optional_targets"].items():
            m = np.array([str(t).lower() == tname for t in tg])
            if m.sum() < nmin:
                keep &= ~m  # reported only, not pooled
        data["affinity_excluded"] = [c for c, k in zip(cases, keep) if not k]
        data["affinity"] = {
            "cases": [c for c, k in zip(cases, keep) if k],
            "target": tg[keep],
            "exp": e[keep],
            "vc": mv["cand"][keep],
            "vr": mv["ref"][keep],
            "pc": pc[keep],
            "pr": pr[keep],
        }
    src = pose_source if pose_source is not None else PoseSource(rows, cfg)
    if src.rows or src.pairs:
        if pairs is None:
            seeds_c = {int(r["seed"]) for r in rc}
            seeds_r = {int(r["seed"]) for r in rr}
            pairs = [(s, s) for s in sorted(seeds_c & seeds_r)]
        cases = sorted({c for (c, m, s, st) in src.rows if m in (cand_mode, ref_mode)} | {k[0] for k in src.pairs})
        tgt = case_targets(rc + rr)
        pose = build_pose(src, cases, [tgt.get(c, c) for c in cases], cand_mode, cand_steps, ref_mode, ref_steps, pairs)
        if len(pose["cases"]):
            data["pose"] = pose
    data["pose_source"] = src
    return data


# --------------------------------------------------------------------------- evaluation of the whole gate set


def evaluate_dataset(data, cfg, ctype, n_boot, seed=0, tau_null=None) -> dict:
    """Run every applicable gate on a dataset (scalar statistics). Returns {gate_id: {"stats", "pass", ...}}."""
    level = cfg["ci_level"]
    stats = {}
    if "protein" in data:
        d = data["protein"]
        stats.update(_scalar(protein_stats(d["lddt_c"], d["lddt_r"], d["tm_c"], d["tm_r"], level)))
    if "affinity" in data:
        d = data["affinity"]
        stats.update(_scalar(affinity_ab_stats(d["vc"], d["vr"], d["pc"], d["pr"], level)))
        ok = np.isfinite(d["exp"])
        if ok.sum() >= 3:
            rng = make_rng(seed, "a3")
            stats.update(
                a3_stats(
                    _nanmean(d["vc"][ok], -1), _nanmean(d["vr"][ok], -1), d["exp"][ok], np.asarray(d["target"])[ok], n_boot, rng, level
                )
            )
    if "pose" in data:
        d = data["pose"]
        stats.update(pose_stats(d["cs"], d["ss"], d["target"], n_boot, make_rng(seed, "pose")))
        stats["chir_cand"], stats["chir_strict"] = float(d["chir_c"].sum()), float(d["chir_s"].sum())
        stats["chir_excess"] = stats["chir_cand"] - stats["chir_strict"]
    out = {}
    for gid in GATE_ORDER:
        needed = {
            "L3P.lddt": "lddt_lo",
            "L3P.tm": "tm_lo",
            "L3L.median": "l3l_med_ub",
            "L3L.p90": "l3l_p90_ub",
            "L3L.chirality": "chir_excess",
            "A1.value": "a1v_lo",
            "A1.prob": "a1p_lo",
            "A2.value": "a2v_ratio",
            "A2.prob": "a2p_ratio",
            "A2p.value": "a2pv_ratio",
            "A3.pearson": "a3_pear_lo",
            "A3.kendall": "a3_tau_lo",
        }[gid]
        active = gid in GATE_SETS[ctype]
        if needed not in stats or not np.isfinite(stats[needed]):
            out[gid] = {"status": "N/A", "reason": "no data", "active": active}
            continue
        ok = bool(gate_pass(gid, stats, cfg, 1.0, tau_null))
        status = "PASS" if ok else "FAIL"
        reason = None
        if gid == "A3.kendall" and (tau_null if tau_null is not None else cfg["l3a"].get("tau_null")) is None:
            status, reason = "UNDETERMINED", "tau_null not set (run --calibrate; PREREG-B sets l3a.tau_null)"
        out[gid] = {
            "status": status,
            "active": active,
            "statistic": gate_stat_text(gid, stats),
            "threshold": gate_threshold_text(gid, cfg, 1.0, tau_null),
            "reason": reason,
        }
    return {"gates": out, "stats": {k: v for k, v in stats.items() if np.ndim(v) == 0}, "arrays": stats}


# --------------------------------------------------------------------------- reporting helpers


def l2_report(rows, cand_mode, ref_mode, cand_steps=None, ref_steps=None, metrics=None) -> dict:
    """Level 2 (diagnostic, not a gate): hard failures = any NaN/inf flag or nonzero guard count in candidate rows;
    flags = a candidate-vs-strict seed-matched delta larger than the strict seed-to-seed spread (range over strict
    seeds) for that case. Optional coordinate RMSD from ``model_cif`` (CA, Kabsch)."""
    rc, rr = select(rows, cand_mode, cand_steps), select(rows, ref_mode, ref_steps)
    hard = []
    for r in rc + rr:
        bad = []
        if r.get("nan") or r.get("inf"):
            bad.append("nan/inf")
        if guard_total(r):
            bad.append(f"guard_count={guard_total(r)}")
        if bad:
            hard.append({"case": r["case"], "seed": r.get("seed"), "mode": r.get("mode"), "reason": ",".join(bad)})
    metrics = metrics or [
        "ca_lddt", "tm", "ptm", "iptm", "plddt", "affinity_pred_value", "affinity_probability_binary", "ligand_rmsd_ref",
    ]  # fmt: skip
    flags = []
    byc = {}
    for r in rr:
        byc.setdefault(r["case"], {})[int(r["seed"])] = r
    for r in rc:
        s = byc.get(r["case"], {})
        if int(r["seed"]) not in s or len(s) < 2:
            continue
        for m in metrics:
            if r.get(m) is None or any(x.get(m) is None for x in s.values()):
                continue
            vals = [float(x[m]) for x in s.values()]
            spread = max(vals) - min(vals)
            delta = abs(float(r[m]) - float(s[int(r["seed"])][m]))
            if delta > spread:
                flags.append(
                    {"case": r["case"], "seed": int(r["seed"]), "metric": m, "delta": delta, "strict_spread": spread}
                )
    # coordinate RMSD where model_cif paths exist
    cm = {}
    for r in rc + rr:
        if r.get("model_cif"):
            cm[(r["case"], r["mode"], int(r["seed"]))] = r["model_cif"]
    for r in rc:
        k = (r["case"], r["mode"], int(r["seed"]))
        kr = (r["case"], ref_mode, int(r["seed"]))
        if k in cm and kr in cm:
            sp = [s for (c, m, s) in cm if c == r["case"] and m == ref_mode]
            if len(sp) < 2:
                continue
            ref_ca = {s: score.load_ca(cm[(r["case"], ref_mode, s)]) for s in sp}
            pr = score.load_ca(cm[k], length=len(ref_ca[int(r["seed"])]))
            d = _ca_rmsd(pr, ref_ca[int(r["seed"])])
            spread = max(_ca_rmsd(ref_ca[a], ref_ca[b]) for a, b in itertools.combinations(sp, 2))
            if d > spread:
                flags.append(
                    {"case": r["case"], "seed": int(r["seed"]), "metric": "ca_coord_rmsd", "delta": d, "strict_spread": spread}
                )
    return {"hard_failures": hard, "flags": flags, "n_candidate_rows": len(rc), "pass": not hard}


def _ca_rmsd(a, b):
    ok = ~(np.isnan(a).any(1) | np.isnan(b).any(1))
    return score.rmsd_after_kabsch(a[ok], b[ok]) if ok.sum() >= 3 else float("nan")


def l3p_report(d, cfg) -> dict:
    """Outliers (|delta| > outlier_abs) and the large-chain subset, reported next to the gate."""
    out = {}
    lo = cfg["l3p"]["outlier_abs"]
    for name in ("lddt", "tm"):
        c, r = d[f"{name}_c"], d[f"{name}_r"]
        both = np.isfinite(c) & np.isfinite(r)
        delta = _nanmean(np.where(both, c, np.nan), -1) - _nanmean(np.where(both, r, np.nan), -1)
        out[name] = {
            "outliers": [
                {"case": cs, "delta": float(x)} for cs, x in zip(d["cases"], delta) if np.isfinite(x) and abs(x) > lo
            ],
            "max_abs_delta": float(np.nanmax(np.abs(delta))) if np.isfinite(delta).any() else float("nan"),
        }
        m = d["large"]
        if m.any():
            mean, l_, h_, n = tci(delta[m], cfg["ci_level"])
            out[name]["large_chains"] = {"n": int(n), "mean": float(mean), "ci": [float(l_), float(h_)]}
    return out


def l3l_report(data, rows, cand_mode, ref_mode, cfg, cand_steps=None, ref_steps=None) -> dict:
    """Success rate (< cutoff A) against a reference pose, per build, labelled crystal / modelled (not a gate)."""
    out = {}
    cutoff = cfg["l3l"]["success_cutoff"]
    for mode, steps in ((ref_mode, ref_steps), (cand_mode, cand_steps)):
        vals, kinds = [], set()
        for r in select(rows, mode, steps):
            v = r.get("ligand_rmsd_ref")
            if v is None and r.get("pose_cif") and r.get("ref_cif"):
                try:
                    v = score.ligand_pose_rmsd(
                        r["pose_cif"],
                        r["ref_cif"],
                        r.get("ligand_chain", "B"),
                        r["smiles"],
                        ref_ligand_chain=r.get("ref_ligand_chain"),
                        ref_ligand_file=r.get("ref_pose"),
                        radius=cfg["l3l"]["pocket_radius"],
                    )["ligand_rmsd"]
                except ValueError:
                    v = None
            if v is not None:
                vals.append(float(v))
                kinds.add(r.get("ref_pose_kind", "unlabelled"))
        if vals:
            out[mode] = {
                "n": len(vals),
                "success_rate": float(np.mean(np.array(vals) < cutoff)),
                "cutoff": cutoff,
                "reference_kind": sorted(kinds),
            }
    # O4: affinity crop residue-set differences
    crops = {}
    for mode, steps in ((ref_mode, ref_steps), (cand_mode, cand_steps)):
        for r in select(rows, mode, steps):
            if r.get("affinity_crop_hash"):
                crops.setdefault((r["case"], int(r["seed"])), {})[mode] = r["affinity_crop_hash"]
    both = [v for v in crops.values() if ref_mode in v and cand_mode in v]
    if both:
        out["O4_crop_differs"] = {"n": len(both), "fraction": float(np.mean([v[ref_mode] != v[cand_mode] for v in both]))}
    return out


def l3a_report(data, rows, cand_mode, ref_mode, cfg, n_boot, seed) -> dict:
    """Reported-only L3-A items: per-target Pearson and tau vs experiment, per-target tau(cand, strict), probability
    distribution (ceiling warning), excluded optional targets."""
    d = data["affinity"]
    out = {"excluded_optional_targets": data.get("affinity_excluded", [])}
    ca, ra = _nanmean(d["vc"], -1), _nanmean(d["vr"], -1)
    if np.isfinite(d["exp"]).all() and len(d["exp"]) >= 3:
        rs = score.ranking_statistics(
            d["exp"], {"strict": ra, "candidate": ca}, d["target"], ref_arm="strict", n_boot=n_boot, seed=seed
        )
        out["per_target"] = {
            arm: {"pearson": v["per_target_pearson"], "tau": v["per_target_tau"]} for arm, v in rs["arms"].items()
        }
        out["per_target"]["candidate"]["tau_vs_strict"] = rs["arms"]["candidate"]["per_target_tau_vs_ref"]
        out["weighted_pearson"] = {a: [v["weighted_pearson"], *v["weighted_pearson_ci"]] for a, v in rs["arms"].items()}
    pq = {}
    for name, a in (("strict", d["pr"]), ("candidate", d["pc"])):
        v = a[np.isfinite(a)]
        if len(v):
            pq[name] = {"quantiles_05_25_50_75_95": [float(x) for x in np.quantile(v, [0.05, 0.25, 0.5, 0.75, 0.95])]}
    out["probability_distribution"] = pq
    if pq.get("strict") and pq["strict"]["quantiles_05_25_50_75_95"][0] > 0.9:
        out["probability_note"] = (
            "affinity_probability_binary is near the ceiling (5th percentile > 0.9): the prob gates are uninformative"
        )
    return out


def a4_report(rows, mode, cfg, n_boot=10000, seed=0, steps=None) -> dict:
    """A4 (reported only, never a gate): FEP+ Pearson vs experiment from seed-averaged predictions of ``mode``.
    Primary = unweighted mean per-target Pearson, MW correction on (labelled "aggregation assumed" unless PREREG says
    otherwise); the others are secondary. BACE1 (config a4.exclude_targets) is excluded. CI = paired within-target
    bootstrap. MW columns: ``value_mw_corrected`` / ``affinity_pred_value_mw_on`` (on), ``affinity_pred_value_mw_off``
    or ``affinity_pred_value`` when ``mw_correction`` is false/absent (off)."""
    rs = select(rows, mode, steps)
    excl = {t.lower() for t in cfg["a4"]["exclude_targets"]}
    rs = [r for r in rs if str(r.get("target", "")).lower() not in excl and r.get("exp_value") is not None]
    cols = {}
    for on, keys in (("mw_on", ("affinity_pred_value_mw_on", "value_mw_corrected")), ("mw_off", ("affinity_pred_value_mw_off",))):
        k = next((k for k in keys if any(r.get(k) is not None for r in rs)), None)
        if k:
            cols[on] = k
    if "mw_off" not in cols and any(r.get("affinity_pred_value") is not None and not r.get("mw_correction") for r in rs):
        cols["mw_off"] = "affinity_pred_value"
    if "mw_on" not in cols and any(r.get("affinity_pred_value") is not None and r.get("mw_correction") for r in rs):
        cols["mw_on"] = "affinity_pred_value"
    msa = sorted({str(r.get("msa", r.get("msa_source", "unlabelled"))) for r in rs})
    res = {"mode": mode, "msa": msa, "columns": cols, "estimates": [], "n_ligands": 0}
    if not rs:
        return res
    for mw, key in cols.items():
        per = {}
        exp = {}
        tgt = {}
        for r in rs:
            if r.get(key) is not None:
                per.setdefault(r["case"], []).append(float(r[key]))
                exp[r["case"]] = float(r["exp_value"])
                tgt[r["case"]] = r["target"]
        cases = sorted(per)
        pred = np.array([np.mean(per[c]) for c in cases])
        e = np.array([exp[c] for c in cases])
        g = np.array([tgt[c] for c in cases])
        res["n_ligands"] = len(cases)
        st = score.ranking_statistics(e, {"x": pred}, g, n_boot=n_boot, seed=seed)["arms"]["x"]
        for agg, v, ci in (
            ("unweighted", st["unweighted_pearson"], st["unweighted_pearson_ci"]),
            ("weighted", st["weighted_pearson"], st["weighted_pearson_ci"]),
        ):
            is_primary = f"{agg}/{mw}" == cfg["a4"]["primary"]
            res["estimates"].append(
                {
                    "aggregation": agg,
                    "mw": mw,
                    "pearson": v,
                    "ci": list(ci),
                    "role": "PRIMARY (" + cfg["a4"]["primary_label"] + ")" if is_primary else "secondary",
                    "n": len(cases),
                    "msa": ",".join(msa),
                    "per_target": st["per_target_pearson"],
                }
            )
    res["published"] = {"pearson": cfg["a4"]["published_pearson"], "source": cfg["a4"]["published_source_url"]}
    if not any(e["role"].startswith("PRIMARY") for e in res["estimates"]):
        res["primary_note"] = f"primary statistic {cfg['a4']['primary']} not available in these rows (needs the MW-corrected column)"
    res["note"] = "n is about 87: the CI on r is about +-0.12; this comparison is coarse. Never a gate."
    if msa == ["empty"]:
        res["msa_note"] = "msa: empty - consistency with published results NOT tested (single-sequence)"
    return res


# --------------------------------------------------------------------------- run mode


def run_gates(rows, cand_mode, ref_mode, cfg, ctype="precision", cand_steps=None, ref_steps=None, n_boot=None, seed=0):
    data = build_dataset(rows, cand_mode, ref_mode, cfg, cand_steps, ref_steps)
    nb = n_boot or cfg["l3a"]["n_boot"]
    ev = evaluate_dataset(data, cfg, ctype, nb, seed)
    res = {
        "meta": {
            "candidate": cand_mode,
            "strict": ref_mode,
            "candidate_type": ctype,
            "cand_steps": cand_steps,
            "strict_steps": ref_steps,
            "n_boot": nb,
            "seed": seed,
            "config_version": cfg.get("version"),
        },
        "L2": l2_report(rows, cand_mode, ref_mode, cand_steps, ref_steps),
        "gates": ev["gates"],
        "stats": ev["stats"],
        "reports": {},
    }
    if "protein" in data:
        res["reports"]["L3P"] = l3p_report(data["protein"], cfg)
    res["reports"]["L3L"] = l3l_report(data, rows, cand_mode, ref_mode, cfg, cand_steps, ref_steps)
    if "affinity" in data:
        res["reports"]["L3A"] = l3a_report(data, rows, cand_mode, ref_mode, cfg, min(nb, 2000), seed)
    res["reports"]["A4"] = {m: a4_report(rows, m, cfg, min(nb, 2000), seed, s) for m, s in ((ref_mode, ref_steps), (cand_mode, cand_steps))}
    active = [g for g in GATE_SETS[ctype] if res["gates"][g]["status"] != "N/A"]
    missing = [g for g in GATE_SETS[ctype] if res["gates"][g]["status"] == "N/A"]
    st = [res["gates"][g]["status"] for g in active]
    res["overall"] = {
        "l2_no_hard_failure": res["L2"]["pass"],
        "gates_evaluated": active,
        "gates_missing_data": missing,
        "all_pass": bool(st) and all(s == "PASS" for s in st) and not missing and res["L2"]["pass"],
        "note": "L1 (GPU module checks) is not computed here; 'essentially lossless' needs L1 + L2 + L3-P/L/A all passing.",
    }
    return res


def format_gate_table(res) -> str:
    lines = []
    ctype = res["meta"]["candidate_type"]
    lines.append(
        f"candidate={res['meta']['candidate']} vs strict={res['meta']['strict']}  type={ctype}  seed={res['meta']['seed']} "
        f"n_boot={res['meta']['n_boot']}  config={res['meta']['config_version']}"
    )
    l2 = res["L2"]
    lines.append(
        f"L2  hard failures: {len(l2['hard_failures'])} -> {'PASS' if l2['pass'] else 'FAIL'};  flags (delta > strict spread): {len(l2['flags'])}"
    )
    for h in l2["hard_failures"][:10]:
        lines.append(f"      HARD {h}")
    for fl in l2["flags"][:10]:
        lines.append(f"      flag {fl}")
    w = max(len(g) for g in GATE_ORDER)
    lines.append(f"{'gate':<{w}}  {'verdict':<12} {'statistic':<68} threshold")
    for gid in GATE_ORDER:
        g = res["gates"][gid]
        if not g.get("active", True) and g["status"] == "N/A":
            continue
        tag = g["status"] if g.get("active", True) else g["status"] + "*"
        lines.append(f"{gid:<{w}}  {tag:<12} {g.get('statistic', g.get('reason', '')):<68} {g.get('threshold', '')}")
        if g.get("reason") and g["status"] != "N/A":
            lines.append(f"{'':<{w}}    ({g['reason']})")
    lines.append("(* = reported, not part of the gate set for this candidate type)")
    ov = res["overall"]
    lines.append(
        f"OVERALL (L2 + L3 gates in this file): {'PASS' if ov['all_pass'] else 'NOT PASSED'}"
        + (f"  missing data for: {ov['gates_missing_data']}" if ov["gates_missing_data"] else "")
    )
    lines.append(ov["note"])
    rep = res["reports"]
    if "L3P" in rep:
        for name, v in rep["L3P"].items():
            lines.append(f"L3-P {name}: max |delta| {v['max_abs_delta']:.4f}; outliers (|delta| > outlier_abs): {v['outliers']}")
            if "large_chains" in v:
                lines.append(f"      large chains: {v['large_chains']}")
    if rep.get("L3L"):
        lines.append(f"L3-L reported: {rep['L3L']}")
    if rep.get("L3A", {}).get("probability_note"):
        lines.append("L3-A: " + rep["L3A"]["probability_note"])
    for mode, a4 in rep.get("A4", {}).items():
        for e in a4.get("estimates", []):
            lines.append(
                f"A4[{mode}] {e['aggregation']}/{e['mw']}: r={e['pearson']:+.3f} CI [{e['ci'][0]:+.3f}, {e['ci'][1]:+.3f}] "
                f"n={e['n']} msa={e['msa']} {e['role']}"
            )
        if a4.get("estimates"):
            lines.append(f"      published {a4['published']}; {a4.get('note', '')} {a4.get('msa_note', '')}")
    return "\n".join(lines)


# =========================================================================== calibration
#
# Model: for each case / ligand the strict runs are N(mu, sigma^2) in the seed dimension, sigma pooled (or per case),
# value & prob (lDDT & TM) correlated by their pooled residual correlation. A "candidate" arm and a "reference" arm
# are simulated with ``kappa`` controlling how much of the seed noise they share: corr = 1 - kappa^2, so that
# RMS(cand_s - ref_s) = kappa * sqrt(2) * sigma.  kappa = 1: independent seeds (the SPEC calibration design: strict
# seeds {5-9} vs {0-4}, or {3-5} vs {0-2}). A2 only makes sense for noise-matched candidates, so its null uses
# kappa = calibration.a2_kappa_null; under independent seeds A2's ratio is ~1 by construction.


class Models:
    pass


def _resid_corr(Xa, Xb):
    ra = Xa - _nanmean(Xa, -1)[:, None]
    rb = Xb - _nanmean(Xb, -1)[:, None]
    ok = np.isfinite(ra) & np.isfinite(rb)
    den = np.sqrt((ra[ok] ** 2).sum() * (rb[ok] ** 2).sum())
    return float((ra[ok] * rb[ok]).sum() / den) if den > 0 else 0.0


def _sd_pair(X, sd_mode):
    mu = _nanmean(X, -1)
    n = np.isfinite(X).sum(1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sd_c = np.nanstd(X, 1, ddof=1)
    pooled = float(np.sqrt(np.nanmean(sd_c[n >= 2] ** 2)))
    if sd_mode == "auto":
        sd_mode = "per_case" if X.shape[1] >= 6 else "pooled"
    sd = np.where(np.isfinite(sd_c), sd_c, pooled) if sd_mode == "per_case" else np.full_like(mu, pooled)
    return mu, sd, pooled, sd_mode


def fit_models(rows, mode, steps, data_obs, pairs_p, pairs_l, src: PoseSource, cfg, sd_mode="auto"):
    """Fit the simulation models from ALL strict runs of ``mode`` (every seed, both arms)."""
    rs = select(rows, mode, steps)
    M = Models()
    M.protein = M.aff = M.pose = None
    if "protein" in data_obs:
        cases, tg, seeds, L = build_all_seed_matrix(rs, "ca_lddt")
        _, _, _, Tm = build_all_seed_matrix(rs, "tm", cases, seeds)
        mu_l, sd_l, pool_l, used = _sd_pair(L, sd_mode)
        mu_t, sd_t, pool_t, _ = _sd_pair(Tm, sd_mode)
        sel = [cases.index(c) for c in data_obs["protein"]["cases"]]
        M.protein = {
            "cases": data_obs["protein"]["cases"],
            "large": data_obs["protein"]["large"],
            "mu_l": mu_l[sel], "sd_l": sd_l[sel], "mu_t": mu_t[sel], "sd_t": sd_t[sel],
            "rho": _resid_corr(L, Tm), "sigma_lddt": pool_l, "sigma_tm": pool_t, "n_seeds": L.shape[1], "sd_mode": used,
        }  # fmt: skip
    if "affinity" in data_obs:
        d = data_obs["affinity"]
        cases, tg, seeds, V = build_all_seed_matrix(rs, "affinity_pred_value")
        _, _, _, Pm = build_all_seed_matrix(rs, "affinity_probability_binary", cases, seeds)
        sel = [cases.index(c) for c in d["cases"]]
        mu_v, sd_v, pool_v, used = _sd_pair(V[sel], sd_mode)
        mu_p, sd_p, pool_p, _ = _sd_pair(Pm[sel], sd_mode)
        M.aff = {
            "cases": d["cases"], "target": d["target"], "exp": d["exp"],
            "mu_v": mu_v, "sd_v": sd_v, "mu_p": mu_p, "sd_p": sd_p,
            "rho": _resid_corr(V[sel], Pm[sel]), "sigma_value": pool_v, "sigma_prob": pool_p,
            "n_seeds": V.shape[1], "sd_mode": used,
        }  # fmt: skip
    if "pose" in data_obs:
        d = data_obs["pose"]
        seeds = sorted({int(r["seed"]) for r in rs})
        allp = list(itertools.combinations(seeds, 2))
        pool = np.full((len(d["cases"]), max(len(allp), 1)), np.nan)
        flips, nposes = 0.0, 0
        for i, c in enumerate(d["cases"]):
            for k, (a, b) in enumerate(allp):
                pool[i, k] = src.rmsd(c, (mode, b, steps), (mode, a, steps))
            for s in seeds:
                fv = src.flips(c, mode, s, steps)
                if np.isfinite(fv):
                    flips += fv
                    nposes += 1
        M.pose = {"cases": d["cases"], "target": d["target"], "pool": pool, "flip_rate": flips / max(nposes, 1), "n_poses": nposes}
    return M


def _extend_idx(n_obs, groups, design_n, rng):
    """Indices (length design_n) drawn with replacement within target, proportional to the observed target sizes."""
    groups = np.asarray(groups)
    if design_n is None or design_n == n_obs:
        return np.arange(n_obs)
    names = sorted(set(groups.tolist()))
    sizes = np.array([(groups == g).sum() for g in names], float)
    alloc = np.maximum(1, np.round(sizes / sizes.sum() * design_n)).astype(int)
    while alloc.sum() > design_n:
        alloc[np.argmax(alloc)] -= 1
    while alloc.sum() < design_n:
        alloc[np.argmax(sizes / np.maximum(alloc, 1))] += 1
    idx = []
    for g, a in zip(names, alloc):
        members = np.where(groups == g)[0]
        idx += list(members) if a >= len(members) and False else list(rng.choice(members, size=a, replace=True))
    return np.array(idx)


def _corr_noise(z1, rho, rng):
    return rho * z1 + math.sqrt(max(1 - rho * rho, 0.0)) * rng.standard_normal(z1.shape)


def _arm_noise(R, shape, rho, kappa, matched, rng):
    """Standard-normal noise for (value-like, prob-like) outputs of a reference arm and a candidate arm.
    Within an arm the two outputs have correlation ``rho``; candidate and reference share a fraction of the noise
    (corr = 1 - kappa^2 per output) when ``matched`` (equal seed counts), else are independent."""
    shape_r, shape_c = shape
    a_r, w_r = rng.standard_normal((R,) + shape_r), rng.standard_normal((R,) + shape_r)
    if matched:
        rm = 1 - kappa * kappa
        sq = math.sqrt(max(1 - rm * rm, 0.0))
        a_c = rm * a_r + sq * rng.standard_normal((R,) + shape_c)
        w_c = rm * w_r + sq * rng.standard_normal((R,) + shape_c)
    else:
        a_c, w_c = rng.standard_normal((R,) + shape_c), rng.standard_normal((R,) + shape_c)
    s = math.sqrt(max(1 - rho * rho, 0.0))
    return a_r, rho * a_r + s * w_r, a_c, rho * a_c + s * w_c


def sim_protein(M, R, rng, nS_c, nS_r, sc):
    p = M.protein
    C = len(p["cases"])
    z1r, z2r, z1c, z2c = _arm_noise(R, ((C, nS_r), (C, nS_c)), p["rho"], sc.get("kappa", 1.0), nS_c == nS_r, rng)
    mu_l, sd_l, mu_t, sd_t = (p[k][None, :, None] for k in ("mu_l", "sd_l", "mu_t", "sd_t"))
    return {
        "lddt_r": np.clip(mu_l + sd_l * z1r, 0, 1),
        "lddt_c": np.clip(mu_l + sc.get("lddt_shift", 0.0) + sd_l * z1c, 0, 1),
        "tm_r": np.clip(mu_t + sd_t * z2r, 0, 1),
        "tm_c": np.clip(mu_t + sc.get("tm_shift", 0.0) + sd_t * z2c, 0, 1),
    }


def sim_affinity(M, R, rng, nS_c, nS_r, sc, lig_idx):
    a = M.aff
    L = len(lig_idx)
    mu_v, sd_v, mu_p, sd_p = (a[k][lig_idx][None, :, None] for k in ("mu_v", "sd_v", "mu_p", "sd_p"))
    z1r, z2r, z1c, z2c = _arm_noise(R, ((L, nS_r), (L, nS_c)), a["rho"], sc.get("kappa", 1.0), nS_c == nS_r, rng)
    extra = sc.get("extra_sd", 0.0) * rng.standard_normal((R, L, 1))  # persistent per-ligand candidate error
    return {
        "vr": mu_v + sd_v * z1r,
        "vc": mu_v + sd_v * z1c + sc.get("value_bias", 0.0) + extra,
        "pr": np.clip(mu_p + sd_p * z2r, 0, 1),
        "pc": np.clip(mu_p + sd_p * z2c + sc.get("prob_bias", 0.0), 0, 1),
    }


def sim_pose(M, R, rng, kc, ks, sc, lig_idx):
    p = M.pose
    pool = p["pool"][lig_idx]
    L, P = pool.shape
    fin = [np.where(np.isfinite(pool[i]))[0] for i in range(L)]
    cs = np.empty((R, L, kc))
    ss = np.empty((R, L, ks))
    for i in range(L):
        f = fin[i] if len(fin[i]) else np.array([0])
        cs[:, i] = pool[i][rng.choice(f, size=(R, kc))]
        ss[:, i] = pool[i][rng.choice(f, size=(R, ks))]
    cs = cs * sc.get("rmsd_scale", 1.0)
    lam = p["flip_rate"]
    chir_c = rng.poisson((lam + sc.get("chir_rate", 0.0)) * kc, size=(R, L))
    chir_s = rng.poisson(lam * kc, size=(R, L))  # strict arm matched seeds: same number of poses as the candidate
    return {"cs": cs, "ss": ss, "chir_c": chir_c.astype(float), "chir_s": chir_s.astype(float)}


def evaluate_batch(batch, gids, cfg, tau_null, rng, inner_boot):
    """Statistics of the requested gates for R replicates. ``batch`` keys: protein{...}, aff{..., exp, groups},
    pose{..., groups}. Returns {stat_name: (R,) array}."""
    level = cfg["ci_level"]
    st = {}
    groups = set(gids)
    if groups & {"L3P.lddt", "L3P.tm"} and "protein" in batch:
        b = batch["protein"]
        st.update(protein_stats(b["lddt_c"], b["lddt_r"], b["tm_c"], b["tm_r"], level))
    if groups & {"A1.value", "A1.prob", "A2.value", "A2.prob", "A2p.value"} and "aff" in batch:
        b = batch["aff"]
        st.update(affinity_ab_stats(b["vc"], b["vr"], b["pc"], b["pr"], level))
    if groups & {"A3.pearson", "A3.kendall"} and "aff3" in batch:
        b = batch["aff3"]
        R = b["vc"].shape[0]
        keys = ("a3_pear_lo", "a3_tau_lo", "a3_pear_delta", "a3_tau")
        acc = {k: np.empty(R) for k in keys}
        ca, ra = _nanmean(b["vc"], -1), _nanmean(b["vr"], -1)
        for r in range(R):
            exp = b["exp"] if b["exp"].ndim == 1 else b["exp"][r]
            g = b["groups"] if b["groups"].ndim == 1 else b["groups"][r]
            s = a3_stats(ca[r], ra[r], exp, g, inner_boot, rng, level)
            for k in keys:
                acc[k][r] = s[k]
        st.update(acc)
    if groups & {"L3L.median", "L3L.p90", "L3L.chirality"} and "pose" in batch:
        b = batch["pose"]
        R = b["cs"].shape[0]
        acc = {k: np.empty(R) for k in ("l3l_med_ub", "l3l_p90_ub", "l3l_med", "l3l_p90")}
        if groups & {"L3L.median", "L3L.p90"}:
            for r in range(R):
                s = pose_stats(b["cs"][r], b["ss"][r], b["groups"], inner_boot, rng)
                for k in acc:
                    acc[k][r] = s[k]
            st.update(acc)
        st["chir_excess"] = b["chir_c"].sum(-1) - b["chir_s"].sum(-1)
    return st


def batch_boot(data, R, rng, gids):
    """Bootstrap-the-cases replicates of the OBSERVED candidate/reference data (resampling within target; protein
    cases are resampled freely)."""
    batch = {}
    if "protein" in data:
        d = data["protein"]
        idx = rng.integers(0, len(d["cases"]), size=(R, len(d["cases"])))
        batch["protein"] = {k: d[k][idx] for k in ("lddt_c", "lddt_r", "tm_c", "tm_r")}
    if "affinity" in data:
        d = data["affinity"]
        boots = score.grouped_bootstrap_indices(d["target"], R, rng)
        idx = np.concatenate([bi for _, bi in boots], 1)
        order = np.concatenate([m for m, _ in boots])
        groups = np.asarray(d["target"])[order]
        ab = {k: d[k][idx] for k in ("vc", "vr", "pc", "pr")}
        batch["aff"] = ab
        batch["aff3"] = {**ab, "exp": d["exp"][idx], "groups": groups}
    if "pose" in data:
        d = data["pose"]
        boots = score.grouped_bootstrap_indices(d["target"], R, rng)
        idx = np.concatenate([bi for _, bi in boots], 1)
        order = np.concatenate([m for m, _ in boots])
        batch["pose"] = {
            "cs": d["cs"][idx], "ss": d["ss"][idx], "chir_c": d["chir_c"][idx], "chir_s": d["chir_s"][idx],
            "groups": np.asarray(d["target"])[order],
        }  # fmt: skip
    return batch


def batch_mc(M, data, R, rng, sc, design_n, nS, gids, lig_idx_cache):
    """Monte Carlo replicates from the fitted models under scenario ``sc``."""
    batch = {}
    need = {GATE_GROUP[g] for g in gids}
    if M.protein is not None and "protein" in need:
        batch["protein"] = sim_protein(M, R, rng, nS["protein_c"], nS["protein_r"], sc)
    if M.aff is not None and "affinity" in need:
        idx = lig_idx_cache("aff", M.aff["target"], design_n, rng)
        sim = sim_affinity(M, R, rng, nS["aff_c"], nS["aff_r"], sc, idx)
        batch["aff"] = sim
        batch["aff3"] = {**sim, "exp": M.aff["exp"][idx], "groups": np.asarray(M.aff["target"])[idx]}
    if M.pose is not None and "pose" in need:
        idx = lig_idx_cache("pose", M.pose["target"], design_n, rng)
        b = sim_pose(M, R, rng, nS["pose_c"], nS["pose_s"], sc, idx)
        b["groups"] = np.asarray(M.pose["target"])[idx]
        batch["pose"] = b
    return batch


def _interp_mde(effects, power, target):
    """Smallest effect (linear interpolation of the power curve) with power >= target; None if never reached."""
    for k, (e, p) in enumerate(zip(effects, power)):
        if p >= target:
            if k == 0:
                return float(e)
            e0, p0 = effects[k - 1], power[k - 1]
            return float(e0 + (e - e0) * (target - p0) / max(p - p0, 1e-12))
    return None


def solve_pearson_noise(M, drop, rng_seed, lig_idx, R0=200):
    """Per-ligand extra noise SD (persistent across seeds) so that the expected delta in weighted mean Pearson vs
    experiment of the seed-averaged candidate is ``drop`` (negative). Bisection with common random numbers."""
    a = M.aff
    groups = np.asarray(a["target"])[lig_idx]
    exp = a["exp"][lig_idx]
    mu = a["mu_v"][lig_idx]
    if not np.isfinite(exp).all():
        return float("nan")
    names = sorted(set(groups.tolist()))
    idxs = [np.where(groups == g)[0] for g in names]
    w = np.array([len(i) for i in idxs], float)
    z = np.random.default_rng(rng_seed).standard_normal((R0, len(lig_idx)))
    base = [score.pearson(mu[i], exp[i]) for i in idxs]
    base_w = score.wmean(base, w)

    def delta(s):
        pred = mu[None] + s * z
        r = np.stack([score._pearson_rows(pred[:, i], np.broadcast_to(exp[i], (R0, len(i)))) for i in idxs], 1)
        ok = np.isfinite(r)
        wr = np.where(ok, w[None], 0)
        return float(((np.where(ok, r, 0) * wr).sum(1) / np.maximum(wr.sum(1), 1e-300)).mean() - base_w)

    lo, hi = 0.0, 10.0 * float(np.nanstd(mu) + 1e-9)
    if delta(hi) > drop:
        return hi
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if delta(mid) > drop:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def prereg_a_view(cfg):
    """cfg with the gate thresholds replaced by the PREREG-A (SPEC) values; everything else kept."""
    out = copy.deepcopy(cfg)
    for sect, keys in (
        ("l3p", ("lddt_ci_lower", "lddt_mean", "tm_ci_lower")),
        ("l3l", ("median_ratio_max", "p90_ratio_max", "chirality_excess_max")),
        ("l3a", ("a1_value", "a1_prob", "a2_ratio_max", "a2p_ratio_max", "a3_pearson_delta_lower", "a3_tau_margin")),
    ):
        for k in keys:
            out[sect][k] = DEFAULTS[sect][k]
    return out


def calibrate(
    rows,
    cfg,
    mode="strict",
    steps=None,
    protein_pairs=None,
    ligand_pairs=None,
    mc_reps=1000,
    boot_reps=300,
    power_reps=400,
    inner_boot=300,
    seed=0,
    design_ligands=None,
    sd_mode="auto",
    pose_source=None,
    keep_raw=False,
    decision="min",
    log=None,
):
    """Null pass probability (bootstrap of the observed cand/ref seed subsets + Monte Carlo from pooled seed SDs),
    power against the defined bad candidates, minimum detectable effect, and the re-specification rule."""
    log = log or (lambda *_: None)
    cfg = prereg_a_view(cfg)  # margins are scaled from the PREREG-A values, whatever thresholds the config holds
    cal = cfg["calibration"]
    src = pose_source if pose_source is not None else PoseSource(rows, cfg)
    # --- observed data: protein rows use protein pairs, ligand rows (affinity + pose) use ligand pairs
    rows_p = [r for r in rows if r.get("ca_lddt") is not None]
    rows_l = [r for r in rows if r.get("ca_lddt") is None and r.get("type") != "ligand_rmsd_pair"]
    data = {}
    if rows_p:
        dp = build_dataset(rows_p, mode, mode, cfg, steps, steps, protein_pairs, PoseSource([], cfg))
        if "protein" in dp:
            data["protein"] = dp["protein"]
    if rows_l:
        dl = build_dataset(rows_l, mode, mode, cfg, steps, steps, ligand_pairs, src)
        for k in ("affinity", "pose"):
            if k in dl:
                data[k] = dl[k]
    if not data:
        raise ValueError("no usable rows for calibration")
    M = fit_models(rows, mode, steps, data, protein_pairs, ligand_pairs, src, cfg, sd_mode)
    nS = {
        "protein_c": data["protein"]["lddt_c"].shape[1] if "protein" in data else 0,
        "protein_r": data["protein"]["lddt_r"].shape[1] if "protein" in data else 0,
        "aff_c": data["affinity"]["vc"].shape[1] if "affinity" in data else 0,
        "aff_r": data["affinity"]["vr"].shape[1] if "affinity" in data else 0,
        "pose_c": data["pose"]["cs"].shape[1] if "pose" in data else 0,
        "pose_s": data["pose"]["ss"].shape[1] if "pose" in data else 0,
    }
    avail = [g for g in GATE_ORDER if (GATE_GROUP[g] == "protein" and "protein" in data)
             or (GATE_GROUP[g] == "pose" and "pose" in data) or (GATE_GROUP[g] == "affinity" and "affinity" in data)]  # fmt: skip
    if "affinity" in data and not np.isfinite(data["affinity"]["exp"]).all():
        avail = [g for g in avail if g not in ("A3.pearson", "A3.kendall")]
    if nS["aff_c"] != nS["aff_r"] or nS["aff_r"] < 2:
        avail = [g for g in avail if g not in ("A2.value", "A2.prob")]
    if "affinity" in data and nS["aff_r"] < 2:
        avail = [g for g in avail if g != "A2p.value"]

    cache = {}

    def lig_idx_cache(key, groups, design_n, rng):
        if key not in cache:
            cache[key] = _extend_idx(len(groups), groups, design_n, make_rng(seed, "design", key))
        return cache[key]

    # --- observed gate set (cand seeds vs ref seeds, all gates)
    observed = evaluate_dataset(data, cfg, "precision", cfg["l3a"]["n_boot"], seed)
    # --- null statistics: bootstrap (observed seed subsets, cases resampled) and Monte Carlo (fixed cases, new seeds)
    def collect(gen, R, label):
        acc = {}
        done = 0
        chunk = 100
        while done < R:
            r = min(chunk, R - done)
            rng = make_rng(seed, label, done)
            b = gen(r, rng)
            s = evaluate_batch(b, avail, cfg, None, rng, inner_boot)
            for k, v in s.items():
                acc.setdefault(k, []).append(np.asarray(v))
            done += r
        return {k: np.concatenate(v) for k, v in acc.items()}

    log("null: bootstrap")
    sb = collect(lambda r, rng: batch_boot(data, r, rng, avail), boot_reps, "boot")
    log("null: Monte Carlo")
    sm = collect(lambda r, rng: batch_mc(M, data, r, rng, {"kappa": 1.0}, design_ligands, nS, avail, lig_idx_cache), mc_reps, "mc")
    # tau_null: strict-vs-strict value of the pooled Kendall tau(cand, strict) (MC mean point estimate)
    tau_null = float(np.nanmean(sm["a3_tau"])) if "a3_tau" in sm else None
    tau_obs = observed["stats"].get("a3_tau")
    # A2 null: noise-matched model with kappa_null
    sa2 = None
    if "A2.value" in avail:
        log("null: A2 (matched kappa model)")
        sa2 = collect(
            lambda r, rng: batch_mc(M, data, r, rng, {"kappa": cal["a2_kappa_null"]}, design_ligands, nS, avail, lig_idx_cache),
            mc_reps, "mc_a2",
        )  # fmt: skip

    def passprob(gid, S, f, tn):
        if gid in ("A2.value", "A2.prob") and S is sm:
            S = sa2  # the independent-seed null is not a meaningful null for A2 (ratio ~ 1 by construction)
        if S is None:
            return None
        try:
            return float(np.mean(gate_pass(gid, S, cfg, f, tn)))
        except KeyError:
            return None

    gates = {}
    cap = cal["respec_cap"]
    for gid in avail:
        pb = passprob(gid, sb, 1.0, tau_null) if gid not in ("A2.value", "A2.prob") else None
        pm = passprob(gid, sm, 1.0, tau_null)
        vals = [v for v in ((pb if decision == "min" else None), pm) if v is not None]
        null = min(vals) if vals else None
        entry = {"threshold_prereg_a": gate_threshold_text(gid, DEFAULTS, 1.0, tau_null), "null_pass_boot": pb, "null_pass_mc": pm, "null_pass": null}
        # re-specification: smallest margin factor with null pass >= 0.9 (min over estimators), capped at 2x
        chosen, status = None, "untestable"
        for f in cal["respec_grid"]:
            if f > cap + 1e-9:
                break
            v = [
                x
                for x in (
                    passprob(gid, sb, f, tau_null) if (gid not in ("A2.value", "A2.prob") and decision == "min") else None,
                    passprob(gid, sm, f, tau_null),
                )
                if x is not None
            ]
            if v and min(v) >= cal["null_pass_min"]:
                chosen = f
                status = "kept" if f == 1.0 else "widened"
                break
        entry["respec"] = {"status": status, "factor": chosen}
        entry["final_factor"] = chosen if chosen is not None else cap
        entry["final_threshold"] = gate_threshold_text(gid, DEFAULTS, entry["final_factor"], tau_null)
        if gid == "L3L.chirality":  # the margin is a count: scaling it by f does nothing, so report the slack it would need
            for m_ in range(0, 200):
                if min(np.mean(sb["chir_excess"] <= m_), np.mean(sm["chir_excess"] <= m_)) >= cal["null_pass_min"]:
                    entry["info"] = {"excess_slack_needed_for_null_pass": m_, "note": "not applied by the re-specification rule"}
                    break
        gates[gid] = entry
    # --- conjunction (essentially-lossless gate set) at the final margins, same replicates
    conj = {}
    for ctype, gset in GATE_SETS.items():
        gs = [g for g in gset if g in gates]
        if not gs:
            continue
        out = {"gates": gs}
        for lab, S in (("boot", sb), ("mc", sm)):
            ok = None
            for g in gs:
                SS = sa2 if (g in ("A2.value", "A2.prob")) else S
                if lab == "boot" and g in ("A2.value", "A2.prob"):
                    continue
                p_ = gate_pass(g, SS, cfg, gates[g]["final_factor"], tau_null)
                ok = p_ if ok is None else (ok & p_)
            out[lab] = float(np.mean(ok)) if ok is not None else None
        # conjunction at the PREREG-A margins (f = 1)
        ok1 = None
        for g in gs:
            SS = sa2 if g in ("A2.value", "A2.prob") else sm
            p_ = gate_pass(g, SS, cfg, 1.0, tau_null)
            ok1 = p_ if ok1 is None else (ok1 & p_)
        out["mc_at_prereg_a_margins"] = float(np.mean(ok1)) if ok1 is not None else None
        vals = [v for v in ((out["boot"] if decision == "min" else None), out["mc"]) if v is not None]
        out["null_pass"] = min(vals) if vals else None
        out["ok"] = bool(vals) and min(vals) >= cal["conjunction_min"]
        conj[ctype] = out
    # --- power and MDE (Monte Carlo, final margins)
    bad = cal["bad"]
    lig_idx = lig_idx_cache("aff", M.aff["target"], design_ligands, None) if M.aff is not None else None
    scenario_effect = {
        "lddt_shift": ("ca_lddt_shift", "lddt_shift", lambda e: e, "CA-lDDT shift"),
        "tm_shift": ("tm_shift", "tm_shift", lambda e: e, "TM-score shift"),
        "value_bias": ("affinity_value_bias", "value_bias", lambda e: e, "affinity_pred_value bias (log10)"),
        "prob_bias": ("affinity_prob_bias", "prob_bias", lambda e: e, "affinity_probability_binary bias"),
        "rmsd_scale": ("rmsd_scale", "rmsd_scale", lambda e: e, "candidate-vs-strict RMSD scale"),
        "chir_rate": ("chirality_extra_flip_rate", "chir_rate", lambda e: e, "extra chirality flips per pose"),
        "pearson_drop": ("weighted_pearson_delta", "pearson_drop", lambda e: e, "delta weighted Pearson"),
    }
    mult = cal["effect_multiples"]
    cache_pow = {}
    for gid in avail:
        scen = GATE_SCENARIO[gid]
        bkey, _, _, label = scenario_effect[scen]
        bad_eff = bad[bkey]
        if scen == "rmsd_scale":
            grid = [1 + m * (bad_eff - 1) for m in mult]
            bad_val = bad_eff
        else:
            grid = [m * bad_eff for m in mult]
            bad_val = bad_eff
        if scen == "pearson_drop" and (M.aff is None or not np.isfinite(M.aff["exp"]).all()):
            continue
        pw = []
        for e in grid:
            key = (scen, round(e, 10), gid in ("A2.value", "A2.prob"))
            if key not in cache_pow:
                sc = {"kappa": cal["a2_kappa_null"] if gid in ("A2.value", "A2.prob") else 1.0}
                if scen == "lddt_shift": sc["lddt_shift"] = e
                elif scen == "tm_shift": sc["tm_shift"] = e
                elif scen == "value_bias": sc["value_bias"] = e
                elif scen == "prob_bias": sc["prob_bias"] = e
                elif scen == "rmsd_scale": sc["rmsd_scale"] = e
                elif scen == "chir_rate": sc["chir_rate"] = e
                elif scen == "pearson_drop":
                    sc["extra_sd"] = solve_pearson_noise(M, e, seed, lig_idx) if e != 0 else 0.0
                rng = make_rng(seed, "power", scen, round(e, 10))
                gset = [g for g in avail if GATE_SCENARIO[g] == scen and (g in ("A2.value", "A2.prob")) == (gid in ("A2.value", "A2.prob"))]
                b = batch_mc(M, data, power_reps, rng, sc, design_ligands, nS, gset, lig_idx_cache)
                cache_pow[key] = evaluate_batch(b, gset, cfg, tau_null, rng, inner_boot)
            S = cache_pow[key]
            pw.append(float(np.mean(~gate_pass(gid, S, cfg, gates[gid]["final_factor"], tau_null))))
        eff_axis = [abs(x) if scen != "rmsd_scale" else x for x in grid]
        mde = _interp_mde(eff_axis, pw, cal["power_min"])
        k_bad = mult.index(1.0) if 1.0 in mult else None
        gates[gid]["power"] = {
            "scenario": label,
            "bad_effect": bad_val,
            "power_at_bad": pw[k_bad] if k_bad is not None else None,
            "power_ok": (pw[k_bad] >= cal["power_min"]) if k_bad is not None else None,
            "effect_grid": grid,
            "power_curve": pw,
            "mde_80": mde,  # magnitude of the effect (RMSD scale: the scale factor itself)
        }
    res = {
        "meta": {
            "mode": mode, "steps": steps, "protein_pairs": protein_pairs, "ligand_pairs": ligand_pairs,
            "mc_reps": mc_reps, "boot_reps": boot_reps, "power_reps": power_reps, "inner_boot": inner_boot,
            "seed": seed, "design_ligands": design_ligands, "sd_mode": sd_mode, "n_seeds": nS,
            "config_version": cfg.get("version"), "decision_estimator": decision,
            "null_model": "independent seeds (kappa=1) for all gates except A2 (kappa=%s)" % cal["a2_kappa_null"],
        },  # fmt: skip
        "estimated": {
            "protein": None if M.protein is None else {k: M.protein[k] for k in ("sigma_lddt", "sigma_tm", "rho", "n_seeds", "sd_mode")} | {"n_cases": len(M.protein["cases"])},
            "affinity": None if M.aff is None else {k: M.aff[k] for k in ("sigma_value", "sigma_prob", "rho", "n_seeds", "sd_mode")} | {"n_ligands": len(M.aff["cases"])},
            "pose": None if M.pose is None else {"flip_rate_per_pose": M.pose["flip_rate"], "n_poses": M.pose["n_poses"], "n_ligands": len(M.pose["cases"])},
        },  # fmt: skip
        "observed_null": {g: {k: v for k, v in e.items() if k != "active"} for g, e in observed["gates"].items()},
        "tau_null": tau_null,
        "tau_null_observed": tau_obs,
        "gates": gates,
        "conjunction": conj,
    }
    res["recommended_config"] = recommended_config(res, cfg)
    if keep_raw:
        res["_raw"] = {"boot": sb, "mc": sm, "mc_a2": sa2}
    res["verdict"] = {
        "untestable_gates": [g for g, e in gates.items() if e["respec"]["status"] == "untestable"],
        "underpowered_gates": [g for g, e in gates.items() if e.get("power", {}).get("power_ok") is False],
        "conjunction_ok": {k: v["ok"] for k, v in conj.items()},
    }
    return res


def recommended_config(res, cfg) -> dict:
    """Threshold overrides implied by the re-specification rule (PREREG-B candidates); untestable gates keep the
    capped value and are listed under ``untestable``."""
    out = {"version": "PREREG-B (proposed)", "l3p": {}, "l3l": {}, "l3a": {}}
    D = DEFAULTS
    for gid, e in res["gates"].items():
        f = e["final_factor"]
        if gid == "L3P.lddt":
            out["l3p"]["lddt_ci_lower"] = round(D["l3p"]["lddt_ci_lower"] * f, 6)
            out["l3p"]["lddt_mean"] = round(D["l3p"]["lddt_mean"] * f, 6)
        elif gid == "L3P.tm":
            out["l3p"]["tm_ci_lower"] = round(D["l3p"]["tm_ci_lower"] * f, 6)
        elif gid == "L3L.median":
            out["l3l"]["median_ratio_max"] = round(1 + (D["l3l"]["median_ratio_max"] - 1) * f, 6)
        elif gid == "L3L.p90":
            out["l3l"]["p90_ratio_max"] = round(1 + (D["l3l"]["p90_ratio_max"] - 1) * f, 6)
        elif gid == "L3L.chirality":
            out["l3l"]["chirality_excess_max"] = D["l3l"]["chirality_excess_max"] * f
        elif gid == "A1.value":
            out["l3a"]["a1_value"] = round(D["l3a"]["a1_value"] * f, 6)
        elif gid == "A1.prob":
            out["l3a"]["a1_prob"] = round(D["l3a"]["a1_prob"] * f, 6)
        elif gid in ("A2.value", "A2.prob"):
            out["l3a"]["a2_ratio_max"] = round(max(out["l3a"].get("a2_ratio_max", 0), D["l3a"]["a2_ratio_max"] * f), 6)
        elif gid == "A2p.value":
            out["l3a"]["a2p_ratio_max"] = round(1 + (D["l3a"]["a2p_ratio_max"] - 1) * f, 6)
        elif gid == "A3.pearson":
            out["l3a"]["a3_pearson_delta_lower"] = round(D["l3a"]["a3_pearson_delta_lower"] * f, 6)
        elif gid == "A3.kendall":
            out["l3a"]["a3_tau_margin"] = round(D["l3a"]["a3_tau_margin"] * f, 6)
    if res.get("tau_null") is not None:
        out["l3a"]["tau_null"] = round(res["tau_null"], 6)
    out["untestable"] = [g for g, e in res["gates"].items() if e["respec"]["status"] == "untestable"]
    return out


def format_calibration(res) -> str:
    L = []
    m = res["meta"]
    L.append(f"CALIBRATION  mode={m['mode']} seeds: protein pairs {m['protein_pairs']} ligand pairs {m['ligand_pairs']}")
    L.append(
        f"  MC reps {m['mc_reps']}, bootstrap reps {m['boot_reps']}, power reps {m['power_reps']}, inner boot {m['inner_boot']}, "
        f"seed {m['seed']}, design ligands {m['design_ligands']}, sd_mode {m['sd_mode']}"
    )
    L.append(f"  null model: {m['null_model']}")
    for k, v in res["estimated"].items():
        if v:
            L.append(f"  estimated {k}: " + ", ".join(f"{a}={b:.4g}" if isinstance(b, float) else f"{a}={b}" for a, b in v.items()))
    L.append(f"  tau_null (strict-vs-strict pooled Kendall tau, MC mean) = {res['tau_null']}  (observed {res['tau_null_observed']})")
    hdr = f"{'gate':<14}{'null(boot)':>11}{'null(MC)':>10}{'respec':>10}{'factor':>8}  {'final threshold':<52}{'power@bad':>10}{'MDE80':>10}  bad candidate"
    L.append(hdr)
    fmt = lambda x, n=3: "   -" if x is None else f"{x:.{n}f}"  # noqa: E731
    for gid in GATE_ORDER:
        e = res["gates"].get(gid)
        if not e:
            continue
        pw = e.get("power", {})
        mde = pw.get("mde_80")
        L.append(
            f"{gid:<14}{fmt(e['null_pass_boot']):>11}{fmt(e['null_pass_mc']):>10}{e['respec']['status']:>10}"
            f"{fmt(e['respec']['factor'], 2):>8}  {e['final_threshold']:<52}{fmt(pw.get('power_at_bad')):>10}"
            f"{('none' if mde is None and pw else fmt(mde, 4)):>10}  {pw.get('scenario', '')} {pw.get('bad_effect', '')}"
        )
    for ctype, c in res["conjunction"].items():
        L.append(
            f"conjunction[{ctype}] ({len(c['gates'])} gates) null pass: boot {fmt(c['boot'])} MC {fmt(c['mc'])} "
            f"(MC at PREREG-A margins {fmt(c['mc_at_prereg_a_margins'])}) -> {'OK' if c['ok'] else 'BELOW 0.8'}"
        )
    v = res["verdict"]
    L.append(f"untestable: {v['untestable_gates']}   underpowered at the defined bad candidate: {v['underpowered_gates']}")
    return "\n".join(L)


# --------------------------------------------------------------------------- CLI


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rows", nargs="+", type=Path, help="jsonl row files (see GATES.md)")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", "--out", dest="out", type=Path, default=None, help="write machine-readable JSON here")
    ap.add_argument("--candidate", default=None)
    ap.add_argument("--strict", default="strict")
    ap.add_argument("--candidate-type", choices=["precision", "sampler"], default="precision")
    ap.add_argument("--cand-steps", type=int, default=None)
    ap.add_argument("--strict-steps", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=None, help="bootstrap resamples (default: config, 10000)")
    ap.add_argument("--a4", action="store_true", help="print only the A4 report")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--mode", default="strict", help="(calibrate) the strict mode name")
    ap.add_argument("--steps", type=int, default=None, help="(calibrate) steps filter")
    ap.add_argument("--protein-cand-seeds", default="5-9")
    ap.add_argument("--protein-ref-seeds", default="0-4")
    ap.add_argument("--ligand-cand-seeds", default="3-5")
    ap.add_argument("--ligand-ref-seeds", default="0-2")
    ap.add_argument("--design-ligands", type=int, default=None, help="(calibrate) simulate this many ligands (e.g. 87)")
    ap.add_argument("--mc-reps", type=int, default=1000)
    ap.add_argument("--boot-reps", type=int, default=300)
    ap.add_argument("--power-reps", type=int, default=400)
    ap.add_argument("--inner-boot", type=int, default=300)
    ap.add_argument("--sd-mode", choices=["auto", "pooled", "per_case"], default="auto")
    ap.add_argument("--decision", choices=["min", "mc"], default="min",
                    help="(calibrate) null pass used for the 0.9 rule: min(bootstrap, Monte Carlo) or Monte Carlo only")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    rows = read_rows(a.rows)
    if a.calibrate:
        pp = list(zip(parse_seeds(a.protein_cand_seeds), parse_seeds(a.protein_ref_seeds)))
        lp = list(zip(parse_seeds(a.ligand_cand_seeds), parse_seeds(a.ligand_ref_seeds)))
        res = calibrate(
            rows, cfg, a.mode, a.steps, pp, lp, a.mc_reps, a.boot_reps, a.power_reps, a.inner_boot, a.seed,
            a.design_ligands, a.sd_mode, decision=a.decision, log=lambda m: print(m, file=sys.stderr),
        )  # fmt: skip
        print(format_calibration(res))
        if a.out:
            a.out.write_text(json.dumps(res, indent=1, default=_json_default))
        return 0
    if a.a4:
        mode = a.candidate or a.strict
        res = a4_report(rows, mode, cfg, a.n_boot or 10000, a.seed, a.strict_steps)
        print(json.dumps(res, indent=1, default=_json_default))
        return 0
    if not a.candidate:
        ap.error("--candidate MODE is required (or use --calibrate / --a4)")
    res = run_gates(rows, a.candidate, a.strict, cfg, a.candidate_type, a.cand_steps, a.strict_steps, a.n_boot, a.seed)
    print(format_gate_table(res))
    if a.out:
        a.out.write_text(json.dumps(res, indent=1, default=_json_default))
    return 0 if res["overall"]["all_pass"] else 1


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return [None if (isinstance(x, float) and not math.isfinite(x)) else x for x in o.tolist()]
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (set, tuple)):
        return list(o)
    raise TypeError(type(o))


if __name__ == "__main__":
    sys.exit(main())
