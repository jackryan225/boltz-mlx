"""Synthetic jsonl rows for the gates tests (no structures needed: ligand RMSDs come as precomputed pair rows)."""

import itertools
import zlib

import numpy as np

TARGETS = {"tyk2": 16, "cdk2": 16, "p38": 34}


def protein_rows(mode, seeds, n_cases=36, sigma=0.01, shift=0.0, seed=0, noise_seed=None, tm_sigma=0.008):
    """CA-lDDT / TM rows; case means are fixed by ``seed``, per-run noise by ``noise_seed``."""
    rng = np.random.default_rng(seed)
    mu_l = rng.uniform(0.6, 0.9, n_cases)
    mu_t = np.clip(mu_l + rng.normal(0, 0.03, n_cases), 0.3, 0.95)
    nres = rng.integers(100, 500, n_cases)
    rows = []
    for s in seeds:
        r = np.random.default_rng([seed if noise_seed is None else noise_seed, 1000 + s, zlib.crc32(mode.encode()) % 1000])
        for c in range(n_cases):
            z = r.standard_normal(2)
            rows.append(
                {
                    "case": f"p{c:02d}", "target": f"p{c:02d}", "seed": s, "mode": mode, "steps": 200, "n_res": int(nres[c]),
                    "ca_lddt": float(np.clip(mu_l[c] + shift + sigma * z[0], 0, 1)),
                    "tm": float(np.clip(mu_t[c] + tm_sigma * (0.8 * z[0] + 0.6 * z[1]), 0, 1)),
                }
            )  # fmt: skip
    return rows


def ligand_rows(mode, seeds, sigma=0.15, prob_sigma=0.01, bias=0.0, corr_noise=0.0, seed=1, noise_seed=None, matched_with=None, kappa=1.0):
    """Affinity rows. ``exp_value`` ~ 0.6 * latent + noise (so Pearson ~ 0.5)."""
    rng = np.random.default_rng(seed)
    rows = []
    lat = {}
    for t, n in TARGETS.items():
        for i in range(n):
            e = rng.normal(0, 1)
            lat[(t, i)] = (e, 0.7 * e + 0.7 * rng.normal(0, 1), rng.uniform(0.85, 0.99))
    ns = seed if noise_seed is None else noise_seed
    for s in seeds:
        r = np.random.default_rng([ns, 2000 + s, zlib.crc32(mode.encode()) % 1000])
        for (t, i), (e, mu, p) in lat.items():
            z = r.standard_normal(2)
            rows.append(
                {
                    "case": f"{t}_{i:02d}", "target": t, "seed": s, "mode": mode, "steps": 200,
                    "exp_value": float(e), "affinity_pred_value": float(mu + bias + sigma * z[0]),
                    "affinity_probability_binary": float(np.clip(p + prob_sigma * z[1], 0, 1)),
                    "msa": "empty",
                }
            )  # fmt: skip
    return rows


def pose_pair_rows(cand_mode, ref_mode, seeds_pairs, scale=1.0, seed=3, flips_c=0, flips_s=0):
    """Pose pair rows: per ligand a lognormal RMSD scale; RMSD(cand, strict) = scale * strict-pair distribution."""
    rng = np.random.default_rng(seed)
    rows = []
    for t, n in TARGETS.items():
        for i in range(n):
            case = f"{t}_{i:02d}"
            base = rng.uniform(1.0, 4.0)
            for sc_, sr_ in seeds_pairs:
                rows.append({"type": "ligand_rmsd_pair", "case": case, "a_mode": cand_mode, "a_seed": sc_, "b_mode": ref_mode,
                             "b_seed": sr_, "rmsd": float(base * scale * rng.lognormal(0, 0.3))})  # fmt: skip
            sr_seeds = sorted({b for _, b in seeds_pairs})
            for a, b in itertools.combinations(sr_seeds, 2):
                rows.append({"type": "ligand_rmsd_pair", "case": case, "a_mode": ref_mode, "a_seed": b, "b_mode": ref_mode,
                             "b_seed": a, "rmsd": float(base * rng.lognormal(0, 0.3))})  # fmt: skip
    return rows


def flip_rows(mode, seeds, rate, seed=5):
    """One pose row per (ligand, seed) carrying only ``chirality_flips``."""
    rng = np.random.default_rng([seed, zlib.crc32(mode.encode()) % 1000])
    rows = []
    for t, n in TARGETS.items():
        for i in range(n):
            for s in seeds:
                rows.append({"case": f"{t}_{i:02d}", "target": t, "seed": s, "mode": mode, "steps": 200,
                             "chirality_flips": int(rng.random() < rate)})  # fmt: skip
    return rows


def derive_candidate(rows, mode, value_bias=0.0, value_sd=0.0, prob_bias=0.0, prob_sd=0.0, lddt_shift=0.0, lddt_sd=0.0, seed=11):
    """A noise-matched candidate: every strict row copied with small independent perturbations (a precision change)."""
    rng = np.random.default_rng([seed, zlib.crc32(mode.encode())])
    out = []
    for r in rows:
        r = dict(r)
        r["mode"] = mode
        if "affinity_pred_value" in r:
            r["affinity_pred_value"] += value_bias + value_sd * rng.standard_normal()
            r["affinity_probability_binary"] = float(
                np.clip(r["affinity_probability_binary"] + prob_bias + prob_sd * rng.standard_normal(), 0, 1)
            )
        if "ca_lddt" in r:
            r["ca_lddt"] = float(np.clip(r["ca_lddt"] + lddt_shift + lddt_sd * rng.standard_normal(), 0, 1))
        out.append(r)
    return out


def strict_pose_rows(mode, seeds, seed=3, sigma_lig=0.3):
    """All pairwise RMSD rows among ``seeds`` of one mode (calibration pools)."""
    rng = np.random.default_rng(seed)
    rows = []
    for t, n in TARGETS.items():
        for i in range(n):
            case = f"{t}_{i:02d}"
            base = rng.uniform(1.0, 4.0)
            for a, b in itertools.combinations(sorted(seeds), 2):
                rows.append(
                    {
                        "type": "ligand_rmsd_pair", "case": case, "a_mode": mode, "a_seed": b, "b_mode": mode,
                        "b_seed": a, "rmsd": float(base * rng.lognormal(0, sigma_lig)),
                    }
                )  # fmt: skip
    return rows
