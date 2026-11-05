"""Unit tests for tools/validation/gates.py on synthetic data (gates, calibration, reporting helpers, CLI)."""

import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "validation"))
sys.path.insert(0, str(ROOT / "tests"))
import gates  # noqa: E402
import gates_synth as gs  # noqa: E402

CFG = copy.deepcopy(gates.DEFAULTS)  # PREREG-A values, independent of whatever validation/gates_config.json holds later


def run(rows, cand="cand", ctype="precision", nboot=600, seed=0, cfg=CFG):
    return gates.run_gates(rows, cand, "strict", cfg, ctype, n_boot=nboot, seed=seed)


def st(res, gid):
    return res["gates"][gid]["status"]


# ------------------------------------------------------------------ configuration


def test_defaults_are_the_spec_values_and_config_file_matches():
    assert CFG["l3p"]["lddt_ci_lower"] == -0.010 and CFG["l3p"]["lddt_mean"] == -0.005 and CFG["l3p"]["tm_ci_lower"] == -0.015
    assert CFG["l3l"]["median_ratio_max"] == 1.25 and CFG["l3l"]["p90_ratio_max"] == 1.5
    a = CFG["l3a"]
    assert (a["a1_value"], a["a1_prob"], a["a2_ratio_max"], a["a2p_ratio_max"]) == (0.05, 0.02, 0.5, 1.25)
    assert (a["a3_pearson_delta_lower"], a["a3_tau_margin"], a["n_boot"]) == (-0.05, 0.05, 10000)
    on_disk = gates.load_config(ROOT / "validation" / "gates_config.json")
    for sect in ("l3p", "l3l", "l3a"):
        assert on_disk[sect] == CFG[sect], sect  # PREREG-A config file == code defaults


def test_threshold_change_through_config_only(tmp_path):
    strict = gs.protein_rows("strict", range(5))
    cand = gs.derive_candidate(strict, "cand", lddt_shift=-0.004, lddt_sd=0.002)
    rows = strict + cand
    assert st(run(rows), "L3P.lddt") == "PASS"
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps({"l3p": {"lddt_mean": -0.001, "lddt_ci_lower": -0.002}}))
    tight = gates.load_config(p)
    assert tight["l3p"]["tm_ci_lower"] == -0.015  # untouched keys keep PREREG-A defaults
    assert st(run(rows, cfg=tight), "L3P.lddt") == "FAIL"


# ------------------------------------------------------------------ L3-P


def test_l3p_null_pass_probability_matches_analytic_and_bias_fails():
    # null candidate with independent seeds: P(pass) for the lDDT gate is analytic (t-interval over n cases)
    sigma, n, S, reps = 0.05, 36, 5, 4000
    rng = np.random.default_rng(0)
    c = rng.normal(0, sigma, (reps, n, S))
    r = rng.normal(0, sigma, (reps, n, S))
    nan = np.full_like(c, np.nan)
    stat = gates.protein_stats(0.7 + c, 0.7 + r, nan, nan)
    p_sim = float(np.mean(gates.gate_pass("L3P.lddt", stat, CFG)))
    sd = sigma * math.sqrt(2.0 / S)
    q = sps.t.ppf(0.975, n - 1)
    p_exact = float(sps.norm.cdf(  # P(mean >= -0.01 + q*sd_hat/sqrt(n)) with sd_hat ~ sd (normal approximation of the mean)
        (0.01 - q * sd / math.sqrt(n)) / (sd / math.sqrt(n))
    ))
    assert p_sim == pytest.approx(p_exact, abs=0.04)
    assert 0.3 < p_sim < 0.6  # an informative (non-saturated) check
    # a candidate with CA-lDDT -0.02 fails with probability ~1 at the realistic noise level
    sigma = 0.01
    c = rng.normal(0, sigma, (reps, n, S)) - 0.02
    r = rng.normal(0, sigma, (reps, n, S))
    assert np.mean(gates.gate_pass("L3P.lddt", gates.protein_stats(0.7 + c, 0.7 + r, nan, nan), CFG)) < 0.001


def test_l3p_full_path_null_and_biased_candidate_and_reports():
    strict = gs.protein_rows("strict", range(5))
    ok = strict + gs.derive_candidate(strict, "cand", lddt_sd=0.006)
    r = run(ok)
    assert st(r, "L3P.lddt") == "PASS" and st(r, "L3P.tm") in ("PASS", "N/A")
    bad = strict + gs.derive_candidate(strict, "cand", lddt_shift=-0.02, lddt_sd=0.004)
    assert st(run(bad), "L3P.lddt") == "FAIL"
    # outliers (|delta| > 0.05) and the large-chain subset are reported
    spike = [dict(x) for x in gs.derive_candidate(strict, "cand", lddt_sd=0.002)]
    for x in spike:
        if x["case"] == "p03":
            x["ca_lddt"] -= 0.08
    rep = run(strict + spike)["reports"]["L3P"]["lddt"]
    assert [o["case"] for o in rep["outliers"]] == ["p03"]
    assert rep["large_chains"]["n"] == int(sum(1 for c in {x["case"]: x["n_res"] for x in strict}.values() if c >= 380))


# ------------------------------------------------------------------ L3-A


def affinity_rows(value_sd=0.01, bias=0.0, prob_bias=0.0, sigma=0.15, extra_rank_noise=0.0, seeds=range(3)):
    strict = gs.ligand_rows("strict", seeds, sigma=sigma)
    cand = gs.derive_candidate(strict, "cand", value_bias=bias, value_sd=value_sd, prob_bias=prob_bias, prob_sd=0.002)
    if extra_rank_noise:
        rng = np.random.default_rng(4)
        persistent = {}
        for r in cand:
            persistent.setdefault(r["case"], rng.normal(0, extra_rank_noise))
            r["affinity_pred_value"] += persistent[r["case"]]
    return strict + cand


def test_a1_a2_a3_pass_for_a_precision_like_candidate_and_fail_for_bias():
    r = run(affinity_rows())
    for g in ("A1.value", "A1.prob", "A2.value", "A2.prob", "A3.pearson"):
        assert st(r, g) == "PASS", (g, r["gates"][g])
    assert st(r, "A3.kendall") == "UNDETERMINED"  # tau_null not set in PREREG-A
    cfg = copy.deepcopy(CFG)
    cfg["l3a"]["tau_null"] = 0.95
    assert st(run(affinity_rows(), cfg=cfg), "A3.kendall") == "PASS"
    cfg["l3a"]["tau_null"] = 1.2  # an unattainable null reference makes the same data fail
    assert st(run(affinity_rows(), cfg=cfg), "A3.kendall") == "FAIL"
    b = run(affinity_rows(bias=0.10, sigma=0.10))
    assert st(b, "A1.value") == "FAIL" and st(b, "A2.value") == "FAIL"
    assert st(b, "A1.prob") == "PASS"
    bp = run(affinity_rows(prob_bias=0.05))
    assert st(bp, "A1.prob") == "FAIL" and st(bp, "A1.value") == "PASS"


def test_a2_ratio_is_rms_matched_over_rms_strict_pairs():
    rows = affinity_rows(value_sd=0.03)
    stats = run(rows)["stats"]
    strict = [x for x in rows if x["mode"] == "strict"]
    cand = [x for x in rows if x["mode"] == "cand"]
    key = lambda x: (x["case"], x["seed"])  # noqa: E731
    cd = {key(x): x["affinity_pred_value"] for x in cand}
    sd = {key(x): x["affinity_pred_value"] for x in strict}
    rms_m = math.sqrt(np.mean([(cd[k] - sd[k]) ** 2 for k in sd]))
    cases = sorted({k[0] for k in sd})
    diffs = [(sd[(c, i)] - sd[(c, j)]) ** 2 for c in cases for i in range(3) for j in range(i + 1, 3)]
    assert stats["a2v_ratio"] == pytest.approx(rms_m / math.sqrt(np.mean(diffs)), rel=1e-9)
    assert stats["a2v_ratio"] < 0.5  # 0.03 vs sqrt(2)*0.15


def test_a2prime_sampler_independent_seeds_pass_and_bias_fails():
    strict = gs.ligand_rows("strict", range(3), sigma=0.15)
    indep = gs.ligand_rows("cand", range(3), sigma=0.15, noise_seed=99)  # same ligands, independent noise
    r = run(strict + indep, ctype="sampler")
    assert st(r, "A2p.value") == "PASS" and 0.7 < r["stats"]["a2pv_ratio"] < 1.25
    assert r["gates"]["A2.value"]["active"] is False  # A2 is reported, not gated, for samplers
    biased = gs.ligand_rows("cand", range(3), sigma=0.15, noise_seed=99, bias=0.15)
    assert st(run(strict + biased, ctype="sampler"), "A2p.value") == "FAIL"


def test_a3_ranking_degradation_detected_and_seedable():
    bad = run(affinity_rows(extra_rank_noise=0.6))
    assert st(bad, "A3.pearson") == "FAIL" and bad["stats"]["a3_pear_hi"] < 0
    a = run(affinity_rows(extra_rank_noise=0.6), seed=3)["stats"]["a3_pear_lo"]
    b = run(affinity_rows(extra_rank_noise=0.6), seed=3)["stats"]["a3_pear_lo"]
    c = run(affinity_rows(extra_rank_noise=0.6), seed=4)["stats"]["a3_pear_lo"]
    assert a == b and a != c


def test_optional_target_pooled_only_if_enough_ligands():
    rows = affinity_rows()
    extra = []
    for x in rows:
        if x["target"] == "tyk2" and int(x["case"][-2:]) < 12:  # only 12 chymase complexes: below the 15 needed to be pooled
            y = dict(x)
            y["target"] = "chymase"
            y["case"] = x["case"].replace("tyk2", "chy")
            extra.append(y)
    d = gates.build_dataset(rows + extra, "cand", "strict", CFG)
    assert len(d["affinity_excluded"]) == 12 and "chymase" not in set(d["affinity"]["target"])
    cfg = copy.deepcopy(CFG)
    cfg["l3a"]["optional_targets"] = {"chymase": 10}
    assert "chymase" in set(gates.build_dataset(rows + extra, "cand", "strict", cfg)["affinity"]["target"])


# ------------------------------------------------------------------ L3-L


def pose_rows(scale, flips_c=0.03, flips_s=0.03):
    pairs = [(0, 0), (1, 1), (2, 2)]
    return (
        gs.pose_pair_rows("cand", "strict", pairs, scale=scale)
        + gs.flip_rows("strict", range(3), flips_s)
        + gs.flip_rows("cand", range(3), flips_c)
    )


def test_l3l_ratio_gates_and_chirality():
    ok = run(pose_rows(1.0))
    assert st(ok, "L3L.median") == "PASS" and st(ok, "L3L.p90") == "PASS"
    assert 0.85 < ok["stats"]["l3l_med"] < 1.15
    bad = run(pose_rows(1.6))
    assert st(bad, "L3L.median") == "FAIL" and bad["stats"]["l3l_med"] == pytest.approx(1.6, rel=0.15)
    # chirality: candidate flips must not exceed strict's (same seeds)
    assert st(run(pose_rows(1.0, flips_c=0.0, flips_s=0.05)), "L3L.chirality") == "PASS"
    r = run(pose_rows(1.0, flips_c=0.3, flips_s=0.0))
    assert st(r, "L3L.chirality") == "FAIL" and r["stats"]["chir_excess"] > 0


def test_pose_source_computes_from_cifs_and_chirality_from_geometry(tmp_path):
    from test_score_ligand_stats import IBU, _receptor, _write_complex, embed
    from scipy.spatial.transform import Rotation

    sym, lig = embed(IBU)
    lig = lig - lig.mean(0)
    rows = []
    for ci in range(3):
        rec = _receptor(ci)
        for mode in ("strict", "cand"):
            for seed in range(2):
                rng = np.random.default_rng([ci, seed, len(mode)])
                x = lig + rng.normal(0, 0.15, lig.shape)
                if mode == "cand" and ci == 2 and seed == 1:
                    x = x * np.array([1, 1, -1])  # a mirror-image pose: one chirality flip
                f = tmp_path / f"{ci}_{mode}_{seed}.cif"
                _write_complex(f, rec, sym, x)
                rows.append({"case": f"c{ci}", "target": "t", "seed": seed, "mode": mode, "steps": 200, "pose_cif": str(f),
                             "ligand_chain": "B", "smiles": IBU})  # fmt: skip
    r = run(rows, nboot=200)
    assert r["stats"]["chir_cand"] == 1 and r["stats"]["chir_strict"] == 0
    assert st(r, "L3L.chirality") == "FAIL"
    assert np.isfinite(r["stats"]["l3l_med"])


# ------------------------------------------------------------------ calibration


def strict_all(sigma_l=0.02, n_cases=36, sigma_v=0.12, seeds_p=range(10), seeds_l=range(6)):
    return (
        gs.protein_rows("strict", seeds_p, n_cases=n_cases, sigma=sigma_l, tm_sigma=0.8 * sigma_l)
        + gs.ligand_rows("strict", seeds_l, sigma=sigma_v)
        + gs.strict_pose_rows("strict", seeds_l)
        + gs.flip_rows("strict", seeds_l, 0.0)
    )


PP = [(s + 5, s) for s in range(5)]
LP = [(3, 0), (4, 1), (5, 2)]


def cal(rows, **kw):
    kw = {"mc_reps": 120, "boot_reps": 60, "power_reps": 60, "inner_boot": 80, **kw}
    return gates.calibrate(rows, CFG, "strict", None, PP, LP, seed=1, **kw)


def test_calibration_recovers_known_seed_sds():
    res = cal(strict_all(sigma_l=0.02, sigma_v=0.12), mc_reps=20, boot_reps=10, power_reps=10, inner_boot=20)
    e = res["estimated"]
    assert e["protein"]["sigma_lddt"] == pytest.approx(0.02, rel=0.05)
    assert e["protein"]["sigma_tm"] == pytest.approx(0.016, rel=0.06)
    assert e["affinity"]["sigma_value"] == pytest.approx(0.12, rel=0.06)
    assert e["affinity"]["sigma_prob"] == pytest.approx(0.01, rel=0.08)
    assert e["protein"]["rho"] == pytest.approx(0.8 * 0.8 / math.sqrt(0.8**2 * 0.64 + 0.36 * 0.64), abs=0.08)  # cov/(sd sd)
    assert e["protein"]["n_seeds"] == 10 and e["affinity"]["n_seeds"] == 6 and e["affinity"]["n_ligands"] == 66


def test_calibration_null_pass_power_and_mde_for_a_well_behaved_gate():
    res = cal(strict_all(sigma_l=0.01, sigma_v=0.05), design_ligands=87)
    g = res["gates"]
    for gid in ("L3P.lddt", "L3P.tm", "A1.value", "A1.prob", "A2.value", "A2.prob", "A2p.value", "A3.pearson"):
        assert g[gid]["null_pass"] >= 0.9, (gid, g[gid]["null_pass_boot"], g[gid]["null_pass_mc"])
        assert g[gid]["respec"]["status"] == "kept"
    # defined bad candidates are detected with power >= 0.8: CA-lDDT -0.02, +0.10 log10 bias, weighted Pearson -0.10
    assert g["L3P.lddt"]["power"]["power_at_bad"] >= 0.8
    assert g["A1.value"]["power"]["power_at_bad"] >= 0.8
    assert g["A3.pearson"]["power"]["power_at_bad"] >= 0.8
    # MDE of the lDDT gate: CI half-width at sigma_delta ~ 0.01*sqrt(2/5) over 36 cases is ~0.0013, so ~0.0087
    assert 0.007 <= g["L3P.lddt"]["power"]["mde_80"] <= 0.0095
    # MDE of A1.value at sigma 0.05 (n=87): ~0.05 - 2*0.05*sqrt(2/3)/sqrt(87) ~ 0.041
    assert 0.03 <= g["A1.value"]["power"]["mde_80"] <= 0.05
    assert res["tau_null"] > 0.9
    assert res["recommended_config"]["l3a"]["tau_null"] == pytest.approx(res["tau_null"], abs=1e-5)


def test_respecification_rule_widens_within_cap_or_declares_untestable():
    # noisy enough that lDDT null pass < 0.9 at PREREG-A margins but recoverable within 2x
    mid = cal(strict_all(sigma_l=0.035), mc_reps=300, boot_reps=100)["gates"]["L3P.lddt"]
    assert mid["null_pass_mc"] < 0.9
    assert mid["respec"]["status"] == "widened" and 1.0 < mid["respec"]["factor"] <= 2.0
    # the margin chosen is the smallest grid value with null pass >= 0.9
    # (checked on the stored statistics, not just the reported number)
    r = cal(strict_all(sigma_l=0.035), mc_reps=300, boot_reps=100, keep_raw=True)
    f = r["gates"]["L3P.lddt"]["respec"]["factor"]
    grid = CFG["calibration"]["respec_grid"]
    below = [x for x in grid if x < f]
    assert min(np.mean(gates.gate_pass("L3P.lddt", r["_raw"]["mc"], CFG, f)), np.mean(gates.gate_pass("L3P.lddt", r["_raw"]["boot"], CFG, f))) >= 0.9
    if below:
        x = below[-1]
        assert min(np.mean(gates.gate_pass("L3P.lddt", r["_raw"]["mc"], CFG, x)), np.mean(gates.gate_pass("L3P.lddt", r["_raw"]["boot"], CFG, x))) < 0.9
    # far too noisy: untestable, the recommended threshold stays at the 2x cap and the gate is listed
    bad = cal(strict_all(sigma_l=0.15), mc_reps=100, boot_reps=40)
    assert bad["gates"]["L3P.lddt"]["respec"]["status"] == "untestable"
    assert "L3P.lddt" in bad["verdict"]["untestable_gates"] and "L3P.lddt" in bad["recommended_config"]["untestable"]
    assert bad["recommended_config"]["l3p"]["lddt_ci_lower"] == pytest.approx(-0.020)  # capped at 2x


def test_chirality_count_gate_cannot_reach_null_pass_with_nonzero_flip_rate():
    rows = strict_all()
    rows = [r for r in rows if "chirality_flips" not in r] + gs.flip_rows("strict", range(6), 0.05)
    res = cal(rows)
    ch = res["gates"]["L3L.chirality"]
    assert ch["null_pass"] < 0.9 and ch["respec"]["status"] == "untestable"
    assert ch["info"]["excess_slack_needed_for_null_pass"] >= 1


def test_calibration_is_deterministic_for_a_fixed_seed():
    rows = strict_all()
    a = json.dumps(cal(rows), sort_keys=True, default=gates._json_default)
    b = json.dumps(cal(rows), sort_keys=True, default=gates._json_default)
    assert a == b
    c = json.dumps(gates.calibrate(rows, CFG, "strict", None, PP, LP, seed=2, mc_reps=120, boot_reps=60, power_reps=60, inner_boot=80),
                   sort_keys=True, default=gates._json_default)  # fmt: skip
    assert a != c


def test_run_gates_is_deterministic_and_calibration_uses_prereg_a_margins_whatever_the_config():
    rows = affinity_rows(value_sd=0.02)
    a = json.dumps(run(rows, seed=5)["gates"], sort_keys=True)
    assert a == json.dumps(run(rows, seed=5)["gates"], sort_keys=True)
    loose = copy.deepcopy(CFG)
    loose["l3p"]["lddt_ci_lower"] = -0.5
    r = gates.calibrate(strict_all(sigma_l=0.15), loose, "strict", None, PP, LP, mc_reps=60, boot_reps=30, power_reps=30, inner_boot=40, seed=1)
    assert r["gates"]["L3P.lddt"]["respec"]["status"] == "untestable"  # the loose config did not leak in


# ------------------------------------------------------------------ reporting helpers


def test_l2_hard_failures_and_flags():
    strict = gs.protein_rows("strict", range(3), n_cases=4)
    cand = gs.derive_candidate(strict, "cand", lddt_sd=0.0)
    cand[0] = dict(cand[0], nan=True)
    cand[1] = dict(cand[1], guard_counts={"tri_mul": 2, "attn": 0})
    out = gates.l2_report(strict + cand, "cand", "strict")
    reasons = sorted(h["reason"] for h in out["hard_failures"])
    assert reasons == ["guard_count=2", "nan/inf"] and not out["pass"]
    # a delta larger than the strict seed-to-seed spread is a flag, not a failure
    big = gs.derive_candidate(strict, "cand", lddt_shift=0.3)
    out = gates.l2_report(strict + big, "cand", "strict")
    assert out["pass"] and any(f["metric"] == "ca_lddt" for f in out["flags"])
    same = gates.l2_report(strict + gs.derive_candidate(strict, "cand"), "cand", "strict")
    assert same["flags"] == [] and same["pass"]


def test_a4_primary_secondary_and_bace1_excluded():
    rows = []
    rng = np.random.default_rng(0)
    for t, n in {"cdk2": 16, "tyk2": 16, "jnk1": 21, "p38": 34, "bace1": 36}.items():
        for i in range(n):
            e = rng.normal()
            rows.append({"case": f"{t}_{i}", "target": t, "seed": 0, "mode": "strict", "exp_value": e,
                         "value_mw_corrected": e + rng.normal(0, 0.8), "affinity_pred_value": e + rng.normal(0, 1.5),
                         "mw_correction": False, "msa": "empty"})  # fmt: skip
    out = gates.a4_report(rows, "strict", CFG, n_boot=500)
    assert out["n_ligands"] == 87  # BACE1 excluded
    by = {(e["aggregation"], e["mw"]): e for e in out["estimates"]}
    assert set(by) == {("unweighted", "mw_on"), ("weighted", "mw_on"), ("unweighted", "mw_off"), ("weighted", "mw_off")}
    assert by[("unweighted", "mw_on")]["role"].startswith("PRIMARY") and "aggregation assumed" in by[("unweighted", "mw_on")]["role"]
    assert by[("weighted", "mw_on")]["role"] == "secondary" and by[("unweighted", "mw_off")]["role"] == "secondary"
    assert by[("unweighted", "mw_on")]["pearson"] > by[("unweighted", "mw_off")]["pearson"]
    lo, hi = by[("unweighted", "mw_on")]["ci"]
    assert lo < by[("unweighted", "mw_on")]["pearson"] < hi
    assert "NOT tested" in out["msa_note"] and out["published"]["pearson"] == 0.66


# ------------------------------------------------------------------ CLI


def test_cli_gates_and_calibrate(tmp_path, capsys):
    rows = affinity_rows(value_sd=0.01) + gs.protein_rows("strict", range(5)) + gs.derive_candidate(gs.protein_rows("strict", range(5)), "cand", lddt_sd=0.004)
    f = tmp_path / "rows.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    out = tmp_path / "res.json"
    rc = gates.main([str(f), "--candidate", "cand", "--n-boot", "300", "--json", str(out), "--config", str(ROOT / "validation" / "gates_config.json")])
    txt = capsys.readouterr().out
    assert "L3P.lddt" in txt and "A3.kendall" in txt and "UNDETERMINED" in txt and rc == 1  # tau_null unset -> not all pass
    res = json.loads(out.read_text())
    assert res["gates"]["L3P.lddt"]["status"] == "PASS" and res["meta"]["candidate"] == "cand"
    # calibration CLI
    g = tmp_path / "strict.jsonl"
    g.write_text("\n".join(json.dumps(r) for r in strict_all(n_cases=12)) + "\n")
    cj = tmp_path / "cal.json"
    rc = gates.main([str(g), "--calibrate", "--protein-cand-seeds", "5-9", "--protein-ref-seeds", "0-4", "--ligand-cand-seeds", "3-5",
                     "--ligand-ref-seeds", "0-2", "--mc-reps", "40", "--boot-reps", "20", "--power-reps", "20", "--inner-boot", "40",
                     "--out", str(cj)])  # fmt: skip
    assert rc == 0
    txt = capsys.readouterr().out
    assert "CALIBRATION" in txt and "conjunction[precision]" in txt
    j = json.loads(cj.read_text())
    assert set(j) >= {"gates", "conjunction", "recommended_config", "estimated", "verdict", "tau_null"}
