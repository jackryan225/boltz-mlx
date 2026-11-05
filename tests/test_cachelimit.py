"""MLX buffer-cache limit policy (boltz_mlx/cachelimit.py) and its call sites. CPU only, no model, no GPU.

The limit only caps how many freed buffers MLX keeps for reuse; these tests pin the policy's outputs, the override rules, the
"small inputs make no mx call at all" property, and that every site that used to set the 4 GB limit goes through the module.
"""

import importlib.util
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from boltz_mlx import cachelimit as C

ROOT = Path(__file__).resolve().parents[1]
_REAL_TOTAL_MEMORY_GB = C.total_memory_gb
GB128 = 128.0   # the M5 Max in the task's units
GB16 = 16.0     # the 16 GB M1 Air
SIZES = (117, 1000, 1600, 2400, 3200)


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    """Policy state reset, mx.set_cache_limit replaced by a recorder (no global MLX state is touched)."""
    monkeypatch.setitem(C._STATE, "armed", False)
    monkeypatch.setitem(C._STATE, "current_gb", None)
    monkeypatch.setitem(C._STATE, "total_gb", None)
    monkeypatch.delenv(C.ENV, raising=False)
    calls = []
    monkeypatch.setattr(C, "_set_limit", lambda gb: calls.append(gb))
    monkeypatch.setattr(C, "total_memory_gb", lambda: GB128)
    return calls


# ------------------------------------------------------------------ the policy function
@pytest.mark.parametrize("n,expect", [(117, 4.0), (1000, 4.0), (1600, 32.0), (2400, 32.0), (3200, 32.0)])
def test_policy_128gb(n, expect):
    # 128 GB: margin max(16, 19.2) = 19.2; 1600 -> 128-19.2-19.2 = 89.6 and 2400 -> 65.6 are capped at 32; 3200 -> 128-76.8-19.2 = 32.0 exactly
    assert C.cache_limit_gb(n, GB128) == pytest.approx(expect)


@pytest.mark.parametrize("n", SIZES)
def test_policy_16gb_never_above_4(n):
    assert C.cache_limit_gb(n, GB16) == 4.0
    assert C.cache_limit_gb(n, 17.18) == 4.0  # a 16 GiB Mac reports 17.18e9 bytes


def test_policy_unrounded_values():
    assert C.estimate_peak_gb(1600) == pytest.approx(19.2)
    assert C.estimate_peak_gb(2400) == pytest.approx(43.2)
    assert C.estimate_peak_gb(3200) == pytest.approx(76.8)
    # between the floor and the cap the formula is RAM - peak - margin
    assert C.cache_limit_gb(1600, 64.0) == pytest.approx(64 - 19.2 - 16)   # 28.8
    assert C.cache_limit_gb(2400, 64.0) == pytest.approx(64 - 43.2 - 16)   # 4.8
    assert C.cache_limit_gb(3200, 64.0) == 4.0                               # negative -> floor
    assert C.cache_limit_gb(2400, 192.0) == 32.0
    # a 128 GiB Mac reports 137.44e9 bytes: margin 20.6, still 32 at 3200
    assert C.cache_limit_gb(3200, 137.44) == 32.0


def test_policy_small_inputs_are_4gb_on_any_machine():
    for total in (8.0, 16.0, 24.0, 128.0, 512.0, 1e6):
        for n in (1, 117, 400, 999, 1000):
            assert C.cache_limit_gb(n, total) == 4.0


def test_policy_bounds_and_monotone_in_ram():
    prev = {}
    for total in (8, 16, 18, 24, 32, 36, 48, 64, 96, 128, 192, 256):
        for n in range(1001, 4001, 150):
            g = C.cache_limit_gb(n, float(total))
            assert 4.0 <= g <= 32.0
            assert g >= prev.get(n, 0.0)  # more RAM never lowers the limit
            prev[n] = g
    for total in (8, 12, 16, 18, 20):  # margin >= 16 GB: machines up to 20 GB always get 4 GB
        assert all(C.cache_limit_gb(n, float(total)) == 4.0 for n in range(1001, 4001, 100))


def test_policy_explicit_peak_estimate():
    assert C.cache_limit_gb(1600, GB128, est_peak_gb=100.0) == pytest.approx(128 - 100 - 19.2)
    assert C.cache_limit_gb(1600, GB128, est_peak_gb=0.0) == 32.0


def test_peak_fit_matches_the_bench_r2_sweep():
    measured = {1000: 8.6, 1600: 19.9, 2400: 43.1}
    for n, gb in measured.items():
        assert abs(C.estimate_peak_gb(n) / gb - 1) < 0.15, n
    assert 67 <= C.estimate_peak_gb(3200) <= 77  # the 3200 point is an ESTIMATE of 67-77 GB


# ------------------------------------------------------------------ override and state machine
def test_env_override_parsing():
    assert C.env_override_gb({}) is None
    assert C.env_override_gb({C.ENV: ""}) is None
    assert C.env_override_gb({C.ENV: "6"}) == 6.0
    assert C.env_override_gb({C.ENV: "0"}) == 0.0
    assert C.env_override_gb({C.ENV: "4"}) == 4.0
    with pytest.raises(ValueError):
        C.env_override_gb({C.ENV: "lots"})


def test_unset_env_is_old_behaviour_for_small_inputs(fresh_state, capsys):
    assert C.install_startup() == 4.0 and fresh_state == [4.0]
    assert C.adapt(117) is None and C.adapt(1000) is None
    assert fresh_state == [4.0]  # no mx call at all after start-up
    assert capsys.readouterr().err == ""


def test_policy_applies_after_n_is_known_and_logs_once(fresh_state, capsys):
    C.install_startup()
    assert C.adapt(1600) == 32.0 and fresh_state == [4.0, 32.0]
    err = capsys.readouterr().err
    assert err.count("MLX buffer cache limit 32.0 GB") == 1 and "1600 tokens" in err
    assert C.adapt(1600) is None and C.adapt(2400) is None   # same value: no call, no log
    assert fresh_state == [4.0, 32.0] and capsys.readouterr().err == ""
    assert C.adapt(117) == 4.0 and fresh_state == [4.0, 32.0, 4.0]   # a later small record is back at today's value
    assert "limit 4.0 GB" in capsys.readouterr().err
    assert C.current_gb() == 4.0


def test_env_var_wins(fresh_state, monkeypatch, capsys):
    monkeypatch.setenv(C.ENV, "6")
    assert C.install_startup() == 6.0 and fresh_state == [6.0]
    assert C.adapt(3200) is None and C.adapt(1600) is None
    assert fresh_state == [6.0] and capsys.readouterr().err == ""
    # an explicit 4 is an override too (it used to mean "the default": it now pins the old behaviour)
    fresh_state.clear()
    monkeypatch.setenv(C.ENV, "4")
    C.install_startup()
    assert C.adapt(3200) is None and fresh_state == [4.0]
    # BOLTZ_MLX_CACHE_GB=0 (cache off) is respected as well
    fresh_state.clear()
    monkeypatch.setenv(C.ENV, "0")
    C.install_startup()
    assert C.adapt(3200) is None and fresh_state == [0.0]


def test_explicit_tool_limit_wins(fresh_state):
    C.set_explicit(8.0)   # bench_mlx --cache-limit-gb 8
    assert C.adapt(3200) is None and fresh_state == [8.0] and C.current_gb() == 8.0


def test_disarmed_by_default_and_feats_are_not_touched(fresh_state):
    class Boom(dict):
        def __getitem__(self, k):
            raise AssertionError("feats read although the policy is not armed")

    assert C.adapt(3200) is None and C.adapt_feats(Boom()) is None and fresh_state == []


def test_16gb_machine_never_exceeds_4gb(fresh_state, monkeypatch):
    monkeypatch.setattr(C, "total_memory_gb", lambda: 17.18)
    C.install_startup()
    for n in SIZES + (4000, 6000):
        assert C.adapt(n) is None
    assert fresh_state == [4.0]


def test_unknown_ram_keeps_4gb(fresh_state, monkeypatch):
    monkeypatch.setattr(C, "total_memory_gb", lambda: None)
    C.install_startup()
    assert C.adapt(3200) is None and fresh_state == [4.0]


def test_128gb_all_sizes_through_adapt(fresh_state):
    C.install_startup()
    got = {n: C.adapt(n, total_mem_gb=GB128) for n in SIZES}
    assert got == {117: None, 1000: None, 1600: 32.0, 2400: None, 3200: None}   # None = already 32
    assert fresh_state == [4.0, 32.0]


def test_adapt_feats_reads_the_token_dimension(fresh_state):
    C.install_startup()
    feats = {"token_pad_mask": np.zeros((1, 1600), np.float32)}
    assert C.n_tokens_of(feats) == 1600 and C.adapt_feats(feats) == 32.0


def test_total_memory_is_positive_gb():
    gb = _REAL_TOTAL_MEMORY_GB()  # the real function (the fixture patched the module attribute)
    assert gb is None or 4.0 < gb < 5000.0


# ------------------------------------------------------------------ call sites
def test_every_cache_limit_site_in_the_package_goes_through_the_module():
    pkg = ROOT / "boltz_mlx"
    offenders = []
    for f in pkg.rglob("*.py"):
        if f.name == "cachelimit.py":
            continue
        if re.search(r"mx\.set_cache_limit|set_cache_limit\(", f.read_text()):
            offenders.append(str(f.relative_to(ROOT)))
    assert offenders == []


def test_cli_wiring():
    import inspect

    from boltz_mlx import cli

    src = inspect.getsource(cli.predict)
    assert src.count("cachelimit.install_startup()") == 1
    assert src.count("cachelimit.adapt_feats(feats)") == 2   # structure and affinity loops
    # N is known (feats built) before the call, and the call is before the model is run
    for m in re.finditer(r"cachelimit\.adapt_feats\(feats\)", src):
        before, after = src[:m.start()], src[m.end():]
        assert before.rstrip().splitlines()[-1].strip().startswith(("_warn_unsupported_guidance(feats)", "feats = _batch_to_mx(batch)"))
        assert "model(feats" in after


def test_screen_wiring(fresh_state, monkeypatch):
    from boltz_mlx.screen import engine as E
    from boltz_mlx.screen import main as M

    import mlx.core as mx

    monkeypatch.setattr(mx, "set_default_device", lambda d: None)
    M.init_mlx(SimpleNamespace(device="cpu"))
    assert fresh_state == [4.0] and C._STATE["armed"]
    import boltz_mlx.cli as cli

    monkeypatch.setattr(cli, "_batch_to_mx", lambda b: b)
    monkeypatch.setattr(cli, "_warn_unsupported_guidance", lambda f: None)
    small = {"token_pad_mask": np.zeros((1, 117), np.float32)}
    big = {"token_pad_mask": np.zeros((1, 1600), np.float32)}
    assert E.UpstreamEngine.model_inputs(None, small) is small and fresh_state == [4.0]
    assert E.UpstreamEngine.model_inputs(None, big) is big and fresh_state == [4.0, 32.0]


def test_screen_env_override(fresh_state, monkeypatch):
    from boltz_mlx.screen import main as M

    import mlx.core as mx

    monkeypatch.setattr(mx, "set_default_device", lambda d: None)
    monkeypatch.setenv(C.ENV, "5")
    M.init_mlx(SimpleNamespace(device="cpu"))
    assert fresh_state == [5.0] and not C._STATE["armed"]


def test_bench_mlx_flag_is_number_or_policy():
    src = (ROOT / "tools" / "bench_mlx.py").read_text()
    assert "set_explicit(a.cache_limit_gb)" in src and 'install_startup(environ={})' in src and "cachelimit.adapt_feats(feats)" in src


# ------------------------------------------------------------------ the A/B tooling (pure functions)
def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def test_ab_tool_digest_is_bytewise():
    ab = _load("tools/prof/cachelimit_ab.py", "cachelimit_ab")
    import mlx.core as mx

    x = mx.array(np.arange(12, dtype=np.float32).reshape(3, 4))
    y = mx.array(np.arange(12, dtype=np.float32).reshape(3, 4))
    assert ab.digest(x) == ab.digest(y)
    z = np.arange(12, dtype=np.float32).reshape(3, 4)
    z.view(np.uint32)[1, 1] ^= 1   # flip the last mantissa bit of one element
    assert ab.digest(mx.array(z)) != ab.digest(x)
    assert ab.digest(x.astype(mx.float16)) != ab.digest(x)   # dtype is part of the digest
    assert ab.digest(x.reshape(4, 3)) != ab.digest(x)        # so is the shape
    assert set(ab.digests({"a": x, "n": 3, "d": {}})) == {"a"}


def _row(label, runs, git="abc1234", tokens=1600):
    return {"git": git, "label": label, "tokens": tokens, "steps": 50, "runs": runs}


def _run(arm, s, digest="d1", limit=None, cache_after=None, sys_share=0.4):
    return {"arm": arm, "predict_s": s, "trunk_s": s * 0.6, "diffusion_s": s * 0.35, "sys_share": sys_share, "digest": digest,
            "digests": {"sample_atom_coords": digest, "z": digest}, "limit_gb": limit or (4.0 if arm == "A" else 32.0),
            "cache_after_gb": cache_after if cache_after is not None else (3.9 if arm == "A" else 25.0), "guard_counts": None}


def _rows_pass(b_digest_b="d1"):
    rows = {
        "cl-b-A": _row("cl-b-A", [_run("A", 100)]),
        "cl-b-B": _row("cl-b-B", [_run("B", 90, digest=b_digest_b)]),
    }
    for n, (a, b) in {"1600": (100, 85), "2400": (400, 340)}.items():
        order = "ABBAABBA" if n == "1600" else "ABBA"
        rows[f"cl-c-{n}"] = _row(f"cl-c-{n}", [_run(ch, (a if ch == "A" else b) + 0.1 * i) for i, ch in enumerate(order)], tokens=int(n))
    return rows


def test_decision_rule_ship():
    S = _load("tools/prof/cachelimit_jobs/summary.py", "cachelimit_summary")
    rows = _rows_pass()
    b, c = S.analyse_b(rows), S.analyse_c(rows)
    assert b["verdict"] == "IDENTICAL"
    assert c["1600"]["verdict"] == "FASTER" and c["1600"]["runs_A"] == 4 and c["1600"]["runs_B"] == 4
    assert c["2400"]["runs_A"] == 2 and c["2400"]["effect_pct"] < 0
    assert S.decide(b, c) == "SHIP"


def test_decision_rule_blocks():
    S = _load("tools/prof/cachelimit_jobs/summary.py", "cachelimit_summary")
    rows = _rows_pass(b_digest_b="d2")   # one differing digest
    assert S.analyse_b(rows)["verdict"] == "DIFFERENT"
    assert S.decide(S.analyse_b(rows), S.analyse_c(rows)).startswith("NO-SHIP (bit-identity")
    rows = _rows_pass()
    rows["cl-c-2400"]["runs"][1]["digest"] = "other"   # a digest differs inside job c
    assert S.decide(S.analyse_b(rows), S.analyse_c(rows)).startswith("NO-SHIP (bit-identity")
    rows = _rows_pass()   # B slower at 2400
    for r in rows["cl-c-2400"]["runs"]:
        if r["arm"] == "B":
            r["predict_s"] = 450
    assert S.decide(S.analyse_b(rows), S.analyse_c(rows)) == "NO-SHIP (B not faster at every size)"
    rows = _rows_pass()   # B's cache not larger: the bitwise test is vacuous
    rows["cl-b-B"]["runs"][0].update(limit_gb=4.0, cache_after_gb=3.9)
    assert S.analyse_b(rows)["verdict"] == "VACUOUS" and S.decide(S.analyse_b(rows), S.analyse_c(rows)) == "INCONCLUSIVE"
    rows = _rows_pass()   # missing / too few
    del rows["cl-c-2400"]
    assert S.decide(S.analyse_b(rows), S.analyse_c(rows)) == "INCONCLUSIVE"
    rows = _rows_pass()
    rows["cl-c-1600"]["runs"] = rows["cl-c-1600"]["runs"][:3]
    assert S.analyse_c(rows)["1600"]["verdict"] == "TOO_FEW"


def test_summary_uses_last_row_of_a_label_for_this_git(tmp_path):
    S = _load("tools/prof/cachelimit_jobs/summary.py", "cachelimit_summary")
    p = tmp_path / "rows.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in [
        {"git": "aaaaaaa", "label": "cl-b-A", "runs": [1]}, {"git": "bbbbbbb", "label": "cl-b-A", "runs": [2]},
        {"git": "bbbbbbb-dirty", "label": "cl-b-A", "runs": [3]}, {"git": "bbbbbbb", "label": "", "runs": [9]}]) + "\nnot json\n")
    assert S.load_rows(p, "bbbbbbb")["cl-b-A"]["runs"] == [3]
    assert S.load_rows(p, "aaaaaaa")["cl-b-A"]["runs"] == [1]
