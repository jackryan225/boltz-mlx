"""The agent-friendly command line: --json shape, exit codes, --preset fast|full (+ the old names), `boltz-mlx check`.

CPU only. No prediction ever runs: `cli.predict` is replaced by a fake that writes Boltz-layout output files, and the screen
runs the real ScreenRunner on tests/screen_fakes.FakeEngine (no checkpoints, no GPU, no network).
"""

import json
import os
import sys
from pathlib import Path

import pytest

from boltz_mlx import check as CK
from boltz_mlx import cli
from boltz_mlx import cliout as CO
from boltz_mlx.screen import main as SM
from boltz_mlx.screen.presets import PRESETS
from screen_fakes import FakeEngine

CELECOXIB = "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1"   # 26 heavy atoms
PROT = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVV"
YAML = ("version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: " + PROT + "\n      msa: empty\n"
        "  - ligand:\n      id: B\n      smiles: '" + CELECOXIB + "'\nproperties:\n  - affinity:\n      binder: B\n")
YAML_NO_MSA = YAML.replace("      msa: empty\n", "")


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    from boltz_mlx import kernels as K

    monkeypatch.setattr(K, "_CAP", K.pinned_status("nax", "test pin"))  # default preset by kernel path: no GPU probe
    monkeypatch.setenv("BOLTZ_MLX_PRECISION", "f16")                      # fast's precision is already set: no re-exec
    monkeypatch.delenv("BOLTZ_MLX_TRACEBACK", raising=False)
    monkeypatch.setattr(CK, "preflight", lambda *a, **k: None)           # weights / mols are not needed by the fakes


def _ready(tmp_path):
    """A cache + weights directory that satisfy `check`."""
    w, c = tmp_path / "weights", tmp_path / "cache"
    w.mkdir()
    (c / "mols").mkdir(parents=True)
    (c / "mols" / "ALA.pkl").write_bytes(b"x")
    for f in CK.WEIGHT_FILES:
        (w / f).write_bytes(b"x")
    return c, w


def _write_yaml(tmp_path, text=YAML, name="cplx.yaml"):
    p = tmp_path / name
    p.write_text(text)
    return p


def _fake_predict(monkeypatch, *, samples=1, affinity=True, raise_=None, noise=True, write=True):
    calls = []

    def fake(args):
        calls.append(args)
        if noise:
            print("stdout noise from a library")
            os.write(1, b"raw fd 1 noise\n")
        if raise_:
            raise raise_
        stem = Path(args.data).stem
        res = Path(args.out_dir) / f"boltz_results_{stem}"
        (res / "processed").mkdir(parents=True, exist_ok=True)
        (res / "processed" / "manifest.json").write_text(json.dumps({"records": [{"id": stem}]}))
        if write:
            d = res / "predictions" / stem
            d.mkdir(parents=True)
            for k in range(samples):
                (d / f"{stem}_model_{k}.cif").write_text("data_x\n")
                (d / f"confidence_{stem}_model_{k}.json").write_text(json.dumps(
                    {"confidence_score": 0.9 - 0.1 * k, "ptm": 0.8, "iptm": 0.7, "complex_plddt": 0.85}))
            if affinity:
                (d / f"affinity_{stem}.json").write_text(json.dumps(
                    {"affinity_pred_value": -1.5, "affinity_probability_binary": 0.93}))
        return {"predict_cplx_s": 2.0, "affinity_cplx_s": 1.0, "total_wall_s": 3.5}

    monkeypatch.setattr(cli, "predict", fake)
    return calls


def _main(argv, capfd):
    """(exit code, parsed stdout JSON or None, stdout text, stderr text)."""
    rc = 0
    try:
        cli.main(argv)
    except SystemExit as ex:
        rc = ex.code if isinstance(ex.code, int) else 1
    out, err = capfd.readouterr()
    try:
        obj = json.loads(out)
    except ValueError:
        obj = None
    return rc, obj, out, err


# ------------------------------------------------------------------------------------------------ predict --json
def test_predict_json_shape_and_clean_stdout(tmp_path, monkeypatch, capfd):
    calls = _fake_predict(monkeypatch)
    y = _write_yaml(tmp_path)
    rc, obj, out, err = _main(["predict", str(y), "--out_dir", str(tmp_path / "o"), "--json", "--seed", "7"], capfd)
    assert rc == 0 and obj is not None, (out, err)
    assert out.count("\n") == len(out.rstrip("\n").splitlines()) and json.loads(out) == obj  # exactly one JSON object
    assert "stdout noise from a library" in err and "raw fd 1 noise" in err                    # logs went to stderr
    assert obj["status"] == "ok" and obj["command"] == "predict" and obj["exit_code"] == 0
    assert obj["inputs"]["data"] == str(y.resolve()) and obj["inputs"]["msa_server"] is None
    s = obj["settings"]
    assert s == {"preset": "fast", "preset_source": "default", "mode": "screen", "precision": "f16", "structure_steps": 20,
                 "structure_steps_reason": None, "affinity_steps": 25, "recycles": 3, "affinity_recycles": 5, "samples": 1, "affinity_samples": 5, "seed": 7,
                 "tf32": False, "diffusion_fp16": False}
    (t,) = obj["targets"]
    d = tmp_path / "o" / "boltz_results_cplx" / "predictions" / "cplx"
    assert t["id"] == "cplx" and t["status"] == "ok" and t["structure"] == str(d / "cplx_model_0.cif")
    assert t["confidence_json"] == str(d / "confidence_cplx_model_0.json") and t["affinity_json"] == str(d / "affinity_cplx.json")
    assert (t["confidence_score"], t["ptm"], t["iptm"], t["complex_plddt"]) == (0.9, 0.8, 0.7, 0.85)
    assert t["affinity"]["affinity_pred_value"] == -1.5 and t["affinity"]["unit"] == "log10(IC50 in uM)"
    assert t["affinity"]["affinity_probability_binary"] == 0.93 and t["affinity"]["ic50_uM"] == pytest.approx(10 ** -1.5)
    assert t["wall_s"] == 3.0 and obj["wall_s"] == 3.5
    assert calls[0].sampling_steps == 20 and calls[0].sampling_steps_affinity == 25 and calls[0].precision == "f16"


def test_predict_json_multiple_samples_and_no_affinity(tmp_path, monkeypatch, capfd):
    _fake_predict(monkeypatch, samples=3, affinity=False)
    rc, obj, *_ = _main(["predict", str(_write_yaml(tmp_path)), "--out_dir", str(tmp_path), "--json"], capfd)
    t = obj["targets"][0]
    assert rc == 0 and t["affinity"] is None and t["affinity_json"] is None
    assert [m["model"] for m in t["models"]] == [0, 1, 2] and t["structure"].endswith("cplx_model_0.cif")


def test_predict_without_json_prints_no_json_summary_on_stdout(tmp_path, monkeypatch, capfd):
    _fake_predict(monkeypatch, noise=False)
    rc, obj, out, err = _main(["predict", str(_write_yaml(tmp_path)), "--out_dir", str(tmp_path)], capfd)
    assert rc == 0 and obj is None and "cplx_model_0.cif" in err


# --------------------------------------------------------------------------------------------------- exit codes
def test_exit_2_bad_input(tmp_path, monkeypatch, capfd):
    _fake_predict(monkeypatch)
    cases = {
        "missing file": [str(tmp_path / "nope.yaml")],
        "unsupported type": [str(_write_yaml(tmp_path, "x", "a.txt"))],
        "bad yaml": [str(_write_yaml(tmp_path, "sequences: [", "b.yaml"))],
        "no sequences": [str(_write_yaml(tmp_path, "version: 1\n", "c.yaml"))],
        "protein without msa": [str(_write_yaml(tmp_path, YAML_NO_MSA, "d.yaml"))],
        "bad flag": [str(_write_yaml(tmp_path)), "--bogus"],
        "conflicting sampler": [str(_write_yaml(tmp_path)), "--sampler", "s25", "--sampling_steps", "50"],
        "samples < 1": [str(_write_yaml(tmp_path)), "--diffusion_samples", "0"],
        "unknown preset": [str(_write_yaml(tmp_path)), "--preset", "ultra"],
    }
    for what, extra in cases.items():
        rc, obj, out, err = _main(["predict", *extra, "--out_dir", str(tmp_path), "--json"], capfd)
        assert rc == 2, (what, err)
        assert obj["status"] == "error" and obj["exit_code"] == 2 and obj["error"]["kind"] == "bad_input", what
        assert obj["error"]["message"] and obj["error"]["fix"], what
        line = err.strip().splitlines()[-1]
        assert line.startswith("boltz-mlx: error: ") and " Fix: " in line and len(err.strip().splitlines()) == 1, (what, err)


def test_missing_msa_is_fine_with_the_server_flag(tmp_path, monkeypatch, capfd):
    calls = _fake_predict(monkeypatch)
    rc, obj, out, err = _main(["predict", str(_write_yaml(tmp_path, YAML_NO_MSA)), "--out_dir", str(tmp_path), "--json",
                               "--use_msa_server"], capfd)
    assert rc == 0 and calls[0].use_msa_server and obj["inputs"]["msa_server"] == "https://api.colabfold.com"


def test_exit_3_setup_problem(tmp_path, monkeypatch, capfd):
    _fake_predict(monkeypatch)

    def no_weights(*a, **k):
        raise CO.setup_problem("weights: /x: missing boltz2_conf.safetensors.", "run `boltz-mlx check`.")

    monkeypatch.setattr(CK, "preflight", no_weights)
    rc, obj, out, err = _main(["predict", str(_write_yaml(tmp_path)), "--out_dir", str(tmp_path), "--json"], capfd)
    assert rc == 3 and obj["error"]["kind"] == "setup" and obj["exit_code"] == 3 and "Fix:" in err


def test_real_preflight_names_the_missing_piece(tmp_path, monkeypatch):
    monkeypatch.undo()  # the autouse stubs off: the real preflight against an empty cache
    monkeypatch.setenv("BOLTZ_MLX_WEIGHTS", str(tmp_path / "none"))
    with pytest.raises(SystemExit) as ex:
        CK.preflight(tmp_path / "cache")
    assert ex.value.code == 3 and "weights" in ex.value.message and ex.value.fix


def test_exit_1_runtime_failure(tmp_path, monkeypatch, capfd):
    _fake_predict(monkeypatch, raise_=RuntimeError("Metal exploded"))
    rc, obj, out, err = _main(["predict", str(_write_yaml(tmp_path)), "--out_dir", str(tmp_path), "--json"], capfd)
    assert rc == 1 and obj["error"]["kind"] == "runtime" and "Metal exploded" in obj["error"]["message"]
    assert "Traceback" not in err and "BOLTZ_MLX_TRACEBACK=1" in err
    monkeypatch.setenv("BOLTZ_MLX_TRACEBACK", "1")
    rc, obj, out, err = _main(["predict", str(_write_yaml(tmp_path)), "--out_dir", str(tmp_path), "--json"], capfd)
    assert rc == 1 and "Traceback" in err


def test_exit_1_when_no_structure_was_written(tmp_path, monkeypatch, capfd):
    _fake_predict(monkeypatch, write=False)
    rc, obj, out, err = _main(["predict", str(_write_yaml(tmp_path)), "--out_dir", str(tmp_path), "--json"], capfd)
    assert rc == 1 and obj["status"] == "error" and obj["targets"][0]["status"] == "error"


def test_missing_manifest_means_bad_input(tmp_path, monkeypatch, capfd):
    # upstream process_inputs swallows its own errors; the manifest is then absent
    _fake_predict(monkeypatch, raise_=FileNotFoundError(2, "No such file", "x/processed/manifest.json"))
    rc, obj, *_ = _main(["predict", str(_write_yaml(tmp_path)), "--out_dir", str(tmp_path), "--json"], capfd)
    assert rc == 2 and obj["error"]["kind"] == "bad_input"


def test_classify_exception():
    c = CO.classify_exception
    assert c(FileNotFoundError(2, "x", "/w/boltz2_conf.safetensors")).code == 3
    assert c(FileNotFoundError(2, "x", "/c/mols/ALA.pkl")).code == 3
    assert c(FileNotFoundError(2, "x", "/some/file.yaml")).code == 2
    assert c(RuntimeError("boltz-mlx requires macOS 26 or later")).code == 3
    assert c(MemoryError()).code == 1 and "memory" in c(MemoryError()).message
    assert c(ValueError("boom")).code == 1
    import yaml

    assert c(yaml.YAMLError("bad")).code == 2


def test_keyboard_interrupt_is_130(tmp_path, monkeypatch, capfd):
    _fake_predict(monkeypatch, raise_=KeyboardInterrupt())
    rc, *_ = _main(["predict", str(_write_yaml(tmp_path)), "--out_dir", str(tmp_path)], capfd)
    assert rc == 130


# ------------------------------------------------------------------------------------------------------- presets
def test_preset_names_and_old_aliases():
    known = set(PRESETS)
    assert CO.canonical_preset("fast", known) == "screen" and CO.canonical_preset("full", known) == "strict"
    for old in ("screen", "strict", "turbo", "f16"):
        assert CO.canonical_preset(old, known) == old
    with pytest.raises(SystemExit) as ex:
        CO.canonical_preset("ultra", known)
    assert ex.value.code == 2 and "--preset fast" in ex.value.fix
    assert CO.public_name("screen") == "fast" and CO.public_name("strict") == "full" and CO.public_name("turbo") == "turbo"
    fast, full = CO.preset_settings("screen"), CO.preset_settings("strict")
    assert (fast["precision"], fast["structure_steps"], fast["affinity_steps"], fast["recycles"]) == ("f16", 20, 25, 3)
    assert (full["precision"], full["structure_steps"], full["affinity_steps"], full["recycles"]) == ("fp32", 200, 200, 3)
    assert full["affinity_samples"] == fast["affinity_samples"] == 5 and full["affinity_recycles"] == 5


def _applied(*flags):
    args = cli.parse_args(["predict", "x.yaml", *flags])
    info = cli.apply_preset(args)
    return args, info, cli.resolve_samplers(args)


@pytest.mark.parametrize("flags,name,prec,steps,asteps", [
    ([], "fast", "f16", 20, 25),                              # default
    (["--preset", "fast"], "fast", "f16", 20, 25),
    (["--preset", "full"], "full", "fp32", 200, 200),
    (["--preset", "screen"], "fast", "f16", 20, 25),         # old names
    (["--preset", "strict"], "full", "fp32", 200, 200),
    (["--preset", "f16"], "f16", "f16", 200, 200),
])
def test_apply_preset(flags, name, prec, steps, asteps):
    args, info, (st, af) = _applied(*flags)
    assert info["preset"] == name and args.precision == prec and (st.steps, af.steps) == (steps, asteps)
    assert info["source"] == ("flag" if flags else "default")


def test_apply_preset_turbo_and_explicit_flags_win():
    args, info, (st, af) = _applied("--preset", "turbo")
    assert args.turbo and args.fast and args.tf32 and (st.steps, af.steps) == (50, 200) and args.precision is None
    args, info, (st, af) = _applied("--preset", "full", "--sampling_steps", "40", "--precision", "f16")  # what the GUI does
    assert (st.steps, af.steps, args.precision) == (40, 200, "f16")
    args, info, (st, af) = _applied("--sampler", "s50+aff25")                                              # sampler presets win
    assert (st.steps, af.steps) == (50, 25)
    args, info, (st, af) = _applied("--turbo")                                                            # explicit --turbo: 50, no precision
    assert (st.steps, af.steps, args.precision) == (50, 25, None)


def test_default_preset_off_nax_is_full(monkeypatch):
    from boltz_mlx import kernels as K

    monkeypatch.setattr(K, "_CAP", K.pinned_status("tensorops", "test pin"))
    args, info, (st, af) = _applied()
    assert info["preset"] == "full" and info["source"] == "default" and args.precision == "fp32" and st.steps == 200


def test_parse_args_still_raw_for_old_callers():
    args = cli.parse_args(["predict", "x.yaml"])
    assert args.preset is None and args.precision is None and args.sampling_steps is None and args.json is False
    for old in (["--fast"], ["--turbo"], ["--tf32"], ["--sampler", "s25"], ["--precision", "f16"], ["--num_workers", "1"],
                ["--replay_noise", "n.npz"], ["--write_embeddings"], ["--msa_server_password", "x"]):
        cli.parse_args(["predict", "x.yaml", *old])


def test_default_precision_triggers_reexec_with_env(tmp_path, monkeypatch):
    monkeypatch.setenv("BOLTZ_MLX_PRECISION", "fp32")
    seen = {}

    def fake_exec(exe, argv, env):
        seen.update(argv=argv, env=env)
        raise SystemExit(0)

    monkeypatch.setattr(cli.os, "execve", fake_exec)
    with pytest.raises(SystemExit):
        cli._main(["predict", str(_write_yaml(tmp_path)), "--json"])
    assert seen["env"]["BOLTZ_MLX_PRECISION"] == "f16" and seen["argv"][-2:] == [str(tmp_path / "cplx.yaml"), "--json"]


# ----------------------------------------------------------------------------------------------------- check
def _patch_system(monkeypatch, mac="26.5", machine="arm64", ram=64):
    import boltz_mlx

    monkeypatch.setattr(boltz_mlx, "macos_version", lambda: mac)
    monkeypatch.setattr(CK.platform, "machine", lambda: machine)
    monkeypatch.setattr(CK.platform, "system", lambda: "Darwin")
    real = CK._sysctl
    monkeypatch.setattr(CK, "_sysctl", lambda n: str(ram * 2**30) if n == "hw.memsize" else ("Apple M5 Max" if "brand" in n else real(n)))


def test_check_ok(tmp_path, monkeypatch, capfd):
    c, w = _ready(tmp_path)
    _patch_system(monkeypatch)
    rc, obj, out, err = _main(["check", "--json", "--cache", str(c), "--weights", str(w)], capfd)
    assert rc == 0 and obj["status"] == "ok" and obj["ok"] is True and obj["exit_code"] == 0 and obj["next"] is None
    assert {c["name"] for c in obj["checks"]} >= {"macos", "apple_silicon", "weights", "cache_mols", "package:mlx", "package:boltz"}
    assert all(c["ok"] and c["fix"] is None for c in obj["checks"])
    assert obj["paths"]["cache"] == str(c) and obj["paths"]["weights"] == str(w) and obj["system"]["chip"] == "Apple M5 Max"
    rc, obj, out, err = _main(["check", "--cache", str(c), "--weights", str(w)], capfd)
    assert rc == 0 and obj is None and "boltz-mlx is ready." in out


@pytest.mark.parametrize("what", ["old_macos", "intel", "no_weights", "no_mols", "empty_mols"])
def test_check_failures_exit_3_with_fix(tmp_path, monkeypatch, capfd, what):
    c, w = _ready(tmp_path)
    _patch_system(monkeypatch, mac="15.4" if what == "old_macos" else "26.5", machine="x86_64" if what == "intel" else "arm64")
    if what == "no_weights":
        (w / "boltz2_aff.safetensors").unlink()
    if what == "no_mols":
        c = tmp_path / "elsewhere"
    if what == "empty_mols":
        (c / "mols" / "ALA.pkl").unlink()
    rc, obj, out, err = _main(["check", "--json", "--cache", str(c), "--weights", str(w)], capfd)
    name = {"old_macos": "macos", "intel": "apple_silicon", "no_weights": "weights", "no_mols": "cache_mols", "empty_mols": "cache_mols"}[what]
    bad = [x for x in obj["checks"] if not x["ok"]]
    assert rc == 3 and obj["status"] == "fail" and obj["exit_code"] == 3 and [x["name"] for x in bad] == [name]
    assert bad[0]["fix"] and obj["next"] == bad[0]["fix"]
    rc, _, out, _ = _main(["check", "--cache", str(c), "--weights", str(w)], capfd)
    assert rc == 3 and f"FAIL {name}" in out and "fix:" in out


def test_check_warns_on_low_memory_and_never_imports_the_gpu(tmp_path, monkeypatch, capfd):
    c, w = _ready(tmp_path)
    _patch_system(monkeypatch, ram=8)
    rc, obj, *_ = _main(["check", "--json", "--cache", str(c), "--weights", str(w)], capfd)
    assert rc == 0 and obj["warnings"] and "8 GB" in obj["warnings"][0]
    src = Path(CK.__file__).read_text()
    assert "mx." not in src and "import mlx" not in src and "kernels" not in src  # no GPU dispatch in check


# ------------------------------------------------------------------------------------------------------ screen
LIGS = "id,smiles\nm1,CCO\nm2,c1ccccc1O\nm3,CCN(CC)CC\nbad,xx(\n"


def _screen_args(tmp_path, *extra, ligs=LIGS):
    lf = tmp_path / "l.csv"
    lf.write_text(ligs)
    return ["screen", "--target", PROT, "--msa", "empty", "--ligands", str(lf), "--out", str(tmp_path / "o"), *extra]


@pytest.fixture
def fake_screen(monkeypatch):
    real = SM.run
    seen = {}

    def fake_run(a, argv=None, engine=None, log=print):
        seen["a"], seen["argv"] = a, argv  # (run() resolves a.preset: fast -> screen)
        return real(a, argv, engine=engine or FakeEngine(Path(a.out) / "work"), log=log)

    monkeypatch.setattr(SM, "run", fake_run)
    return seen


def test_screen_json_shape(tmp_path, fake_screen, capfd):
    rc, obj, out, err = _main(_screen_args(tmp_path, "--json", "--preset", "fast", "--rank-by", "value"), capfd)
    assert rc == 0 and obj is not None and json.loads(out) == obj, (out, err)
    assert "[plan]" in err and "[plan]" not in out                                   # progress on stderr only
    assert obj["status"] == "ok" and obj["command"] == "screen" and obj["exit_code"] == 0
    assert obj["inputs"]["target"] == PROT and obj["inputs"]["out"] == str((tmp_path / "o").resolve())
    s = obj["settings"]
    assert (s["preset"], s["mode"], s["precision"], s["structure_steps"], s["affinity_steps"], s["recycles"]) == (
        "fast", "screen", "f16", 20, 25, 3)
    assert s["affinity_samples"] == 5 and s["seed"] == 0 and s["rank_by"] == "value" and fake_screen["a"].preset == "screen"
    assert obj["counts"] == {"ligands": 4, "ok": 3, "invalid": 1, "error": 0, "remaining": 0}
    assert obj["top_n"] == 20 and [r["rank"] for r in obj["top"]] == [1, 2, 3]
    vals = [r["affinity"]["affinity_pred_value"] for r in obj["top"]]
    assert vals == sorted(vals)                                                      # ranked by value: low (strong) first
    r0 = obj["top"][0]
    assert set(r0) >= {"rank", "ligand_id", "smiles", "affinity", "confidence_score", "ptm", "iptm", "complex_plddt"}
    assert r0["affinity"]["unit"] == "log10(IC50 in uM)" and "affinity_probability_binary" in r0["affinity"]
    assert Path(obj["results_csv"]).is_file() and obj["results_csv"].endswith("results.csv") and Path(obj["results_jsonl"]).is_file()
    assert obj["wall_s"] is not None and obj["complete"] is True


def test_screen_top_n(tmp_path, fake_screen, capfd):
    rc, obj, *_ = _main(_screen_args(tmp_path, "--json", "--top", "2"), capfd)
    assert rc == 0 and obj["top_n"] == 2 and len(obj["top"]) == 2 and obj["counts"]["ok"] == 3


def test_screen_top_default_is_20(tmp_path, fake_screen, capfd):
    many = "id,smiles\n" + "".join(f"m{i},{'C' * (i + 1)}O\n" for i in range(25))
    rc, obj, *_ = _main(_screen_args(tmp_path, "--json", "--save-structures", "0", ligs=many), capfd)
    assert rc == 0 and obj["top_n"] == 20 and len(obj["top"]) == 20 and obj["counts"]["ok"] == 25


@pytest.mark.parametrize("name,mode,steps", [("fast", "screen", 20), ("full", "strict", 200), ("screen", "screen", 20),
                                             ("strict", "strict", 200), ("f16", "f16", 200), ("turbo", "turbo", 50)])
def test_screen_preset_names(tmp_path, fake_screen, capfd, name, mode, steps):
    rc, obj, out, err = _main(_screen_args(tmp_path, "--json", "--preset", name), capfd)
    assert rc == 0, err
    assert fake_screen["a"].preset == mode and obj["settings"]["mode"] == mode and obj["settings"]["structure_steps"] == steps
    assert obj["settings"]["preset"] == CO.public_name(mode)


def test_screen_default_preset_is_fast_on_nax(tmp_path, fake_screen, capfd):
    rc, obj, out, err = _main(_screen_args(tmp_path, "--json"), capfd)
    assert rc == 0 and obj["settings"]["preset"] == "fast" and "the default for kernel path nax" in err


def test_screen_dry_run_json(tmp_path, capfd):
    rc, obj, out, err = _main(_screen_args(tmp_path, "--json", "--dry-run"), capfd)
    assert rc == 0 and obj["status"] == "dry_run" and any("ESTIMATE" in l for l in obj["plan"]) and "top" not in obj
    assert not (tmp_path / "o" / "results.jsonl").exists()


def test_screen_exit_codes(tmp_path, fake_screen, capfd, monkeypatch):
    rc, obj, out, err = _main(_screen_args(tmp_path, "--json", "--preset", "ultra"), capfd)
    assert rc == 2 and obj["error"]["kind"] == "bad_input" and "--preset fast" in obj["error"]["fix"]
    rc, obj, out, err = _main(["screen", "--target", PROT, "--ligands", str(tmp_path / "missing.csv"), "--out", str(tmp_path / "o"),
                               "--json"], capfd)
    assert rc == 2 and obj["status"] == "error"
    rc, obj, *_ = _main(_screen_args(tmp_path, "--json", "--chunk", "0"), capfd)
    assert rc == 2
    rc, obj, *_ = _main(["screen", "--json"], capfd)  # required flags missing
    assert rc == 2 and obj["error"]["fix"].startswith("see `boltz-mlx screen --help`")
    rc, obj, out, err = _main(_screen_args(tmp_path, "--json", "--out", str(tmp_path / "o3"), ligs="id,smiles\nbad,xx(\nbad2,yy(\n"), capfd)
    assert rc == 2 and obj["status"] == "error" and obj["counts"]["ok"] == 0 and obj["counts"]["invalid"] == 2


def test_screen_setup_problem_exits_3(tmp_path, monkeypatch, capfd):
    monkeypatch.undo()  # real preflight, real run(): no fake engine, so the missing weights / mols stop it before any model work
    monkeypatch.setenv("BOLTZ_MLX_WEIGHTS", str(tmp_path / "none"))
    rc, obj, out, err = _main(_screen_args(tmp_path, "--json", "--preset", "full", "--cache", str(tmp_path / "nocache")), capfd)
    assert rc == 3 and obj["error"]["kind"] == "setup" and "Fix:" in err


def test_screen_reexec_gets_the_real_stdout(tmp_path, monkeypatch, capfd):
    """--json redirects fd 1 to stderr; the re-exec'd child must get the real stdout back."""
    import boltz_mlx.check

    monkeypatch.setattr(boltz_mlx.check, "preflight", lambda *a, **k: None)
    seen = {}

    def fake_exec(exe, argv, env):
        seen["fd1_is_stderr"] = os.fstat(1).st_ino == os.fstat(2).st_ino and os.fstat(1).st_dev == os.fstat(2).st_dev
        raise SystemExit(0)

    monkeypatch.setattr(SM.os, "execve", fake_exec)
    with pytest.raises(SystemExit):
        cli.main(_screen_args(tmp_path, "--json", "--preset", "full"))
    capfd.readouterr()
    assert seen["fd1_is_stderr"] is False


# ------------------------------------------------------------------------------------------------------- help
def test_help_is_short_and_accurate(capsys):
    ap = cli.build_parser()
    top = ap.format_help()
    assert len(top.splitlines()) < 30 and "check" in top and "--preset fast" in top and "Exit codes" in top and "AGENTS.md" in top
    sub = ap._subparsers._group_actions[0].choices
    assert set(sub) == {"check", "setup", "predict", "screen", "gui"}
    pred, scr = sub["predict"].format_help(), sub["screen"].format_help()
    for text in (pred, scr):
        assert len(text.splitlines()) < 90 and "--json" in text and "--preset" in text and "fast" in text and "full" in text
    assert "--use_msa_server" in pred and "ColabFold" in pred and "msa: empty" in pred
    for hidden in ("--sampler", "--turbo", "--replay_noise", "--tf32", "--write_embeddings", "--msa_server_password"):
        assert f"\n  {hidden}" not in pred
    assert "--presets-file" not in scr and "--measure" not in scr and "--top" in scr
    assert "--target" in scr and "--dry-run" in scr


def test_agents_md_contract():
    root = Path(__file__).resolve().parents[1]
    md = (root / "AGENTS.md").read_text()
    assert len(md.splitlines()) <= 80
    for needle in ("boltz-mlx check", "--json", "--use_msa_server", "--preset fast", "--preset full", "affinity_pred_value",
                   "log10(IC50", "macOS 26", "ColabFold"):
        assert needle in md, needle
    for code in ("0", "1", "2", "3"):
        assert f"| {code} " in md or f"`{code}`" in md or f"{code} =" in md


# ------------------------------------------------------------------------------- Fast: tiny-ligand rule (< 10 heavy atoms)
# Owner decision: at 20 structure steps a 4-atom ligand (BCT, 8gpp) collapsed; Fast runs 200 structure steps when any ligand has
# fewer than 10 heavy atoms and the user did not set the steps. Affinity stays 25. Full is unchanged. CPU only, predictions faked.
BCT = "OC(=O)[O-]"            # 4 heavy atoms
NONANE, DECANE = "C" * 9, "C" * 10   # 9 -> tiny, 10 -> not


def _yaml_lig(lig_line: str, lid: str = "B") -> str:
    return (f"version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: {PROT}\n      msa: empty\n"
            f"  - ligand:\n      id: {lid}\n      {lig_line}\nproperties:\n  - affinity:\n      binder: {lid}\n")


def _mols(tmp_path):
    """A fake upstream mols/ dictionary: <CCD>.pkl RDKit mols (BCT 4 heavy atoms, ATP-sized 31, ZN 1)."""
    import pickle

    from rdkit import Chem

    d = tmp_path / "cache" / "mols"
    d.mkdir(parents=True)
    for code, smi in (("BCT", BCT), ("CEL", CELECOXIB), ("ZN", "[Zn+2]"),
                      ("ATP", "Nc1ncnc2c1ncn2C1OC(COP(=O)(O)OP(=O)(O)OP(=O)(O)O)C(O)C1O")):
        (d / f"{code}.pkl").write_bytes(pickle.dumps(Chem.AddHs(Chem.MolFromSmiles(smi))))   # with H, as upstream's pickles may be
    return d.parent


def test_predict_tiny_smiles_ligand_runs_200_steps(tmp_path, monkeypatch, capfd):
    calls = _fake_predict(monkeypatch)
    y = _write_yaml(tmp_path, _yaml_lig(f"smiles: '{BCT}'"))
    rc, obj, out, err = _main(["predict", str(y), "--out_dir", str(tmp_path / "o"), "--json"], capfd)
    assert rc == 0, err
    s = obj["settings"]
    assert s["structure_steps"] == 200 and s["affinity_steps"] == 25 and s["structure_steps_reason"] == "ligand B has 4 heavy atoms (< 10)"
    assert calls[0].sampling_steps == 200 and calls[0].sampling_steps_affinity == 25 and calls[0].precision == "f16"
    assert "200 structure steps instead of 20" in err and "ligand B has 4 heavy atoms (< 10)" in err   # the stderr note
    assert "200 structure steps" not in out.replace(json.dumps(obj), "")


def test_predict_26_atom_ligand_stays_at_20_steps(tmp_path, monkeypatch, capfd):
    calls = _fake_predict(monkeypatch)
    rc, obj, out, err = _main(["predict", str(_write_yaml(tmp_path)), "--out_dir", str(tmp_path / "o"), "--json"], capfd)
    assert rc == 0 and obj["settings"]["structure_steps"] == 20 and obj["settings"]["structure_steps_reason"] is None
    assert calls[0].sampling_steps == 20 and "instead of 20" not in err


@pytest.mark.parametrize("smi,steps", [(NONANE, 200), (DECANE, 20)])
def test_predict_threshold_is_below_ten_heavy_atoms(tmp_path, monkeypatch, capfd, smi, steps):
    _fake_predict(monkeypatch)
    rc, obj, *_ = _main(["predict", str(_write_yaml(tmp_path, _yaml_lig(f"smiles: '{smi}'"))), "--out_dir", str(tmp_path), "--json"], capfd)
    assert rc == 0 and obj["settings"]["structure_steps"] == steps


def test_predict_counts_heavy_atoms_not_hydrogens(tmp_path, monkeypatch, capfd):
    # explicit hydrogens and [H] atoms are not heavy atoms: 4 carbons + 6 explicit H = 4 heavy
    _fake_predict(monkeypatch)
    rc, obj, *_ = _main(["predict", str(_write_yaml(tmp_path, _yaml_lig("smiles: '[H]C([H])([H])C([H])([H])C([H])([H])C'"))),
                         "--out_dir", str(tmp_path), "--json"], capfd)
    assert rc == 0 and obj["settings"]["structure_steps"] == 200 and "4 heavy atoms" in obj["settings"]["structure_steps_reason"]


def test_predict_explicit_steps_always_win(tmp_path, monkeypatch, capfd):
    calls = _fake_predict(monkeypatch)
    y = _write_yaml(tmp_path, _yaml_lig(f"smiles: '{BCT}'"))
    rc, obj, out, err = _main(["predict", str(y), "--out_dir", str(tmp_path / "o"), "--json", "--sampling_steps", "20"], capfd)
    assert rc == 0 and obj["settings"]["structure_steps"] == 20 and obj["settings"]["structure_steps_reason"] is None
    assert calls[-1].sampling_steps == 20 and "heavy atoms" not in err
    rc, obj, *_ = _main(["predict", str(y), "--out_dir", str(tmp_path / "o2"), "--json", "--sampling_steps", "75"], capfd)
    assert obj["settings"]["structure_steps"] == 75 and calls[-1].sampling_steps == 75


def test_predict_full_preset_is_unchanged(tmp_path, monkeypatch, capfd):
    monkeypatch.setenv("BOLTZ_MLX_PRECISION", "fp32")   # Full is fp32: already set, so no re-exec
    calls = _fake_predict(monkeypatch)
    y = _write_yaml(tmp_path, _yaml_lig(f"smiles: '{BCT}'"))
    rc, obj, out, err = _main(["predict", str(y), "--out_dir", str(tmp_path), "--json", "--preset", "full"], capfd)
    assert rc == 0 and obj["settings"]["structure_steps"] == 200 and obj["settings"]["structure_steps_reason"] is None
    assert calls[0].sampling_steps == 200 and calls[0].sampling_steps_affinity == 200 and "heavy atoms" not in err


@pytest.mark.parametrize("name", ["fast", "screen"])
def test_predict_rule_applies_to_both_names_of_fast(tmp_path, monkeypatch, capfd, name):
    _fake_predict(monkeypatch)
    y = _write_yaml(tmp_path, _yaml_lig(f"smiles: '{BCT}'"))
    rc, obj, *_ = _main(["predict", str(y), "--out_dir", str(tmp_path), "--json", "--preset", name], capfd)
    assert rc == 0 and obj["settings"]["structure_steps"] == 200


def test_predict_ccd_ligand_uses_the_mols_dictionary(tmp_path, monkeypatch, capfd):
    cache = _mols(tmp_path)
    calls = _fake_predict(monkeypatch)
    for code, steps in (("BCT", 200), ("ZN", 200), ("CEL", 20), ("ATP", 20)):
        y = _write_yaml(tmp_path, _yaml_lig(f"ccd: {code}"), name=f"{code}.yaml")
        rc, obj, out, err = _main(["predict", str(y), "--out_dir", str(tmp_path / code), "--json", "--cache", str(cache)], capfd)
        assert rc == 0 and obj["settings"]["structure_steps"] == steps and calls[-1].sampling_steps == steps, (code, err)
        if steps == 200:
            assert obj["settings"]["structure_steps_reason"] == (
                "ligand B has 4 heavy atoms (< 10)" if code == "BCT" else "ligand B has 1 heavy atom (< 10)")


def test_predict_ccd_not_in_dictionary_is_noted_and_not_tiny(tmp_path, monkeypatch, capfd):
    cache = _mols(tmp_path)
    _fake_predict(monkeypatch)
    y = _write_yaml(tmp_path, _yaml_lig("ccd: QQQ"))
    rc, obj, out, err = _main(["predict", str(y), "--out_dir", str(tmp_path / "o"), "--json", "--cache", str(cache)], capfd)
    assert rc == 0 and obj["settings"]["structure_steps"] == 20 and "could not count the heavy atoms of ligand B" in err


def test_predict_directory_any_tiny_ligand_names_its_file(tmp_path, monkeypatch, capfd):
    calls = _fake_predict(monkeypatch)
    d = tmp_path / "in"
    d.mkdir()
    _write_yaml(d, _yaml_lig(f"smiles: '{CELECOXIB}'"), name="big.yaml")
    _write_yaml(d, _yaml_lig(f"smiles: '{BCT}'", lid="L"), name="tiny.yaml")
    rc, obj, out, err = _main(["predict", str(d), "--out_dir", str(tmp_path / "o"), "--json"], capfd)
    assert rc == 0 and calls[0].sampling_steps == 200
    assert obj["settings"]["structure_steps_reason"] == "tiny.yaml: ligand L has 4 heavy atoms (< 10)"


def test_predict_fasta_ligands(tmp_path, monkeypatch, capfd):
    cache = _mols(tmp_path)
    _fake_predict(monkeypatch)
    f = tmp_path / "x.fasta"
    f.write_text(f">A|protein|empty\n{PROT}\n>B|ccd|\nBCT\n")
    rc, obj, *_ = _main(["predict", str(f), "--out_dir", str(tmp_path / "o"), "--json", "--cache", str(cache)], capfd)
    assert rc == 0 and obj["settings"]["structure_steps"] == 200 and "ligand B has 4" in obj["settings"]["structure_steps_reason"]


def test_predict_help_mentions_the_rule(capfd):
    rc, _, out, err = _main(["predict", "--help"], capfd)
    assert "heavy atoms" in out


# ------------------------------------------------------------------------------------------ screen: per ligand
# A screen's structure pass is per ligand, so the rule is per ligand: only a ligand with < 10 heavy atoms runs 200 steps; the others
# keep Fast's 20. The affinity pass is 25 for everyone. The config key records the rule, so a resume cannot mix configurations.
TINY_LIGS = f"id,smiles\ntiny,{BCT}\nbig,{CELECOXIB}\nethanol,CCO\nmid,CC(=O)Nc1ccc(O)cc1\n"


def _rows(tmp_path):
    from boltz_mlx.screen import results as R

    return {r["ligand_id"]: r for r in R.latest_rows(R.read_rows(tmp_path / "o" / "results.jsonl")).values()}


def test_screen_only_tiny_ligands_run_200_steps(tmp_path, fake_screen, capfd):
    from screen_fakes import FakeModel

    FakeModel.reset()
    rc, obj, out, err = _main(_screen_args(tmp_path, "--json", "--chunk", "10", ligs=TINY_LIGS), capfd)
    assert rc == 0, err
    rows = _rows(tmp_path)
    assert {k: r["steps"] for k, r in rows.items()} == {"tiny": 200, "big": 20, "ethanol": 200, "mid": 20}
    assert rows["tiny"]["steps_reason"] == "ligand tiny has 4 heavy atoms (< 10)" and rows["big"]["steps_reason"] is None
    assert {r["affinity_steps"] for r in rows.values()} == {25}
    assert sorted(s for n, s in FakeModel.steps_log if n == "conf") == [20, 20, 200, 200]   # one structure pass per ligand
    assert [s for n, s in FakeModel.steps_log if n == "aff"] == [25, 25, 25, 25]
    s = obj["settings"]
    assert s["structure_steps"] == 20 and s["structure_steps_tiny_ligands"] == 200 and s["ligands_at_tiny_steps"] == 2
    assert s["structure_steps_reason"].startswith("2 of 4 ligands have < 10 heavy atoms") and "tiny" in s["structure_steps_reason"]
    assert {r["ligand_id"]: r["structure_steps"] for r in obj["top"]} == {"tiny": 200, "big": 20, "ethanol": 200, "mid": 20}
    assert "Fast preset: 2 of 4 ligands have < 10 heavy atoms" in err and "run 200 structure steps" in err   # the stderr note
    cfg = json.loads((tmp_path / "o" / "screen_config.json").read_text())
    assert cfg["steps"] == 20 and cfg["tiny_ligand_steps"] == {"heavy_atoms_lt": 10, "steps": 200}


def test_screen_no_tiny_ligand_changes_nothing(tmp_path, fake_screen, capfd):
    rc, obj, out, err = _main(_screen_args(tmp_path, "--json", ligs=f"id,smiles\nbig,{CELECOXIB}\nmid,CC(=O)Nc1ccc(O)cc1\n"), capfd)
    assert rc == 0 and {r["steps"] for r in _rows(tmp_path).values()} == {20}
    assert obj["settings"]["structure_steps_reason"] is None and obj["settings"]["ligands_at_tiny_steps"] == 0
    assert "heavy atoms" not in err.split("[plan]")[0]


def test_screen_explicit_steps_win_and_full_is_unchanged(tmp_path, fake_screen, capfd):
    rc, obj, out, err = _main(_screen_args(tmp_path, "--json", "--steps", "20", ligs=TINY_LIGS), capfd)
    assert rc == 0 and {r["steps"] for r in _rows(tmp_path).values()} == {20}
    assert "structure_steps_reason" not in obj["settings"] and "tiny_ligand_steps" not in json.loads(
        (tmp_path / "o" / "screen_config.json").read_text())
    full = tmp_path / "full"
    full.mkdir()
    rc, obj, *_ = _main(_screen_args(full, "--json", "--preset", "full", ligs=TINY_LIGS), capfd)
    assert rc == 0 and obj["settings"]["structure_steps"] == 200 and "structure_steps_reason" not in obj["settings"]


def test_screen_dry_run_plan_says_which_ligands_are_tiny(tmp_path, capfd):
    rc, obj, out, err = _main(_screen_args(tmp_path, "--json", "--dry-run", ligs=TINY_LIGS), capfd)
    assert rc == 0 and any(l.strip().startswith("tiny ligands: 2 of 4 valid ligands have < 10 heavy atoms and run 200 structure steps")
                           for l in obj["plan"])
    assert obj["settings"]["ligands_at_tiny_steps"] == 2


def test_screen_resume_keeps_per_ligand_steps(tmp_path, fake_screen, capfd):
    from screen_fakes import FakeModel

    assert _main(_screen_args(tmp_path, "--json", ligs=TINY_LIGS), capfd)[0] == 0
    FakeModel.reset()
    rc, obj, out, err = _main(_screen_args(tmp_path, "--json", ligs=TINY_LIGS), capfd)   # the same command again: nothing to do
    assert rc == 0 and FakeModel.steps_log == [] and obj["counts"]["ok"] == 4
    rc, obj, out, err = _main(_screen_args(tmp_path, "--json", "--steps", "20", ligs=TINY_LIGS), capfd)   # a different configuration
    assert rc != 0 or obj["status"] == "error" or "different configuration" in err + out
