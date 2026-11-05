"""Offline: a screen run with socket connections disabled completes (precomputed MSA or msa: empty only).

Runs the REAL CPU side of `boltz-mlx screen` (target/ligand parsing, upstream process_input with RDKit conformers, the
receptor-MSA cache, seeded upstream featurization for both passes incl. the affinity crop, BoltzWriter /
BoltzAffinityWriter) with the model replaced by tests/screen_fakes.FakeModel (no checkpoints: CPU-only, fast).
Needs the upstream molecule dictionary (<cache>/mols); skipped when it is not installed.
"""

import argparse
import csv
import json
import socket
from pathlib import Path

import numpy as np
import pytest

from boltz_mlx.paths import default_cache
from boltz_mlx.screen import results as R
from boltz_mlx.screen.engine import UpstreamEngine
from boltz_mlx.screen.main import add_parser, run
from screen_fakes import FakeModel

PROT = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQ"


def _cache() -> Path:
    for c in (default_cache(),):
        if (c / "mols").is_dir():
            return c
    pytest.skip("upstream mols/ dictionary not installed (set BOLTZ_MLX_CACHE)")


class MockModelEngine(UpstreamEngine):
    def load_model(self, name):
        return FakeModel(name)


@pytest.fixture
def no_network(monkeypatch):
    attempts = []

    def deny(self, *a, **kw):
        attempts.append(a)
        raise OSError("network disabled by test_offline")

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket.socket, "connect_ex", deny)
    monkeypatch.setattr(socket, "create_connection", lambda *a, **kw: deny(None, *a))
    return attempts


def _args(argv):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    add_parser(sub)
    return ap.parse_args(["screen", *argv])


@pytest.mark.parametrize("msa", ["empty", "a3m"])
def test_screen_offline_real_featurization(tmp_path, no_network, msa):
    FakeModel.reset()
    cache = _cache()
    lig = tmp_path / "ligs.csv"
    lig.write_text("id,smiles\nphenol,Oc1ccccc1\nbad,C1CC\nacetamide,CC(N)=O\nsalt,CC(=O)[O-].[Na+]\n")
    target = tmp_path / "rec.yaml"
    if msa == "a3m":
        a3m = tmp_path / "rec.a3m"
        a3m.write_text(f">query\n{PROT}\n>hit1\n{PROT[:-3]}GGG\n>hit2\n{'A' + PROT[1:]}\n")
        target.write_text(f"version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: {PROT}\n      msa: rec.a3m\n")
    else:
        target.write_text(f"version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: {PROT}\n")  # no msa -> empty + warning
    out = tmp_path / "out"
    a = _args(["--target", str(target), "--ligands", str(lig), "--out", str(out), "--device", "cpu", "--chunk", "2",
               "--cache", str(cache)])
    from boltz_mlx.screen.main import build_config, load_inputs, load_presets

    tgt, _ = load_inputs(a)
    cfg = build_config(a, load_presets(None))
    eng = MockModelEngine(out / "work", cache, tgt, cfg.max_msa_seqs, cfg.output_format, cfg.seed)
    logs = []
    rc = run(a, [], engine=eng, log=logs.append)
    assert rc == 0, "\n".join(logs)
    assert no_network == []
    rows = R.latest_rows(R.read_rows(out / "results.jsonl"))
    assert rows["bad"]["status"] == "invalid"
    for k in ("phenol", "acetamide", "salt"):
        assert rows[k]["status"] == "ok", rows[k]
        assert rows[k]["affinity_crop_tokens"] > 0 and rows[k]["n_tokens"] == len(PROT) + rows[k]["n_atoms"]
        assert rows[k]["msa_source"] == ("empty" if msa == "empty" else "a3m:rec.a3m")
        assert rows[k]["affinity_mw"] > 0 and rows[k]["ligand_plddt"] is not None
        d = out / "work" / "predictions" / rows[k]["record_id"]
        assert (d / f"{rows[k]['record_id']}_model_0.cif").exists() and (d / f"affinity_{rows[k]['record_id']}.json").exists()
        assert not list(d.glob("pae_*.npz"))  # cleaned up after the row
    assert rows["salt"]["smiles_used"] == "CC(=O)[O-]"
    ranked = list(csv.DictReader(open(out / "results.csv")))
    assert [r["rank"] for r in ranked] == ["1", "2", "3", ""]
    if msa == "a3m":
        # receptor MSA parsed once and linked per record (same bytes as upstream's own per-record parse)
        cached = list((out / "work" / "msa_cache").glob("*.npz"))
        assert len(cached) == 1
        for k in ("phenol", "acetamide"):
            linked = out / "work" / "processed" / "msa" / f"{rows[k]['record_id']}_0.npz"
            assert linked.exists() and linked.stat().st_ino == cached[0].stat().st_ino
        from boltz.data.parse.a3m import parse_a3m

        ref = tmp_path / "ref.npz"
        parse_a3m(a3m, taxonomy=None, max_seqs=8192).dump(ref)
        za, zb = np.load(ref), np.load(cached[0])
        assert sorted(za.files) == sorted(zb.files) and all(np.array_equal(za[f], zb[f]) for f in za.files)
    # features are per-record seeded: a second engine re-featurizes phenol bit-identically
    lig_obj = next(l for l in load_inputs(a)[1] if l.ligand_id == "phenol")
    b1 = eng.featurize(lig_obj, affinity=False)
    b2 = MockModelEngine(out / "work", cache, tgt, cfg.max_msa_seqs, cfg.output_format, cfg.seed).featurize(lig_obj, affinity=False)
    for k_, v in b1.items():
        if hasattr(v, "numpy"):
            assert np.array_equal(v.numpy(), b2[k_].numpy()), k_
    st = json.loads((out / "screen_status.json").read_text())
    assert st["complete"] and st["ok_total"] == 3


def test_upstream_preprocessing_failure_becomes_ligand_failure(tmp_path, no_network):
    """upstream process_input swallows exceptions; the engine turns a missing record into a LigandFailure with the reason."""
    from boltz_mlx.screen.engine import LigandFailure
    from boltz_mlx.screen.inputs import Ligand, load_target

    tgt = load_target(PROT, msa="empty")
    eng = MockModelEngine(tmp_path / "work", _cache(), tgt)
    lig = Ligand("broken", "C1CC", 0, record_id="broken", smiles_used="C1CC")  # bypasses validate_ligand on purpose
    with pytest.raises(LigandFailure, match="preprocessing failed"):
        eng.prepare(lig)
    assert "broken" in eng._failures
