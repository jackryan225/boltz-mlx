"""`boltz-mlx screen`: SIGTERM stops the run cleanly at a ligand boundary (the GUI's Stop button), a second SIGTERM stops at once, and
the per-ligand `[start]` log lines the GUI reads for its progress view.

CPU only; the model and the upstream featurizer are replaced by tests/screen_fakes.py.
"""

import json
import os
import signal

import pytest

from boltz_mlx.screen import results as R
from boltz_mlx.screen.inputs import Ligand, load_target, validate_ligand
from boltz_mlx.screen.presets import PRESETS
from boltz_mlx.screen.runner import ScreenConfig, ScreenRunner
from screen_fakes import FakeEngine, FakeModel

SMILES = ["CCO", "c1ccccc1O", "CC(=O)Nc1ccc(O)cc1", "CCN(CC)CC", "OC(=O)c1ccccc1", "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "CCOC(=O)C", "c1ccncc1"]
PROT = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQW"


@pytest.fixture(autouse=True)
def _reset():
    FakeModel.reset()
    yield
    FakeModel.reset()


class TermEngine(FakeEngine):
    """Sends SIGTERM (`n` times) to this process at a chosen ligand's affinity featurization (as the GUI's Stop does to a screen)."""

    def __init__(self, *a, at=None, n=1, **kw):
        super().__init__(*a, **kw)
        self.at, self.n = at, n

    def featurize(self, lig, affinity=False):
        if affinity and lig.record_id == self.at and self.n:
            n, self.n = self.n, 0
            for _ in range(n):
                os.kill(os.getpid(), signal.SIGTERM)
        return super().featurize(lig, affinity)


def _runner(tmp_path, eng_cls=FakeEngine, **eng_kw):
    out = tmp_path / "out"
    from boltz_mlx.screen.runner import ScreenConfig

    p = PRESETS["strict"]
    cfg = ScreenConfig(out=out, preset=p, steps=p.steps, recycling=p.recycling, affinity_steps=p.affinity_steps, chunk=3, device="cpu",
                       save_structures=1, prefetch=False)
    ligs = [validate_ligand(Ligand(f"cpd{i}", SMILES[i % len(SMILES)], i, f"cpd{i}")) for i in range(6)]
    logs = []
    return ScreenRunner(cfg, load_target(PROT[:60], msa="empty"), ligs, eng_cls(out / "work", **eng_kw), log=logs.append), logs, out


def test_sigterm_stops_at_ligand_boundary_and_run_resumes(tmp_path):
    prev = signal.getsignal(signal.SIGTERM)
    r, logs, out = _runner(tmp_path, TermEngine, at="cpd1")
    st = r.run()
    assert signal.getsignal(signal.SIGTERM) == prev                      # handler restored
    assert st["stopped_by_signal"] is True and st["complete"] is False and st["interrupted"] is False
    rows = R.read_rows(out / "results.jsonl")
    assert [x["ligand_id"] for x in rows] == ["cpd0", "cpd1"]            # the ligand in progress finished; the next did not start
    assert any(l.startswith("[stop] SIGTERM") for l in logs) and not any(l.startswith("[interrupted]") for l in logs)
    assert any(l.startswith("[start] structure cpd0 1/3") for l in logs) and any(l.startswith("[start] affinity cpd1 2/3") for l in logs)
    assert json.loads((out / "screen_status.json").read_text())["remaining"] == 4
    FakeModel.reset()
    r2, logs2, _ = _runner(tmp_path)
    st2 = r2.run()
    assert st2["complete"] and st2["stopped_by_signal"] is False and st2["ok_total"] == 6
    assert len({x["ligand_id"] for x in R.read_rows(out / "results.jsonl")}) == 6


def test_second_sigterm_stops_immediately(tmp_path):
    prev = signal.getsignal(signal.SIGTERM)
    r, logs, out = _runner(tmp_path, TermEngine, at="cpd1", n=2)
    with pytest.raises(KeyboardInterrupt):
        r.run()
    assert signal.getsignal(signal.SIGTERM) == prev
    assert any(l.startswith("[interrupted]") for l in logs)
    st = json.loads((out / "screen_status.json").read_text())
    assert st["interrupted"] is True and st["complete"] is False
    assert [x["ligand_id"] for x in R.read_rows(out / "results.jsonl")] == ["cpd0"]       # cpd1 was in flight when the 2nd signal came
