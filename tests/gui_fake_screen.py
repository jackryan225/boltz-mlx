"""Stand-in for `python -m boltz_mlx.cli screen` in the GUI tests (and for trying the GUI without a GPU):
the real `screen` argument parser and runner, with the fake engine of tests/screen_fakes.py (CPU, no checkpoints).

  python tests/gui_fake_screen.py [screen] --target ... --ligands ... --out ...

Environment:  GUI_FAKE_DELAY   seconds added to every structure / affinity pass (default 0), so a run can be stopped midway
              GUI_FAKE_POSE    an mmCIF to copy as every ligand's pose (default: a tiny placeholder file)
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from screen_fakes import FakeEngine  # noqa: E402  (sets the CPU device before MLX initialises)

from boltz_mlx.screen.main import add_parser, run  # noqa: E402


class SlowFakeEngine(FakeEngine):
    def __init__(self, work, delay=0.0, pose=None, **kw):
        super().__init__(work, **kw)
        self.delay = delay
        self.pose = pose

    def write_structure(self, out, batch):
        time.sleep(self.delay)
        super().write_structure(out, batch)
        if self.pose:
            shutil.copyfile(self.pose, self.pose_path(batch["lig"]))

    def write_affinity(self, out, batch):
        time.sleep(self.delay)
        super().write_affinity(out, batch)


def main(argv: list[str]) -> int:
    if argv and argv[0] == "screen":
        argv = argv[1:]
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    add_parser(sub)
    args = ap.parse_args(["screen", *argv])
    eng = SlowFakeEngine(Path(args.out) / "work", delay=float(os.environ.get("GUI_FAKE_DELAY", "0")),
                         pose=os.environ.get("GUI_FAKE_POSE") or None)
    return run(args, argv, engine=eng, log=print)


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except KeyboardInterrupt:
        sys.exit(130)
