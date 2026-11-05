"""`boltz predict` of an upstream-style install (RBVI fork, stock PyPI boltz, or the Novel-Therapeutics boltz-community fork) with model-call timing; nothing else changes.

  <venv>/bin/python -I tools/macbench/up_predict.py predict INPUT.yaml --out_dir D --cache C --accelerator gpu ... [boltz args]
  UP_TIMINGS=FILE   where to write the timing record (JSON); default: <out_dir>/up_timings.json

The arguments after the script name are handed unchanged to the installed package's click CLI (`boltz.main.cli`), so the run is the
one `boltz predict ...` would do. Instrumentation (outside the timed numerics, it only reads clocks):
  * `Boltz2.predict_step` wrapped: torch.mps.synchronize() before and after, perf_counter span per call -> model-call time M,
    split into structure / affinity by `self.affinity_prediction`; one span per record (the T400 estimate uses them);
  * `boltz.main.process_inputs` wrapped: input processing + MSA parsing + featurisation prep seconds;
  * `Boltz2.load_from_checkpoint` wrapped: checkpoint load seconds (structure, then affinity).
The JSON also carries the python start epoch, versions and the argv, so a row can be traced to its environment.
"""
import atexit
import json
import os
import sys
import time

T_PROC = time.time()
T_IMPORT0 = time.perf_counter()

import torch  # noqa: E402

try:
    import boltz  # noqa: E402
    import boltz.main as bm  # noqa: E402
    from boltz.model.models.boltz2 import Boltz2  # noqa: E402
except Exception as e:  # pragma: no cover
    sys.stderr.write(f"up_predict: cannot import boltz: {e!r}\n")
    raise

REC = {"python_start_epoch": T_PROC, "argv": sys.argv[1:], "boltz_file": boltz.__file__, "torch": torch.__version__,
       "calls": [], "process_inputs_s": None, "loads": [], "import_s": None, "t_main_start": None}


def _sync():
    try:
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
    except Exception:  # noqa: BLE001
        pass


def _out_path():
    p = os.environ.get("UP_TIMINGS")
    if p:
        return p
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == "--out_dir" and i + 1 < len(argv):
            return os.path.join(argv[i + 1], "up_timings.json")
    return "up_timings.json"


def _write():
    calls = REC["calls"]
    for k in ("structure", "affinity"):
        sp = [c for c in calls if c["kind"] == k]
        REC[f"{k}_model_s"] = round(sum(c["t1"] - c["t0"] for c in sp), 4) if sp else None
        REC[f"{k}_calls"] = len(sp)
    REC["model_s"] = round(sum(c["t1"] - c["t0"] for c in calls), 4) if calls else None
    REC["t_end_epoch"] = time.time()
    try:
        import importlib.metadata as md
        for dist in ("boltz", "boltz-community"):  # the community fork's distribution is named boltz-community (package boltz)
            try:
                REC["boltz_version"] = f"{dist} {md.version(dist)}"
                break
            except md.PackageNotFoundError:
                continue
        else:
            REC["boltz_version"] = getattr(boltz, "__version__", None)
    except Exception:  # noqa: BLE001
        REC["boltz_version"] = getattr(boltz, "__version__", None)
    p = _out_path()
    os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    with open(p, "w") as f:
        json.dump(REC, f, indent=1)


atexit.register(_write)

_orig_step = Boltz2.predict_step


def _step(self, *a, **k):
    _sync()
    t0 = time.perf_counter()
    out = _orig_step(self, *a, **k)
    _sync()
    t1 = time.perf_counter()
    try:
        rid = a[0]["record"][0].id
    except Exception:  # noqa: BLE001
        rid = None
    REC["calls"].append({"kind": "affinity" if getattr(self, "affinity_prediction", False) else "structure", "record": rid,
                         "t0": t0, "t1": t1, "exception": bool(isinstance(out, dict) and out.get("exception"))})
    return out


Boltz2.predict_step = _step

_orig_proc = bm.process_inputs


def _proc(*a, **k):
    t0 = time.perf_counter()
    try:
        return _orig_proc(*a, **k)
    finally:
        REC["process_inputs_s"] = round(time.perf_counter() - t0, 3)


bm.process_inputs = _proc

_orig_load = Boltz2.load_from_checkpoint  # already bound to the class (Lightning's restricted classmethod)


def _load(*a, **k):
    t0 = time.perf_counter()
    try:
        return _orig_load(*a, **k)
    finally:
        REC["loads"].append(round(time.perf_counter() - t0, 3))


Boltz2.load_from_checkpoint = staticmethod(_load)

if __name__ == "__main__":
    REC["import_s"] = round(time.perf_counter() - T_IMPORT0, 3)
    REC["t_main_start"] = time.perf_counter()
    bm.cli.main(args=sys.argv[1:], prog_name="boltz", standalone_mode=False)
