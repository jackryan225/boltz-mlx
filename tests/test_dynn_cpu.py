"""R8-coldN DYNN, CPU only (nothing is compiled or dispatched): mx.fast.metal_kernel is replaced by a recorder, every
DYNN kernel wrapper (tools/prof/metal_check.dynn_builds: tri-att NAX core, diffusion NAX attention, linear_addln /
tri-mult tails (+ tri bias), register transition, add_ln) is built at several token counts N, and
* DYNN on: the instantiations (kernel name, template args, source, header) are the same set for every N (one Metal
  library per kernel variant, whatever the structure size);
* DYNN off: they differ per N (the shipped static-size kernels: the cold-compile cost this round removes);
* each DYNN source is its static twin plus the run-time size declarations only, and only size parameters left the
  template (same tile shapes / loops / accumulation order -> bitwise; the GPU check is tests/test_dynn_gpu.py).
"""

import hashlib
import sys
from pathlib import Path

import mlx.core as mx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "prof"))

SIZE_NAMES = {"NN", "I_DIM", "J_DIM", "KP"}


def _record(monkeypatch, build):
    import metal_check
    from boltz_mlx import kernels, kernels_nax, kernels_nax2, kernels_nax3

    rec = []

    def fake(name, input_names, output_names, source, header="", **kw):
        def call(inputs, template=None, output_shapes=(), output_dtypes=(), **ckw):
            assert len(inputs) == len(input_names), (name, len(inputs), input_names)
            rec.append({"name": name, "template": tuple((str(k), str(v)) for k, v in (template or [])),
                        "source": source, "header": header, "inputs": tuple(input_names)})
            return [mx.zeros(s, d) for s, d in zip(output_shapes, output_dtypes)]

        return call

    for m in (kernels, kernels_nax, kernels_nax2, kernels_nax3):
        monkeypatch.setattr(m, "_kernels", {})
    monkeypatch.setattr(mx.fast, "metal_kernel", fake)
    build()
    assert metal_check  # noqa
    return rec


def _key(r):
    h = hashlib.sha1((r["source"] + "\x00" + r["header"]).encode()).hexdigest()[:12]
    return r["name"], r["template"], h


def _builds(monkeypatch, N, on):
    import metal_check

    out = []
    with metal_check.dynn(on):
        for label, build in metal_check.dynn_builds(N):
            recs = _record(monkeypatch, build)
            assert recs, label
            out.append((label, recs))
    return out


def test_dynn_one_instantiation_for_every_n(monkeypatch):
    sets = {}
    for N in (40, 72, 117):
        b = _builds(monkeypatch, N, True)
        sets[N] = {_key(r) for _, recs in b for r in recs}
        for _, recs in b:
            for r in recs:
                t = dict(r["template"])
                assert not {"NN", "J_DIM", "KP"} & set(t) and t.get("I_DIM", "1") == "1", r["name"]
    assert sets[40] == sets[72] == sets[117]
    print(f"DYNN on: {len(sets[40])} instantiations, identical for N = 40 / 72 / 117")


def test_static_instantiates_per_n(monkeypatch):
    a = {_key(r) for _, recs in _builds(monkeypatch, 40, False) for r in recs}
    b = {_key(r) for _, recs in _builds(monkeypatch, 72, False) for r in recs}
    new = b - a
    assert new, "the static-size kernels should instantiate per N"
    print(f"DYNN off: {len(new)} of {len(b)} instantiations are new at N=72 after N=40")


@pytest.mark.parametrize("N", [40, 117])
def test_dyn_source_is_static_plus_size_locals(monkeypatch, N):
    st = _builds(monkeypatch, N, False)
    dy = _builds(monkeypatch, N, True)
    assert [lab for lab, _ in st] == [lab for lab, _ in dy]
    for (lab, rs), (_, rd) in zip(st, dy):
        assert len(rs) == len(rd), lab
        for s, d in zip(rs, rd):
            assert d["name"] == s["name"] + "_dyn", (lab, s["name"], d["name"])
            assert d["header"] == s["header"], lab
            # only size parameters left the template; the rest is unchanged and in the same order
            st_t = [kv for kv in s["template"] if kv[0] not in SIZE_NAMES or kv == ("I_DIM", "1")]
            assert list(d["template"]) == st_t, lab
            gone = {k for k, _ in s["template"]} - {k for k, _ in d["template"]}
            assert gone <= SIZE_NAMES, (lab, gone)
            # inputs: the static list (+ the size buffer, or the row count buffer reused / renamed)
            assert d["inputs"][: len(s["inputs"]) - 1] == s["inputs"][:-1] or d["inputs"][:len(s["inputs"])] == \
                s["inputs"], lab
            # source: the static source plus the run-time size declarations (add_ln: + the renamed row count)
            dl = [ln for ln in d["source"].splitlines()
                  if not ln.strip().startswith(("const int I_DIM", "const int J_DIM", "const int NN",
                                                "const uint n_rows_v"))]
            assert "\n".join(dl).replace("n_rows_v", "n_rows") == s["source"].rstrip("\n") or \
                "\n".join(dl) + "\n" == s["source"] or "\n".join(dl) == s["source"], lab
            uses = any(n in s["source"] for n in ("NN", "J_DIM", "KP"))  # rtrans: never reads NN (template only)
            assert len(d["source"].splitlines()) - len(dl) in ((1, 2) if uses else (0,)), lab
