"""Round-2 L3 validation runner: one jsonl row per (case, seed, mode).

Emits the row schema documented in tools/validation/GATES.md, which tools/validation/gates.py consumes.

  python tools/validation/run_l3.py --set v36 --seeds 0 1 2 3 4 --mode strict --steps 200 --out logs/l3/v36_strict.jsonl
  python tools/validation/run_l3.py --set fep --targets tyk2 cdk2 --seeds 0 1 2 --mode turbo --steps 200 \
         --affinity-steps 200 --out logs/l3/fep_turbo.jsonl --max-minutes 120

Sets
  v36       the 36 V36 chains (validation/manifest_v36.json); one structure pass per (chain, seed), scored (CA-lDDT, TM) against
            the experimental structure.
  parity    the 8 parity cases (cases/*.yaml; no experimental structure): protein cases give metric-only rows (ptm, plddt, nan/inf,
            guard counts, model_cif for the L2 coordinate spread); lig_aff runs the two-phase affinity flow; ex_ligand has no
            affinity property (structure phase only).
  fep       FEP+ ligand complexes (data/external/fep/yaml/<target>/<id>.yaml; --targets cdk2 tyk2 jnk1 p38 bace1).
  thrombin  the 11 thrombin screen smoke-test ligands (not gated).      chymase  the 17 CASP16 chymase complexes (internal only).

Mode = an environment preset (MODES below, extend with --modes-file) applied by re-exec before MLX initialises: every BOLTZ_MLX_* /
MLX_ENABLE_TF32 variable of the calling shell is dropped first (except BOLTZ_MLX_CACHE_GB), then the preset is applied.

Sampler = a named preset (boltz_mlx.sampler.SAMPLER_PRESETS; `--list-samplers`) composing with any mode: `--mode f16 --sampler s25`.
A preset sets the structure pass (steps, rho, step_scale, euler|heun) and/or the affinity pass (aff100/aff50/aff25); '+' composes
one of each (`--sampler s50+aff25`). A preset's steps replace --steps / --affinity-steps (an explicit conflicting value is an error).
Rows of a sampler run are labelled `mode = "<mode>+<sampler>"` (e.g. "strict+s25", "f16+aff50"), so gates.py takes them as the
candidate arm (`--candidate strict+s25 --strict strict --cand-steps 25 --strict-steps 200 --candidate-type sampler`) even when they
share a jsonl with strict rows; `precision_mode`, `sampler`, `sampler_structure` / `sampler_affinity` ({steps, rho, step_scale,
solver, heun_euler_first, nfe, gamma_0, gamma_min, noise_scale}) are recorded on every row. Output dirs are keyed by what was computed
(`<mode>_s<steps>[_<sampler tag>]`, `aff_a<N>[_<tag>]`; tag '' = upstream schedule and Euler), so e.g. an affinity-only preset reuses
the finished strict@200 structure passes of the same build.

Protein cases (V36, parity proteins), per (case, seed): features = feature_cache.load(case, seed) (per-record seeded featurizer, shared by
every mode); noise = record_noise(seed, case, "structure", 1); conf model, `--recycling` recycles (3), `--steps` sampling steps;
the mmCIF is written with the upstream BoltzWriter to <work>/<mode>_s<steps>/seed<seed>/<case>/.

Ligand cases (fep / thrombin / chymase / lig_aff), per (target, seed) batch, in TWO PHASES so only one checkpoint is resident:
  1. structure phase (conf model, every ligand of the batch): features feature_cache(id, seed), noise record_noise(seed, id, "structure", 1),
     BoltzWriter -> <work>/<mode>_s<steps>/seed<seed>/<id>/{<id>_model_0.cif, pre_affinity_<id>.npz, confidence_*.json, ...};
  2. affinity phase (aff model): affinity features from THIS run's predictions dir
     (feature_cache.load(id, seed, affinity=True, pred_dir=<seed dir>, tag="<mode>_s<steps>")), noise record_noise(seed, id, "affinity", 5),
     `--affinity-recycling` (5) recycles, `--affinity-steps` steps, 5 diffusion samples, BoltzAffinityWriter -> <seed dir>/aff_a<N>/<id>/affinity_<id>.json.
Which pose feeds the affinity pass: the structure-phase pose `<id>_model_0.cif` (= rank-0 model; the same coordinates are saved as
pre_affinity_<id>.npz) defines the affinity CROP (tokens around the ligand); the affinity model then re-samples its own 5 structures on that crop
and its highest-iptm sample's coordinates feed the affinity heads. `pose_cif` in the row is the structure-phase mmCIF; `affinity_best_sample` and
`affinity_iptm` record the affinity-pass sample choice.

MW correction (upstream Boltz2.forward, `--affinity_mw_correction`; off in the model default): applied to the ensemble mean value
  value_mw_corrected = 1.03525938 * affinity_pred_value - 0.59992683 * MW**0.3 + 2.83288489,  MW = Record.affinity.mw (meta "affinity_mw")
`affinity_pred_value` is the MW-off value (model default); per-member values are never corrected.

Resume: rows already in --out (key = case, seed, mode, steps, affinity_steps) are skipped; each row is flushed as it completes; per-item
sidecar json files (l3_struct.json / l3_aff.json) let a killed run reuse finished structure passes (same boltz_mlx tree + steps + recycling)
and let --collect-only rebuild rows from existing output dirs without a model. --max-minutes stops cleanly at an item boundary.
Run one GPU job at a time; --device cpu is for plumbing smoke tests only.

Data root (`--data-root DIR`, default data/ (or $BOLTZ_MLX_DATA)): where per-case processed inputs (<DIR>/<id>/processed, written once
by upstream process_inputs), the feature cache (<DIR>/cache/features, tools/feature_cache.py) and the default --work
(<DIR>/validation/runs_l3) live. A new DIR re-processes every case it runs from its YAML, so the run never reads or writes the
processed inputs, feature caches or run dirs of the default root (e.g. a re-check of the conformer-seed fix on freshly embedded
ligands). Reference files (experimental structures, <default root>/validation/cif) are always read from the default root.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # the editable install points at the main checkout: always run this tree's boltz_mlx
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "validation"))

DEFAULT_DATA = Path(os.environ.get("BOLTZ_MLX_DATA", Path(__file__).resolve().parents[2] / "data")).expanduser()
REF_DATA = DEFAULT_DATA  # reference files (validation/cif): never moved by --data-root
DATA = DEFAULT_DATA  # per-case processed inputs, feature cache and default --work: set_data_root (--data-root)
FEP = Path(os.environ.get("BOLTZ_MLX_EXT_DATA", Path(__file__).resolve().parents[2] / "data" / "external")).expanduser() / "fep"
CASP = Path(os.environ.get("BOLTZ_MLX_EXT_DATA", Path(__file__).resolve().parents[2] / "data" / "external")).expanduser() / "casp16" / "yaml_labels"
CACHE = Path(os.environ.get("BOLTZ_MLX_CACHE", Path.home() / ".boltz")).expanduser()
WORK = DATA / "validation" / "runs_l3"

LARGE_CHAIN_MIN = 380
CHILD_ENV = "BOLTZ_L3_CHILD"

# Mode = environment preset (applied by re-exec before MLX initialises). Extend: add a row here or pass --modes-file (json {name: {VAR: value}}).
# Future modes (f16, f16k, ...) only add environment variables.
MODES: dict[str, dict[str, str]] = {
    "strict": {},
    "fast": {"BOLTZ_MLX_DIFF_DTYPE": "fp16"},
    "turbo": {"BOLTZ_MLX_DIFF_DTYPE": "fp16", "BOLTZ_MLX_TF32": "1"},
    # validation: residual guards (per-recycle z/s check + post-trunk count, written to rows). Per-op mode is for L1
    # only: its lazy counts are flushed only in the trunk, so over the confidence/affinity passes they retain every
    # guarded tensor (2026-10-28 02:15: f16 thrombin smoke grew swap to 39 GB).
    "f16": {"BOLTZ_MLX_PRECISION": "f16"},
}
ENV_PASSTHROUGH = {"BOLTZ_MLX_CACHE_GB", "BOLTZ_MLX_NAX", "BOLTZ_MLX_KERNEL_PATH"}  # BOLTZ_MLX_* variables kept from the caller's environment
TARGET_ALIAS = {"bace": "bace1"}  # labels.csv target -> gates target name
FEP_TARGETS = ("cdk2", "tyk2", "jnk1", "p38", "bace")
PARITY_CASES = ("prot117", "prot400", "msa136", "msa384", "lig_aff", "ex_cyclic", "ex_multimer", "ex_ligand")
SETS = ("v36", "v66", "fep", "parity", "chymase", "thrombin")


# =============================================================================================================== pure helpers


def load_modes(modes_file: str | os.PathLike | None = None) -> dict[str, dict[str, str]]:
    modes = {k: dict(v) for k, v in MODES.items()}
    if modes_file:
        extra = json.loads(Path(modes_file).read_text())
        for k, v in extra.items():
            if not isinstance(v, dict):
                raise ValueError(f"modes file: {k!r} must map to an object of environment variables")
            modes[k] = {str(a): str(b) for a, b in v.items()}
    return modes


def mode_environ(base: dict, preset: dict) -> dict:
    """Environment for a mode: the caller's env without MLX numerics switches, then the preset."""
    env = {k: v for k, v in base.items()
           if not ((k.startswith("BOLTZ_MLX_") and k not in ENV_PASSTHROUGH) or k == "MLX_ENABLE_TF32")}
    env.update(preset)
    return env


def run_name(mode: str, steps: int, run_tag: str = "", sampler_tag: str = "") -> str:
    """Structure-pass run dir: precision mode + steps + sampler tag ('' = upstream schedule / Euler) + optional run tag."""
    return f"{mode}_s{int(steps)}" + (f"_{sampler_tag}" if sampler_tag else "") + (f"_{run_tag}" if run_tag else "")


def mode_label(mode: str, sampler: str | None) -> str:
    """Row `mode` field: the precision mode, plus '+<sampler preset>' for sampler runs (the gates' arm name)."""
    return f"{mode}+{sampler}" if sampler else mode


def resolve_run_samplers(sampler: str | None, steps: int | None, affinity_steps: int | None):
    """(structure SamplerConfig, affinity SamplerConfig) for a run; a preset's steps replace --steps / --affinity-steps."""
    from boltz_mlx.sampler import SamplerConfig, resolve_sampler

    pre = resolve_sampler(sampler) if sampler else None
    st = pre.structure if pre else None
    af = pre.affinity if pre else None
    if st is not None and steps is not None and int(steps) != st.steps:
        raise ValueError(f"--sampler {sampler} sets {st.steps} structure steps; --steps {steps} conflicts")
    if af is not None and affinity_steps is not None and int(affinity_steps) != af.steps:
        raise ValueError(f"--sampler {sampler} sets {af.steps} affinity steps; --affinity-steps {affinity_steps} conflicts")
    st = st or SamplerConfig(200 if steps is None else int(steps))
    af = af or SamplerConfig(200 if affinity_steps is None else int(affinity_steps))
    return st, af


def _sampler_desc(info: dict | None, steps) -> dict | None:
    """Sampler description from a sidecar; sidecars written before sampler options ran the upstream sampler."""
    if info is None or steps is None:
        return None
    if info.get("sampler_cfg"):
        return info["sampler_cfg"]
    from boltz_mlx.sampler import SamplerConfig

    return SamplerConfig(int(steps)).describe()


def row_key(r: dict) -> tuple:
    """Resume key. affinity_steps distinguishes affinity-sampler variants (same structure pass)."""
    a = r.get("affinity_steps")
    return (r["case"], int(r["seed"]), r["mode"], None if r.get("steps") is None else int(r["steps"]), None if a is None else int(a))


def load_done(path: str | os.PathLike) -> set:
    done = set()
    p = Path(path)
    if not p.exists():
        return done
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue  # torn last line of a killed run
        if "case" in r and "seed" in r and "mode" in r and not r.get("type"):
            done.add(row_key(r))
    return done


def _jsonable(o):
    import numpy as np

    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not json serialisable: {type(o)}")


def append_row(path: str | os.PathLike, row: dict) -> None:
    """Append one json line and flush it to disk (rows are written as they complete)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(row, default=_jsonable) + "\n")
        f.flush()
        os.fsync(f.fileno())


def mw_correct(value: float, mw: float) -> float:
    """Upstream Boltz2.forward affinity_mw_correction on the ensemble value (constants from boltz_mlx.modules.affinity)."""
    from boltz_mlx.modules.affinity import MW_BIAS, MW_MODEL_COEF, MW_MW_COEF

    return float(MW_MODEL_COEF * float(value) + MW_MW_COEF * float(mw) ** 0.3 + MW_BIAS)


def crop_hash(asym_id, residue_index, token_pad_mask=None) -> str:
    """sha256[:12] of the sorted (asym_id, residue_index) pairs of the (non-padded) tokens of the affinity crop (the residue set)."""
    import numpy as np

    a = np.asarray(asym_id).reshape(-1)
    r = np.asarray(residue_index).reshape(-1)
    if token_pad_mask is not None:
        keep = np.asarray(token_pad_mask).reshape(-1) > 0
        a, r = a[keep], r[keep]
    pairs = sorted({(int(x), int(y)) for x, y in zip(a, r)})
    return hashlib.sha256(repr(pairs).encode()).hexdigest()[:12]


def nonfinite_flags(arrays) -> tuple[bool, bool]:
    """(any NaN, any inf) over float arrays."""
    import numpy as np

    nan = inf = False
    for a in arrays:
        if a is None:
            continue
        a = np.asarray(a)
        if a.dtype.kind != "f":
            continue
        nan |= bool(np.isnan(a).any())
        inf |= bool(np.isinf(a).any())
    return nan, inf


def to_counts(gc):
    """Guard counts (mx arrays / numbers / dict of them) -> int or {name: int}; None if the model returned none."""
    import numpy as np

    if gc is None:
        return None
    if isinstance(gc, dict):
        return {str(k): int(np.array(v).sum()) for k, v in gc.items()}
    return int(np.array(gc).sum())


def merge_counts(**named):
    """Combine per-phase guard counts into the flat {name: int} dict the gates read; None if no phase had any."""
    out = {}
    for phase, gc in named.items():
        if gc is None:
            continue
        if isinstance(gc, dict):
            for k, v in gc.items():
                out[f"{phase}.{k}"] = int(v)
        else:
            out[phase] = int(gc)
    return out or None


def scalar(out: dict, key: str):
    import numpy as np

    if key not in out or out[key] is None:
        return None
    return float(np.array(out[key]).astype(np.float64).reshape(-1)[0])


def fingerprint_for(pkg: str, mode: str, path: Path | None = None) -> str:
    """tools/fingerprint.py fingerprint recorded for this boltz_mlx tree hash and mode (logs/fingerprints.jsonl), else 'none'."""
    fp = "none"
    p = path or ROOT / "logs" / "fingerprints.jsonl"
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("pkg_tree") == pkg and r.get("mode") == mode:
                fp = r.get("fingerprint", fp)
    return fp


def build_identity() -> dict:
    import fingerprint as fpm

    pkg = fpm.pkg_tree()
    git = fpm.git_hash()
    return {"pkg_tree": pkg + ("-dirty" if git.endswith("-dirty") else ""), "git": git}


# ================================================================================================================ case model


@dataclass
class Case:
    id: str
    kind: str                      # "protein" | "ligand"
    target: str
    yaml: Path | None = None
    set: str = ""
    # protein
    exp_cif: Path | None = None
    exp_chain: str | None = None
    n_res: int | None = None
    # ligand
    smiles: str | None = None
    exp_value: float | None = None
    ligand_chain: str = "L"
    has_affinity: bool = False
    msa: str = "empty"
    extra: dict = field(default_factory=dict)

    @property
    def data_dir(self) -> Path:
        return DATA / self.id


def yaml_info(path: Path | None) -> dict:
    """msa label, ligand chain/smiles and affinity flag of a Boltz input YAML."""
    import yaml

    info = {"msa": "empty", "has_affinity": False, "ligand_chain": None, "smiles": None}
    if path is None or not Path(path).exists():
        return info
    d = yaml.safe_load(Path(path).read_text())
    msas = [str(s["protein"].get("msa", "none")) for s in d.get("sequences", []) if "protein" in s]
    if msas and all(m == "empty" for m in msas):
        info["msa"] = "empty"
    elif msas and all(m not in ("empty", "none") for m in msas):
        info["msa"] = "a3m"
    elif msas:
        info["msa"] = "mixed"
    ligs = [s["ligand"] for s in d.get("sequences", []) if "ligand" in s]
    binder = None
    for p in d.get("properties", []) or []:
        if "affinity" in p:
            binder = p["affinity"].get("binder")
            info["has_affinity"] = True
    chosen = next((l for l in ligs if l.get("id") == binder), ligs[-1] if ligs else None)
    if chosen is not None:
        info["ligand_chain"] = chosen.get("id")
        info["smiles"] = chosen.get("smiles")
    return info


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def resolve_cases(set_name: str, targets=None, cases=None) -> list[Case]:
    if set_name not in SETS:
        raise ValueError(f"unknown set {set_name!r}; one of {SETS}")
    out: list[Case] = []
    if set_name in ("v36", "v66"):
        # v66 = the 30-chain extension only (PREREG deviation 13); L3-P on "V66" pools v36 + v66 rows
        man = json.loads((ROOT / "validation" / ("manifest_v36.json" if set_name == "v36" else "manifest_v66_ext.json")).read_text())
        for c in man["cases"]:
            y = ROOT / c["yaml"]
            out.append(Case(id=c["id"], kind="protein", target=c["id"], yaml=y, set=set_name,
                            exp_cif=REF_DATA / "validation" / "cif" / f"{c['id']}.cif", exp_chain=c.get("chain"),
                            n_res=int(c["length_seqres"]), msa=yaml_info(y)["msa"]))
    elif set_name == "parity":
        for cid in PARITY_CASES:
            y = ROOT / "cases" / f"{cid}.yaml"
            yi = yaml_info(y)
            if cid in ("lig_aff", "ex_ligand"):
                out.append(Case(id=cid, kind="ligand", target=cid, yaml=y, set="parity", smiles=yi["smiles"],
                                ligand_chain=yi["ligand_chain"] or "B", has_affinity=yi["has_affinity"], msa=yi["msa"]))
            else:
                out.append(Case(id=cid, kind="protein", target=cid, yaml=y, set="parity", msa=yi["msa"]))
    elif set_name in ("fep", "thrombin"):
        want = {"thrombin": ("thrombin",)}.get(set_name)
        if want is None:
            want = tuple({"bace1": "bace"}.get(t, t) for t in (targets or FEP_TARGETS))
            bad = [t for t in want if t not in FEP_TARGETS]
            if bad:
                raise ValueError(f"unknown FEP+ target(s) {bad}; choose from {FEP_TARGETS} (bace1 = bace)")
        rows = [r for r in _read_csv(FEP / "labels.csv") if r["target"] in want]
        for r in sorted(rows, key=lambda r: (want.index(r["target"]), r["id"])):
            y = FEP / "yaml" / r["target"] / f"{r['id']}.yaml"
            if not y.exists():
                continue
            yi = yaml_info(y)
            out.append(Case(id=r["id"], kind="ligand", target=TARGET_ALIAS.get(r["target"], r["target"]), yaml=y, set=set_name,
                            smiles=r["smiles"], exp_value=float(r["exp_dg_kcal_mol"]), ligand_chain=yi["ligand_chain"] or "L",
                            has_affinity=yi["has_affinity"], msa=yi["msa"]))
    elif set_name == "chymase":
        for r in sorted(_read_csv(CASP / "labels.csv"), key=lambda r: r["id"]):
            y = CASP / "yaml" / f"{r['id']}.yaml"
            yi = yaml_info(y)
            out.append(Case(id=r["id"], kind="ligand", target="chymase", yaml=y, set="chymase", smiles=r["smiles"],
                            exp_value=float(r["binding_affinity"]), ligand_chain=yi["ligand_chain"] or "L",
                            has_affinity=yi["has_affinity"], msa=yi["msa"]))
    if cases:
        keep = set(cases)
        missing = keep - {c.id for c in out}
        if missing:
            raise ValueError(f"cases not in set {set_name}: {sorted(missing)}")
        out = [c for c in out if c.id in keep]
    return out


# ================================================================================================================ paths + rows


def seed_dir(run_root: Path, seed: int) -> Path:
    return run_root / f"seed{int(seed)}"


def struct_sidecar(sdir: Path, case_id: str) -> Path:
    return sdir / case_id / "l3_struct.json"


def aff_dir(sdir: Path, affinity_steps: int, sampler_tag: str = "") -> Path:
    return sdir / (f"aff_a{int(affinity_steps)}" + (f"_{sampler_tag}" if sampler_tag else ""))


def aff_sidecar(sdir: Path, case_id: str, affinity_steps: int, sampler_tag: str = "") -> Path:
    return aff_dir(sdir, affinity_steps, sampler_tag) / case_id / "l3_aff.json"


def _read_json(p: Path):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, json.JSONDecodeError):
        return None


def struct_reusable(info: dict | None, case: Case, mode: str, steps: int, recycling: int, ident: dict, sampler_tag: str = "") -> bool:
    """A finished structure pass can be reused iff same mode/steps/sampler/recycling and the same clean boltz_mlx tree, and its files exist."""
    if not info or str(ident["pkg_tree"]).endswith("-dirty"):
        return False
    if (info.get("mode"), info.get("steps"), info.get("recycling"), info.get("pkg_tree")) != (mode, steps, recycling, ident["pkg_tree"]):
        return False
    if info.get("sampler_tag", "") != sampler_tag:
        return False
    if not Path(info.get("model_cif", "")).exists():
        return False
    if case.has_affinity and not Path(info.get("pre_affinity", "")).exists():
        return False
    return True


def hw_mode(mode: str, hw: dict | None) -> str:
    """Precision mode as it ran: f16 off the NAX kernel path (M1-M4, or BOLTZ_MLX_KERNEL_PATH) is 'f16-portable' /
    'f16-tensorops', distinct arms (not gated with the NAX build's f16 rows)."""
    path = (hw or {}).get("path", "nax")
    return f"f16-{path}" if mode == "f16" and path != "nax" else mode


def _common(case: Case, seed: int, mode: str, steps: int, info: dict, sampler: str | None = None) -> dict:
    mode = hw_mode(mode, info.get("hw"))
    return {
        "case": case.id, "target": case.target, "seed": int(seed), "mode": mode_label(mode, sampler), "steps": int(steps),
        "precision_mode": mode, "sampler": sampler, "sampler_structure": _sampler_desc(info, steps),
        "recycling": info.get("recycling"), "device": info.get("device"), "hw": info.get("hw"),
    }


def build_protein_row(case: Case, seed: int, mode: str, steps: int, info: dict, scorer=None, fp_path: Path | None = None,
                      sampler: str | None = None) -> dict:
    """Row from a structure sidecar (+ scoring vs the experimental structure when the case has one).

    `mode` is the precision mode (environment preset; fingerprint lookup); the row's `mode` field is mode_label(mode, sampler)."""
    n_res = case.n_res if case.n_res is not None else info.get("n_tokens")
    row = _common(case, seed, mode, steps, info, sampler)
    row.update({
        "n_res": n_res, "large_chain": bool(n_res is not None and n_res >= LARGE_CHAIN_MIN),
        "msa": case.msa,
    })
    if case.exp_cif is not None and scorer is not None:
        try:
            s = scorer(info["model_cif"], case.exp_cif, exp_chain=case.exp_chain)
        except Exception as ex:  # noqa: BLE001
            row["score_error"] = repr(ex)
        else:
            row.update(ca_lddt=s["lddt_ca"], tm=s["tm_score"], rmsd_ca=s["rmsd_ca"])
    for k_out, k_in in (("ptm", "ptm"), ("iptm", "iptm"), ("plddt", "complex_plddt")):
        if info.get(k_in) is not None:
            row[k_out] = info[k_in]
    row.update(nan=bool(info.get("nan", False)), inf=bool(info.get("inf", False)))
    if info.get("guard_counts") is not None:
        row["guard_counts"] = info["guard_counts"]
    row.update(fingerprint=fingerprint_for(info.get("pkg_tree", ""), mode, fp_path), pkg_tree=info.get("pkg_tree"), git=info.get("git"),
               env=info.get("env"), model_cif=info["model_cif"], seconds=info.get("seconds"))
    return row


def build_ligand_row(case: Case, seed: int, mode: str, steps: int, affinity_steps: int | None, sinfo: dict, ainfo: dict | None,
                     fp_path: Path | None = None, sampler: str | None = None) -> dict:
    """Row from the structure sidecar and (when the case has an affinity property) the affinity sidecar."""
    row = _common(case, seed, mode, steps, sinfo, sampler)
    row.update({"affinity_steps": None if ainfo is None else int(affinity_steps), "msa": case.msa,
                "sampler_affinity": None if ainfo is None else _sampler_desc(ainfo, affinity_steps)})
    if ainfo is not None:
        row.update({
            "affinity_pred_value": ainfo["affinity_pred_value"],
            "affinity_pred_value_mw_off": ainfo["affinity_pred_value"],
            "value_mw_corrected": ainfo["value_mw_corrected"],
            "mw_correction": False,
            "mw": ainfo.get("mw"),
            "affinity_probability_binary": ainfo["affinity_probability_binary"],
            "affinity_pred_value1": ainfo.get("affinity_pred_value1"), "affinity_pred_value2": ainfo.get("affinity_pred_value2"),
            "affinity_probability_binary1": ainfo.get("affinity_probability_binary1"),
            "affinity_probability_binary2": ainfo.get("affinity_probability_binary2"),
            "affinity_crop_hash": ainfo.get("crop_hash"), "affinity_crop_tokens": ainfo.get("crop_tokens"),
            "affinity_best_sample": ainfo.get("best_sample"), "affinity_iptm": ainfo.get("iptm_samples"),
            "affinity_recycling": ainfo.get("recycling"),
        })
        if case.exp_value is not None:
            row["exp_value"] = case.exp_value
    row.update({
        "pose_cif": sinfo["model_cif"], "ligand_chain": case.ligand_chain, "smiles": case.smiles, "n_tokens": sinfo.get("n_tokens"),
        "iptm": sinfo.get("iptm"), "ligand_iptm": sinfo.get("ligand_iptm"), "ptm": sinfo.get("ptm"),
        "complex_plddt": sinfo.get("complex_plddt"), "plddt": sinfo.get("complex_plddt"),
    })
    nan = bool(sinfo.get("nan", False)) or bool(ainfo and ainfo.get("nan", False))
    inf = bool(sinfo.get("inf", False)) or bool(ainfo and ainfo.get("inf", False))
    row.update(nan=nan, inf=inf)
    gc = merge_counts(structure=sinfo.get("guard_counts"), affinity=(ainfo or {}).get("guard_counts"))
    if gc is not None:
        row["guard_counts"] = gc
    sec_s, sec_a = sinfo.get("seconds"), (ainfo or {}).get("seconds")
    row.update(fingerprint=fingerprint_for(sinfo.get("pkg_tree", ""), mode, fp_path), pkg_tree=sinfo.get("pkg_tree"), git=sinfo.get("git"),
               env=sinfo.get("env"), seconds_structure=sec_s, seconds_affinity=sec_a,
               seconds=None if sec_s is None else round(sec_s + (sec_a or 0.0), 3))
    return row


def collect_rows(cases, seeds, mode, steps, affinity_steps, run_root: Path, scorer=None, fp_path: Path | None = None,
                 sampler: str | None = None, aff_tag: str = ""):
    """Rebuild rows from existing output dirs (sidecars). Yields (row | None, case, seed, why_not)."""
    for seed in seeds:
        sdir = seed_dir(run_root, seed)
        for c in cases:
            sinfo = _read_json(struct_sidecar(sdir, c.id))
            if sinfo is None:
                yield None, c, seed, "no structure sidecar"
                continue
            if c.kind == "protein":
                yield build_protein_row(c, seed, mode, steps, sinfo, scorer, fp_path, sampler), c, seed, ""
                continue
            if c.has_affinity:
                ainfo = _read_json(aff_sidecar(sdir, c.id, affinity_steps, aff_tag))
                if ainfo is None:
                    yield None, c, seed, "no affinity sidecar"
                    continue
                yield build_ligand_row(c, seed, mode, steps, affinity_steps, sinfo, ainfo, fp_path, sampler), c, seed, ""
            else:
                yield build_ligand_row(c, seed, mode, steps, None, sinfo, None, fp_path, sampler), c, seed, ""


def row_key_for(c: Case, seed: int, mode: str, steps: int, affinity_steps: int | None) -> tuple:
    """affinity_steps None = structure-only rows (--structure-only)."""
    aff = c.kind == "ligand" and c.has_affinity and affinity_steps is not None
    return (c.id, int(seed), mode, int(steps), int(affinity_steps) if aff else None)


# ================================================================================================================ model runs


def set_data_root(root: str | os.PathLike | None) -> Path:
    """--data-root: move the per-case processed inputs (Case.data_dir), the feature cache (feature_cache.DATA / CACHE_DIR)
    and the default --work (WORK) under ``root``; None keeps the default root. Returns the data root in use."""
    global DATA, WORK
    if root is None:
        return DATA
    import feature_cache

    root = Path(root).expanduser().resolve()
    DATA, WORK = root, root / "validation" / "runs_l3"
    feature_cache.DATA, feature_cache.CACHE_DIR = root, root / "cache" / "features"
    return root


def ensure_processed(c: Case) -> None:
    """Upstream process_inputs (CPU, mode independent) into data/<id>/processed, once."""
    if (c.data_dir / "processed" / "manifest.json").exists():
        return
    from boltz.main import process_inputs

    from boltz_mlx.seeding import install_conformer_seed

    install_conformer_seed()  # SMILES-ligand conformers independent of the cases processed before (seeding.py)
    from boltz_mlx.upfix import install_upstream_fixes

    install_upstream_fixes()
    c.data_dir.mkdir(parents=True, exist_ok=True)
    process_inputs(
        data=[c.yaml.resolve()], out_dir=c.data_dir, ccd_path=CACHE / "ccd.pkl", mol_dir=CACHE / "mols",
        msa_server_url="", msa_pairing_strategy="greedy", use_msa_server=False, boltz2=True, preprocessing_threads=1,
    )


class Budget:
    def __init__(self, minutes: float | None, clock=time.monotonic):
        self.clock = clock
        self.t0 = clock()
        self.limit = None if not minutes else float(minutes) * 60.0

    def exhausted(self) -> bool:
        return self.limit is not None and (self.clock() - self.t0) >= self.limit

    def elapsed_min(self) -> float:
        return (self.clock() - self.t0) / 60.0


class Runner:
    def __init__(self, a, cases, modes):
        self.a = a
        self.cases = cases
        self.modes = modes
        self.preset = modes[a.mode]
        self.label = mode_label(a.mode, a.sampler)
        self.s_tag, self.a_tag = a.s_cfg.tag(), a.a_cfg.tag()
        self.run_root = Path(a.work) / run_name(a.mode, a.steps, a.run_tag, self.s_tag)
        self.tag = run_name(a.mode, a.steps, a.run_tag, self.s_tag)
        self.budget = Budget(a.max_minutes)
        self.done = load_done(a.out)
        self.ident = build_identity()
        self.n_rows = 0
        self.n_err = 0
        self.stopped_early = False
        self.t_start = time.time()

    # ---- helpers
    def todo(self, c: Case, seed: int) -> bool:
        return row_key_for(c, seed, self.label, self.a.steps, self.a.key_aff) not in self.done

    def emit(self, row: dict) -> None:
        append_row(self.a.out, row)
        self.done.add(row_key(row))
        self.n_rows += 1
        bits = [f"{row['case']} s{row['seed']}"]
        for k in ("ca_lddt", "tm", "affinity_pred_value", "affinity_probability_binary", "iptm"):
            if isinstance(row.get(k), float):
                bits.append(f"{k}={row[k]:.4f}")
        bits.append(f"{row.get('seconds')}s")
        print("[row] " + " ".join(bits), flush=True)

    def fail(self, what: str, ex: BaseException) -> None:
        self.n_err += 1
        tb = traceback.format_exc()
        print(f"[error] {what}: {ex!r}\n{tb}", file=sys.stderr, flush=True)
        err = Path(str(self.a.out) + ".errors.jsonl")
        err.parent.mkdir(parents=True, exist_ok=True)
        with open(err, "a") as f:
            f.write(json.dumps({"what": what, "error": repr(ex), "mode": self.a.mode, "date": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
        if self.a.stop_on_error:
            raise ex

    def out_of_time(self) -> bool:
        if self.budget.exhausted():
            if not self.stopped_early:
                print(f"[budget] --max-minutes {self.a.max_minutes} reached ({self.budget.elapsed_min():.1f} min): stopping cleanly at an "
                      f"item boundary; rerun the same command to resume.", flush=True)
            self.stopped_early = True
            return True
        return False

    # ---- MLX side
    def init_mlx(self):
        import boltz_mlx  # noqa: F401  (TF32 env handling before MLX init)
        import mlx.core as mx

        if not Path(boltz_mlx.__file__).resolve().is_relative_to(ROOT):
            sys.exit(f"run_l3: boltz_mlx imported from {boltz_mlx.__file__}, not from this tree ({ROOT})")
        mx.set_default_device(mx.gpu if self.a.device == "gpu" else mx.cpu)
        mx.set_cache_limit(int(float(os.environ.get("BOLTZ_MLX_CACHE_GB", "4")) * 1e9))
        if self.a.device == "gpu":
            mx.set_memory_limit(int(self.a.mem_limit_gb * 1e9))
        self.mx = mx
        from boltz_mlx import kernels as K

        self.hw = K.device_tag()  # chip / architecture / nax flag, recorded in every sidecar and row
        eff = {k: v for k, v in sorted(os.environ.items()) if k.startswith("BOLTZ_MLX_") or k == "MLX_ENABLE_TF32"}
        print(f"[init] boltz_mlx={boltz_mlx.__file__} mlx={mx.__version__} device={self.a.device} mode={self.a.mode} "
              f"preset={self.preset} effective_env={eff} pkg_tree={self.ident['pkg_tree']} hw={self.hw}", flush=True)
        print(f"[init] sampler={self.a.sampler} label={self.label} structure={self.a.s_cfg.describe()} "
              f"affinity={self.a.a_cfg.describe()}", flush=True)

    def structure_pass(self, model, c: Case, seed: int, writer, sdir: Path) -> dict:
        """One conf-model prediction + BoltzWriter output + sidecar. Returns the sidecar dict."""
        import numpy as np
        import torch
        from boltz_mlx.cli import _structure_pred_dict
        from boltz_mlx.model import PredictArgs
        from boltz_mlx.sampler import record_noise
        import feature_cache

        mx = self.mx
        arrays, meta = feature_cache.load(c.id, seed)
        feats = {k: mx.array(v) for k, v in arrays.items()}
        batch = {k: torch.from_numpy(v) for k, v in arrays.items()}
        batch.update(meta)
        noise = record_noise(seed, c.id, "structure", 1)
        t0 = time.perf_counter()
        out = model(feats, PredictArgs.from_sampler(self.a.s_cfg, self.a.recycling, 1), noise)
        mx.eval(out["sample_atom_coords"])
        seconds = time.perf_counter() - t0
        nan, inf = nonfinite_flags([np.array(out[k]) for k in ("sample_atom_coords", "plddt", "pae") if k in out])
        for k in ("ptm", "iptm", "ligand_iptm", "complex_plddt"):
            if k in out:
                nan_k, inf_k = nonfinite_flags([np.array(out[k])])
                nan |= nan_k
                inf |= inf_k
        writer.write_on_batch_end(None, None, _structure_pred_dict(out, batch), None, batch, 0, 0)
        # the writer names outputs by the record id (the input YAML's stem), which can differ from the case id
        rec = meta.get("record")
        rid = getattr(rec[0], "id", c.id) if isinstance(rec, (list, tuple)) and rec else getattr(rec, "id", c.id)
        cdir = sdir / rid
        info = {
            "case": c.id, "seed": seed, "mode": self.a.mode, "steps": self.a.steps, "recycling": self.a.recycling,
            "sampler": self.a.sampler, "sampler_tag": self.s_tag, "sampler_cfg": self.a.s_cfg.describe(),
            "device": self.a.device, "hw": getattr(self, "hw", None), "env": self.preset, "pkg_tree": self.ident["pkg_tree"],
            "git": self.ident["git"], "n_tokens": int(arrays["token_pad_mask"].sum()), "n_atoms": int(arrays["atom_pad_mask"].sum()),
            "ptm": scalar(out, "ptm"), "iptm": scalar(out, "iptm"), "ligand_iptm": scalar(out, "ligand_iptm"),
            "complex_plddt": scalar(out, "complex_plddt"), "nan": nan, "inf": inf,
            "guard_counts": to_counts(out.get("guard_counts")), "seconds": round(seconds, 3),
            "model_cif": str(cdir / f"{rid}_model_0.cif"), "pre_affinity": str(cdir / f"pre_affinity_{rid}.npz"),
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if not Path(info["model_cif"]).exists():
            raise RuntimeError(f"BoltzWriter did not write {info['model_cif']}")
        (sdir / c.id).mkdir(parents=True, exist_ok=True)  # sidecar under the case id (resume lookup), outputs under rid
        (sdir / c.id / "l3_struct.json").write_text(json.dumps(info, indent=1, default=_jsonable))
        del out, feats
        return info

    def affinity_pass(self, model, c: Case, seed: int, sdir: Path, sinfo: dict, writer) -> dict:
        import numpy as np
        import torch
        from boltz_mlx.cli import _to_torch
        from boltz_mlx.model import PredictArgs
        from boltz_mlx.sampler import record_noise
        import feature_cache

        mx = self.mx
        arrays, meta = feature_cache.load(c.id, seed, affinity=True, pred_dir=sdir, tag=self.tag)
        feats = {k: mx.array(v) for k, v in arrays.items()}
        batch = {k: torch.from_numpy(v) for k, v in arrays.items()}
        batch.update(meta)
        mw = float(meta["affinity_mw"][0])
        noise = record_noise(seed, c.id, "affinity", 5)
        t0 = time.perf_counter()
        out = model(feats, PredictArgs.from_sampler(self.a.a_cfg, self.a.affinity_recycling, 5), noise, mw_correction=False)
        mx.eval(out["affinity_pred_value"], out["affinity_probability_binary"])
        seconds = time.perf_counter() - t0
        keys = ("affinity_pred_value", "affinity_probability_binary", "affinity_pred_value1", "affinity_probability_binary1",
                "affinity_pred_value2", "affinity_probability_binary2")
        vals = {k: scalar(out, k) for k in keys}
        iptm = np.array(out["iptm"]).astype(np.float64).reshape(-1)
        nan, inf = nonfinite_flags([np.array(out[k]) for k in ("sample_atom_coords", "plddt", "pae", "iptm", *keys) if k in out])
        pred = {"exception": False}
        for k in keys:
            pred[k] = _to_torch(out[k])
        writer.write_on_batch_end(None, None, pred, None, batch, 0, 0)
        info = {
            "case": c.id, "seed": seed, "mode": self.a.mode, "steps": self.a.steps, "affinity_steps": self.a.affinity_steps,
            "sampler": self.a.sampler, "sampler_tag": self.a_tag, "sampler_cfg": self.a.a_cfg.describe(),
            "recycling": self.a.affinity_recycling, "pkg_tree": self.ident["pkg_tree"], "struct_date": sinfo.get("date"),
            "hw": getattr(self, "hw", None),
            **vals, "mw": mw, "value_mw_corrected": mw_correct(vals["affinity_pred_value"], mw),
            "crop_hash": crop_hash(arrays["asym_id"], arrays["residue_index"], arrays.get("token_pad_mask")),
            "crop_tokens": int(arrays["token_pad_mask"].sum()),
            "best_sample": int(np.argmax(iptm)), "iptm_samples": [float(x) for x in iptm],
            "nan": nan, "inf": inf, "guard_counts": to_counts(out.get("guard_counts")), "seconds": round(seconds, 3),
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        adir = aff_sidecar(sdir, c.id, self.a.affinity_steps, self.a_tag)
        adir.parent.mkdir(parents=True, exist_ok=True)
        adir.write_text(json.dumps(info, indent=1, default=_jsonable))
        del out, feats
        return info

    def scorer(self):
        import score

        return score.score_files

    # ---- drivers
    def run_protein_seed(self, cases: list[Case], seed: int) -> None:
        from boltz.data.write.writer import BoltzWriter
        from boltz_mlx.model import load_model

        todo = [c for c in cases if self.todo(c, seed)]
        if not todo:
            return
        sdir = seed_dir(self.run_root, seed)
        model = None
        for c in todo:
            if self.out_of_time():
                break
            try:
                info = _read_json(struct_sidecar(sdir, c.id))
                if not struct_reusable(info, c, self.a.mode, self.a.steps, self.a.recycling, self.ident, self.s_tag):
                    ensure_processed(c)
                    if model is None:
                        t0 = time.perf_counter()
                        model = load_model("conf")
                        print(f"[load] conf model {time.perf_counter() - t0:.1f}s", flush=True)
                    writer = BoltzWriter(data_dir=c.data_dir / "processed" / "structures", output_dir=sdir, output_format="mmcif", boltz2=True)
                    info = self.structure_pass(model, c, seed, writer, sdir)
                self.emit(build_protein_row(c, seed, self.a.mode, self.a.steps, info, self.scorer(), sampler=self.a.sampler))
            except Exception as ex:  # noqa: BLE001
                self.fail(f"{c.id} seed {seed}", ex)
        del model
        self.mx.clear_cache()

    def run_ligand_batch(self, cases: list[Case], seed: int) -> None:
        """One (target, seed) batch: structure phase for all ligands, then affinity phase (one checkpoint resident at a time)."""
        from boltz.data.write.writer import BoltzAffinityWriter, BoltzWriter
        from boltz_mlx.model import load_model
        import feature_cache

        todo = [c for c in cases if self.todo(c, seed)]
        if not todo:
            return
        sdir = seed_dir(self.run_root, seed)
        # ---- phase 1: structure
        sinfos: dict[str, dict] = {}
        need = []
        for c in todo:
            info = _read_json(struct_sidecar(sdir, c.id))
            if struct_reusable(info, c, self.a.mode, self.a.steps, self.a.recycling, self.ident, self.s_tag):
                sinfos[c.id] = info
            else:
                need.append(c)
        if need and not self.out_of_time():
            model = None
            for c in need:
                if self.out_of_time():
                    break
                try:
                    ensure_processed(c)
                    if model is None:
                        t0 = time.perf_counter()
                        model = load_model("conf")
                        print(f"[load] conf model {time.perf_counter() - t0:.1f}s", flush=True)
                    for p in (feature_cache._path(c.id, seed, True, self.tag), feature_cache._path(c.id, seed, True, self.tag).with_suffix(".meta.pkl")):
                        p.unlink(missing_ok=True)  # affinity features depend on this pose: never reuse a stale crop
                    writer = BoltzWriter(data_dir=c.data_dir / "processed" / "structures", output_dir=sdir, output_format="mmcif", boltz2=True)
                    sinfos[c.id] = self.structure_pass(model, c, seed, writer, sdir)
                    print(f"[structure] {c.id} s{seed} {sinfos[c.id]['seconds']}s iptm={sinfos[c.id]['iptm']}", flush=True)
                except Exception as ex:  # noqa: BLE001
                    self.fail(f"structure {c.id} seed {seed}", ex)
            del model
            self.mx.clear_cache()
        # ligands without an affinity property (or --structure-only): the structure pass is the row
        for c in todo:
            if c.id in sinfos and (not c.has_affinity or self.a.structure_only):
                self.emit(build_ligand_row(c, seed, self.a.mode, self.a.steps, None, sinfos[c.id], None, sampler=self.a.sampler))
        # ---- phase 2: affinity
        aff_todo = [] if self.a.structure_only else [c for c in todo if c.has_affinity and c.id in sinfos]
        if aff_todo and not self.out_of_time():
            model = None
            writer = BoltzAffinityWriter(data_dir=DATA, output_dir=aff_dir(sdir, self.a.affinity_steps, self.a_tag))
            for c in aff_todo:
                if self.out_of_time():
                    break
                try:
                    ainfo = _read_json(aff_sidecar(sdir, c.id, self.a.affinity_steps, self.a_tag))
                    if not (ainfo and ainfo.get("pkg_tree") == self.ident["pkg_tree"] and not self.ident["pkg_tree"].endswith("-dirty")
                            and ainfo.get("recycling") == self.a.affinity_recycling
                            and ainfo.get("sampler_tag", "") == self.a_tag
                            and ainfo.get("struct_date") == sinfos[c.id].get("date")):
                        if model is None:
                            t0 = time.perf_counter()
                            model = load_model("aff")
                            print(f"[load] aff model {time.perf_counter() - t0:.1f}s", flush=True)
                        ainfo = self.affinity_pass(model, c, seed, sdir, sinfos[c.id], writer)
                    self.emit(build_ligand_row(c, seed, self.a.mode, self.a.steps, self.a.affinity_steps, sinfos[c.id], ainfo,
                                               sampler=self.a.sampler))
                except Exception as ex:  # noqa: BLE001
                    self.fail(f"affinity {c.id} seed {seed}", ex)
            del model
            self.mx.clear_cache()


# ======================================================================================================================== CLI


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=SETS, help="case set (required unless --list-samplers)")
    ap.add_argument("--targets", nargs="*", default=None, help="fep only: cdk2 tyk2 jnk1 p38 bace1 (default all five)")
    ap.add_argument("--cases", nargs="*", default=None, help="restrict to these case ids")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--mode", default="strict", help="environment preset name (strict|fast|turbo, or one from --modes-file)")
    ap.add_argument("--modes-file", default=None, help="json {mode: {ENV_VAR: value}} adding/overriding presets (future f16, f16k, ...)")
    ap.add_argument("--steps", type=int, default=None, help="structure-pass sampling steps (default 200, or the --sampler preset's)")
    ap.add_argument("--affinity-steps", type=int, default=None, help="affinity-pass sampling steps (default 200, or the --sampler preset's)")
    ap.add_argument("--sampler", default=None,
                    help="sampler preset (boltz_mlx.sampler.SAMPLER_PRESETS, see --list-samplers), composes with --mode; '+' combines a "
                         "structure and an affinity preset (s50+aff25). Rows get mode '<mode>+<sampler>'")
    ap.add_argument("--list-samplers", action="store_true", help="print the sampler presets and exit")
    ap.add_argument("--recycling", type=int, default=3, help="structure-pass recycling steps")
    ap.add_argument("--affinity-recycling", type=int, default=5)
    ap.add_argument("--out", type=Path, help="jsonl to append rows to (resumable; required unless --list-samplers)")
    ap.add_argument("--max-minutes", type=float, default=120.0, help="stop cleanly after this many minutes (0 = no limit)")
    ap.add_argument("--work", type=Path, default=None, help=f"parent of <mode>_s<steps>/seed<seed>/<case>/ (default <data root>/validation/runs_l3, "
                    f"i.e. {WORK})")
    ap.add_argument("--data-root", type=Path, default=None,
                    help=f"root of the per-case processed inputs, the feature cache and the default --work (default {DEFAULT_DATA}); "
                         "a new root re-processes every case from its YAML and never touches the default root's files")
    ap.add_argument("--run-tag", default="", help="suffix of the run dir / affinity-feature tag (use a new tag when the code under a mode name changes)")
    ap.add_argument("--collect-only", action="store_true", help="rebuild rows from existing output dirs (sidecars); no model")
    ap.add_argument("--prefeaturize", action="store_true", help="CPU only: process inputs and build the (case, seed) structure feature cache; no model")
    ap.add_argument("--dry-run", action="store_true", help="print the plan (items to run / already done) and exit")
    ap.add_argument("--device", choices=["gpu", "cpu"], default="gpu", help="cpu = plumbing smoke tests only")
    ap.add_argument("--mem-limit-gb", type=float, default=18.0)
    ap.add_argument("--allow-dirty", action="store_true", help="run with uncommitted changes in boltz_mlx/ (rows get pkg_tree '...-dirty', fingerprint none)")
    ap.add_argument("--stop-on-error", action="store_true")
    ap.add_argument("--structure-only", action="store_true",
                    help="ligand sets: structure pass only (rows with affinity_steps null; a later run with the affinity pass reuses "
                         "these structure passes). For sampler tuning stage A (PREREG deviation 17)")
    return ap


def reexec_with_mode(argv: list[str], env: dict) -> None:
    env = dict(env)
    env[CHILD_ENV] = "1"
    os.execve(sys.executable, [sys.executable, str(Path(__file__).resolve()), *argv], env)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    a = build_parser().parse_args(argv)
    set_data_root(a.data_root)
    if a.work is None:
        a.work = WORK
    if a.list_samplers:
        print(sampler_table())
        return 0
    if not a.set or a.out is None:
        sys.exit("run_l3: --set and --out are required")
    try:
        a.s_cfg, a.a_cfg = resolve_run_samplers(a.sampler, a.steps, a.affinity_steps)
    except ValueError as ex:
        sys.exit(f"run_l3: {ex}")
    a.steps, a.affinity_steps = a.s_cfg.steps, a.a_cfg.steps
    a.key_aff = None if a.structure_only else a.affinity_steps
    label = mode_label(a.mode, a.sampler)
    modes = load_modes(a.modes_file)
    if a.mode not in modes:
        sys.exit(f"unknown mode {a.mode!r}; known: {sorted(modes)} (add with --modes-file)")
    a.out = Path(a.out)
    cases = resolve_cases(a.set, a.targets, a.cases)
    if not cases:
        sys.exit("no cases selected")
    ident = build_identity()

    plan_only = a.collect_only or a.prefeaturize or a.dry_run
    if not plan_only and os.environ.get(CHILD_ENV) != "1":
        if ident["pkg_tree"].endswith("-dirty") and not a.allow_dirty:
            sys.exit("run_l3: boltz_mlx/ has uncommitted changes; commit first (or --allow-dirty for a smoke test)")
        reexec_with_mode(argv, mode_environ(os.environ, modes[a.mode]))  # does not return

    if a.collect_only:
        return collect(a, cases)
    if a.prefeaturize:
        return prefeaturize(a, cases)

    runner = Runner(a, cases, modes) if not a.dry_run else None
    seeds = list(dict.fromkeys(a.seeds))
    done = load_done(a.out)
    n_todo = sum(1 for s in seeds for c in cases if row_key_for(c, s, label, a.steps, a.key_aff) not in done)
    n_all = len(seeds) * len(cases)
    print(f"[plan] set={a.set} mode={a.mode} sampler={a.sampler} (rows: mode={label}) steps={a.steps} (NFE {a.s_cfg.nfe}) "
          f"affinity_steps={a.affinity_steps} (NFE {a.a_cfg.nfe}) seeds={seeds} cases={len(cases)}: "
          f"{n_todo} to run, {n_all - n_todo} already in {a.out}; data_root={DATA} work={a.work}", flush=True)
    if a.dry_run:
        for s in seeds:
            for c in cases:
                if row_key_for(c, s, label, a.steps, a.key_aff) not in done:
                    print(f"  todo: {c.id} seed {s} ({c.kind}, target {c.target})")
        return 0

    runner.init_mlx()
    # seeds outermost so a time-limited batch finishes whole seeds; ligand sets batch per (target, seed)
    for seed in seeds:
        if runner.out_of_time():
            break
        prot = [c for c in cases if c.kind == "protein"]
        if prot:
            runner.run_protein_seed(prot, seed)
        by_target: dict[str, list[Case]] = {}
        for c in cases:
            if c.kind == "ligand":
                by_target.setdefault(c.target, []).append(c)
        for tgt, group in by_target.items():
            if runner.out_of_time():
                break
            print(f"[batch] target {tgt} seed {seed}: {len(group)} ligands", flush=True)
            runner.run_ligand_batch(group, seed)

    done = load_done(a.out)
    remaining = sum(1 for s in seeds for c in cases if row_key_for(c, s, label, a.steps, a.key_aff) not in done)
    status = {"complete": remaining == 0, "remaining": remaining, "rows_written": runner.n_rows, "errors": runner.n_err,
              "stopped_by_budget": runner.stopped_early, "minutes": round(runner.budget.elapsed_min(), 2), "out": str(a.out),
              "date": time.strftime("%Y-%m-%d %H:%M:%S")}
    Path(str(a.out) + ".status.json").write_text(json.dumps(status, indent=1))
    print("[status] " + json.dumps(status), flush=True)
    return 1 if runner.n_err else 0


def sampler_table() -> str:
    """Human-readable table of the sampler presets (PREREG F candidates)."""
    from boltz_mlx.sampler import SAMPLER_PRESETS

    lines = [f"{'preset':<18} {'PREREG':<11} {'pass':<9} {'steps':>5} {'rho':>4} {'ss':>5} {'solver':<7} {'NFE':>4}  note"]
    for name, p in SAMPLER_PRESETS.items():
        for kind, cfg in (("structure", p.structure), ("affinity", p.affinity)):
            if cfg is not None:
                lines.append(f"{name:<18} {p.candidate:<11} {kind:<9} {cfg.steps:>5} {cfg.rho:>4g} {cfg.step_scale:>5g} {cfg.solver:<7} "
                             f"{cfg.nfe:>4}  {p.note}")
    lines.append("All presets: gamma_0 0.8 on steps whose next sigma > gamma_min 1.0, noise_scale 1.003, sigma 2560 -> 0.0016 A -> 0 "
                 "(upstream). Unset pass = unchanged (200 steps, Euler).")
    return "\n".join(lines)


def collect(a, cases) -> int:
    run_root = Path(a.work) / run_name(a.mode, a.steps, a.run_tag, a.s_cfg.tag())
    scorer = None
    if any(c.kind == "protein" and c.exp_cif is not None for c in cases):
        import score

        scorer = score.score_files
    done = load_done(a.out)
    added = skipped = missing = 0
    for row, c, seed, why in collect_rows(cases, list(dict.fromkeys(a.seeds)), a.mode, a.steps, a.affinity_steps, run_root, scorer,
                                          sampler=a.sampler, aff_tag=a.a_cfg.tag()):
        if row is None:
            missing += 1
            print(f"[collect] {c.id} seed {seed}: {why}")
            continue
        if row_key(row) in done:
            skipped += 1
            continue
        append_row(a.out, row)
        done.add(row_key(row))
        added += 1
    print(f"[collect] {added} rows added to {a.out}, {skipped} already present, {missing} not available")
    return 0


def prefeaturize(a, cases) -> int:
    """CPU-only: process_inputs + structure-feature cache for every (case, seed). Safe to run while the GPU is busy."""
    import feature_cache

    print(f"[prefeaturize] data_root={DATA} feature_cache={feature_cache.CACHE_DIR}", flush=True)
    for c in cases:
        ensure_processed(c)
        for s in dict.fromkeys(a.seeds):
            t0 = time.perf_counter()
            p = feature_cache.build(c.id, s)
            print(f"[prefeaturize] {c.id} seed {s} -> {p} ({time.perf_counter() - t0:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
