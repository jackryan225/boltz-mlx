"""One table from the macbench runs: MEASURED / ESTIMATE rows with the PUBLISHED rows beside them (stdlib only).

  python tools/macbench/summary.py [--logs logs/macbench] [--tags TAG ...] [--md OUT.md] [--json OUT.json]

Reads <logs>/<tag>/<arm>/<input>/seed<k>/run.json (tools/macbench/driver.py), <logs>/<tag>/t400_mlx/summary_t400cold.json (tools/bench_r2/
summary.py t400cold; the T400 cold-screen job script is not included) and <logs>/<tag>/t400up/*/run.json + t400up_decision.json (driver t400up). Labels:
  MEASURED  a run.json (or summary json) exists: median over the seeds that finished ok, min-max shown, n ok / n run
  ESTIMATE  computed from MEASURED numbers by a stated rule (the 199-ligand upstream wall, from the 5-ligand run)
  PUBLISHED read from a web page (tools/macbench/published.json, with URL); never MEASURED here, never in a ratio
Ratios (upstream up200 or boltz-community comm200 over an MLX arm) are MEASURED / MEASURED and labelled so; no ratio uses a PUBLISHED number.

The first table is the headline: ONE ROW PER CASE with ours at Standard (preset `screen`) and at f16-200, boltz-community on MPS (comm200 =
fork defaults incl. its MSA subsampling; comm200_full = --no_subsample_msa, where run), ChimeraX published (M2 Ultra, M1 Max: other chips,
Boltz 2.1.1, steps and MSA accounting unstated), and H100 upstream PyTorch from logs/h100 when that exists (it does not: not run).
The MSA arm is used for every case (the fork arm is MSA-only); the per-arm detail table follows.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARM_ORDER = ["up200", "stock200", "comm200", "comm200_full", "strict200", "f16_200", "f16_200_aff25", "screen", "f16_50_aff25", "f16_50"]
TOKENS = {"nim_8eil": 194, "nim_7qsj": 399, "nim_8cpk": 410, "nim_8gpp": 628, "prot117": 117, "8rf4": 129, "8rf4aff": 129, "lig_aff": 397, "prot400": 400, "ntca_aff": 516, "9moj": 660, "9b3h": 911}
ORDER = ["nim_8eil", "nim_7qsj", "nim_8cpk", "nim_8gpp", "prot117", "8rf4", "8rf4aff", "lig_aff", "prot400", "ntca_aff", "9moj", "9b3h"]


def case_of(inp: str) -> str:
    for s in ("_empty", "_msa"):
        if inp.endswith(s):
            return inp[: -len(s)]
    return inp


def load_runs(logs: Path, tags):
    runs = {}
    for td in sorted(p for p in logs.iterdir() if p.is_dir()):
        if tags and td.name not in tags:
            continue
        for rj in td.glob("*/*/seed*/run.json"):
            r = json.loads(rj.read_text())
            runs.setdefault((r["input"], r["arm"]), []).append(r)
    return runs


def fmt(v, d=1):
    return "-" if v is None else f"{v:.{d}f}"


def med(vals):
    vals = [v for v in vals if v is not None]
    return (st.median(vals), min(vals), max(vals)) if vals else (None, None, None)


def published_for(pub: dict, inp: str) -> str:
    c = case_of(inp)
    cx = pub["chimerax_runtimes"]["rows"]
    if c == "8rf4" and "8rf4" in cx:
        b = cx["8rf4"]["b2"]
        return f"M1 16 GB {b['M1_16GB']} / M1 Max {b['M1_Max_32GB']} / M2 Ultra {b['M2_Ultra_64GB']} min (Boltz 2.1.1, structure only)"
    if c == "8rf4aff":
        b = cx["8rf4"]["b2a_with_affinity"]
        return f"M1 16 GB {b['M1_16GB']} / M1 Max {b['M1_Max_32GB']} / M2 Ultra {b['M2_Ultra_64GB']} min (Boltz 2.1.1 with affinity)"
    if c in ("9moj", "9b3h"):
        b = cx[c]["b2"]
        return f"M1 16 GB {b['M1_16GB']} / M1 Max {b['M1_Max_32GB']} / M2 Ultra {b['M2_Ultra_64GB']} min (Boltz 2.1.1)"
    if c == "ntca_aff":
        m = pub["chimerax_pncc_workshop"]["minutes"]
        return f"M2 Ultra {m['Mac_Studio_M2_Ultra']} / M1 Max {m['MacBook_Pro_M1_Max']} / M1 {m['Mac_Mini_M1_16GB']} min (Boltz 2.2, structure + affinity)"
    if c == "prot117":
        return f"Apple M3 about {pub['moleculedesk']['seconds_M3']} s (MoleculeDesk, boltz-community fork, 'about 88 s', no run count)"
    return ""


def _cell(rs):
    """W median [min-max] (M median), n ok / n run, for one (input, arm) run list; None if no runs."""
    if not rs:
        return None
    ok = [r for r in rs if r.get("ok")]
    w, lo, hi = med([r["W_s"] for r in ok])
    m, _, _ = med([r.get("M_s") for r in ok])
    pk = max([r.get("peak_footprint_gb") or 0 for r in rs] or [0])
    if w is None:
        return f"FAILED 0 / {len(rs)} ok (see run.json)"
    return f"W {w:.1f} s [{lo:.1f}-{hi:.1f}], M {fmt(m, 1)} s, peak {pk:.1f} GB, n {len(ok)}/{len(rs)}", w


def chimerax_cells(pub: dict, case: str):
    """(M2 Ultra, M1 Max) PUBLISHED strings for a case, or ('-', '-')."""
    cx = pub["chimerax_runtimes"]["rows"]
    def one(v):
        return "-" if v is None else (f"{v * 60:.0f} s ({v} min)" if isinstance(v, (int, float)) else str(v))
    if case in ("9moj", "9b3h", "8rf4"):
        b = cx[case]["b2"]
        return one(b["M2_Ultra_64GB"]), one(b["M1_Max_32GB"])
    if case == "8rf4aff":
        b = cx["8rf4"]["b2a_with_affinity"]
        return one(b["M2_Ultra_64GB"]), one(b["M1_Max_32GB"])
    if case == "ntca_aff":
        m = pub["chimerax_pncc_workshop"]["minutes"]
        return one(m["Mac_Studio_M2_Ultra"]), one(m["MacBook_Pro_M1_Max"])
    return "-", "-"


def h100_cell(logs: Path, inp: str) -> str:
    """H100 upstream PyTorch from logs/h100 (sibling of the macbench logs dir) if that exists, else say it was not run."""
    d = logs.parent / "h100"
    if not d.is_dir():
        return "not run (no logs/h100)"
    ws = []
    for rj in d.glob("**/run.json"):
        try:
            r = json.loads(rj.read_text())
        except Exception:  # noqa: BLE001
            continue
        if r.get("input") == inp and r.get("ok") and r.get("W_s") is not None:
            ws.append(r["W_s"])
    return f"W {st.median(ws):.1f} s (n {len(ws)}, logs/h100)" if ws else "logs/h100 exists, no ok run for this input"


def headline_table(runs, pub, logs):
    """One row per case. Returns (markdown lines, json rows)."""
    L = ["| case | tokens | input | ours Standard (screen: f16, 20 + 25 steps) | ours f16-200 | boltz-community MPS fp32 (comm200, fork defaults) | boltz-community, all MSA rows (comm200_full) | comm200 / ours Standard, comm200 / ours f16-200 (MEASURED W ratio) | ChimeraX M2 Ultra (PUBLISHED) | ChimeraX M1 Max (PUBLISHED) | H100 upstream PyTorch |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    out = []
    cases = sorted({case_of(k[0]) for k in runs if k[1] in ("screen", "f16_200", "comm200", "comm200_full") and not k[0].startswith("nim_")},
                   key=lambda c: ORDER.index(c) if c in ORDER else 99)
    for c in cases:
        inp = next((i for i in (c + "_msa", c + "_empty", c) if any(k[0] == i for k in runs)), c + "_msa")
        cells = {a: _cell(runs.get((inp, a))) for a in ("screen", "f16_200", "comm200", "comm200_full")}
        txt = {a: ("not measured" if v is None else (v if isinstance(v, str) else v[0])) for a, v in cells.items()}
        wv = {a: (v[1] if isinstance(v, tuple) else None) for a, v in cells.items()}
        rat = " / ".join(f"{wv['comm200'] / wv[a]:.2f}x" if (wv["comm200"] and wv[a]) else "-" for a in ("screen", "f16_200"))
        m2, m1 = chimerax_cells(pub, c)
        L.append(f"| {c} | {TOKENS.get(c, '')} | {inp} | {txt['screen']} | {txt['f16_200']} | {txt['comm200']} | {txt['comm200_full']} | {rat} | {m2} | {m1} | {h100_cell(logs, inp)} |")
        out.append({"case": c, "input": inp, **{f"W_{a}_s": wv[a] for a in wv}, "chimerax_m2_ultra": m2, "chimerax_m1_max": m1, "h100": h100_cell(logs, inp)})
    return L, out


NIM_LIG = {"nim_8eil": "SIN (8 atoms)", "nim_7qsj": "celecoxib (26)", "nim_8cpk": "celecoxib (26)", "nim_8gpp": "BCT (4)"}
NIM_RES = {"nim_8eil": "186", "nim_7qsj": "373 (table: 375)", "nim_8cpk": "384", "nim_8gpp": "3 x 208 = 624 (table: 628)"}


def _nim_cell(rs):
    """(structure s, affinity s, W s, text) for the runs of an arm on a NIM case. One run: its values. Repeats (job nim-ours-rep): the MEDIAN of the
    ok runs with [min-max] and n ok / n run for structure s, affinity s and W. Failures are shown, never retried."""
    if not rs:
        return None, None, None, "not measured"
    ok = sorted([r for r in rs if r.get("ok")], key=lambda x: (x.get("repeat") or 0, x.get("seed", 0)))
    if not ok:
        r = rs[0]
        why = f"timeout after {r.get('timeout_s'):.0f} s" if r.get("timed_out") else f"rc {r.get('rc')}, cif {r.get('has_cif')}"
        return None, None, r.get("W_s"), f"FAILED ({why}); W {fmt(r.get('W_s'))} s, peak {fmt(r.get('peak_footprint_gb'))} GB"
    pk = max(r.get("peak_footprint_gb") or 0 for r in rs)
    if len(rs) == 1:
        r = ok[0]
        return r.get("M_structure_s"), r.get("M_affinity_s"), r["W_s"], (f"structure {fmt(r.get('M_structure_s'), 1)} s, affinity {fmt(r.get('M_affinity_s'), 1)} s, "
                                                                         f"W {fmt(r['W_s'])} s, peak {fmt(pk)} GB")
    def band(key, d=1):
        m, lo, hi = med([r.get(key) for r in ok])
        return m, (f"{fmt(m, d)} [{fmt(lo, d)}-{fmt(hi, d)}]")
    ms, ts = band("M_structure_s")
    ma, ta = band("M_affinity_s")
    mw, tw = band("W_s")
    return ms, ma, mw, f"structure {ts} s, affinity {ta} s, W {tw} s, median [min-max] of n {len(ok)}/{len(rs)}, peak {fmt(pk)} GB"


def nim_table(runs, pub):
    """NVIDIA-size cases: ours (f16_50, the fast preset) and boltz-community (comm200_full: its defaults, 200 steps, all MSA rows), one timed run
    each (seed 0), structure and affinity model time reported separately beside NVIDIA's published H100 numbers (PUBLISHED, settings undisclosed)."""
    nv = pub.get("nvidia_nim_h100", {}).get("seconds", {})
    L = ["| case (NIM id) | residues | tokens (with ligand) | affinity ligand | ours Fast: Standard preset `screen` (f16, 20 + 25 steps) or `f16_50` (MEASURED) | ours Full: `strict200` (fp32, 200 + 200 steps) (MEASURED) | boltz-community MPS fp32, 200 steps, all MSA rows (MEASURED) | community W / ours W (MEASURED) | NVIDIA H100 OSS structure / affinity s (PUBLISHED) | NVIDIA H100 TensorRT structure / affinity s (PUBLISHED) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    out = []
    for c in ("nim_8eil", "nim_7qsj", "nim_8cpk", "nim_8gpp"):
        inp = c + "_msa"
        # the Fast arm that was run: f16_200_aff25 (Fast's steps for a ligand with < 10 heavy atoms, 8eil / 8gpp) if present, else screen (the 20 + 25 Fast),
        # else f16_50_aff25 (the drafted 50 + 25 variant), else f16_50
        fast = next((x for x in ("f16_200_aff25", "screen", "f16_50_aff25") if runs.get((inp, x))), "f16_50")
        os_, oa, ow, ot = _nim_cell(runs.get((inp, fast)))
        _, _, _, ft = _nim_cell(runs.get((inp, "strict200")))
        cs, ca, cw, ct = _nim_cell(runs.get((inp, "comm200_full")))
        ratio = f"{cw / ow:.1f}x" if (cw and ow and os_ is not None and cs is not None) else "-"
        n = nv.get(c.replace("nim_", ""), {})
        oss = f"{n['structure']['oss']} / {n['affinity']['oss']}" if n else "-"
        trt = f"{n['structure']['trt']} / {n['affinity']['trt']}" if n else "-"
        L.append(f"| {c.replace('nim_', '')} | {NIM_RES[c]} | {TOKENS[c]} | {NIM_LIG[c]} | {ot} ({fast}) | {ft} | {ct} | {ratio} | {oss} | {trt} |")
        out.append({"case": c, "ours_fast": {"arm": fast, "structure_s": os_, "affinity_s": oa, "W_s": ow}, "community_200": {"structure_s": cs, "affinity_s": ca, "W_s": cw}, "nvidia": n})
    return L, out


def timing_table(runs, pub):
    L = ["| input | tokens | arm | n ok / run | W median s [min-max] | M median s (struct / aff) | load s | peak GB (max) | AC power | up200 W / arm W (MEASURED) | label |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    rows_json = []
    inputs = sorted({k[0] for k in runs}, key=lambda i: (ORDER.index(case_of(i)) if case_of(i) in ORDER else 99, i))
    for inp in inputs:
        up = [r for r in runs.get((inp, "up200"), []) if r.get("ok")]
        upW = st.median([r["W_s"] for r in up]) if up else None
        for arm in ARM_ORDER:
            rs = runs.get((inp, arm))
            if not rs:
                continue
            ok = [r for r in rs if r.get("ok")]
            w, wlo, whi = med([r["W_s"] for r in ok])
            m, _, _ = med([r.get("M_s") for r in ok])
            ms, _, _ = med([r.get("M_structure_s") for r in ok])
            ma, _, _ = med([r.get("M_affinity_s") for r in ok])
            ld = [x for r in ok for x in (r.get("load_s") or []) if x]
            pk = max([r.get("peak_footprint_gb") or 0 for r in rs] or [0])
            ac = f"{sum(1 for r in rs if r.get('all_on_ac'))}/{len(rs)}"
            ratio = f"{upW / w:.2f}x" if (upW and w and arm != "up200") else "-"
            wtxt = f"{fmt(w)} [{fmt(wlo)}-{fmt(whi)}]" if w is not None else "-"
            note = "" if len(ok) >= 3 else (" (n < 3: FAILED runs, see run.json)" if len(ok) < len(rs) else " (n < 3)")
            L.append(f"| {inp} | {TOKENS.get(case_of(inp), '')} | {arm} | {len(ok)} / {len(rs)} | {wtxt} | {fmt(m, 2)} ({fmt(ms, 2)} / {fmt(ma, 2)}) | "
                     f"{fmt(st.median(ld) if ld else None, 1)} | {fmt(pk)} | {ac} | {ratio} | MEASURED{note} |")
            rows_json.append({"input": inp, "arm": arm, "n_ok": len(ok), "n_run": len(rs), "W_median_s": w, "W_min_s": wlo, "W_max_s": whi,
                              "M_median_s": m, "peak_gb": pk, "label": "MEASURED"})
        p = published_for(pub, inp)
        if p and (inp.endswith("_msa") or case_of(inp) in ("prot117",) or not any(i == case_of(inp) + "_msa" for i in inputs)):
            L.append(f"| {inp} | {TOKENS.get(case_of(inp), '')} | published | - | {p} | | | | | | PUBLISHED (other chips; settings partly unstated; MSA accounting unstated) |")
    return L, rows_json


def t400_section(logs: Path, tags):
    L = ["| T400 run | what | wall | per ligand | peak footprint GB | label |", "|---|---|---|---|---|---|"]
    out = {}
    for td in sorted(p for p in logs.iterdir() if p.is_dir()):
        if tags and td.name not in tags:
            continue
        s = td / "t400_mlx" / "summary_t400cold.json"
        if s.exists():
            d = json.loads(s.read_text())
            L.append(f"| MLX screen (r2-m5max-4, preset screen = f16 + 20 + 25) | full cold screen, {d.get('ligands_ok')} ligands ok | "
                     f"{fmt(d.get('cold_wall_s'), 0)} s | {fmt(d.get('s_per_ligand_cold'), 2)} s | {fmt(d.get('peak_footprint_gb'), 2)} | MEASURED; power rows: {d.get('power_rows')} |")
            out["mlx"] = d
        for rj in sorted((td / "t400up").glob("*/run.json")) if (td / "t400up").exists() else []:
            r = json.loads(rj.read_text())
            n = r["n_ligands"]
            L.append(f"| upstream up200 (RBVI, MPS fp32, 200 / 200) {r['part']} | {n} ligands, one run, rc {r['rc']} | {fmt(r['W_s'], 0)} s | "
                     f"{fmt(r['W_s'] / n, 1)} s (whole run incl. startup) | {fmt(r.get('peak_footprint_gb'), 2)} | MEASURED |")
            out.setdefault("up", []).append(r)
        dec = td / "t400up_decision.json"
        if dec.exists():
            d = json.loads(dec.read_text())
            L.append(f"| upstream up200, all 199 ligands | est_wall_199 = (W5 - span5) + 199 x span5 / 5 | {fmt(d['est_wall_199_s'], 0)} s ({d['est_wall_199_s'] / 3600:.1f} h) | "
                     f"{fmt(d['est_wall_199_s'] / 199, 1)} s | - | ESTIMATE (from the 5-ligand MEASURED run); verdict {d['verdict']} at the 6 h limit |")
            out["decision"] = d
    return L, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs/macbench")
    ap.add_argument("--tags", nargs="*")
    ap.add_argument("--md")
    ap.add_argument("--json")
    a = ap.parse_args()
    logs = Path(a.logs)
    pub = json.loads((HERE / "published.json").read_text())
    runs = load_runs(logs, a.tags)
    hl, hrows = headline_table(runs, pub, logs)
    nl, nrows = nim_table(runs, pub)
    tl, rows = timing_table(runs, pub)
    t4, t4json = t400_section(logs, a.tags)
    md = ["# macbench summary (MEASURED / ESTIMATE / PUBLISHED)\n",
          "W = whole-process wall (fresh process, spawn to exit); M = model-call time; median over the seeds that ended ok, [min-max]. "
          "Ratio = MEASURED up200 median W / this arm's median W, same input (the headline table has the comm200 / ours ratios). PUBLISHED rows are other chips and unstated settings: not in any ratio.\n",
          "## One row per case: ours vs boltz-community on this Mac (MEASURED) beside the published ChimeraX numbers\n",
          "MEASURED cells: median over the seeds that ended ok, [min-max], n ok / n run; same Mac, same pinned MSA csv, 3 recycles, 1 sample. "
          "comm200 is the fork as shipped (it subsamples the MSA to 1024 rows by default; ours and upstream use up to 8192): read the comm200_full column "
          "for the row-count-matched comparison. PUBLISHED cells: ChimeraX runtimes.html, Boltz 2.1.1, other chips, sampling steps and MSA accounting unstated, "
          "never in a ratio. H100: read from logs/h100 when it exists.\n",
          *hl, "\n## NVIDIA-size cases (structure + affinity): ours fast preset vs boltz-community on this Mac, beside NVIDIA's published H100 numbers\n",
          "One timed run per cell (seed 0, no repeats, so no median or range), fresh process, structure and affinity model-call time separate (M), W = whole process. "
          "NVIDIA's settings (steps, recycles, MSA accounting, ligand, warm or cold) are undisclosed; their cells are PUBLISHED, H100, never in a ratio. "
          "Both builds use the same ColabFold MSA csv, the same ligand and 3 recycles / 1 sample; ours is f16 at 50 steps, the community fork fp32 at its default 200 steps "
          "with --no_subsample_msa, so the speed ratio mixes a step-count and a precision difference and is not an implementation-only comparison.\n",
          *nl, "\n## Per-arm detail\n", *tl, "\n## T400 (199 ligands: 406-421 tokens; target lig_aff + seq1.a3m)\n", *t4,
          "\nPublished sources: ChimeraX help https://www.rbvi.ucsf.edu/chimerax/data/boltz-apr2025/boltz_help.html and runtimes.html; PNCC workshop "
          "https://www.rbvi.ucsf.edu/chimerax/data/pncc-workshop-sep2025/pncc_boltz.html; MoleculeDesk https://www.moleculedesk.com/docs/models/boltz.\n"]
    text = "\n".join(md)
    print(text)
    if a.md:
        Path(a.md).write_text(text)
    if a.json:
        Path(a.json).write_text(json.dumps({"headline": hrows, "nim": nrows, "timing": rows, "t400": t4json}, indent=1, default=str) + "\n")


if __name__ == "__main__":
    main()
