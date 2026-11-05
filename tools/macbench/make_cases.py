"""Writes the macbench Boltz inputs (tools/macbench/cases/*.yaml) from the committed RCSB files in tools/macbench/ref/.

  python tools/macbench/make_cases.py [--msa-dir macbench-env/msa] [--out tools/macbench/cases]

Case names (the YAML stem is the Boltz record id and the results directory name):
  <case>_empty.yaml   every protein `msa: empty` (the no-MSA arm)
  <case>_msa.yaml     every protein with the cached ColabFold-server MSA csv (the MSA arm; paths absolute on this machine)
NIM cases nim_8eil / nim_7qsj / nim_8cpk / nim_8gpp (structure + affinity, see build()).
Cases: 8rf4 (nsp1 118 aa + EQT, structure only), 8rf4aff (same + affinity of EQT), 9moj, 9b3h, ntca_aff (NTCA dimer + 2 x 31-nt DNA +
AKG, structure + affinity), prot117 (= upstream examples/prot_no_msa.yaml = MoleculeDesk's protein.yaml), prot400, lig_aff.
The MSA arm of prot117 / prot400 does not exist (continuity cases, no-MSA only); lig_aff's MSA arm uses upstream's
examples/msa/seq1.a3m, the file the T400 screen uses. The MSA csv names are <case>_<entity index>.csv in --msa-dir (see fetch_msa.sh);
8rf4aff reuses 8rf4's.
"""
import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
REF = HERE / "ref"
ROOT = HERE.parents[1]
SEQ1 = "cases/msa/seq1.a3m"  # upstream boltz examples/msa/seq1.a3m (copy in cases/msa/), relative to the checkout


def fasta(name):
    out, h = {}, None
    for ln in (REF / f"{name}.fasta").read_text().splitlines():
        if ln.startswith(">"):
            h = ln[1:].split("|")
            ent = int(h[0].split("_")[1])
            chains = [c.strip() for c in h[1].replace("Chains", "").replace("Chain", "").split(",")]
            out[ent] = {"chains": chains, "seq": ""}
        else:
            out[ent]["seq"] += ln.strip()
    return [out[k] for k in sorted(out)]


def prot_block(ids, seq, msa):
    ids = ids if len(ids) > 1 else ids[0]
    idtxt = f"[{', '.join(ids)}]" if isinstance(ids, list) else ids
    s = f"  - protein:\n      id: {idtxt}\n      sequence: {seq}\n"
    return s + f"      msa: {msa}\n"


def yaml_text(entities, msa_paths, ligands=(), dna=(), affinity=None):
    """entities: [(ids, seq)] proteins; msa_paths: list parallel to entities (path str or 'empty')."""
    t = "version: 1\nsequences:\n"
    for (ids, seq), m in zip(entities, msa_paths):
        t += prot_block(ids, seq, m)
    for ids, seq in dna:
        idtxt = f"[{', '.join(ids)}]" if len(ids) > 1 else ids[0]
        t += f"  - dna:\n      id: {idtxt}\n      sequence: {seq}\n"
    for lid, kind, val in ligands:
        t += f"  - ligand:\n      id: {lid}\n      {kind}: {val}\n"
    if affinity:
        t += f"properties:\n  - affinity:\n      binder: {affinity}\n"
    return t


def build(msa_dir):
    msa_dir = Path(msa_dir).expanduser()
    cases = {}
    # 8rf4: one chain A (118 aa) + the deposited ligand EQT (C7H6ClN3, 11 heavy atoms) -> 129 tokens
    e = fasta("8RF4")
    assert len(e) == 1 and len(e[0]["seq"]) == 118
    ent = [(["A"], e[0]["seq"])]
    cases["8rf4"] = dict(entities=ent, msa=[msa_dir / "8rf4_0.csv"], ligands=[("B", "ccd", "EQT")], affinity=None)
    cases["8rf4aff"] = dict(entities=ent, msa=[msa_dir / "8rf4_0.csv"], ligands=[("B", "ccd", "EQT")], affinity="B")
    # 9moj: entity 1 chains A,B (151 aa), entity 2 chains C,D (179 aa) -> 660 tokens
    e = fasta("9MOJ")
    assert [len(x["seq"]) for x in e] == [151, 179]
    cases["9moj"] = dict(entities=[(x["chains"], x["seq"]) for x in e], msa=[msa_dir / f"9moj_{i}.csv" for i in range(2)])
    # 9b3h: entity 1 chain A (600 aa, bovine factor I), entity 2 chain B (311 aa); NAG glycans not included -> 911 tokens
    e = fasta("9B3H")
    assert [len(x["seq"]) for x in e] == [600, 311]
    cases["9b3h"] = dict(entities=[(x["chains"], x["seq"]) for x in e], msa=[msa_dir / f"9b3h_{i}.csv" for i in range(2)])
    # NTCA workshop input: NTCA dimer (9GQU sequence, 2 x 222) + 2 x the 31-nt palindromic DNA + AKG, affinity of AKG
    e = fasta("9GQU")
    ws = "".join(ln.strip() for ln in (REF / "ntca_workshop_sequence.fasta.txt").read_text().splitlines() if not ln.startswith(">"))
    assert ws == e[0]["seq"] and len(ws) == 222, "workshop NTCA sequence differs from 9GQU"
    dna = "".join(ln.strip() for ln in (REF / "ntca_workshop_dna.fasta.txt").read_text().splitlines() if not ln.startswith(">"))
    assert dna == "CATTTTTATGTATCAGCTGATACATAAAAAT" and len(dna) == 31
    cases["ntca_aff"] = dict(entities=[(["A", "B"], ws)], msa=[msa_dir / "ntca_0.csv"], dna=[(["C", "D"], dna)],
                             ligands=[("E", "ccd", "AKG")], affinity="E")
    # NVIDIA-size cases (2026-11-02): the RCSB entity of the NIM test id, structure + affinity of one fixed ligand per case.
    # Ligand rule: the largest bound organic small molecule of the entry (not an ion, not a crystallisation additive such as GOL);
    # entries without one get CELECOXIB (26 heavy atoms), the same fixed drug-like ligand for both builds.
    celecoxib = "'Cc1ccc(cc1)-c1cc(nn1-c1ccc(cc1)S(N)(=O)=O)C(F)(F)F'"
    for pid, lig in (("8EIL", ("B", "ccd", "SIN")), ("7QSJ", ("B", "smiles", celecoxib)), ("8CPK", ("B", "smiles", celecoxib)),
                     ("8GPP", ("D", "ccd", "BCT"))):
        e = fasta(pid)
        assert len(e) == 1
        # NVIDIA's length is one chain for 8eil / 7qsj / 8cpk (the entity, modelled as chain A); for 8gpp 628 = the 3 x 208 trimer
        # (the deposited biological assembly) + the 4 heavy atoms of BCT as tokens
        chains = e[0]["chains"] if pid == "8GPP" else ["A"]
        cases[f"nim_{pid.lower()}"] = dict(entities=[(chains, e[0]["seq"])], msa=[msa_dir / f"nim_{pid.lower()}_0.csv"],
                                           ligands=[lig], affinity=lig[0])
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msa-dir", default="macbench-env/msa", help="written into the YAMLs as given (relative = to the checkout, where driver.py runs)")
    ap.add_argument("--out", default=str(HERE / "cases"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, c in build(a.msa_dir).items():
        n = len(c["entities"])
        kw = dict(ligands=c.get("ligands", ()), dna=c.get("dna", ()), affinity=c.get("affinity"))
        (out / f"{name}_empty.yaml").write_text(yaml_text(c["entities"], ["empty"] * n, **kw))
        (out / f"{name}_msa.yaml").write_text(yaml_text(c["entities"], [str(p) for p in c["msa"]], **kw))
        # server input: no msa line at all (upstream --use_msa_server fills it in); written next to the others, used by fetch_msa.sh
        srv = yaml_text(c["entities"], ["__DROP__"] * n, **kw).replace("      msa: __DROP__\n", "")
        (out / f"{name}.yaml").write_text(srv)
    # continuity cases: the repo's own YAMLs, byte-for-byte (prot117 == upstream examples/prot_no_msa.yaml)
    for name in ("prot117", "prot400", "lig_aff"):
        (out / f"{name}_empty.yaml").write_text((ROOT / "cases" / f"{name}.yaml").read_text())
    txt = (ROOT / "cases" / "lig_aff.yaml").read_text()
    assert "msa: empty" in txt
    (out / "lig_aff_msa.yaml").write_text(txt.replace("msa: empty", f"msa: {SEQ1}"))
    print("wrote", len(list(out.glob('*.yaml'))), "files to", out)


if __name__ == "__main__":
    main()
