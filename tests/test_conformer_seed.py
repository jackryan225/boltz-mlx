"""SMILES-ligand reference conformers do not depend on what the process embedded before (seeding.install_conformer_seed).

Diagnosis 2026-10-29: upstream compute_3d_conformer leaves ETKDG's randomSeed
at -1, so RDKit's process-wide random stream made a screen ligand's reference conformer (featurized ref_pos, the only
one of 78 batch tensors that differed) depend on every ligand embedded before it in the process. The fix embeds every
SMILES ligand with randomSeed = CONFORMER_SEED (42), which is where RDKit's default stream starts: each ligand gets the
conformer it got as the first embedding in a fresh process.
"""

import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from boltz_mlx.paths import default_cache
from boltz_mlx.seeding import CONFORMER_SEED, install_conformer_seed

# three T400 (FEP+ BACE) ligands of the diagnosis: 13c gave a different ref_pos fresh / after 13b / after 17d
SMI = {
    "bace_cat-13b": "CC[C@]1(c2cccc(-c3cccc(Cl)c3)c2)[NH+]=C(N)N(C)C1=O",
    "bace_cat-13c": "CC(C)[C@]1(c2cccc(-c3cccc(Cl)c3)c2)[NH+]=C(N)N(C)C1=O",
    "bace_cat-17d": "CN1C(=O)[C@](c2cccc(-c3cccnc3)c2)(C2CC2)[NH+]=C1N",
}
PROT = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQ"

# the molecule preparation of upstream parse_boltz_schema's SMILES branch, then its compute_3d_conformer; prints a hash
# of the conformer coordinates. Run in a fresh interpreter, this is the first (default-seed) embedding of that process.
_EMBED = r"""
import hashlib, sys
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import boltz.data.parse.schema as schema

def embed(smi):
    mol = AllChem.AddHs(AllChem.MolFromSmiles(smi))
    Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
    assert schema.compute_3d_conformer(mol)
    return hashlib.sha256(np.asarray(mol.GetConformer().GetPositions()).tobytes()).hexdigest()

print(embed(sys.argv[1]))
"""


def _embed_here(smi: str) -> str:
    import boltz.data.parse.schema as schema
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = AllChem.AddHs(AllChem.MolFromSmiles(smi))
    Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
    assert schema.compute_3d_conformer(mol)
    return hashlib.sha256(np.asarray(mol.GetConformer().GetPositions()).tobytes()).hexdigest()


def _advance_default_stream():
    """An unseeded RDKit embedding: moves the process-wide default stream on (what an earlier ligand used to do)."""
    from rdkit.Chem import AllChem

    m = AllChem.AddHs(AllChem.MolFromSmiles("CCOc1ccc(CC(=O)N)cc1"))
    AllChem.EmbedMolecule(m, AllChem.ETKDGv3())


def test_install_is_idempotent_and_marks_the_wrapper():
    import boltz.data.parse.schema as schema

    install_conformer_seed()
    fn = schema.compute_3d_conformer
    install_conformer_seed()
    assert schema.compute_3d_conformer is fn
    assert fn._boltz_mlx_conformer_seed == CONFORMER_SEED == 42
    assert fn.__wrapped__.__module__ == "boltz.data.parse.schema"


def test_unseeded_upstream_depends_on_history():
    """The bug: upstream's own function, called twice in one process on the same molecule, gives two conformers."""
    import boltz.data.parse.schema as schema
    from rdkit import Chem
    from rdkit.Chem import AllChem

    install_conformer_seed()
    upstream = schema.compute_3d_conformer.__wrapped__
    out = []
    for _ in range(2):
        mol = AllChem.AddHs(AllChem.MolFromSmiles(SMI["bace_cat-13c"]))
        Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
        assert upstream(mol)
        out.append(np.asarray(mol.GetConformer().GetPositions()))
    assert not np.array_equal(out[0], out[1])


@pytest.mark.parametrize("lid", sorted(SMI))
def test_seeded_embedding_equals_first_default_embedding_of_a_fresh_process(lid):
    """RDKit pin: the seeded conformer (here, after the default stream has moved on) is bit-identical to what upstream's
    unseeded compute_3d_conformer gives as the first embedding in a fresh interpreter. Breaks if RDKit's default stream
    stops starting at seed 42 (order independence would still hold; the match with single-input runs would not)."""
    fresh = subprocess.run([sys.executable, "-c", _EMBED, SMI[lid]], capture_output=True, text=True, timeout=300)
    assert fresh.returncode == 0, fresh.stderr[-2000:]
    install_conformer_seed()
    _advance_default_stream()
    assert _embed_here(SMI[lid]) == fresh.stdout.split()[-1]


def _cache() -> Path:
    for c in (default_cache(),):
        if (c / "mols").is_dir():
            return c
    pytest.skip("upstream mols/ dictionary not installed (set BOLTZ_MLX_CACHE)")


def test_featurized_ref_pos_of_13c_is_independent_of_the_ligands_before_it(tmp_path):
    """13c's structure-pass features (all tensors, ref_pos included) are bit-identical when it is featurized first,
    after 13b and after 17d, each in a fresh work dir of this one process, with unseeded embeddings in between."""
    from boltz_mlx.screen.engine import UpstreamEngine
    from boltz_mlx.screen.inputs import load_target, read_ligands, validate_ligand

    cache = _cache()
    csv = tmp_path / "ligs.csv"
    csv.write_text("id,smiles\n" + "".join(f"{k},{v}\n" for k, v in SMI.items()))
    ligs = {l.ligand_id: l for l in (validate_ligand(x) for x in read_ligands(str(csv), None, None))}
    tgt = load_target(PROT, msa="empty")
    feats = {}
    for name, order in (("fresh", ["bace_cat-13c"]), ("after_13b", ["bace_cat-13b", "bace_cat-13c"]),
                        ("after_17d", ["bace_cat-17d", "bace_cat-13c"])):
        _advance_default_stream()
        eng = UpstreamEngine(tmp_path / name / "work", cache, tgt, 8192, "mmcif", 0, keep_extras=False)
        for lid in order:
            eng.prepare(ligs[lid])
            b = eng.featurize(ligs[lid], affinity=False)
        feats[name] = {k: v.numpy() for k, v in b.items() if hasattr(v, "numpy")}
    ref = feats["fresh"]
    assert "ref_pos" in ref and ref["ref_pos"].shape[-2] == ref["atom_pad_mask"].shape[-1]
    for name in ("after_13b", "after_17d"):
        diff = sorted(k for k in ref if not np.array_equal(ref[k], feats[name][k]))
        assert diff == [], f"13c {name}: tensors differ from fresh: {diff}"
