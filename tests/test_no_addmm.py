"""ml-explore/mlx #4769: mx.addmm rounds prematurely in half precision. Never use it in boltz_mlx."""

import re
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "boltz_mlx"


def test_no_addmm_in_package():
    hits = []
    for f in PKG.rglob("*.py"):
        for i, line in enumerate(f.read_text().splitlines(), 1):
            if re.search(r"\baddmm\s*\(", line) and not line.lstrip().startswith("#"):
                hits.append(f"{f.relative_to(PKG.parent)}:{i}: {line.strip()}")
    assert not hits, "mx.addmm is banned (premature fp16 rounding, mlx #4769):\n" + "\n".join(hits)
