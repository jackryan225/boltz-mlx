"""sha256 + size of downloaded files (a data manifest). Usage: python -I tools/data/hash_files.py OUT.json FILE [FILE ...]"""
import hashlib
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
res = {}
for f in sys.argv[2:]:
    p = Path(f)
    res[p.name] = {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
out.write_text(json.dumps(res, indent=1, sort_keys=True))
print(len(res), "files ->", out)
