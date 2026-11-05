"""Every tool that imports boltz_mlx imports THIS checkout's package, not the editable install's (CPU, no model).

`python tools/x.py` puts the script's directory (tools/, not the repo root) first on sys.path, and the editable install's
import finder maps boltz_mlx to the main checkout. So a tool run from a worktree silently used the main checkout's code
unless it put its own repo root on sys.path first (found 2026-10-29: the boltz-preset2 pre-merge fingerprints, run from a
worktree, measured the main checkout's boltz_mlx).

The check builds a scratch "checkout": a copy of tools/ next to a stub boltz_mlx package that is not installed anywhere.
Each tool's module top level then runs as `python <tool>` would run it (sys.path[0] = the tool's directory, the module
named so that its `if __name__ == "__main__"` block does not run). The run stops at the first boltz_mlx import. For a
tool that imports boltz_mlx only inside functions, the check asks after the top level where `import boltz_mlx` would
resolve. Every tool must resolve to the scratch checkout's stub, never to the editable install, so the check is
meaningful in the main checkout too.
"""

import ast
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_PROBE = r"""
import importlib.util, os, runpy, sys
tool = os.path.abspath(sys.argv[1])
sys.argv = [tool]
sys.path[0] = os.path.dirname(tool)  # what `python <tool>` puts first


class _Stop(BaseException):
    pass


class _Hook:
    def find_spec(self, name, path=None, target=None):
        if name == "boltz_mlx":
            sys.meta_path.remove(self)
            print("BOLTZ_MLX_FROM", importlib.util.find_spec("boltz_mlx").origin, "top-level", flush=True)
            raise _Stop
        return None


sys.meta_path.insert(0, _Hook())
try:
    runpy.run_path(tool, run_name="__tool_import_check__")
except _Stop:
    sys.exit(0)
sys.meta_path[:] = [h for h in sys.meta_path if not isinstance(h, _Hook)]
spec = importlib.util.find_spec("boltz_mlx")
print("BOLTZ_MLX_FROM", spec.origin if spec else None, "lazy", flush=True)
"""


def _imports_boltz_mlx(path: Path) -> bool:
    for node in ast.walk(ast.parse(path.read_text(), str(path))):
        if isinstance(node, ast.Import) and any(a.name == "boltz_mlx" or a.name.startswith("boltz_mlx.") for a in node.names):
            return True
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module and (
                node.module == "boltz_mlx" or node.module.startswith("boltz_mlx.")):
            return True
    return False


def tools_importing_boltz_mlx() -> list[str]:
    return sorted(str(p.relative_to(ROOT)) for p in (ROOT / "tools").rglob("*.py")
                  if "__pycache__" not in p.parts and _imports_boltz_mlx(p))


def test_the_tool_list_is_not_empty():
    tools = tools_importing_boltz_mlx()
    assert len(tools) >= 40 and "tools/fingerprint.py" in tools and "tools/validation/run_l3.py" in tools


def test_every_tool_imports_its_own_checkouts_boltz_mlx(tmp_path):
    scratch = tmp_path / "checkout"
    shutil.copytree(ROOT / "tools", scratch / "tools", ignore=shutil.ignore_patterns("__pycache__"))
    (scratch / "boltz_mlx").mkdir()
    stub = scratch / "boltz_mlx" / "__init__.py"
    stub.write_text("raise ImportError('stub boltz_mlx of the tool import-path test: never executed')\n")
    (scratch / "probe.py").write_text(_PROBE)
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONSTARTUP", "PYTHONSAFEPATH")}

    def probe(rel: str):
        p = subprocess.run([sys.executable, str(scratch / "probe.py"), str(scratch / rel)], capture_output=True,
                           text=True, timeout=300, cwd=tmp_path, env=env)
        line = next((ln for ln in p.stdout.splitlines() if ln.startswith("BOLTZ_MLX_FROM")), None)
        return rel, p.returncode, line, (p.stderr or p.stdout)[-600:]

    tools = tools_importing_boltz_mlx()
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
        results = list(ex.map(probe, tools))
    bad = []
    for rel, rc, line, tail in results:
        origin = line.split()[1] if line else None
        if origin is None or Path(origin).resolve() != stub.resolve():
            bad.append(f"{rel}: rc={rc} {line or 'no import resolved'}" + ("" if line else f"\n{tail}"))
    assert not bad, f"{len(bad)} of {len(tools)} tools do not import their own checkout's boltz_mlx:\n" + "\n".join(bad)
