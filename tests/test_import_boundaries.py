"""Enforce one-way dependencies (docs/ARCHITECTURE.md §2).

likho/ must not import api/, app/, eval/ or UI/web frameworks.
api/ must not import app/ or the UI framework.
app/ talks to the API over HTTP only: no likho/, no model libraries (CLAUDE.md rule 1).
eval/ must not import api/ or app/ (CLAUDE.md rule 4).
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FORBIDDEN = {
    "likho": {"api", "app", "eval", "gradio", "fastapi", "uvicorn"},
    "api": {"app", "gradio"},
    "app": {"likho", "eval", "torch", "transformers", "ai4bharat", "IndicTransToolkit"},
    "eval": {"api", "app", "gradio"},
}


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_import_direction():
    violations = []
    for package, forbidden in FORBIDDEN.items():
        for path in sorted((ROOT / package).rglob("*.py")):
            bad = _imported_roots(path) & forbidden
            if bad:
                violations.append(f"{path.relative_to(ROOT)} imports {sorted(bad)}")
    assert not violations, "\n".join(violations)
