"""Boundary check: keep backend and engine cleanly separated.

Run from repo root:

    python scripts/check_engine_boundary.py

Rules enforced:

1) In backend/app/**/*.py, the only allowed Engine imports are:
   - engine.api
   - engine.contracts...

2) In engine/**/*.py, forbid imports from:
   - backend...
   - frontend...
   - fastapi... (unless allow-listed below)

This script is intentionally small and dependency-free so it can run in CI.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path


# If you *intentionally* add an Engine adapter layer that imports fastapi,
# list file globs (relative to repo root) here, e.g. ["engine/adapters/**"].
ALLOW_FASTAPI_IN_ENGINE: tuple[str, ...] = ()


@dataclass(frozen=True)
class Violation:
    file: Path
    lineno: int
    kind: str
    detail: str


def _repo_root() -> Path:
    # This file is <repo_root>/scripts/check_engine_boundary.py
    return Path(__file__).resolve().parents[1]


def _iter_py_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*.py") if "__pycache__" not in p.parts]


def _matches_any_glob(path: Path, globs: tuple[str, ...], repo_root: Path) -> bool:
    rel = path.relative_to(repo_root).as_posix()
    return any(Path(rel).match(g) for g in globs)


def _parse_imports(py_file: Path) -> list[tuple[int, str]]:
    """Return a list of (lineno, imported_module_prefix) for each import."""

    src = py_file.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(py_file))
    out: list[tuple[int, str]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                out.append((node.lineno or 1, a.name))
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            out.append((node.lineno or 1, node.module))

    return out


def _check_backend_imports(repo_root: Path) -> list[Violation]:
    backend_dir = repo_root / "backend" / "app"
    if not backend_dir.exists():
        return []

    violations: list[Violation] = []
    for py_file in _iter_py_files(backend_dir):
        try:
            imports = _parse_imports(py_file)
        except SyntaxError as e:
            violations.append(
                Violation(py_file, int(getattr(e, "lineno", 1) or 1), "syntax", str(e))
            )
            continue

        for lineno, mod in imports:
            # Catch both `import engine.x` and `from engine.x import y`
            if mod == "engine" or mod.startswith("engine."):
                allowed = mod == "engine.api" or mod.startswith("engine.contracts")
                if not allowed:
                    violations.append(
                        Violation(
                            py_file,
                            lineno,
                            "backend->engine",
                            f"Disallowed Engine import '{mod}'. Allowed: engine.api, engine.contracts.*",
                        )
                    )

    return violations


def _check_engine_imports(repo_root: Path) -> list[Violation]:
    engine_dir = repo_root / "engine"
    if not engine_dir.exists():
        return []

    violations: list[Violation] = []
    for py_file in _iter_py_files(engine_dir):
        try:
            imports = _parse_imports(py_file)
        except SyntaxError as e:
            violations.append(
                Violation(py_file, int(getattr(e, "lineno", 1) or 1), "syntax", str(e))
            )
            continue

        for lineno, mod in imports:
            if mod == "backend" or mod.startswith("backend."):
                violations.append(
                    Violation(
                        py_file,
                        lineno,
                        "engine->backend",
                        f"Engine must not import backend ('{mod}').",
                    )
                )
            if mod == "frontend" or mod.startswith("frontend."):
                violations.append(
                    Violation(
                        py_file,
                        lineno,
                        "engine->frontend",
                        f"Engine must not import frontend ('{mod}').",
                    )
                )

            if mod == "fastapi" or mod.startswith("fastapi."):
                if not _matches_any_glob(py_file, ALLOW_FASTAPI_IN_ENGINE, repo_root):
                    violations.append(
                        Violation(
                            py_file,
                            lineno,
                            "engine->fastapi",
                            "Engine must not import fastapi (keep framework adapters outside engine, or allow-list explicitly).",
                        )
                    )

    return violations


def main() -> int:
    repo_root = _repo_root()
    all_violations = []
    all_violations.extend(_check_backend_imports(repo_root))
    all_violations.extend(_check_engine_imports(repo_root))

    if not all_violations:
        print("ENGINE BOUNDARY CHECK: OK")
        return 0

    print("ENGINE BOUNDARY CHECK: FAILED\n")
    for v in sorted(all_violations, key=lambda x: (str(x.file), x.lineno)):
        rel = v.file.relative_to(repo_root)
        print(f"- {rel}:{v.lineno} [{v.kind}] {v.detail}")

    print(
        "\nFix boundary violations by importing Engine through engine.api and contracts, "
        "and keeping engine free from backend/frontend/framework deps."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
