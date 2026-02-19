"""Repo root discovery for local development convenience paths."""

from __future__ import annotations

import os


def find_repo_root(start_dir: str, max_hops: int = 10) -> str:
    """Find the repository root directory.

    In dev mode we accept convenience paths like ``data/<...>`` and map them to
    ``<repo>/data/<...>``. The backend lives under ``<repo>/backend/...``, so a
    fixed relative hop is brittle.

    Strategy:
      - Walk up parents until we find an ``engine/`` directory (and preferably a
        ``requirements.txt``) which reliably indicates the repo root.
      - Fall back to a conservative relative hop.
    """
    cur = os.path.abspath(start_dir)
    for _ in range(max_hops):
        has_engine = os.path.isdir(os.path.join(cur, "engine"))
        has_reqs = os.path.isfile(os.path.join(cur, "requirements.txt"))
        if has_engine and has_reqs:
            return cur
        if has_engine:
            return cur

        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent

    # Fallback: backend/app/adapters/io -> backend/app/adapters -> backend/app -> backend -> repo
    return os.path.abspath(os.path.join(start_dir, "../../../../"))
