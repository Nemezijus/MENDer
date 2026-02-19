"""Path normalization and allowlisting for user-provided inputs."""

from __future__ import annotations

from typing import Optional
import os

from .errors import LoadError
from .environment import DATA_ROOT, UPLOAD_DIR, is_running_in_docker
from .repo_root import find_repo_root


def _looks_like_windows_abs(p: str) -> bool:
    # e.g. "C:\\..." or "D:/..."
    return len(p) >= 2 and p[1] == ":" and p[0].isalpha()


def resolve_user_path(path: Optional[str]) -> Optional[str]:
    """Normalize and validate a user-provided path.

    Docker/containers:
      - Only allow files under DATA_ROOT (/app/data by default) or UPLOAD_DIR.
      - Accept convenient shorthands:
          * "data/<...>"    -> "{DATA_ROOT}/<...>" (and fall back to /app/data)
          * "uploads/<...>" -> "{UPLOAD_DIR}/<...>"
          * "<filename>"    -> "{UPLOAD_DIR}/<filename>" (default for relative)

    Dev/local:
      - Accept absolute paths.
      - Map repo-relative "data/<...>" -> "<repo>/data/<...>".
      - If a relative path doesn't exist as-is, also try "{UPLOAD_DIR}/<...>".
    """
    if not path:
        return None

    p = path.strip()
    if not p:
        return None

    if is_running_in_docker():
        up = p.replace("\\", "/")

        # Resolve shorthand paths
        if up.startswith("data/"):
            ap = os.path.abspath(os.path.join(DATA_ROOT, up[len("data/"):]))
            # If DATA_ROOT is overridden (e.g. /data) but image datasets are in /app/data, fall back.
            if not os.path.exists(ap) and DATA_ROOT != os.path.abspath("/app/data"):
                ap2 = os.path.abspath(os.path.join("/app/data", up[len("data/"):]))
                if os.path.exists(ap2):
                    ap = ap2
        elif up.startswith("uploads/"):
            ap = os.path.abspath(os.path.join(UPLOAD_DIR, up[len("uploads/"):]))
        else:
            # Relative path defaults to UPLOAD_DIR
            if (not os.path.isabs(p)) and (not _looks_like_windows_abs(p)):
                ap = os.path.abspath(os.path.join(UPLOAD_DIR, p))
            else:
                ap = os.path.abspath(p)

        # Allowed roots
        allowed_roots = []
        if DATA_ROOT:
            allowed_roots.append(DATA_ROOT)

        app_data = os.path.abspath("/app/data")
        if os.path.isdir(app_data) and app_data not in allowed_roots:
            allowed_roots.append(app_data)

        if UPLOAD_DIR:
            allowed_roots.append(UPLOAD_DIR)

        if not any(ap == root or ap.startswith(root + os.sep) for root in allowed_roots):
            raise LoadError(
                f"Path must be under one of: {', '.join(allowed_roots)} (got: {path!r})"
            )

        if not os.path.exists(ap):
            raise LoadError(f"File not found: {path} (resolved to: {ap})")

        return ap

    # Dev mode
    norm = p.replace("\\", "/")
    if norm.startswith("data/"):
        repo_root = find_repo_root(os.path.abspath(os.path.dirname(__file__)))
        ap = os.path.abspath(os.path.join(repo_root, norm))
    else:
        ap = os.path.abspath(p)

    if os.path.exists(ap):
        return ap

    if (not os.path.isabs(p)) and (not _looks_like_windows_abs(p)):
        ap2 = os.path.abspath(os.path.join(UPLOAD_DIR, p))
        if os.path.exists(ap2):
            return ap2

    raise LoadError(f"File not found: {path}")
