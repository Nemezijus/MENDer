"""Environment detection and filesystem roots for the backend I/O boundary."""

from __future__ import annotations

import os


def is_running_in_docker() -> bool:
    """Best-effort detection for containerized runs."""
    try:
        if os.path.exists("/.dockerenv"):
            return True
        if os.path.exists("/proc/1/cgroup"):
            with open("/proc/1/cgroup", "rt", encoding="utf-8", errors="ignore") as fh:
                cgroup = fh.read()
            return any(tok in cgroup for tok in ("docker", "kubepod", "containerd"))
    except Exception:
        return False
    return False


def get_data_root() -> str:
    """Read-only datasets root inside container image (optionally overridden)."""
    return os.path.abspath(os.getenv("DATA_ROOT", "/app/data"))


def get_upload_dir() -> str:
    """Read-write uploads directory (Docker: usually /data/uploads; Dev: ./uploads)."""
    return os.path.abspath(os.getenv("UPLOAD_DIR", os.path.abspath("./uploads")))


# Convenience constants (computed at import time)
DATA_ROOT = get_data_root()
UPLOAD_DIR = get_upload_dir()
