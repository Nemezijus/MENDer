from typing import Optional, Tuple
import os
import numpy as np

from utils.configs.configs import DataConfig
from utils.factories.data_loading_factory import make_data_loader

DATA_ROOT = os.getenv("DATA_ROOT", "/data")       # RO datasets in Docker
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads") # RW uploads (Docker:/uploads; Dev: ./uploads)

class LoadError(Exception):
    pass


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))


def _coerce_dev_relative(p: str) -> str:
    # For dev: turn "data/..." into "<repo>/data/..."
    norm = p.replace("\\", "/")
    if norm.startswith("data/"):
        return os.path.join(_repo_root(), norm)
    return p


def _normalize_and_check(path: Optional[str]) -> Optional[str]:
    """
    Normalize user paths.
    - In Docker (when DATA_ROOT or UPLOAD_DIR exist): only allow files under
      DATA_ROOT (e.g. /data) or UPLOAD_DIR (e.g. /uploads). Also map 'data/...'
      to '/data/...'.
    - In dev (no container mounts): accept absolute paths and repo-relative 'data/...'.
    """
    if not path:
        return None

    p = path.strip()
    if not p:
        return None

    running_in_docker = os.path.isdir(DATA_ROOT) or os.path.isdir(UPLOAD_DIR)

    if running_in_docker:
        up = p.replace("\\", "/")
        if up.startswith("data/"):
            up = "/" + up  # -> "/data/..."
        ap = os.path.abspath(up)

        allowed_roots = []
        if DATA_ROOT:
            allowed_roots.append(os.path.abspath(DATA_ROOT))
        if UPLOAD_DIR:
            allowed_roots.append(os.path.abspath(UPLOAD_DIR))

        # must be inside at least one allowed root
        if not any(ap == root or ap.startswith(root + os.sep) for root in allowed_roots):
            raise LoadError(f"Path must be under one of: {', '.join(allowed_roots)} (got: {path!r})")

        if not os.path.exists(ap):
            raise LoadError(f"File not found: {path}")

        return ap

    # Dev mode
    ap = os.path.abspath(_coerce_dev_relative(p))
    if not os.path.exists(ap):
        raise LoadError(f"File not found: {path}")
    return ap


def build_data_config(
    npz_path: Optional[str],
    x_key: Optional[str],
    y_key: Optional[str],
    x_path: Optional[str],
    y_path: Optional[str],
) -> DataConfig:
    npz_p = _normalize_and_check(npz_path)
    x_p = _normalize_and_check(x_path)
    y_p = _normalize_and_check(y_path)
    return DataConfig(
        npz_path=npz_p,
        x_path=x_p,
        y_path=y_p,
        x_key=x_key or "X",
        y_key=y_key or "y",
    )


def load_X_y(
    npz_path: Optional[str],
    x_key: Optional[str],
    y_key: Optional[str],
    x_path: Optional[str],
    y_path: Optional[str],
) -> Tuple[np.ndarray, np.ndarray]:
    try:
        cfg = build_data_config(npz_path, x_key, y_key, x_path, y_path)
        loader = make_data_loader(cfg)
        X, y = loader.load()
        return X, y
    except LoadError:
        raise
    except Exception as e:
        raise LoadError(str(e))
