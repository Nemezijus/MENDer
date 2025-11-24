from typing import Optional, Tuple
import os
import numpy as np

from shared_schemas.run_config import DataModel
from utils.factories.data_loading_factory import make_data_loader
from utils.strategies.data_loaders import _coerce_shapes
from utils.parse.data_read import load_mat_variable

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


def _is_running_in_docker() -> bool:
    """
    More reliable detection for containerized runs:
      - check for /.dockerenv
      - inspect /proc/1/cgroup for docker/container indicators
    Fallback to existing directory checks on platforms where /proc may not exist (e.g., Windows).
    """
    try:
        if os.path.exists("/.dockerenv"):
            return True
        if os.path.exists("/proc/1/cgroup"):
            try:
                with open("/proc/1/cgroup", "rt", encoding="utf-8", errors="ignore") as fh:
                    cgroup = fh.read()
                if any(tok in cgroup for tok in ("docker", "kubepod", "containerd")):
                    return True
            except Exception:
                # ignore read errors and fall back to directory checks
                pass
    except Exception:
        # ignore platform-specific errors and fall back
        pass

    return False


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

    # running_in_docker = os.path.isdir(DATA_ROOT) or os.path.isdir(UPLOAD_DIR)
    running_in_docker = _is_running_in_docker()

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
) -> DataModel:
    npz_p = _normalize_and_check(npz_path)
    x_p = _normalize_and_check(x_path)
    y_p = _normalize_and_check(y_path)
    return DataModel(
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
    """
    Training/inspect helper: requires both X and y to be present somewhere.
    """
    try:
        cfg = build_data_config(npz_path, x_key, y_key, x_path, y_path)
        loader = make_data_loader(cfg)
        X, y = loader.load()
        return X, y
    except LoadError:
        raise
    except Exception as e:
        raise LoadError(str(e))


def load_X_optional_y(
    npz_path: Optional[str],
    x_key: Optional[str],
    y_key: Optional[str],
    x_path: Optional[str],
    y_path: Optional[str],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Prediction helper: X is required, y is optional.

    Cases
    -----
    - Single .npz (npz_path or x_path ending with .npz):
        * Always load X from x_key.
        * If y_key is present -> run full _coerce_shapes(X, y).
        * If y_key is missing -> return X (rows as observations) and y=None.
    - MAT pair (x_path + y_path provided):
        * Delegate to the standard loader via make_data_loader (training semantics).
    - Single MAT (x_path only):
        * Load variable from x_path, ensure X is 2D, return y=None.
    """
    try:
        cfg = build_data_config(npz_path, x_key, y_key, x_path, y_path)

        # --- NPZ-based loading ------------------------------------------------
        if cfg.npz_path or (cfg.x_path and cfg.x_path.lower().endswith(".npz")):
            path = cfg.npz_path or cfg.x_path
            if not path:
                raise LoadError("NPZ loading requires npz_path or x_path pointing to a .npz file.")
            data = np.load(path, allow_pickle=True)

            if cfg.x_key not in data:
                raise LoadError(
                    f"Key '{cfg.x_key}' not found in {path}. Available keys: {list(data.keys())}"
                )

            X_raw = np.asarray(data[cfg.x_key])

            # If y is present in the file, use the full training-like alignment.
            if cfg.y_key in data.files:
                y_raw = np.asarray(data[cfg.y_key]).ravel()
                X, y = _coerce_shapes(X_raw, y_raw)
                return X, y

            # X-only NPZ: be conservative, just ensure X is 2D.
            X = np.asarray(X_raw)
            if X.ndim == 1:
                X = X[:, None]
            if X.ndim != 2:
                raise LoadError(f"X must be 1D or 2D array; got shape {X.shape}")
            return X, None

        # --- MAT pair: both X and y provided ---------------------------------
        if cfg.x_path and cfg.y_path:
            loader = make_data_loader(cfg)  # AutoLoader -> MatPairLoader or NPZLoader
            X, y = loader.load()
            return X, y

        # --- Single MAT: X only ----------------------------------------------
        if cfg.x_path and not cfg.y_path:
            X_raw = np.asarray(load_mat_variable(cfg.x_path))
            X = X_raw
            if X.ndim == 1:
                X = X[:, None]
            if X.ndim != 2:
                raise LoadError(f"X must be 1D or 2D array; got shape {X.shape}")
            return X, None

        raise LoadError(
            "You must provide at least one feature source (npz_path or x_path) for prediction."
        )

    except LoadError:
        raise
    except Exception as e:
        raise LoadError(str(e))
