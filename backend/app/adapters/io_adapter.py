from typing import Optional, Tuple
import os
import numpy as np

from shared_schemas.run_config import DataModel
from utils.factories.data_loading_factory import make_data_loader
from utils.strategies.data_loaders import _coerce_shapes
from utils.parse.data_read import load_mat_variable

# Read-only datasets in Docker (typically mounted at /data)
DATA_ROOT = os.getenv("DATA_ROOT", "/data")

# Read-write uploads (Docker: /uploads; Dev: ./uploads)
# Keep this consistent with backend/app/routers/files.py.
UPLOAD_DIR = os.getenv("UPLOAD_DIR", os.path.abspath("./uploads"))

# Normalize to absolute paths early so downstream checks are consistent.
DATA_ROOT = os.path.abspath(DATA_ROOT) if DATA_ROOT else DATA_ROOT
UPLOAD_DIR = os.path.abspath(UPLOAD_DIR) if UPLOAD_DIR else UPLOAD_DIR


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


def _looks_like_windows_abs(p: str) -> bool:
    # e.g. "C:\\..." or "D:/..."
    return len(p) >= 2 and p[1] == ":" and p[0].isalpha()


def _normalize_and_check(path: Optional[str]) -> Optional[str]:
    """
    Normalize user paths.
    Rules
    -----
    Docker/containers:
      - Only allow files under DATA_ROOT (e.g. /data) or UPLOAD_DIR (e.g. /uploads).
      - Accept convenient shorthands:
          * "data/<...>"    -> "{DATA_ROOT}/<...>"
          * "uploads/<...>" -> "{UPLOAD_DIR}/<...>"
          * "<filename>"    -> "{UPLOAD_DIR}/<filename>" (default)

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

    running_in_docker = _is_running_in_docker()
    data_root_abs = os.path.abspath(DATA_ROOT) if DATA_ROOT else ""
    upload_dir_abs = os.path.abspath(UPLOAD_DIR) if UPLOAD_DIR else ""

    if running_in_docker:
        # In Docker, only allow reads from mounted roots.
        # Also support convenient relative shorthands like:
        #   - "data/foo/bar.mat"      -> "${DATA_ROOT}/foo/bar.mat"
        #   - "uploads/abc.mat"       -> "${UPLOAD_DIR}/abc.mat"
        #   - "abc.mat"               -> "${UPLOAD_DIR}/abc.mat" (default)
        up = p.replace("\\", "/")

        if up.startswith("data/") and data_root_abs:
            ap = os.path.abspath(os.path.join(data_root_abs, up[len("data/"):]))
        elif up.startswith("uploads/") and upload_dir_abs:
            ap = os.path.abspath(os.path.join(upload_dir_abs, up[len("uploads/"):]))
        else:
            # If the user passed a relative path, treat it as relative to UPLOAD_DIR.
            # This makes the API more robust if callers store only the saved filename.
            if (not os.path.isabs(p)) and (not _looks_like_windows_abs(p)) and upload_dir_abs:
                ap = os.path.abspath(os.path.join(upload_dir_abs, p))
            else:
                ap = os.path.abspath(p)

        allowed_roots = []
        if data_root_abs:
            allowed_roots.append(data_root_abs)
        if upload_dir_abs:
            allowed_roots.append(upload_dir_abs)

        # must be inside at least one allowed root
        if not any(ap == root or ap.startswith(root + os.sep) for root in allowed_roots):
            raise LoadError(f"Path must be under one of: {', '.join(allowed_roots)} (got: {path!r})")

        if not os.path.exists(ap):
            raise LoadError(f"File not found: {path}")

        return ap

    # Dev mode
    # - accept absolute paths
    # - accept repo-relative "data/..." -> "<repo>/data/..."
    # - accept bare filenames/relative paths as "<UPLOAD_DIR>/<name>" if present
    ap = os.path.abspath(_coerce_dev_relative(p))
    if os.path.exists(ap):
        return ap

    if (not os.path.isabs(p)) and (not _looks_like_windows_abs(p)) and upload_dir_abs:
        ap2 = os.path.abspath(os.path.join(upload_dir_abs, p))
        if os.path.exists(ap2):
            return ap2

    raise LoadError(f"File not found: {path}")


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
    expected_n_features: Optional[int] = None,
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
        
            if expected_n_features is not None:
                try:
                    exp = int(expected_n_features)
                    if X.shape[1] != exp and X.shape[0] == exp:
                        X = X.T
                except Exception:
                    pass

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
            if expected_n_features is not None:
                try:
                    exp = int(expected_n_features)
                    if X.shape[1] != exp and X.shape[0] == exp:
                        X = X.T
                except Exception:
                    pass

            return X, None

        raise LoadError(
            "You must provide at least one feature source (npz_path or x_path) for prediction."
        )

    except LoadError:
        raise
    except Exception as e:
        raise LoadError(str(e))
