from typing import Optional, Tuple
import os
import numpy as np

from engine.contracts.run_config import DataModel

# Use the stable Engine public surface.
from engine.api import load_from_data_model

# Read-only datasets inside the container image (and/or optionally overridden by env)
DATA_ROOT = os.path.abspath(os.getenv("DATA_ROOT", "/app/data"))

# Read-write uploads (Docker: /uploads; Dev: ./uploads)
# Keep this consistent with backend/app/routers/files.py.
UPLOAD_DIR = os.path.abspath(os.getenv("UPLOAD_DIR", os.path.abspath("./uploads")))


class LoadError(Exception):
    pass


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))


def _is_running_in_docker() -> bool:
    """
    Detect containerized runs:
      - check for /.dockerenv
      - inspect /proc/1/cgroup for docker/container indicators
    """
    try:
        if os.path.exists("/.dockerenv"):
            return True
        if os.path.exists("/proc/1/cgroup"):
            with open("/proc/1/cgroup", "rt", encoding="utf-8", errors="ignore") as fh:
                cgroup = fh.read()
            return any(tok in cgroup for tok in ("docker", "kubepod", "containerd"))
    except Exception:
        return False


def _looks_like_windows_abs(p: str) -> bool:
    # e.g. "C:\\..." or "D:/..."
    return len(p) >= 2 and p[1] == ":" and p[0].isalpha()


def _normalize_and_check(path: Optional[str]) -> Optional[str]:
    """
    Normalize user paths.

    Docker/containers:
      - Only allow files under DATA_ROOT (/app/data by default) or UPLOAD_DIR (/uploads).
      - Accept convenient shorthands:
          * "data/<...>"    -> "{DATA_ROOT}/<...>"  (and fall back to /app/data if DATA_ROOT differs)
          * "uploads/<...>" -> "{UPLOAD_DIR}/<...>"
          * "<filename>"    -> "{UPLOAD_DIR}/<filename>" (default for relative paths)

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

    if running_in_docker:
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
        ap = os.path.abspath(os.path.join(_repo_root(), norm))
    else:
        ap = os.path.abspath(p)

    if os.path.exists(ap):
        return ap

    if (not os.path.isabs(p)) and (not _looks_like_windows_abs(p)):
        ap2 = os.path.abspath(os.path.join(UPLOAD_DIR, p))
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
        X, y, _feature_names = load_from_data_model(cfg)
        if y is None:
            raise LoadError("y is required for this operation, but was not found in the provided inputs.")
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
    """Prediction helper: X is required, y is optional.

    Notes
    -----
    - Parsing is delegated to Engine readers via the stable API
      (:func:`engine.api.load_from_data_model`).
    - If expected_n_features is provided, we apply a best-effort transpose fix for
      X-only inputs when the data appears to be (n_features, n_samples).
    """
    try:
        cfg = build_data_config(npz_path, x_key, y_key, x_path, y_path)

        X, y, _feature_names = load_from_data_model(cfg)

        if expected_n_features is not None:
            try:
                exp = int(expected_n_features)
                if X.ndim == 2 and X.shape[1] != exp and X.shape[0] == exp:
                    X = X.T
            except Exception:
                pass

        return X, y

    except LoadError:
        raise
    except Exception as e:
        raise LoadError(str(e))
