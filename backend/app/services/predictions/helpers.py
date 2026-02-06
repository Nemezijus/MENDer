from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple
import math
import numpy as np

from shared_schemas.eval_configs import EvalModel

from utils.factories.sanity_factory import make_sanity_checker
from engine.runtime.caches.artifact_cache import artifact_cache


def safe_float_optional(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except Exception:
        return None
    if not math.isfinite(f):
        return None
    return f


def build_preview_rows(
    *,
    indices: Sequence[int],
    y_pred: np.ndarray,
    task: str,
    y_true: Optional[np.ndarray] = None,
) -> list[Dict[str, Any]]:
    """Build a compact list of dicts compatible with PredictionRow."""
    rows: list[Dict[str, Any]] = []

    y_true_arr: Optional[np.ndarray] = None
    if y_true is not None:
        y_true_arr = np.asarray(y_true).ravel()
        if y_true_arr.shape[0] != len(indices):
            y_true_arr = None

    for i, idx in enumerate(indices):
        row: Dict[str, Any] = {
            "index": int(idx),
            "y_pred": y_pred[i].item() if hasattr(y_pred[i], "item") else y_pred[i],
        }

        if y_true_arr is not None:
            y_true_val = y_true_arr[i]
            row["y_true"] = y_true_val.item() if hasattr(y_true_val, "item") else y_true_val

            if task == "regression":
                try:
                    resid = float(y_true_val) - float(y_pred[i])
                    row["residual"] = resid
                    row["abs_error"] = abs(resid)
                except Exception:
                    pass
            else:
                row["correct"] = bool(y_pred[i] == y_true_val)

        rows.append(row)

    return rows


def setup_prediction_common(
    *,
    artifact_uid: str,
    artifact_meta: Any,
    X: np.ndarray,
) -> Tuple[Any, np.ndarray, str]:
    """Shared preparation for apply/export flows that do not need EvalModel.

    This is used by unsupervised artifacts where the stored eval config is not
    `EvalModel`, and by any other flow where we only need pipeline retrieval and
    X-shape validation.
    """

    pipeline = artifact_cache.get(artifact_uid)
    if pipeline is None:
        raise ValueError(
            f"No cached model pipeline found for artifact_uid={artifact_uid!r}. "
            "Train a model or load an artifact first."
        )

    X_arr = np.asarray(X)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    if X_arr.ndim != 2:
        raise ValueError(f"Expected 2D X for prediction; got shape {X_arr.shape}.")

    n_features_meta = getattr(artifact_meta, "n_features_in", None) or getattr(artifact_meta, "n_features", None)
    if n_features_meta is not None:
        n_features_meta = int(n_features_meta)
        if X_arr.shape[1] == n_features_meta:
            pass
        elif X_arr.shape[0] == n_features_meta:
            X_arr = X_arr.T
        if X_arr.shape[1] != n_features_meta:
            raise ValueError(
                f"Feature mismatch: model expects {n_features_meta} features, "
                f"but X has shape {X_arr.shape}."
            )

    task = getattr(artifact_meta, "kind", None) or "classification"

    return pipeline, X_arr, task


def setup_prediction(
    *,
    artifact_uid: str,
    artifact_meta: Any,
    X: np.ndarray,
    y: Optional[np.ndarray],
) -> Tuple[Any, np.ndarray, Optional[np.ndarray], str, EvalModel, str]:
    """Shared preparation for apply/export flows."""
    pipeline = artifact_cache.get(artifact_uid)
    if pipeline is None:
        raise ValueError(
            f"No cached model pipeline found for artifact_uid={artifact_uid!r}. "
            "Train a model or load an artifact first."
        )

    X_arr = np.asarray(X)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    if X_arr.ndim != 2:
        raise ValueError(f"Expected 2D X for prediction; got shape {X_arr.shape}.")

    y_arr: Optional[np.ndarray] = None
    if y is not None:
        y_arr = np.asarray(y).ravel()
        make_sanity_checker().check(X_arr, y_arr)

    n_features_meta = getattr(artifact_meta, "n_features_in", None) or getattr(artifact_meta, "n_features", None)
    if n_features_meta is not None:
        n_features_meta = int(n_features_meta)
        if X_arr.shape[1] == n_features_meta:
            pass
        elif X_arr.shape[0] == n_features_meta:
            X_arr = X_arr.T
        if X_arr.shape[1] != n_features_meta:
            raise ValueError(
                f"Feature mismatch: model expects {n_features_meta} features, "
                f"but X has shape {X_arr.shape}."
            )

    task = getattr(artifact_meta, "kind", None) or "classification"
    eval_dict = getattr(artifact_meta, "eval", None) or {}
    ev = EvalModel.parse_obj(eval_dict)
    eval_kind = "regression" if task == "regression" else "classification"

    return pipeline, X_arr, y_arr, task, ev, eval_kind
