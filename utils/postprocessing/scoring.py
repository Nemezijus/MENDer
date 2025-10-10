# utils/postprocessing/scoring.py
from __future__ import annotations

from typing import Literal, Optional, Sequence
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    log_loss,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    r2_score,
    explained_variance_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)


# ------------------------- Helpers -------------------------

def _as_1d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    return a.ravel()

def _check_len(y_true: np.ndarray, y_pred_like: np.ndarray, name: str):
    if y_true.shape[0] != y_pred_like.shape[0]:
        raise ValueError(f"Length mismatch: y_true({y_true.shape[0]}) vs {name}({y_pred_like.shape[0]}).")

def _infer_kind(y_true: np.ndarray) -> Literal["classification", "regression"]:
    """
    Heuristic:
      - If y_true has <= 20 unique values and looks discrete → classification
      - Else → regression
    Prefer passing `kind` explicitly in critical code.
    """
    y_true = _as_1d(y_true)
    uniq = np.unique(y_true)
    if y_true.dtype.kind in "iu" or (len(uniq) <= 20):
        return "classification"
    return "regression"


# ------------------------- Classification -------------------------

_CLASS_METRICS = {
    # hard-label metrics (need y_pred)
    "accuracy": lambda y, yhat, **kw: accuracy_score(y, yhat),
    "balanced_accuracy": lambda y, yhat, **kw: balanced_accuracy_score(y, yhat),
    "f1_macro": lambda y, yhat, **kw: f1_score(y, yhat, average="macro"),
    "f1_micro": lambda y, yhat, **kw: f1_score(y, yhat, average="micro"),
    "f1_weighted": lambda y, yhat, **kw: f1_score(y, yhat, average="weighted"),
    "precision_macro": lambda y, yhat, **kw: precision_score(y, yhat, average="macro", zero_division=0),
    "recall_macro": lambda y, yhat, **kw: recall_score(y, yhat, average="macro", zero_division=0),
    # probability / score-based metrics (need y_proba or y_score)
    "log_loss": "log_loss",                # needs y_proba
    "roc_auc_ovr": "roc_auc_ovr",          # needs y_score or y_proba
    "roc_auc_ovo": "roc_auc_ovo",          # needs y_score or y_proba
    "avg_precision_macro": "avg_precision_macro",  # needs y_score or y_proba (one-vs-rest)
}

def _classification_score(
    y_true: np.ndarray,
    *,
    metric: str = "accuracy",
    y_pred: Optional[np.ndarray] = None,
    y_proba: Optional[np.ndarray] = None,
    y_score: Optional[np.ndarray] = None,
    labels: Optional[Sequence] = None,
) -> float:
    """
    Compute a classification metric. Use y_pred for hard-label metrics.
    For probabilistic metrics, pass y_proba (preferred) or y_score.
    """
    y_true = _as_1d(y_true)

    if metric not in _CLASS_METRICS:
        raise ValueError(f"Unknown classification metric '{metric}'. Supported: {list(_CLASS_METRICS)}")

    fn = _CLASS_METRICS[metric]

    # Hard-label metrics
    if callable(fn):
        if y_pred is None:
            raise ValueError(f"Metric '{metric}' requires y_pred (hard labels).")
        y_pred = _as_1d(y_pred)
        _check_len(y_true, y_pred, "y_pred")
        return float(fn(y_true, y_pred))

    # Probabilistic / score-based metrics
    # Prefer probabilities if available; else decision scores.
    Z = None
    if y_proba is not None:
        Z = np.asarray(y_proba)
        _check_len(y_true, Z, "y_proba")
    elif y_score is not None:
        Z = np.asarray(y_score)
        _check_len(y_true, Z, "y_score")
    else:
        raise ValueError(f"Metric '{metric}' requires y_proba or y_score.")

    # Align labels for multi-class cases
    # For binary with probabilities, expect shape (n_samples, 2) or (n_samples,) for positive class.
    # We convert to a consistent form as needed below.
    if metric == "log_loss":
        # log_loss expects class probabilities (n_samples, n_classes)
        return float(log_loss(y_true, Z, labels=labels))
    elif metric in ("roc_auc_ovr", "roc_auc_ovo"):
        multi_class = "ovr" if metric.endswith("ovr") else "ovo"
        return float(roc_auc_score(y_true, Z, multi_class=multi_class, labels=labels, average="macro"))
    elif metric == "avg_precision_macro":
        # One-vs-rest average precision (macro)
        return float(average_precision_score(y_true, Z, average="macro"))

    raise RuntimeError("Unreachable branch in _classification_score.")


# ------------------------- Regression -------------------------

_REG_METRICS = {
    "r2": lambda y, yhat, **kw: r2_score(y, yhat),
    "explained_variance": lambda y, yhat, **kw: explained_variance_score(y, yhat),
    "mse": lambda y, yhat, **kw: mean_squared_error(y, yhat),
    "rmse": lambda y, yhat, **kw: np.sqrt(mean_squared_error(y, yhat)),
    "mae": lambda y, yhat, **kw: mean_absolute_error(y, yhat),
    "mape": lambda y, yhat, **kw: mean_absolute_percentage_error(y, yhat),
}

def _regression_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    metric: str = "r2",
) -> float:
    y_true = _as_1d(y_true)
    y_pred = _as_1d(y_pred)
    _check_len(y_true, y_pred, "y_pred")

    if metric not in _REG_METRICS:
        raise ValueError(f"Unknown regression metric '{metric}'. Supported: {list(_REG_METRICS)}")
    return float(_REG_METRICS[metric](y_true, y_pred))


# ------------------------- Public API -------------------------

def score(
    y_true: np.ndarray,
    y_pred: Optional[np.ndarray] = None,
    *,
    kind: Literal["auto", "classification", "regression"] = "auto",
    metric: str = "accuracy",
    y_proba: Optional[np.ndarray] = None,
    y_score: Optional[np.ndarray] = None,
    labels: Optional[Sequence] = None,
) -> float:
    """
    Universal scorer for classification and regression.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        Ground truth targets.
    y_pred : array-like, optional
        Predicted hard outputs. Required for hard-label metrics and all regression metrics.
    kind : {'auto','classification','regression'}, default='auto'
        Problem type. If 'auto', we infer from `y_true` (heuristic).
    metric : str, default='accuracy'
        Classification: 'accuracy','balanced_accuracy','f1_macro','f1_micro','f1_weighted',
                        'precision_macro','recall_macro','log_loss','roc_auc_ovr','roc_auc_ovo',
                        'avg_precision_macro'
        Regression: 'r2','explained_variance','mse','rmse','mae','mape'
    y_proba : array-like, optional
        Class probabilities (n_samples, n_classes) or (n_samples,) for positive class (binary).
    y_score : array-like, optional
        Decision scores/margins. Used if y_proba is not provided.
    labels : sequence, optional
        Label ordering for some metrics (e.g., log_loss, roc_auc in multiclass).

    Returns
    -------
    float
        The requested metric value.

    Notes
    -----
    - For classification probability-based metrics, prefer `y_proba`. If absent, `y_score` is used.
    - For regression, `y_pred` is required.
    - If `kind='auto'`, a simple heuristic is used; pass `kind` explicitly when in doubt.

    Examples
    --------
    1. score_val = score(
        y_test, y_pred,
        kind="classification",
        metric="accuracy",   # or 'balanced_accuracy', 'f1_macro', ...
    )
    
    2. y_proba = model.predict_proba(X_test)  # shape (n_samples, n_classes)
    ll = score(y_test, kind="classification", metric="log_loss", y_proba=y_proba)
    auc = score(y_test, kind="classification", metric="roc_auc_ovr", y_proba=y_proba)

    3. r2 = score(y_test, y_pred, kind="regression", metric="r2")
    rmse = score(y_test, y_pred, kind="regression", metric="rmse")
    """
    y_true = _as_1d(y_true)
    if kind == "auto":
        kind = _infer_kind(y_true)

    if kind == "classification":
        return _classification_score(
            y_true,
            metric=metric,
            y_pred=y_pred,
            y_proba=y_proba,
            y_score=y_score,
            labels=labels,
        )
    elif kind == "regression":
        if y_pred is None:
            raise ValueError("Regression scoring requires y_pred.")
        return _regression_score(y_true, y_pred, metric=metric)
    else:
        raise ValueError("kind must be one of {'auto','classification','regression'}.")


def classification_report_quick(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """
    Small convenience: return a dict of common classification scores.
    """
    y_true = _as_1d(y_true); y_pred = _as_1d(y_pred)
    _check_len(y_true, y_pred, "y_pred")
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }
