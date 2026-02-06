from __future__ import annotations

from typing import Literal, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    explained_variance_score,
    log_loss,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from .helpers import _as_1d, _check_len


def _infer_kind(y_true: np.ndarray) -> Literal["classification", "regression"]:
    """Heuristic kind inference.

    Prefer passing `kind` explicitly in critical code.
    """
    y_true = _as_1d(y_true)
    uniq = np.unique(y_true)
    if y_true.dtype.kind in "iu" or (len(uniq) <= 20):
        return "classification"
    return "regression"


PROBA_METRICS = {"log_loss", "roc_auc_ovr", "roc_auc_ovo", "avg_precision_macro"}


_CLASS_METRICS = {
    # hard-label metrics (need y_pred)
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "f1_macro": "f1_macro",
    "f1_micro": "f1_micro",
    "f1_weighted": "f1_weighted",
    "precision_macro": "precision_macro",
    "recall_macro": "recall_macro",
    # probability / score-based metrics (need y_proba or y_score)
    "log_loss": "log_loss",
    "roc_auc_ovr": "roc_auc_ovr",
    "roc_auc_ovo": "roc_auc_ovo",
    "avg_precision_macro": "avg_precision_macro",
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
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

    y_true = _as_1d(y_true)

    if metric not in _CLASS_METRICS:
        raise ValueError(
            f"Unknown classification metric '{metric}'. Supported: {list(_CLASS_METRICS)}"
        )

    # Hard-label metrics
    if metric not in PROBA_METRICS:
        if y_pred is None:
            raise ValueError(f"Metric '{metric}' requires y_pred (hard labels).")
        y_pred = _as_1d(y_pred)
        _check_len(y_true, y_pred, "y_pred")

        if metric == "accuracy":
            return float(accuracy_score(y_true, y_pred))
        if metric == "balanced_accuracy":
            return float(balanced_accuracy_score(y_true, y_pred))
        if metric == "f1_macro":
            return float(f1_score(y_true, y_pred, average="macro"))
        if metric == "f1_micro":
            return float(f1_score(y_true, y_pred, average="micro"))
        if metric == "f1_weighted":
            return float(f1_score(y_true, y_pred, average="weighted"))
        if metric == "precision_macro":
            return float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        if metric == "recall_macro":
            return float(recall_score(y_true, y_pred, average="macro", zero_division=0))

        raise RuntimeError("Unreachable branch in classification hard metrics")

    # Probabilistic / score-based metrics
    Z = None
    if y_proba is not None:
        Z = np.asarray(y_proba)
        _check_len(y_true, Z, "y_proba")
    elif y_score is not None:
        Z = np.asarray(y_score)
        _check_len(y_true, Z, "y_score")
    else:
        raise ValueError(f"Metric '{metric}' requires y_proba or y_score.")

    if metric == "log_loss":
        return float(log_loss(y_true, Z, labels=labels))
    if metric in ("roc_auc_ovr", "roc_auc_ovo"):
        multi_class = "ovr" if metric.endswith("ovr") else "ovo"
        return float(
            roc_auc_score(
                y_true,
                Z,
                multi_class=multi_class,
                labels=labels,
                average="macro",
            )
        )
    if metric == "avg_precision_macro":
        return float(average_precision_score(y_true, Z, average="macro"))

    raise RuntimeError("Unreachable branch in _classification_score")


_REG_METRICS = {
    "r2": lambda y, yhat: r2_score(y, yhat),
    "explained_variance": lambda y, yhat: explained_variance_score(y, yhat),
    "mse": lambda y, yhat: mean_squared_error(y, yhat),
    "rmse": lambda y, yhat: np.sqrt(mean_squared_error(y, yhat)),
    "mae": lambda y, yhat: mean_absolute_error(y, yhat),
    "mape": lambda y, yhat: mean_absolute_percentage_error(y, yhat),
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
        raise ValueError(
            f"Unknown regression metric '{metric}'. Supported: {list(_REG_METRICS)}"
        )
    return float(_REG_METRICS[metric](y_true, y_pred))


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
    """Universal scorer for classification and regression."""
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
    if kind == "regression":
        if y_pred is None:
            raise ValueError("Regression scoring requires y_pred.")
        return _regression_score(y_true, y_pred, metric=metric)

    raise ValueError("kind must be one of {'auto','classification','regression'}.")
