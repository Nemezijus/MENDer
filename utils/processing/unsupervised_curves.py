from __future__ import annotations

"""Curve utilities for unsupervised (clustering) tuning.

These helpers keep the tuning strategy layer small by encapsulating the
implementation details of learning/validation curves for clustering models.

We intentionally *do not* invent a brand-new API: we leverage sklearn's
learning_curve / validation_curve with a custom unsupervised scorer. When a
model cannot predict labels for unseen samples (no ``predict``), validation
scores are reported as NaN so the UI can show N/A and a short explanatory note.
"""

from typing import Any, Dict, List, Optional, Sequence

import math
import numpy as np
from sklearn.model_selection import learning_curve as sk_learning_curve
from sklearn.model_selection import validation_curve as sk_validation_curve

from utils.postprocessing.scoring import make_estimator_scorer


UNSUPERVISED_METRICS = {"silhouette", "davies_bouldin", "calinski_harabasz"}


def _coerce_metric(metric: str) -> str:
    return metric if metric in UNSUPERVISED_METRICS else "silhouette"


def _sanitize(values: Sequence[float]) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    for v in values:
        try:
            fv = float(v)
            out.append(fv if math.isfinite(fv) else None)
        except Exception:
            out.append(None)
    return out


def _predict_supported(estimator) -> bool:
    """Return True iff the *final estimator* supports predict()."""
    try:
        if hasattr(estimator, "steps") and estimator.steps:
            return hasattr(estimator.steps[-1][1], "predict")
    except Exception:
        pass
    return hasattr(estimator, "predict")


def compute_unsupervised_learning_curve(
    *,
    estimator: Any,
    X: np.ndarray,
    metric: str,
    cv: Any,
    train_sizes: Optional[Sequence[float | int]] = None,
    n_steps: int = 5,
    n_jobs: int = 1,
    shuffle: bool = False,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute a learning curve for an unsupervised estimator.

    Returns arrays already sanitized to Optional[float] so the frontend can
    render N/A when validation scores are unavailable.
    """
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[0] < 2:
        raise ValueError("X must be a 2D array with at least 2 samples.")

    metric = _coerce_metric(str(metric))
    scorer = make_estimator_scorer("unsupervised", metric)

    if train_sizes is not None:
        ts = np.asarray(list(train_sizes))
    else:
        ts = np.linspace(0.1, 1.0, int(n_steps))

    sizes_abs, train_scores, val_scores = sk_learning_curve(
        estimator=estimator,
        X=X,
        y=None,
        train_sizes=ts,
        cv=cv,
        scoring=scorer,
        n_jobs=n_jobs,
        shuffle=shuffle,
        random_state=random_state,
        return_times=False,
    )

    train_mean = np.nanmean(train_scores, axis=1).tolist()
    train_std = np.nanstd(train_scores, axis=1).tolist()
    val_mean = np.nanmean(val_scores, axis=1).tolist()
    val_std = np.nanstd(val_scores, axis=1).tolist()

    predict_supported = _predict_supported(estimator)
    note = None
    if not predict_supported:
        note = (
            "Validation scores are unavailable for this model because it does not support predicting labels for unseen samples (no predict())."
        )

    return {
        "metric_used": metric,
        "note": note,
        "train_sizes": [int(s) for s in sizes_abs.tolist()],
        "train_scores_mean": _sanitize(train_mean),
        "train_scores_std": _sanitize(train_std),
        "val_scores_mean": _sanitize(val_mean),
        "val_scores_std": _sanitize(val_std),
    }


def compute_unsupervised_validation_curve(
    *,
    estimator: Any,
    X: np.ndarray,
    metric: str,
    cv: Any,
    param_name: str,
    param_range: Sequence[Any],
    n_jobs: int = 1,
) -> Dict[str, Any]:
    """Compute a validation curve for an unsupervised estimator."""
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[0] < 2:
        raise ValueError("X must be a 2D array with at least 2 samples.")

    metric = _coerce_metric(str(metric))
    scorer = make_estimator_scorer("unsupervised", metric)

    train_scores, val_scores = sk_validation_curve(
        estimator=estimator,
        X=X,
        y=None,
        param_name=param_name,
        param_range=list(param_range),
        cv=cv,
        scoring=scorer,
        n_jobs=n_jobs,
        error_score="raise",
    )

    train_mean = np.nanmean(train_scores, axis=1).tolist()
    train_std = np.nanstd(train_scores, axis=1).tolist()
    val_mean = np.nanmean(val_scores, axis=1).tolist()
    val_std = np.nanstd(val_scores, axis=1).tolist()

    predict_supported = _predict_supported(estimator)
    note = None
    if not predict_supported:
        note = (
            "Validation scores are unavailable for this model because it does not support predicting labels for unseen samples (no predict())."
        )

    return {
        "metric_used": metric,
        "note": note,
        "param_name": param_name,
        "param_range": list(param_range),
        "train_scores_mean": _sanitize(train_mean),
        "train_scores_std": _sanitize(train_std),
        "val_scores_mean": _sanitize(val_mean),
        "val_scores_std": _sanitize(val_std),
    }
