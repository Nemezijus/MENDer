from __future__ import annotations

"""Scoring helpers for unsupervised (clustering) SearchCV."""

from typing import Any, Optional

import numpy as np

from engine.components.evaluation.unsupervised_scoring import compute_unsupervised_metrics

from .param_space import coerce_metric


def supports_predict(pipe: Any) -> bool:
    """Return True iff the *final estimator* supports predicting labels."""

    try:
        if hasattr(pipe, "steps") and pipe.steps:
            return hasattr(pipe.steps[-1][1], "predict")
    except Exception:
        pass
    return hasattr(pipe, "predict")


def transform_features(pipe: Any, X: np.ndarray) -> np.ndarray:
    """Transform X using Pipeline preprocessing if available."""

    try:
        return np.asarray(pipe[:-1].transform(X))
    except Exception:
        return np.asarray(X)


def labels_for_training(pipe: Any, X_train: np.ndarray) -> Optional[np.ndarray]:
    """Get labels for the training set after fitting.

    For many clustering estimators, labels are available via ``labels_``.
    If not, try predicting labels on the training data.
    """

    est = None
    try:
        if hasattr(pipe, "named_steps"):
            est = pipe.named_steps.get("clf")
    except Exception:
        est = None

    if est is None and hasattr(pipe, "steps") and pipe.steps:
        est = pipe.steps[-1][1]

    labels = getattr(est, "labels_", None) if est is not None else None
    if labels is not None:
        labels = np.asarray(labels)
        if labels.shape[0] == X_train.shape[0]:
            return labels

    if supports_predict(pipe):
        try:
            return np.asarray(pipe.predict(X_train))
        except Exception:
            return None

    return None


def score_from_labels(Z: np.ndarray, labels: np.ndarray, metric: str) -> float:
    metric = coerce_metric(metric)
    m, _warnings = compute_unsupervised_metrics(Z, labels, [metric])
    v = m.get(metric)
    return float(v) if v is not None else float("nan")
