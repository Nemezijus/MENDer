from __future__ import annotations

"""Shared utilities for unsupervised (clustering) tuning."""

from typing import Any, List, Optional, Sequence

import math

import numpy as np
from sklearn.model_selection import KFold


UNSUPERVISED_METRICS = {"silhouette", "davies_bouldin", "calinski_harabasz"}


def coerce_metric(metric: str) -> str:
    """Return a supported metric name, falling back to ``silhouette``."""
    m = str(metric)
    return m if m in UNSUPERVISED_METRICS else "silhouette"


def prefers_higher(metric: str) -> bool:
    """Whether larger scores are better for the given intrinsic metric."""
    # Davies–Bouldin: lower is better; the others: higher is better.
    return coerce_metric(metric) != "davies_bouldin"


def predict_supported(estimator) -> bool:
    """Return True iff the *final estimator* supports predict()."""
    try:
        if hasattr(estimator, "steps") and estimator.steps:
            return hasattr(estimator.steps[-1][1], "predict")
    except Exception:
        pass
    return hasattr(estimator, "predict")


def make_kfold(cv: int, *, shuffle: bool, random_state: Optional[int]):
    return KFold(n_splits=int(cv), shuffle=bool(shuffle), random_state=random_state)


def sanitize_float_list(values: Sequence[float]) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    for v in values:
        try:
            fv = float(v)
            out.append(fv if math.isfinite(fv) else None)
        except Exception:
            out.append(None)
    return out


def to_py(v: Any) -> Any:
    """Convert numpy scalars/arrays into pure python types for JSON."""
    if isinstance(v, (np.generic,)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v
