from __future__ import annotations

"""Parameter-space helpers for unsupervised tuning."""

from typing import Optional

from sklearn.model_selection import KFold


UNSUPERVISED_METRICS = {"silhouette", "davies_bouldin", "calinski_harabasz"}


def coerce_metric(metric: str) -> str:
    return metric if metric in UNSUPERVISED_METRICS else "silhouette"


def prefers_higher(metric: str) -> bool:
    """Return True iff higher values mean better quality."""

    # Davies–Bouldin: lower is better; the others: higher is better.
    return metric != "davies_bouldin"


def make_kfold(cv: int, *, shuffle: bool, random_state: Optional[int]):
    return KFold(n_splits=int(cv), shuffle=bool(shuffle), random_state=random_state)
