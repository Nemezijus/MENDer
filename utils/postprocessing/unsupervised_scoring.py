from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from shared_schemas.types import UnsupervisedMetricName

DEFAULT_UNSUPERVISED_METRICS: List[UnsupervisedMetricName] = [
    "silhouette",
    "davies_bouldin",
    "calinski_harabasz",
]


def compute_unsupervised_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    metrics: Optional[Iterable[UnsupervisedMetricName]] = None,
    *,
    ignore_noise_label: bool = True,
) -> Tuple[Dict[str, Optional[float]], List[str]]:
    """Compute common unsupervised validation metrics (post-fit diagnostics).

    Parameters
    ----------
    X : np.ndarray
        Input features used for unsupervised learning.

    labels : np.ndarray
        Cluster labels for each sample.

    metrics : Iterable[UnsupervisedMetricName] | None
        Which metrics to compute. If None, the default metric pack is used.

    ignore_noise_label : bool, default True
        If True and labels contain -1, noise points are excluded from metric computation.

    Returns
    -------
    (metrics, warnings)
        metrics is a dict[str, float|None]. Undefined / failed metrics are returned as None.
        warnings is a list of human-readable strings explaining undefined / failed metrics.
    """
    warnings: List[str] = []

    metrics_to_compute = list(metrics) if metrics is not None else list(DEFAULT_UNSUPERVISED_METRICS)

    X_arr = np.asarray(X)
    y = np.asarray(labels)

    mask = np.ones(y.shape[0], dtype=bool)
    if ignore_noise_label and np.any(y == -1):
        mask = y != -1
        warnings.append("Excluded noise points (label=-1) from unsupervised metric computation.")

    X_use = X_arr[mask]
    y_use = y[mask]

    n_samples = int(X_use.shape[0])
    unique = np.unique(y_use) if n_samples > 0 else np.array([])
    n_clusters = int(unique.shape[0])

    out: Dict[str, Optional[float]] = {m: None for m in metrics_to_compute}

    if n_samples < 2:
        warnings.append("Unsupervised metrics are undefined: fewer than 2 samples available after filtering.")
        return out, warnings

    if n_clusters < 2:
        warnings.append("Unsupervised metrics are undefined: fewer than 2 clusters found.")
        return out, warnings

    if n_clusters >= n_samples:
        warnings.append(
            "Silhouette score is undefined when the number of clusters is greater than or equal to the number of samples."
        )

    for m in metrics_to_compute:
        try:
            if m == "silhouette":
                if n_clusters >= n_samples:
                    out[m] = None
                else:
                    out[m] = float(silhouette_score(X_use, y_use))
            elif m == "davies_bouldin":
                out[m] = float(davies_bouldin_score(X_use, y_use))
            elif m == "calinski_harabasz":
                out[m] = float(calinski_harabasz_score(X_use, y_use))
            else:
                out[m] = None
                warnings.append(f"Unknown unsupervised metric name: {m}")
        except Exception as e:
            out[m] = None
            warnings.append(f"Failed to compute unsupervised metric '{m}': {e}")

    return out, warnings
