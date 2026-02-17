from __future__ import annotations

from typing import Any

import numpy as np

from engine.contracts.results.prediction import UnsupervisedApplyResult, UnsupervisedApplyRow


def apply_unsupervised(
    *,
    pipeline: Any,
    X_arr: np.ndarray,
    max_preview_rows: int,
) -> UnsupervisedApplyResult:
    """Apply an unsupervised pipeline to X and return cluster IDs."""

    n_samples = int(X_arr.shape[0])
    n_features = int(X_arr.shape[1])

    try:
        cluster_ids = np.asarray(pipeline.predict(X_arr)).reshape(-1)
    except Exception as e:
        raise ValueError(f"Unsupervised pipeline does not support predict(...): {e}") from e

    n_preview = min(int(max_preview_rows), n_samples)
    rows = [
        UnsupervisedApplyRow(index=int(i), cluster_id=int(cluster_ids[i]))
        for i in range(n_preview)
    ]

    return UnsupervisedApplyResult(
        n_samples=n_samples,
        n_features=n_features,
        preview=rows,
        notes=[],
    )
