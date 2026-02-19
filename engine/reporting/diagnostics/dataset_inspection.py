from __future__ import annotations

"""Dataset inspection helpers.

This module produces *UI-facing* summaries of datasets prior to training or
prediction. It is intentionally lightweight:

- No file IO (callers should load arrays elsewhere)
- No backend/FastAPI concepts

The primary consumer is the backend upload/inspect endpoints.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np


def compute_missingness(X: np.ndarray) -> Tuple[int, list[int]]:
    """Compute missing values for numeric 2D arrays."""

    if np.issubdtype(X.dtype, np.number):
        missing_mask = np.isnan(X)
        total = int(missing_mask.sum())
        by_col = missing_mask.sum(axis=0).astype(int).tolist()
        return total, by_col
    return 0, []


def infer_task_and_y_summary(y: np.ndarray) -> Tuple[str, Dict[str, Any]]:
    """Infer task kind and produce a compact y-summary.

    Heuristic:
      - If y is string/object dtype -> classification
      - If y is numeric:
          - classification if n_unique <= min(50, max(2, 0.05 * n_samples))
          - else regression
    """

    y = np.asarray(y).ravel()
    n = int(y.shape[0])
    y_dtype = y.dtype

    uniq, counts = np.unique(y, return_counts=True)
    n_unique = int(uniq.shape[0])

    is_numeric = np.issubdtype(y_dtype, np.number)

    uniq_cap = 50
    ratio_cap = 0.05
    uniq_thresh = min(uniq_cap, max(2, int(np.ceil(ratio_cap * n))))

    if not is_numeric:
        task = "classification"
    else:
        task = "classification" if n_unique <= uniq_thresh else "regression"

    y_summary: Dict[str, Any] = {"n": n, "n_unique": n_unique, "dtype": str(y_dtype)}
    if task == "classification":
        classes = [c.item() if hasattr(c, "item") else c for c in uniq]
        class_counts = {str(k): int(v) for k, v in zip(classes, counts)}
        y_summary.update({"classes": classes[:uniq_cap], "class_counts": class_counts})
    else:
        y_num = y.astype(float)
        y_summary.update(
            {
                "min": float(np.min(y_num)),
                "max": float(np.max(y_num)),
                "mean": float(np.mean(y_num)),
                "std": float(np.std(y_num)),
            }
        )

    return task, y_summary


def build_inspection_payload(
    *,
    X: np.ndarray,
    y: Optional[np.ndarray],
    treat_missing_y_as_unsupervised: bool,
) -> Dict[str, Any]:
    """Build a JSON-serializable inspection payload for loaded arrays."""

    n_samples = int(X.shape[0])
    n_features = int(X.shape[1])

    total_missing, by_column = compute_missingness(X)

    recommend_pca = n_features > 2 * n_samples
    reason = None
    if recommend_pca:
        reason = (
            f"Number of features ({n_features}) is much larger than the number of samples ({n_samples})."
        )

    base: Dict[str, Any] = {
        "n_samples": n_samples,
        "n_features": n_features,
        "missingness": {"total": total_missing, "by_column": by_column},
        "suggestions": {"recommend_pca": recommend_pca, "reason": reason},
    }

    if y is None:
        task = "unsupervised" if treat_missing_y_as_unsupervised else None
        return {
            **base,
            "classes": [],  # legacy
            "class_counts": {},  # legacy
            "task_inferred": task,
            "y_summary": {"present": False},
        }

    task_inferred, y_summary = infer_task_and_y_summary(y)
    classes = y_summary.get("classes", [])
    class_counts = y_summary.get("class_counts", {})

    return {
        **base,
        "classes": classes,  # legacy
        "class_counts": class_counts,  # legacy
        "task_inferred": task_inferred,
        "y_summary": y_summary,
    }
