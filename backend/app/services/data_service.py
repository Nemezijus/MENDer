# backend/app/services/data_service.py
import numpy as np
from typing import Dict, Any, Tuple
from ..adapters.io_adapter import load_X_y

def compute_missingness(X: np.ndarray) -> Tuple[int, list]:
    if np.issubdtype(X.dtype, np.number):
        missing_mask = np.isnan(X)
        total = int(missing_mask.sum())
        by_col = missing_mask.sum(axis=0).astype(int).tolist()
        return total, by_col
    return 0, []

def _infer_task_and_y_summary(y: np.ndarray) -> Tuple[str, Dict[str, Any]]:
    """
    Heuristic:
      - If y is string/object dtype -> classification
      - If y is numeric:
          - classification if n_unique <= min(50, max(2, 0.05 * n_samples))
          - else regression
    """
    y = np.asarray(y).ravel()
    n = int(y.shape[0])
    y_dtype = y.dtype

    # Gather uniques with cap to avoid expensive operations on massive y
    uniq, counts = np.unique(y, return_counts=True)
    n_unique = int(uniq.shape[0])

    is_numeric = np.issubdtype(y_dtype, np.number)

    # Thresholds
    uniq_cap = 50
    ratio_cap = 0.05
    uniq_thresh = min(uniq_cap, max(2, int(np.ceil(ratio_cap * n))))

    if not is_numeric:
        task = "classification"
    else:
        task = "classification" if n_unique <= uniq_thresh else "regression"

    y_summary: Dict[str, Any] = {"n": n, "n_unique": n_unique, "dtype": str(y_dtype)}
    if task == "classification":
        # Provide classes (capped for safety) and counts
        classes = [c.item() if hasattr(c, "item") else c for c in uniq]
        class_counts = {str(k): int(v) for k, v in zip(classes, counts)}
        # cap classes list for huge cardinalities (though by rule we keep it small)
        y_summary.update({
            "classes": classes[:uniq_cap],
            "class_counts": class_counts,
        })
    else:
        # Regression stats
        y_num = y.astype(float)
        y_summary.update({
            "min": float(np.min(y_num)),
            "max": float(np.max(y_num)),
            "mean": float(np.mean(y_num)),
            "std": float(np.std(y_num)),
        })

    return task, y_summary

def inspect_data(payload) -> Dict[str, Any]:
    X, y = load_X_y(
        payload.npz_path,
        payload.x_key,
        payload.y_key,
        payload.x_path,
        payload.y_path,
    )

    n_samples = int(X.shape[0])
    n_features = int(X.shape[1])

    total_missing, by_column = compute_missingness(X)
    recommend_pca = n_features > 2 * n_samples
    reason = None
    if recommend_pca:
        reason = f"n_features ({n_features}) >> n_samples ({n_samples}); consider PCA."

    # --- NEW: infer task and build y_summary
    task_inferred, y_summary = _infer_task_and_y_summary(y)

    # Keep legacy fields to avoid breaking UI:
    # For classification, keep classes/class_counts; for regression, keep them empty
    classes = y_summary.get("classes", [])
    class_counts = y_summary.get("class_counts", {})

    return {
        "n_samples": n_samples,
        "n_features": n_features,
        "classes": classes,                 # legacy
        "class_counts": class_counts,       # legacy
        "missingness": {"total": total_missing, "by_column": by_column},
        "suggestions": {"recommend_pca": recommend_pca, "reason": reason},
        # NEW fields:
        "task_inferred": task_inferred,     # "classification" | "regression"
        "y_summary": y_summary,             # compact, task-aware summary
    }
