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

    # y alignment is already enforced in your loaders via _coerce_shapes
    # We only compute summary here:
    uniq, counts = np.unique(y, return_counts=True)
    classes = [c.item() if hasattr(c, "item") else c for c in uniq]
    class_counts = {str(k): int(v) for k, v in zip(classes, counts)}

    total_missing, by_column = compute_missingness(X)

    recommend_pca = n_features > 2 * n_samples
    reason = None
    if recommend_pca:
        reason = f"n_features ({n_features}) >> n_samples ({n_samples}); consider PCA."

    return {
        "n_samples": n_samples,
        "n_features": n_features,
        "classes": classes,
        "class_counts": class_counts,
        "missingness": {"total": total_missing, "by_column": by_column},
        "suggestions": {"recommend_pca": recommend_pca, "reason": reason},
    }
