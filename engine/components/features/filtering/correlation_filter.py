import numpy as np
from engine.core.stats.correlations import pearson

def filter_out_uncorrelating_features(
    feature_matrix: np.ndarray,
    label_vector: np.ndarray,
    corr_method: str = "pearson",
    thr: float = 0.0,
) -> np.ndarray:
    """
    Filter out features (columns) that do not correlate with the label vector.

    Args:
        feature_matrix (np.ndarray): 2D array (n_samples, n_features).
        label_vector (np.ndarray): 1D array (n_samples,).
        corr_method (str): correlation method ("pearson" supported for now).
        thr (float): absolute correlation threshold; keep features with |r| > thr.

    Returns:
        np.ndarray: Filtered feature matrix with selected columns.

    Raises:
        ValueError: If shapes mismatch or unsupported corr_method.
    """
    if feature_matrix.ndim != 2:
        raise ValueError("feature_matrix must be 2D (n_samples, n_features).")
    if label_vector.ndim != 1:
        raise ValueError("label_vector must be 1D.")
    if feature_matrix.shape[0] != label_vector.shape[0]:
        raise ValueError("feature_matrix rows must match label_vector length.")

    if corr_method != "pearson":
        raise NotImplementedError(f"Correlation method '{corr_method}' is not supported yet.")

    selected_cols = []
    for j in range(feature_matrix.shape[1]):
        r = pearson(feature_matrix[:, j], label_vector)
        if abs(r) > thr:
            selected_cols.append(j)

    if not selected_cols:
        # return empty matrix with same number of rows
        return np.empty((feature_matrix.shape[0], 0))

    return feature_matrix[:, selected_cols]