import numpy as np

def pearson(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Pearson correlation coefficient between two 1D vectors.

    Args:
        x (np.ndarray): First vector.
        y (np.ndarray): Second vector.

    Returns:
        float: Pearson correlation coefficient in [-1, 1].

    Raises:
        ValueError: If inputs are not 1D or have different lengths.
    """
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("Both inputs must be numpy arrays.")
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Both inputs must be 1D arrays.")
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"Vectors must have the same length, got {x.shape[0]} vs {y.shape[0]}.")

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    num = np.sum((x - x_mean) * (y - y_mean))
    den = np.sqrt(np.sum((x - x_mean) ** 2) * np.sum((y - y_mean) ** 2))

    if den == 0:
        return 0.0  # define correlation as 0 if denominator is zero
    return float(num / den)