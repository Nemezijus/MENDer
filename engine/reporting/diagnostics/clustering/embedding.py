from __future__ import annotations

from typing import Any, List, Mapping, Optional, Tuple

from engine.reporting.common.json_safety import ReportError, add_report_error

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:
    from sklearn.decomposition import PCA  # type: ignore
except Exception:  # pragma: no cover
    PCA = None  # type: ignore

from .core_metrics import _as_2d


def embedding_2d(
    X: Any,
    *,
    method: str = "pca",
    max_points: int = 5000,
    seed: int = 0,
) -> Tuple[Optional[Mapping[str, List[float]]], List[str]]:
    warnings: List[str] = []
    if np is None:
        return None, ["numpy unavailable; cannot compute embedding_2d."]

    try:
        Xa = _as_2d(X)
    except Exception as e:
        return None, [f"Invalid X for embedding_2d: {e}"]

    n = int(Xa.shape[0])
    if n == 0:
        return None, ["Empty X; cannot compute embedding_2d."]

    idx = np.arange(n)
    if max_points is not None and int(max_points) > 0 and n > int(max_points):
        try:
            rng = np.random.default_rng(int(seed))
            idx = rng.choice(n, size=int(max_points), replace=False)
            idx = np.sort(idx)
            warnings.append(f"Downsampled embedding_2d to {int(max_points)} points.")
        except Exception:
            idx = idx[: int(max_points)]
            warnings.append(f"Downsampled embedding_2d to first {int(max_points)} points.")

    Xs = Xa[idx]

    if method != "pca":
        warnings.append(f"Unknown embedding method '{method}'. Falling back to PCA.")
        method = "pca"

    if PCA is None:
        return None, ["scikit-learn unavailable; cannot compute PCA embedding."]

    try:
        pca = PCA(n_components=2, random_state=int(seed))
        Z = pca.fit_transform(Xs)
        return {
            "x": [float(v) for v in Z[:, 0].tolist()],
            "y": [float(v) for v in Z[:, 1].tolist()],
            "idx": [int(i) for i in idx.tolist()],
        }, warnings
    except Exception as e:
        return None, [f"Failed to compute PCA embedding_2d: {type(e).__name__}: {e}"]
