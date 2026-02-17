from __future__ import annotations

"""Optional heavy dependencies for clustering plot payloads.

This module centralizes best-effort imports so individual plot builders can
remain small and focused.
"""

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:
    from sklearn.cluster import KMeans, MiniBatchKMeans  # type: ignore
except Exception:  # pragma: no cover
    KMeans = None  # type: ignore
    MiniBatchKMeans = None  # type: ignore

try:
    from sklearn.neighbors import NearestNeighbors  # type: ignore
except Exception:  # pragma: no cover
    NearestNeighbors = None  # type: ignore

try:
    from sklearn.metrics import silhouette_samples  # type: ignore
except Exception:  # pragma: no cover
    silhouette_samples = None  # type: ignore

try:
    from scipy.cluster.hierarchy import dendrogram  # type: ignore
except Exception:  # pragma: no cover
    dendrogram = None  # type: ignore
