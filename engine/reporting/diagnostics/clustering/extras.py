from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

from engine.reporting.common.json_safety import ReportError, add_report_error

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:
    from sklearn.cluster import KMeans  # type: ignore
except Exception:  # pragma: no cover
    KMeans = None  # type: ignore

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

from .core_metrics import _as_2d, _final_estimator, _transform_through_pipeline


def _histogram_payload(values: Any, *, bins: int = 30) -> Optional[Mapping[str, Any]]:
    """Return a small histogram payload {x, y} for JSON transport."""
    if np is None:
        return None
    try:
        v = np.asarray(values, dtype=float).reshape(-1)
        v = v[np.isfinite(v)]
        if v.size < 2:
            return None
        vmin = float(np.min(v))
        vmax = float(np.max(v))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            return None

        n_bins = int(max(5, min(int(bins), int(np.floor(np.sqrt(v.size) * 2)))))
        hist, edges = np.histogram(v, bins=n_bins)
        centers = (edges[:-1] + edges[1:]) / 2.0
        return {
            "x": [float(x) for x in centers.tolist()],
            "y": [int(c) for c in hist.tolist()],
        }
    except Exception:
        return None


def _agglomerative_dendrogram_payload(
    model: Any,
    labels: Any = None,
    *,
    max_leaves: int = 200,
) -> Tuple[Optional[Mapping[str, Any]], List[str]]:
    warnings: List[str] = []
    if np is None:
        return None, ["numpy unavailable; cannot compute dendrogram."]
    if dendrogram is None:
        return None, ["scipy unavailable; cannot compute dendrogram."]

    est = _final_estimator(model)
    children = getattr(est, "children_", None)
    distances = getattr(est, "distances_", None)
    if children is None:
        return None, []
    if distances is None:
        return None, [
            "AgglomerativeClustering has no distances_. Enable compute_distances=True to render a dendrogram."
        ]

    try:
        children = np.asarray(children, dtype=float)
        distances = np.asarray(distances, dtype=float).reshape(-1)
        if children.ndim != 2 or children.shape[1] != 2:
            return None, ["Invalid children_ shape for dendrogram."]

        n_merges = int(children.shape[0])
        n_leaves = int(n_merges + 1)

        if n_leaves > int(max_leaves):
            return None, [
                f"Dendrogram skipped: too many leaves ({n_leaves}) for display (limit={int(max_leaves)})."
            ]

        counts = np.zeros((n_merges,), dtype=float)
        for i in range(n_merges):
            c1, c2 = int(children[i, 0]), int(children[i, 1])
            cnt1 = 1.0 if c1 < n_leaves else counts[c1 - n_leaves]
            cnt2 = 1.0 if c2 < n_leaves else counts[c2 - n_leaves]
            counts[i] = cnt1 + cnt2

        Z = np.column_stack([children, distances[:n_merges], counts])
        dd = dendrogram(Z, no_plot=True)

        leaf_order = [int(v) for v in dd.get("leaves", [])]
        leaf_labels = [str(v) for v in dd.get("ivl", [])]
        leaf_x = [float(5 + 10 * i) for i in range(len(leaf_order))]

        leaf_cluster_ids: Optional[List[int]] = None
        if labels is not None:
            try:
                y = np.asarray(labels).reshape(-1)
                if y.size >= len(leaf_order) and len(leaf_order) > 0:
                    leaf_cluster_ids = [int(y[i]) for i in leaf_order]
            except Exception:
                leaf_cluster_ids = None

        segments = []
        for xs, ys in zip(dd.get("icoord", []), dd.get("dcoord", [])):
            segments.append({"x": [float(v) for v in xs], "y": [float(v) for v in ys]})

        payload: Dict[str, Any] = {
            "segments": segments,
            "leaf_order": leaf_order,
            "leaf_labels": leaf_labels,
            "leaf_x": leaf_x,
        }
        if leaf_cluster_ids is not None:
            payload["leaf_cluster_ids"] = leaf_cluster_ids

        return payload, warnings
    except Exception as e:
        return None, [f"Failed to compute dendrogram: {type(e).__name__}: {e}"]


def build_plot_data(
    *,
    model: Any,
    X: Any,
    labels: Any,
    per_sample: Optional[Mapping[str, Any]] = None,
    embedding: Optional[Mapping[str, Any]] = None,
    max_points: int = 5000,
    seed: int = 0,
) -> Tuple[Mapping[str, Any], List[str]]:
    """Build JSON-friendly plot payloads for the frontend.

    Derived values for plots only. Never raises in normal use.

    This is intentionally conservative and best-effort.
    """
    warnings: List[str] = []
    out: Dict[str, Any] = {}

    if np is None:
        return out, ["numpy unavailable; cannot build plot data."]

    try:
        y = np.asarray(labels).reshape(-1)
    except Exception as e:
        return out, [f"Invalid labels for plot_data: {e}"]

    try:
        Xt = _transform_through_pipeline(model, X)
        Xa = _as_2d(Xt)
    except Exception:
        try:
            Xa = _as_2d(X)
        except Exception as e:
            return out, [f"Invalid inputs for plot_data: {e}"]

    n = int(Xa.shape[0])
    if n == 0:
        return out, ["Empty X; cannot build plot data."]
    if y.size != n:
        m = int(min(y.size, n))
        warnings.append(
            f"Length mismatch between X ({n}) and labels ({int(y.size)}). Truncating to {m}."
        )
        Xa = Xa[:m]
        y = y[:m]
        n = m

    # Ensure per_sample and embedding are present if caller didn't provide them.
    # We treat them as inputs to plotting only; if absent, we compute minimally.
    if embedding is None:
        embedding = None

    out["embedding_2d"] = embedding

    # --- silhouette sample histogram (if available) ---
    if silhouette_samples is not None:
        try:
            sil = silhouette_samples(Xa, y)
            out["silhouette_hist"] = _histogram_payload(sil, bins=40)
        except Exception as e:
            warnings.append(f"Failed silhouette_samples: {type(e).__name__}: {e}")

    # --- k-distance plot for DBSCAN intuition (k=4 by default) ---
    if NearestNeighbors is not None:
        try:
            k = 4
            nn = NearestNeighbors(n_neighbors=min(k, n))
            nn.fit(Xa)
            d, _ = nn.kneighbors(Xa)
            # distance to k-th neighbor
            dk = np.sort(d[:, -1])
            if dk.size:
                out["k_distance"] = {
                    "x": [int(i) for i in range(int(dk.size))],
                    "y": [float(v) for v in dk.tolist()],
                    "k": int(k),
                }
        except Exception as e:
            warnings.append(f"Failed k-distance: {type(e).__name__}: {e}")

    # --- elbow curve for KMeans-style models (small k-range) ---
    if KMeans is not None:
        try:
            est = _final_estimator(model)
            if hasattr(est, "n_clusters"):
                k0 = int(getattr(est, "n_clusters", 0) or 0)
                k_min = max(2, min(2, k0))
                k_max = max(3, min(12, max(k0 + 3, 6)))
                ks = list(range(k_min, k_max + 1))

                # subsample for elbow curve
                idx = np.arange(n)
                if max_points and n > int(max_points):
                    rng = np.random.default_rng(int(seed))
                    idx = rng.choice(n, size=int(max_points), replace=False)
                Xsub = Xa[idx]

                inertias = []
                for k in ks:
                    km = KMeans(n_clusters=int(k), random_state=int(seed), n_init="auto")
                    km.fit(Xsub)
                    inertias.append(float(getattr(km, "inertia_", float("nan"))))

                out["elbow"] = {"k": ks, "inertia": inertias}
        except Exception as e:
            warnings.append(f"Failed elbow curve: {type(e).__name__}: {e}")

    # --- dendrogram for Agglomerative (best-effort) ---
    dendro, w = _agglomerative_dendrogram_payload(model, y)
    if dendro is not None:
        out["dendrogram"] = dendro
    warnings.extend(w)

    return out, warnings
