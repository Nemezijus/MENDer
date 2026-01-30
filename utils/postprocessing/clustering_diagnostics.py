from __future__ import annotations

"""Unsupervised (clustering) diagnostics (business-logic only).

This module computes compact, JSON-friendly diagnostics from:
  - the input features X
  - the fitted model (often a sklearn Pipeline)
  - per-sample cluster labels

It intentionally avoids backend/frontend imports.
Outputs are plain Python types (dict/list/float/int/bool).

The module is defensive:
  - Any computation failure returns None for that field and adds a warning.
  - It never raises in normal use.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:
    from sklearn.decomposition import PCA  # type: ignore
except Exception:  # pragma: no cover
    PCA = None  # type: ignore

try:
    from sklearn.metrics import silhouette_samples  # type: ignore
except Exception:  # pragma: no cover
    silhouette_samples = None  # type: ignore

try:
    from scipy.cluster.hierarchy import dendrogram  # type: ignore
except Exception:  # pragma: no cover
    dendrogram = None  # type: ignore


def _as_2d(X: Any) -> Any:
    if np is None:
        return X
    a = np.asarray(X)
    if a.ndim != 2:
        raise ValueError(f"X must be 2D; got shape {a.shape}.")
    return a


def _final_estimator(model: Any) -> Any:
    """Return the final estimator for a Pipeline-like model, else the model itself."""
    if hasattr(model, "steps") and isinstance(getattr(model, "steps"), list):
        try:
            return model.steps[-1][1]
        except Exception:
            return model
    return model


def _transform_through_pipeline(model: Any, X: Any) -> Any:
    """If model is a fitted sklearn Pipeline, transform X through all steps except last."""
    if np is None:
        return X
    try:
        if not (hasattr(model, "steps") and isinstance(getattr(model, "steps"), list) and len(model.steps) > 1):
            return X
        Xt = np.asarray(X)
        for _, step in model.steps[:-1]:
            if step is None or step == "passthrough":
                continue
            if hasattr(step, "transform"):
                Xt = step.transform(Xt)
            else:
                return X
        return Xt
    except Exception:
        return X


def _histogram_payload(values: Any, *, bins: int = 30) -> Optional[Mapping[str, Any]]:
    """Return a small histogram payload {x, y} for JSON transport.

    - x: bin centers
    - y: counts
    """
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
    """Build a dendrogram payload for AgglomerativeClustering.

    Uses scipy.cluster.hierarchy.dendrogram to get JSON-friendly segments.
    Returns (payload, warnings).

    Notes
    -----
    Plotly does not provide rich hover for dendrogram "segments" (lines) by default,
    so we additionally emit leaf coordinates + leaf cluster ids so the frontend
    can render leaf markers with hover tooltips and cluster coloring.
    """
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
            return None, [f"Dendrogram skipped: too many leaves ({n_leaves}) for display (limit={int(max_leaves)})."]
        # Compute counts per node (required by linkage matrix)
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

        # SciPy uses fixed x positions: 5, 15, 25, ... in leaf-order.
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


def cluster_summary(labels: Any) -> Mapping[str, Any]:
    """Return basic cluster counts and sizes.

    Notes:
      - If labels contain -1, it is treated as a "noise" label.
      - n_clusters excludes noise.
    """
    if np is None:
        return {
            "n_clusters": None,
            "n_noise": None,
            "noise_ratio": None,
            "cluster_sizes": None,
        }
    try:
        y = np.asarray(labels).reshape(-1)
        if y.size == 0:
            return {
                "n_clusters": 0,
                "n_noise": 0,
                "noise_ratio": 0.0,
                "cluster_sizes": {},
            }

        unique, counts = np.unique(y, return_counts=True)
        sizes = {int(k): int(v) for k, v in zip(unique.tolist(), counts.tolist())}
        n_noise = int(sizes.get(-1, 0))
        # clusters excluding noise
        n_clusters = int(sum(1 for k in sizes.keys() if int(k) != -1))
        noise_ratio = float(n_noise / y.size) if int(y.size) > 0 else 0.0
        return {
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "noise_ratio": noise_ratio,
            "cluster_sizes": sizes,
        }
    except Exception:
        return {
            "n_clusters": None,
            "n_noise": None,
            "noise_ratio": None,
            "cluster_sizes": None,
        }


def embedding_2d(
    X: Any,
    *,
    method: str = "pca",
    max_points: int = 5000,
    seed: int = 0,
) -> Tuple[Optional[Mapping[str, List[float]]], List[str]]:
    """Compute a simple 2D embedding payload for plotting.

    Returns (payload, warnings). payload is {x, y, idx}.
    """
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

    # Downsample to keep payload small
    idx = np.arange(n)
    if max_points is not None and int(max_points) > 0 and n > int(max_points):
        try:
            rng = np.random.default_rng(int(seed))
            idx = rng.choice(n, size=int(max_points), replace=False)
            idx = np.sort(idx)
            warnings.append(f"Downsampled embedding_2d to {int(max_points)} points.")
        except Exception:
            # If RNG fails, just take first max_points
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


def model_diagnostics(
    model: Any,
    X: Any,
    labels: Any,
) -> Tuple[Mapping[str, Any], List[str]]:
    """Return model-specific diagnostics.

    This function inspects the final estimator and gathers scalar diagnostics.
    It never raises; unknown fields are omitted.
    """
    warnings: List[str] = []
    out: Dict[str, Any] = {}

    if np is None:
        return out, ["numpy unavailable; cannot compute model diagnostics."]

    est = _final_estimator(model)
    Xa = _transform_through_pipeline(model, X)
    try:
        Xa = _as_2d(Xa)
    except Exception:
        # If transformation fails, try raw X
        try:
            Xa = _as_2d(X)
        except Exception:
            Xa = None

    # KMeans-style
    try:
        inertia = getattr(est, "inertia_", None)
        if inertia is not None:
            out["inertia"] = float(inertia)
        n_iter = getattr(est, "n_iter_", None)
        if n_iter is not None:
            out["n_iter"] = int(n_iter)
    except Exception:
        pass

    # GaussianMixture: aic/bic require X
    if Xa is not None:
        try:
            if hasattr(est, "aic"):
                out["aic"] = float(est.aic(Xa))
            if hasattr(est, "bic"):
                out["bic"] = float(est.bic(Xa))
        except Exception as e:
            warnings.append(f"Failed to compute AIC/BIC: {type(e).__name__}: {e}")

        # Mixture score_samples (log-likelihood per sample)
        try:
            if hasattr(est, "score_samples"):
                ll = np.asarray(est.score_samples(Xa)).reshape(-1)
                if ll.size:
                    out["mean_log_likelihood"] = float(np.mean(ll))
                    out["std_log_likelihood"] = float(np.std(ll))
        except Exception:
            pass

    # Convergence info (mixtures)
    try:
        conv = getattr(est, "converged_", None)
        if conv is not None:
            out["converged"] = bool(conv)
        nit = getattr(est, "n_iter_", None)
        if nit is not None and "n_iter" not in out:
            out["n_iter"] = int(nit)
        lb = getattr(est, "lower_bound_", None)
        if lb is not None:
            out["lower_bound"] = float(lb)
    except Exception:
        pass

    # Label summary (redundant but useful)
    try:
        out["label_summary"] = dict(cluster_summary(labels))
    except Exception:
        pass

    return out, warnings


def per_sample_outputs(
    model: Any,
    X: Any,
    labels: Any,
    *,
    include_cluster_probabilities: bool = False,
) -> Tuple[Mapping[str, Any], List[str]]:
    """Extract per-sample unsupervised outputs.

    Returns (payload, warnings). payload is a dict with at least:
      - cluster_id: list[int]

    Optional keys (if available):
      - is_noise: list[bool]
      - distance_to_center: list[float] (KMeans: min distance to centroid)
      - log_likelihood: list[float] (Mixtures: score_samples)
      - max_membership_prob: list[float] (Mixtures: max responsibility)
      - cluster_probabilities: list[list[float]] (Mixtures, optional)
    """
    warnings: List[str] = []
    out: Dict[str, Any] = {}

    if np is None:
        return out, ["numpy unavailable; cannot extract per-sample outputs."]

    y = np.asarray(labels).reshape(-1)
    out["cluster_id"] = [int(v) for v in y.tolist()]
    out["is_noise"] = [bool(int(v) == -1) for v in y.tolist()]

    est = _final_estimator(model)
    Xa = _transform_through_pipeline(model, X)
    try:
        Xa = _as_2d(Xa)
    except Exception:
        try:
            Xa = _as_2d(X)
        except Exception as e:
            warnings.append(f"Invalid X for per_sample_outputs: {e}")
            return out, warnings

    # KMeans distance to nearest centroid
    try:
        if hasattr(est, "transform"):
            d = np.asarray(est.transform(Xa))
            if d.ndim == 2 and d.shape[0] == Xa.shape[0] and d.shape[1] >= 1:
                out["distance_to_center"] = [float(v) for v in np.min(d, axis=1).tolist()]
    except Exception as e:
        warnings.append(f"Failed to compute distance_to_center: {type(e).__name__}: {e}")

    # Mixtures: responsibilities and per-sample log-likelihood
    try:
        if hasattr(est, "predict_proba"):
            P = np.asarray(est.predict_proba(Xa))
            if P.ndim == 2 and P.shape[0] == Xa.shape[0]:
                out["max_membership_prob"] = [float(v) for v in np.max(P, axis=1).tolist()]
                if include_cluster_probabilities:
                    out["cluster_probabilities"] = [[float(v) for v in row.tolist()] for row in P]
    except Exception as e:
        warnings.append(f"Failed to compute membership probabilities: {type(e).__name__}: {e}")

    try:
        if hasattr(est, "score_samples"):
            ll = np.asarray(est.score_samples(Xa)).reshape(-1)
            if ll.size == Xa.shape[0]:
                out["log_likelihood"] = [float(v) for v in ll.tolist()]
    except Exception as e:
        warnings.append(f"Failed to compute log_likelihood: {type(e).__name__}: {e}")

    # DBSCAN: core sample indicator (if available)
    try:
        core_idx = getattr(est, "core_sample_indices_", None)
        if core_idx is not None:
            core = np.zeros((Xa.shape[0],), dtype=bool)
            core[np.asarray(core_idx, dtype=int)] = True
            out["is_core"] = [bool(v) for v in core.tolist()]
    except Exception:
        pass

    return out, warnings


@dataclass
class UnsupervisedDiagnostics:
    """Convenience container for unsupervised diagnostics."""

    cluster_summary: Mapping[str, Any]
    model_diagnostics: Mapping[str, Any]
    per_sample: Mapping[str, Any]
    embedding_2d: Optional[Mapping[str, List[float]]] = None
    warnings: List[str] = None  # type: ignore[assignment]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_summary": dict(self.cluster_summary) if self.cluster_summary is not None else None,
            "model_diagnostics": dict(self.model_diagnostics) if self.model_diagnostics is not None else None,
            "per_sample": dict(self.per_sample) if self.per_sample is not None else None,
            "embedding_2d": dict(self.embedding_2d) if self.embedding_2d is not None else None,
            "warnings": list(self.warnings) if self.warnings is not None else [],
        }


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
    """
    warnings: List[str] = []
    out: Dict[str, Any] = {}
    if np is None:
        return out, ["numpy unavailable; cannot build plot data."]

    try:
        Xa = _as_2d(X)
        y = np.asarray(labels).reshape(-1)
    except Exception as e:
        return out, [f"Invalid inputs for plot_data: {e}"]

    n = int(Xa.shape[0])
    if n == 0:
        return out, ["Empty X; cannot build plot data."]

    idx = np.arange(n, dtype=int)
    if int(max_points) > 0 and n > int(max_points):
        try:
            rng = np.random.default_rng(int(seed))
            idx = rng.choice(n, size=int(max_points), replace=False)
            idx = np.sort(idx)
            warnings.append(f"Downsampled plot_data arrays to {int(max_points)} points.")
        except Exception:
            idx = idx[: int(max_points)]
            warnings.append(f"Downsampled plot_data arrays to first {int(max_points)} points.")

    # Cluster sizes + Lorenz/Gini
    try:
        cs = cluster_summary(y).get("cluster_sizes")
        if cs is not None:
            out["cluster_sizes"] = cs
            sizes = []
            for k, v in dict(cs).items():
                if int(k) == -1:
                    continue
                sizes.append(int(v))
            sizes = [s for s in sizes if s > 0]
            if sizes:
                srt = np.sort(np.asarray(sizes, dtype=float))
                cum = np.cumsum(srt)
                total = float(cum[-1])
                lor_x = np.linspace(0.0, 1.0, num=int(srt.size) + 1)
                lor_y = np.concatenate([[0.0], cum / total])
                gini = float(1.0 - 2.0 * np.trapz(lor_y, lor_x))
                out["lorenz"] = {
                    "x": [float(v) for v in lor_x.tolist()],
                    "y": [float(v) for v in lor_y.tolist()],
                    "gini": gini,
                }
    except Exception:
        pass

    # Distance-to-center values (if present)
    try:
        if per_sample is not None and per_sample.get("distance_to_center") is not None:
            d = np.asarray(per_sample.get("distance_to_center")).reshape(-1).astype(float)
            out["distance_to_center"] = {"values": [float(v) for v in d[idx].tolist()]}
    except Exception:
        pass

    # Centroids + separation matrix (exclude noise)
    try:
        cluster_ids = [int(v) for v in np.unique(y).tolist() if int(v) != -1]
        if cluster_ids:
            centroids = []
            kept_ids = []
            for cid in cluster_ids:
                mask = (y == cid)
                if int(np.sum(mask)) == 0:
                    continue
                centroids.append(np.mean(Xa[mask], axis=0))
                kept_ids.append(cid)
            if centroids:
                C = np.vstack(centroids)
                p = int(C.shape[1])
                top_k = min(30, p)
                var = np.var(C, axis=0)
                feat_idx = np.argsort(var)[::-1][:top_k]
                C_small = C[:, feat_idx]
                out["centroids"] = {
                    "cluster_ids": kept_ids,
                    "feature_idx": [int(i) for i in feat_idx.tolist()],
                    "values": [[float(v) for v in row.tolist()] for row in C_small],
                }
                diff = C[:, None, :] - C[None, :, :]
                D = np.sqrt(np.sum(diff * diff, axis=2))
                out["separation_matrix"] = {
                    "cluster_ids": kept_ids,
                    "values": [[float(v) for v in row.tolist()] for row in D],
                }
    except Exception:
        pass

    # Silhouette samples grouped per cluster (exclude noise)
    try:
        if silhouette_samples is not None:
            mask = (y != -1)
            y2 = y[mask]
            if y2.size > 1 and np.unique(y2).size >= 2:
                s = silhouette_samples(Xa[mask], y2)
                full_idx = np.where(mask)[0]
                keep = np.isin(full_idx, idx)
                s_keep = s[keep]
                y_keep = y2[keep]
                groups: Dict[int, List[float]] = {}
                for val, lab in zip(s_keep.tolist(), y_keep.tolist()):
                    groups[int(lab)] = groups.get(int(lab), []) + [float(val)]
                if groups:
                    keys = sorted(groups.keys())
                    out["silhouette"] = {
                        "cluster_ids": keys,
                        "values": [groups[k] for k in keys],
                        "avg": float(np.mean(s)) if s.size else None,
                    }
    except Exception as e:
        warnings.append(f"Silhouette computation failed: {type(e).__name__}: {e}")

    # Decoder-style payload
    dec: Dict[str, Any] = {}
    try:
        if per_sample is not None:
            if per_sample.get("max_membership_prob") is not None:
                v = np.asarray(per_sample.get("max_membership_prob")).reshape(-1).astype(float)
                conf_vals = v[idx]
                dec["confidence"] = {"values": [float(x) for x in conf_vals.tolist()]}
                h = _histogram_payload(conf_vals, bins=30)
                if h is not None:
                    dec["confidence_hist"] = dict(h)
            if per_sample.get("log_likelihood") is not None:
                v = np.asarray(per_sample.get("log_likelihood")).reshape(-1).astype(float)
                ll_vals = v[idx]
                dec["log_likelihood"] = {"values": [float(x) for x in ll_vals.tolist()]}
                h = _histogram_payload(ll_vals, bins=30)
                if h is not None:
                    dec["log_likelihood_hist"] = dict(h)
            if per_sample.get("is_noise") is not None:
                v = np.asarray(per_sample.get("is_noise")).reshape(-1).astype(bool)
                x = np.arange(v.size, dtype=int)
                cum = np.cumsum(v.astype(int))
                frac = cum / np.maximum(1, (x + 1))
                dec["noise_trend"] = {
                    "x": [int(i) for i in x[idx].tolist()],
                    "y": [float(f) for f in frac[idx].tolist()],
                }
    except Exception:
        pass

    if dec:
        out["decoder"] = dec

    # Dendrogram (AgglomerativeClustering), when available
    try:
        payload, w = _agglomerative_dendrogram_payload(model, labels=labels)
        if w:
            warnings.extend(w)
        if payload is not None:
            out["dendrogram"] = dict(payload)
    except Exception:
        pass

    # Embedding labels aligned to embedding.idx
    try:
        if embedding is not None and "idx" in embedding:
            emb_idx = np.asarray(embedding.get("idx"), dtype=int)
            out["embedding_labels"] = [int(v) for v in y[emb_idx].tolist()]
    except Exception:
        pass

    return out, warnings
