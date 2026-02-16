from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

from engine.reporting.common.json_safety import ReportError, add_report_error

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

from .core_metrics import _as_2d, _final_estimator, _transform_through_pipeline, cluster_summary


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
    # NOTE: This function intentionally mirrors the pre-refactor plot payloads
    # so the existing frontend can render the richer diagnostics without
    # requiring UI refactors.

    warnings: List[str] = []
    out: Dict[str, Any] = {}

    if np is None:
        return out, ["numpy unavailable; cannot build plot data."]

    # --- validate inputs ---------------------------------------------------
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
            f"Label length ({int(y.size)}) does not match X rows ({int(n)}); using first {m} samples."
        )
        y = y[:m]
        Xa = Xa[:m]
        n = int(m)

    # Downsample indices for payload-heavy arrays
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

    est = _final_estimator(model)
    est_name = type(est).__name__

    # ---------------------------------------------------------------------
    # Global/common plots
    # ---------------------------------------------------------------------

    # Cluster sizes + Lorenz/Gini (exclude noise)
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

    # Distance-to-center values (if present in per_sample)
    try:
        if per_sample is not None and per_sample.get("distance_to_center") is not None:
            d = np.asarray(per_sample.get("distance_to_center")).reshape(-1).astype(float)
            if d.size == n:
                out["distance_to_center"] = {"values": [float(v) for v in d[idx].tolist()]}
    except Exception:
        pass

    # Centroids + separation matrix + compactness/separation scatter (exclude noise)
    try:
        cluster_ids = [int(v) for v in np.unique(y).tolist() if int(v) != -1]
        if cluster_ids:
            centroids = []
            kept_ids = []
            compactness = []
            for cid in cluster_ids:
                mask = (y == cid)
                cnt = int(np.sum(mask))
                if cnt == 0:
                    continue
                Xc = Xa[mask]
                mu = np.mean(Xc, axis=0)
                centroids.append(mu)
                kept_ids.append(int(cid))
                # Compactness: mean distance to centroid
                try:
                    dd = np.sqrt(np.sum((Xc - mu) ** 2, axis=1))
                    compactness.append(float(np.mean(dd)) if dd.size else 0.0)
                except Exception:
                    compactness.append(float("nan"))

            if centroids:
                C = np.vstack(centroids)
                p = int(C.shape[1])
                top_k = min(30, p)

                # Feature profiles: show top-variance features across centroids
                var = np.var(C, axis=0)
                feat_idx = np.argsort(var)[::-1][:top_k]
                C_small = C[:, feat_idx]
                out["centroids"] = {
                    "cluster_ids": kept_ids,
                    "feature_idx": [int(i) for i in feat_idx.tolist()],
                    "values": [[float(v) for v in row.tolist()] for row in C_small],
                }

                # Separation matrix between centroids
                diff = C[:, None, :] - C[None, :, :]
                D = np.sqrt(np.sum(diff * diff, axis=2))
                out["separation_matrix"] = {
                    "cluster_ids": kept_ids,
                    "values": [[float(v) for v in row.tolist()] for row in D],
                }

                # Compactness vs separation per cluster
                try:
                    if len(kept_ids) >= 2:
                        sep_min = []
                        for i in range(D.shape[0]):
                            row = np.asarray(D[i]).astype(float).reshape(-1)
                            row = row[np.isfinite(row)]
                            if row.size <= 1:
                                sep_min.append(float("nan"))
                            else:
                                row2 = np.sort(row)
                                # first is 0 (self), second is nearest neighbor centroid
                                sep_min.append(float(row2[1]))

                        filt = []
                        for cid, comp, sepv in zip(kept_ids, compactness, sep_min):
                            if np.isfinite(comp) and np.isfinite(sepv):
                                filt.append((int(cid), float(comp), float(sepv)))

                        if len(filt) >= 2:
                            out["compactness_separation"] = {
                                "cluster_ids": [c for c, _, _ in filt],
                                "compactness": [v for _, v, _ in filt],
                                "separation": [v for _, _, v in filt],
                            }
                except Exception:
                    pass
    except Exception:
        pass

    # Silhouette samples grouped per cluster (exclude noise)
    try:
        if silhouette_samples is not None:
            mask = (y != -1)
            y2 = y[mask]
            if y2.size > 1 and np.unique(y2).size >= 2:
                s = silhouette_samples(Xa[mask], y2)
                # Align to downsampled indices
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

    # Decoder-style payload (confidence / likelihood / noise trend)
    dec: Dict[str, Any] = {}
    try:
        if per_sample is not None:
            if per_sample.get("max_membership_prob") is not None:
                v = np.asarray(per_sample.get("max_membership_prob")).reshape(-1).astype(float)
                if v.size == n:
                    conf_vals = v[idx]
                    dec["confidence"] = {"values": [float(x) for x in conf_vals.tolist()]}
                    h = _histogram_payload(conf_vals, bins=30)
                    if h is not None:
                        dec["confidence_hist"] = dict(h)

            if per_sample.get("log_likelihood") is not None:
                v = np.asarray(per_sample.get("log_likelihood")).reshape(-1).astype(float)
                if v.size == n:
                    ll_vals = v[idx]
                    dec["log_likelihood"] = {"values": [float(x) for x in ll_vals.tolist()]}
                    h = _histogram_payload(ll_vals, bins=30)
                    if h is not None:
                        dec["log_likelihood_hist"] = dict(h)

            if per_sample.get("is_noise") is not None:
                v = np.asarray(per_sample.get("is_noise")).reshape(-1).astype(bool)
                if v.size == n:
                    x = np.arange(v.size, dtype=int)
                    cum = np.cumsum(v.astype(int))
                    frac = cum / np.maximum(1, (x + 1))
                    dec["noise_trend"] = {
                        "x": [int(i) for i in x[idx].tolist()],
                        "y": [float(z) for z in frac[idx].tolist()],
                    }
    except Exception:
        pass

    if dec:
        out["decoder"] = dec

    # ---------------------------------------------------------------------
    # Model-specific plots
    # ---------------------------------------------------------------------

    # Elbow curve for KMeans / MiniBatchKMeans (approximate; fits small k-range on a subset)
    try:
        if KMeans is not None and est_name in {"KMeans", "MiniBatchKMeans"}:
            # Use up to ~2000 points for elbow to keep runtime reasonable.
            n_elbow = int(min(n, 2000))
            Xa_elbow = Xa
            if n > n_elbow:
                rng = np.random.default_rng(int(seed))
                pick = rng.choice(n, size=n_elbow, replace=False)
                Xa_elbow = Xa[pick]
                warnings.append(f"Elbow curve computed on a subsample of {n_elbow} points.")
            base_k = int(getattr(est, "n_clusters", 2) or 2)
            k_max = int(min(max(6, base_k + 4), 12, max(3, n_elbow - 1)))
            ks = list(range(2, k_max + 1))
            ys = []
            for k in ks:
                try:
                    km = KMeans(n_clusters=int(k), random_state=int(seed), n_init=5, max_iter=200)
                    km.fit(Xa_elbow)
                    ys.append(float(getattr(km, "inertia_", float("nan"))))
                except Exception:
                    ys.append(float("nan"))
            if ks and any(np.isfinite(v) for v in ys):
                out["elbow_curve"] = {
                    "x": [int(k) for k in ks],
                    "y": [float(v) if np.isfinite(v) else None for v in ys],
                }
    except Exception:
        pass

    # DBSCAN helpers: k-distance curve + core/border/noise counts
    try:
        if NearestNeighbors is not None and est_name == "DBSCAN":
            k = int(getattr(est, "min_samples", 5) or 5)
            k = max(2, k)
            n_kd = int(min(n, 3000))
            Xa_kd = Xa
            if n > n_kd:
                rng = np.random.default_rng(int(seed))
                pick = rng.choice(n, size=n_kd, replace=False)
                Xa_kd = Xa[pick]
                warnings.append(f"k-distance computed on a subsample of {n_kd} points.")
            nn = NearestNeighbors(n_neighbors=k)
            nn.fit(Xa_kd)
            dists, _ = nn.kneighbors(Xa_kd)
            kd = np.sort(dists[:, -1].astype(float))
            out["k_distance"] = {"k": int(k), "y": [float(v) for v in kd.tolist()]}

            core_idx = getattr(est, "core_sample_indices_", None)
            if core_idx is not None:
                core = np.zeros((n,), dtype=bool)
                try:
                    core[np.asarray(core_idx, dtype=int)] = True
                    noise = (y == -1)
                    border = (~core) & (~noise)
                    out["core_border_noise_counts"] = {
                        "core": int(np.sum(core)),
                        "border": int(np.sum(border)),
                        "noise": int(np.sum(noise)),
                    }
                except Exception:
                    pass
    except Exception:
        pass

    # Spectral clustering eigenvalue spectrum (optional; small n only)
    try:
        if est_name == "SpectralClustering":
            A = getattr(est, "affinity_matrix_", None)
            if A is not None:
                try:
                    if hasattr(A, "toarray"):
                        A = A.toarray()
                except Exception:
                    pass
                A = np.asarray(A, dtype=float)
                nA = int(A.shape[0])
                if A.ndim == 2 and A.shape[0] == A.shape[1] and nA >= 2:
                    if nA > 400:
                        warnings.append(
                            f"Spectral eigenvalues skipped: affinity matrix too large ({nA}x{nA})."
                        )
                    else:
                        A = 0.5 * (A + A.T)
                        d = np.sum(A, axis=1)
                        d = np.maximum(d, 1e-12)
                        D_inv_sqrt = np.diag(1.0 / np.sqrt(d))
                        L = np.eye(nA) - (D_inv_sqrt @ A @ D_inv_sqrt)
                        w = np.linalg.eigvalsh(L)
                        w = np.sort(np.asarray(w, dtype=float))
                        m_keep = int(min(25, w.size))
                        out["spectral_eigenvalues"] = {
                            "values": [float(v) for v in w[:m_keep].tolist()]
                        }
    except Exception:
        pass

    # Dendrogram (AgglomerativeClustering), when available
    try:
        payload, w = _agglomerative_dendrogram_payload(model, labels=y)
        if w:
            warnings.extend(w)
        if payload is not None:
            out["dendrogram"] = dict(payload)
    except Exception:
        pass

    # Embedding labels aligned to embedding.idx (used for coloring)
    try:
        if embedding is not None and "idx" in embedding:
            emb_idx = np.asarray(embedding.get("idx"), dtype=int)
            if emb_idx.size and int(np.max(emb_idx)) < n:
                out["embedding_labels"] = [int(v) for v in y[emb_idx].tolist()]
    except Exception:
        pass

    return out, warnings
