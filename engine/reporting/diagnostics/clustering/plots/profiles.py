from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from engine.reporting.common.report_errors import record_error

from .context import PlotContext
from .deps import np


@dataclass
class CentroidStats:
    cluster_ids: List[int]
    centroids: Any  # ndarray (k, p)
    compactness: List[float]


def compute_centroid_stats(ctx: PlotContext) -> Optional[CentroidStats]:
    """Compute centroids and per-cluster compactness (exclude noise).

    Cached on the context for reuse by other plot builders.
    """
    if np is None:
        return None

    cached = ctx.cache.get("centroid_stats")
    if isinstance(cached, CentroidStats):
        return cached

    try:
        cluster_ids = [int(v) for v in np.unique(ctx.y).tolist() if int(v) != -1]
        if not cluster_ids:
            return None

        centroids = []
        kept_ids: List[int] = []
        compactness: List[float] = []

        for cid in cluster_ids:
            mask = (ctx.y == cid)
            cnt = int(np.sum(mask))
            if cnt == 0:
                continue

            Xc = ctx.Xa[mask]
            mu = np.mean(Xc, axis=0)
            centroids.append(mu)
            kept_ids.append(int(cid))

            try:
                dd = np.sqrt(np.sum((Xc - mu) ** 2, axis=1))
                compactness.append(float(np.mean(dd)) if dd.size else 0.0)
            except Exception:
                compactness.append(float("nan"))

        if not centroids:
            return None

        C = np.vstack(centroids)
        stats = CentroidStats(cluster_ids=kept_ids, centroids=C, compactness=compactness)
        ctx.cache["centroid_stats"] = stats
        return stats

    except Exception as e:
        # We don't have access to a payload dict here; leave breadcrumbs for the orchestrator.
        ctx.warnings.append(f"Failed to compute centroid stats: {type(e).__name__}: {e}")
        ctx.cache["centroid_stats"] = None
        return None


def add_centroid_profiles(out: Dict[str, Any], ctx: PlotContext, *, top_k: int = 30) -> None:
    """Per-cluster feature profiles derived from centroid differences."""
    if np is None:
        return

    stats = compute_centroid_stats(ctx)
    if stats is None:
        return

    try:
        C = stats.centroids
        p = int(C.shape[1])
        k = min(int(top_k), p)

        # Feature profiles: show top-variance features across centroids
        var = np.var(C, axis=0)
        feat_idx = np.argsort(var)[::-1][:k]
        C_small = C[:, feat_idx]

        out["centroids"] = {
            "cluster_ids": stats.cluster_ids,
            "feature_idx": [int(i) for i in feat_idx.tolist()],
            "values": [[float(v) for v in row.tolist()] for row in C_small],
        }

    except Exception as e:
        record_error(out, where="reporting.clustering.plots.centroid_profiles", exc=e)
        return
