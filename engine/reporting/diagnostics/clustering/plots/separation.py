from __future__ import annotations

from typing import Any, Dict, List, Tuple

from engine.reporting.common.report_errors import record_error

from .context import PlotContext
from .deps import np
from .profiles import compute_centroid_stats


def _centroid_distance_matrix(C: Any) -> Any:
    diff = C[:, None, :] - C[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def add_separation(out: Dict[str, Any], ctx: PlotContext) -> None:
    """Centroid separation matrix + compactness vs separation."""
    if np is None:
        return

    stats = compute_centroid_stats(ctx)
    if stats is None:
        return

    try:
        C = stats.centroids
        D = _centroid_distance_matrix(C)

        out["separation_matrix"] = {
            "cluster_ids": stats.cluster_ids,
            "values": [[float(v) for v in row.tolist()] for row in D],
        }

        # Compactness vs separation per cluster
        if len(stats.cluster_ids) < 2:
            return

        sep_min: List[float] = []
        for i in range(D.shape[0]):
            row = np.asarray(D[i]).astype(float).reshape(-1)
            row = row[np.isfinite(row)]
            if row.size <= 1:
                sep_min.append(float("nan"))
            else:
                row2 = np.sort(row)
                sep_min.append(float(row2[1]))

        filt: List[Tuple[int, float, float]] = []
        for cid, comp, sepv in zip(stats.cluster_ids, stats.compactness, sep_min):
            if np.isfinite(comp) and np.isfinite(sepv):
                filt.append((int(cid), float(comp), float(sepv)))

        if len(filt) >= 2:
            out["compactness_separation"] = {
                "cluster_ids": [c for c, _, _ in filt],
                "compactness": [v for _, v, _ in filt],
                "separation": [v for _, _, v in filt],
            }

    except Exception as e:
        record_error(out, where="reporting.clustering.plots.separation", exc=e)
        return
