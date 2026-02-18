from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

from engine.reporting.common.report_errors import record_error

from .context import PlotContext
from .deps import dendrogram, np


def _agglomerative_dendrogram_payload(
    est: Any,
    labels: Any = None,
    *,
    max_leaves: int = 200,
) -> Tuple[Optional[Mapping[str, Any]], List[str]]:
    warnings: List[str] = []
    if np is None:
        return None, ["numpy unavailable; cannot compute dendrogram."]
    if dendrogram is None:
        return None, ["scipy unavailable; cannot compute dendrogram."]

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


def add_dendrogram(out: Dict[str, Any], ctx: PlotContext) -> None:
    """Agglomerative dendrogram (when available)."""
    try:
        payload, w = _agglomerative_dendrogram_payload(ctx.est, labels=ctx.y)
        if w:
            ctx.warnings.extend(w)
        if payload is not None:
            out["dendrogram"] = dict(payload)
    except Exception as e:
        record_error(out, where="reporting.clustering.plots.dendrogram", exc=e)
        return
