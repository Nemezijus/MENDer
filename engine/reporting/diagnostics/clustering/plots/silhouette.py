from __future__ import annotations

from typing import Any, Dict

from .context import PlotContext
from .deps import np, silhouette_samples


def add_silhouette(out: Dict[str, Any], ctx: PlotContext) -> None:
    """Silhouette samples grouped per cluster (exclude noise)."""
    if np is None or silhouette_samples is None:
        return

    try:
        mask = (ctx.y != -1)
        y2 = ctx.y[mask]
        if y2.size <= 1 or np.unique(y2).size < 2:
            return

        s = silhouette_samples(ctx.Xa[mask], y2)

        # Align to downsampled indices
        full_idx = np.where(mask)[0]
        keep = np.isin(full_idx, ctx.idx)
        s_keep = s[keep]
        y_keep = y2[keep]

        groups: Dict[int, list] = {}
        for val, lab in zip(s_keep.tolist(), y_keep.tolist()):
            groups[int(lab)] = groups.get(int(lab), []) + [float(val)]

        if not groups:
            return

        keys = sorted(groups.keys())
        out["silhouette"] = {
            "cluster_ids": keys,
            "values": [groups[k] for k in keys],
            "avg": float(np.mean(s)) if s.size else None,
        }

    except Exception as e:
        ctx.warnings.append(f"Silhouette computation failed: {type(e).__name__}: {e}")
