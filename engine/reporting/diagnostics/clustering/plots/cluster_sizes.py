from __future__ import annotations

from typing import Any, Dict

from engine.reporting.common.report_errors import record_error

from ..core_metrics import cluster_summary
from .context import PlotContext
from .deps import np


def add_cluster_sizes(out: Dict[str, Any], ctx: PlotContext) -> None:
    """Cluster sizes + Lorenz/Gini (exclude noise)."""
    if np is None:
        return

    try:
        cs = cluster_summary(ctx.y).get("cluster_sizes")
        if cs is None:
            return

        out["cluster_sizes"] = cs

        sizes = []
        for k, v in dict(cs).items():
            if int(k) == -1:
                continue
            sizes.append(int(v))
        sizes = [s for s in sizes if s > 0]

        if not sizes:
            return

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
    except Exception as e:
        record_error(out, where="reporting.clustering.plots.cluster_sizes", exc=e)
        return
