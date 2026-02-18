from __future__ import annotations

from typing import Any, Dict

from engine.reporting.common.report_errors import record_error

from .context import PlotContext
from .deps import np


def add_distance_to_center(out: Dict[str, Any], ctx: PlotContext) -> None:
    """Distance-to-center values (if present in per_sample)."""
    if np is None:
        return

    try:
        if ctx.per_sample is None or ctx.per_sample.get("distance_to_center") is None:
            return

        d = np.asarray(ctx.per_sample.get("distance_to_center")).reshape(-1).astype(float)
        if d.size != ctx.n:
            return

        out["distance_to_center"] = {"values": [float(v) for v in d[ctx.idx].tolist()]}
    except Exception as e:
        record_error(out, where="reporting.clustering.plots.distance_to_center", exc=e)
        return
