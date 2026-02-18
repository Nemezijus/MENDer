from __future__ import annotations

from typing import Any, Dict

from engine.reporting.common.report_errors import record_error

from .context import PlotContext
from .deps import np


def add_embedding_labels(out: Dict[str, Any], ctx: PlotContext) -> None:
    """Embedding labels aligned to embedding.idx (used for coloring)."""
    if np is None:
        return

    try:
        if ctx.embedding is None or "idx" not in ctx.embedding:
            return

        emb_idx = np.asarray(ctx.embedding.get("idx"), dtype=int)
        if not emb_idx.size:
            return

        if int(np.max(emb_idx)) < ctx.n:
            out["embedding_labels"] = [int(v) for v in ctx.y[emb_idx].tolist()]

    except Exception as e:
        record_error(out, where="reporting.clustering.plots.embedding_labels", exc=e)
        return
