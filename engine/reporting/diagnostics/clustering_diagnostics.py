from __future__ import annotations

"""Unsupervised diagnostics facade.

Segment 7 split:
- Core metrics + per-sample outputs: engine.reporting.diagnostics.clustering.core_metrics
- Embedding/projection helpers: engine.reporting.diagnostics.clustering.embedding
- Optional/expensive extras: engine.reporting.diagnostics.clustering.extras

This module remains the stable import path.
"""

from engine.reporting.diagnostics.clustering import (
    UnsupervisedDiagnostics,
    build_plot_data,
    cluster_summary,
    embedding_2d,
    model_diagnostics,
    per_sample_outputs,
)

__all__ = [
    "UnsupervisedDiagnostics",
    "cluster_summary",
    "embedding_2d",
    "model_diagnostics",
    "per_sample_outputs",
    "build_plot_data",
]
