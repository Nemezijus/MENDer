from __future__ import annotations

"""Scoring facade.

Segment 7 split:
- metric registry and core scoring live in engine.components.evaluation.metrics.registry
- sklearn glue (scorer factories) live in engine.components.evaluation.metrics.sklearn_glue
- shared helpers live in engine.components.evaluation.metrics.helpers

This module remains the stable import path for the rest of the codebase.
"""

from engine.components.evaluation.metrics import (
    PROBA_METRICS,
    binary_roc_curve_from_scores,
    confusion_matrix_metrics,
    make_estimator_scorer,
    multiclass_roc_curves_from_scores,
    score,
)

__all__ = [
    "PROBA_METRICS",
    "score",
    "make_estimator_scorer",
    "confusion_matrix_metrics",
    "binary_roc_curve_from_scores",
    "multiclass_roc_curves_from_scores",
]
