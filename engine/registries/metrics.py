from __future__ import annotations

from typing import Callable, Literal

import numpy as np

from engine.components.evaluation.metrics.registry import (
    PROBA_METRICS,
    score as score_metric,
)

# NOTE: the canonical metric implementations are in engine.components.evaluation.metrics.
# This module exposes a "registry-like" interface (names -> scorer) for OCP.


EvalKind = Literal["classification", "regression"]


def is_proba_metric(metric_name: str) -> bool:
    return metric_name in PROBA_METRICS


def list_metrics(kind: EvalKind) -> list[str]:
    from engine.components.evaluation.metrics.registry import _CLASS_METRICS, _REG_METRICS

    if kind == "classification":
        return sorted(list(_CLASS_METRICS.keys()))
    return sorted(list(_REG_METRICS.keys()))


def get_scorer(kind: EvalKind, metric_name: str) -> Callable[..., float]:
    """Return a scorer callable for the given metric name.

    The returned callable matches the signature of engine.components.evaluation.metrics.registry.score
    but with kind and metric bound.
    """

    def _scorer(
        y_true: np.ndarray,
        y_pred: np.ndarray | None = None,
        *,
        y_proba: np.ndarray | None = None,
        y_score: np.ndarray | None = None,
        labels=None,
    ) -> float:
        return float(
            score_metric(
                y_true,
                y_pred,
                kind=kind,
                metric=metric_name,
                y_proba=y_proba,
                y_score=y_score,
                labels=labels,
            )
        )

    return _scorer
