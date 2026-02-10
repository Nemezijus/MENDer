from __future__ import annotations

from typing import Literal

from engine.contracts.metrics_configs import MetricsModel
from engine.components.interfaces import MetricsComputer
from engine.components.evaluation.metrics_computer import SklearnMetrics


def make_metrics_computer(
    *,
    kind: Literal["classification", "regression"] = "classification",
    compute_confusion: bool = True,
    compute_roc: bool = True,
) -> MetricsComputer:
    """Factory for the metrics strategy.

    This keeps the backend isolated from the concrete implementation in
    engine.components.evaluation.scoring.

    Parameters
    ----------
    kind : {"classification", "regression"}, default "classification"
        Evaluation kind.

    compute_confusion : bool, default True
        Whether to compute confusion-matrix-based metrics.

    compute_roc : bool, default True
        Whether to compute ROC curves when scores/probabilities are provided.
    """
    cfg = MetricsModel(
        kind=kind,
        compute_confusion=compute_confusion,
        compute_roc=compute_roc,
    )
    return SklearnMetrics(cfg=cfg)
