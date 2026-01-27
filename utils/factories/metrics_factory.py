from __future__ import annotations

from typing import Literal

from shared_schemas.metrics_configs import MetricsModel
from utils.strategies.interfaces import MetricsComputer
from utils.strategies.metrics import SklearnMetrics


def make_metrics_computer(
    *,
    kind: Literal["classification", "regression"] = "classification",
    compute_confusion: bool = True,
    compute_roc: bool = True,
) -> MetricsComputer:
    """Factory for the metrics strategy.

    This keeps the backend isolated from the concrete implementation in
    utils.postprocessing.scoring and utils.strategies.metrics.

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
