# shared_schemas/metrics_configs.py
from __future__ import annotations

from pydantic import BaseModel

from .choices import ProblemKind


class MetricsModel(BaseModel):
    """
    Configuration for post-hoc metrics computation (confusion matrix and ROC)
    used by the SklearnMetrics strategy.

    Attributes
    ----------
    kind:
        Problem type. For now, only "classification" produces confusion/ROC
        metrics. "regression" will return empty metrics structures so the
        strategy can still be called safely from generic code paths.

    compute_confusion:
        If True, confusion-matrix-based metrics are computed.

    compute_roc:
        If True, ROC curves (and AUC) are computed when scores/probabilities
        are provided.
    """
    kind: ProblemKind = "classification"
    compute_confusion: bool = True
    compute_roc: bool = True
