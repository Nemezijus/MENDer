from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import Field

from engine.contracts.types import MetricName

from .common import ResultModel, JSONDict
from .decoder import DecoderOutputs


class EvalResult(ResultModel):
    """Core evaluation outputs common across use-cases."""

    metric_name: MetricName
    metric_value: float

    # Metrics payloads are still represented as JSON-friendly dicts
    # until we fully converge on typed metric contracts.
    confusion: JSONDict = Field(default_factory=dict)
    roc: Optional[JSONDict] = None
    regression: Optional[JSONDict] = None


class TrainResult(EvalResult):
    """Supervised training result (single model)."""

    n_train: int
    n_test: int
    notes: List[str] = Field(default_factory=list)

    shuffled_scores: Optional[List[float]] = None
    p_value: Optional[float] = None

    fold_scores: Optional[List[float]] = None
    mean_score: Optional[float] = None
    std_score: Optional[float] = None
    n_splits: Optional[int] = None

    artifact: Optional[JSONDict] = None

    decoder_outputs: Optional[DecoderOutputs] = None


class EnsembleResult(EvalResult):
    """Training result for supervised ensembles."""

    n_train: int
    n_test: int
    notes: List[str] = Field(default_factory=list)

    shuffled_scores: Optional[List[float]] = None
    p_value: Optional[float] = None

    fold_scores: Optional[List[float]] = None
    mean_score: Optional[float] = None
    std_score: Optional[float] = None
    n_splits: Optional[int] = None

    artifact: Optional[JSONDict] = None

    ensemble_report: Optional[JSONDict] = None

    decoder_outputs: Optional[DecoderOutputs] = None
