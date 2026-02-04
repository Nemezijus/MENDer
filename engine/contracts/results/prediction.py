from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal

from pydantic import ConfigDict, Field

from .common import Label, ResultModel, JSONDict
from .decoder import DecoderOutputs


class PredictionRow(ResultModel):
    """One per-sample prediction preview row.

    Allow extra columns (e.g. fold_id, proba, score, etc.)
    """

    model_config = ConfigDict(extra="allow")

    index: int
    y_pred: Label
    y_true: Optional[Label] = None

    # Regression convenience fields
    residual: Optional[float] = None
    abs_error: Optional[float] = None

    # Classification convenience
    correct: Optional[bool] = None


class PredictionResult(ResultModel):
    """Apply a supervised model to a dataset."""

    n_samples: int
    n_features: int

    task: Literal["classification", "regression"]
    has_labels: bool

    metric_name: Optional[str] = None
    metric_value: Optional[float] = None

    preview: List[PredictionRow] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)

    decoder_outputs: Optional[DecoderOutputs] = None


class UnsupervisedApplyRow(ResultModel):
    model_config = ConfigDict(extra="allow")

    index: int
    cluster_id: int


class UnsupervisedApplyResult(ResultModel):
    n_samples: int
    n_features: int

    task: Literal["unsupervised"] = "unsupervised"

    preview: List[UnsupervisedApplyRow] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)
