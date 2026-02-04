"""Business-layer result contracts.

These models define the *output* shapes produced by BL computations and
orchestration (use-cases). They are independent from backend response models.

Callers are expected to serialize via `model_dump()` at the boundary.
"""

from .common import Label, ResultModel
from .decoder import DecoderOutputs, DecoderOutputRow, DecoderSummary
from .training import EvalResult, TrainResult, EnsembleResult
from .unsupervised import UnsupervisedResult, UnsupervisedOutputs, UnsupervisedOutputRow
from .prediction import (
    PredictionResult,
    PredictionRow,
    UnsupervisedApplyResult,
    UnsupervisedApplyRow,
)
from .tuning import (
    TuningResult,
    LearningCurveResult,
    ValidationCurveResult,
    GridSearchResult,
    RandomSearchResult,
)

__all__ = [
    "Label",
    "ResultModel",
    "DecoderOutputs",
    "DecoderOutputRow",
    "DecoderSummary",
    "EvalResult",
    "TrainResult",
    "EnsembleResult",
    "UnsupervisedResult",
    "UnsupervisedOutputs",
    "UnsupervisedOutputRow",
    "PredictionResult",
    "PredictionRow",
    "UnsupervisedApplyResult",
    "UnsupervisedApplyRow",
    "TuningResult",
    "LearningCurveResult",
    "ValidationCurveResult",
    "GridSearchResult",
    "RandomSearchResult",
]
