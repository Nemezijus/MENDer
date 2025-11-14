from typing import Optional, List, Union
from pydantic import BaseModel, Field
from .shared import (
    DataModel,
    ScaleModel,
    EvalModel,
    FeaturesModel,
    ModelModel,
    SplitHoldoutModel,
    SplitCVModel,
    MetricName,
)
from .model_artifact import ModelArtifactMeta


class TrainRequest(BaseModel):
    data: DataModel
    # Support both hold-out and k-fold in a single request model
    split: Union[SplitHoldoutModel, SplitCVModel]
    scale: ScaleModel
    features: FeaturesModel
    model: ModelModel
    eval: EvalModel


class ConfusionMatrix(BaseModel):
    labels: List[Union[int, float, str]]
    matrix: List[List[int]]


class TrainResponse(BaseModel):
    # Common / legacy train fields
    metric_name: MetricName
    metric_value: float
    confusion: ConfusionMatrix
    n_train: int
    n_test: int
    notes: List[str] = Field(default_factory=list)

    # Shuffle-baseline fields (used by both train & cv use-cases)
    shuffled_scores: Optional[List[float]] = None
    p_value: Optional[float] = None

    # Cross-validation-style fields (k-fold case; may be None for pure hold-out)
    fold_scores: Optional[List[float]] = None
    mean_score: Optional[float] = None
    std_score: Optional[float] = None
    n_splits: Optional[int] = None

    # Model description
    artifact: Optional[ModelArtifactMeta] = None
