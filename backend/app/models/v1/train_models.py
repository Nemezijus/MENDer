from typing import Optional, List, Union
from pydantic import BaseModel, Field
from .shared import (
    DataModel, ScaleModel, EvalModel,
    FeaturesModel, ModelModel,
    SplitHoldoutModel, MetricName
)

class TrainRequest(BaseModel):
    data: DataModel
    split: SplitHoldoutModel
    scale: ScaleModel
    features: FeaturesModel
    model: ModelModel
    eval: EvalModel

class ConfusionMatrix(BaseModel):
    labels: List[Union[int, float, str]]
    matrix: List[List[int]]

class TrainResponse(BaseModel):
    metric_name: MetricName
    metric_value: float
    confusion: ConfusionMatrix
    n_train: int
    n_test: int
    notes: List[str] = Field(default_factory=list)
    shuffled_scores: Optional[List[float]] = None
    p_value: Optional[float] = None