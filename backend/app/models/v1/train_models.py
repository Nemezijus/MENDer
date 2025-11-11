from typing import Optional, Literal, List, Dict, Any, Union
from pydantic import BaseModel, Field
from .shared import FeaturesModel, ModelModel


ScaleName = Literal["standard", "robust", "minmax", "maxabs", "quantile", "none"]

class ScaleTrain(BaseModel):
    method: ScaleName = "standard"

MetricName = Literal["accuracy", "balanced_accuracy", "f1_macro"]

class EvalTrain(BaseModel):
    metric: MetricName = "accuracy"
    n_shuffles: int = 0
    seed: Optional[int] = 42
    n_shuffles: int = 0
    progress_id: Optional[str] = None

class SplitTrain(BaseModel):
    mode: Literal["holdout"] = "holdout"
    train_frac: float = 0.75
    stratified: bool = True
    shuffle: bool = True

class DataSpec(BaseModel):
    npz_path: Optional[str] = None
    x_key: Optional[str] = "X"
    y_key: Optional[str] = "y"
    x_path: Optional[str] = None
    y_path: Optional[str] = None

class TrainRequest(BaseModel):
    data: DataSpec
    split: SplitTrain
    scale: ScaleTrain
    features: FeaturesModel
    model: ModelModel
    eval: EvalTrain

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
    # Optional shuffle-baseline outputs
    shuffled_scores: Optional[List[float]] = None
    # One-sided p-value: P(null >= real), simple +1 smoothing
    p_value: Optional[float] = None