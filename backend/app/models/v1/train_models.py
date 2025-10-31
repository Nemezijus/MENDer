from typing import Optional, Literal, List, Dict, Any, Union
from pydantic import BaseModel, Field
from .shared import FeaturesModel

# These should mirror what you already expose in configs

ScaleName = Literal["standard", "robust", "minmax", "maxabs", "quantile", "none"]

class ScaleTrain(BaseModel):
    method: ScaleName = "standard"

FeatureName = Literal["none", "pca", "lda", "sfs"]

PenaltyName = Literal["l2", "none"]

class ModelTrain(BaseModel):
    algo: Literal["logreg"] = "logreg"
    C: float = 1.0
    penalty: PenaltyName = "l2"
    solver: str = "lbfgs"
    max_iter: int = 1000
    class_weight: Optional[Literal["balanced"]] = None

MetricName = Literal["accuracy", "balanced_accuracy", "f1_macro"]

class EvalTrain(BaseModel):
    metric: MetricName = "accuracy"
    n_shuffles: int = 0
    seed: Optional[int] = 42

class SplitTrain(BaseModel):
    train_frac: float = 0.75
    stratified: bool = True

class DataSpec(BaseModel):
    # We allow either npz or x/y pair, same as /data/inspect
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
    model: ModelTrain
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
