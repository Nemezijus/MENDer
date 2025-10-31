# backend/app/models/v1/cv_models.py
from pydantic import BaseModel
from typing import Optional, List, Literal
from .shared import FeaturesModel

# Reuse the same shapes you used for TrainRequest to avoid duplication,
# only Split is different for CV.
class DataModel(BaseModel):
    x_path: Optional[str] = None
    y_path: Optional[str] = None
    npz_path: Optional[str] = None
    x_key: str = "X"
    y_key: str = "y"

class SplitCVModel(BaseModel):
    mode: Literal["kfold"] = "kfold"
    n_splits: int = 5
    stratified: bool = True
    shuffle: bool = True

class ScaleModel(BaseModel):
    method: Literal["standard","robust","minmax","maxabs","quantile","none"] = "standard"

class ModelModel(BaseModel):
    algo: Literal["logreg"] = "logreg"
    C: float = 1.0
    penalty: Literal["l1","l2","elasticnet","none"] = "l2"
    solver: Literal["lbfgs","liblinear","saga","newton-cg","newton-cholesky","sag"] = "lbfgs"
    max_iter: int = 1000
    class_weight: Optional[Literal["balanced"]] = None

class EvalModel(BaseModel):
    metric: Literal["accuracy","balanced_accuracy","f1_macro"] = "accuracy"
    seed: Optional[int] = None

class CVRequest(BaseModel):
    data: DataModel
    split: SplitCVModel
    scale: ScaleModel
    features: FeaturesModel
    model: ModelModel
    eval: EvalModel

class CVResponse(BaseModel):
    metric_name: str
    fold_scores: List[float]
    mean_score: float
    std_score: float
    n_splits: int
