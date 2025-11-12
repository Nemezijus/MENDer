from pydantic import BaseModel
from typing import Optional, List
from .shared import (
    DataModel, ScaleModel, EvalModel,
    FeaturesModel, ModelModel, SplitCVModel
)

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
    notes: List[str] = []
    shuffled_scores: Optional[List[float]] = None
    p_value: Optional[float] = None
