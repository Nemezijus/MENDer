from typing import Optional, Literal, Union, List, Dict, Any
from pydantic import BaseModel, Field
from .shared import FeaturesModel

# ---- Request models (mirror config knobs for a dry-fit) ----

ScaleName = Literal["standard", "robust", "minmax", "maxabs", "quantile", "none"]

class ScalePreview(BaseModel):
    method: ScaleName = "standard"

FeatureName = Literal["none", "pca", "lda", "sfs"]

PenaltyName = Literal["l2", "none"]

class ModelPreview(BaseModel):
    algo: Literal["logreg"] = "logreg"
    C: float = 1.0
    penalty: PenaltyName = "l2"
    solver: str = "lbfgs"
    max_iter: int = 1000
    class_weight: Optional[Literal["balanced"]] = None

MetricName = Literal["accuracy", "balanced_accuracy", "f1_macro"]

class EvalPreview(BaseModel):
    metric: MetricName = "accuracy"
    n_shuffles: int = 0
    seed: Optional[int] = None

class PipelinePreviewRequest(BaseModel):
    scale: ScalePreview
    features: FeaturesModel
    model: ModelPreview
    eval: EvalPreview

# ---- Response models ----

class PipelineStep(BaseModel):
    name: Literal["scale", "feat", "clf"]
    class_path: str                       # e.g., "sklearn.preprocessing._data.StandardScaler"
    params: Dict[str, Any] = Field(default_factory=dict)

class PipelinePreviewResponse(BaseModel):
    ok: bool
    steps: List[PipelineStep] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
