from typing import Optional, Literal, Union, List, Dict, Any
from pydantic import BaseModel, Field
from .shared import FeaturesModel, ModelModel

# ---- Request models (mirror config knobs for a dry-fit) ----

ScaleName = Literal["standard", "robust", "minmax", "maxabs", "quantile", "none"]

class ScalePreview(BaseModel):
    method: ScaleName = "standard"

MetricName = Literal["accuracy", "balanced_accuracy", "f1_macro"]

class EvalPreview(BaseModel):
    metric: MetricName = "accuracy"
    n_shuffles: int = 0
    seed: Optional[int] = None

class PipelinePreviewRequest(BaseModel):
    scale: ScalePreview
    features: FeaturesModel
    model: ModelModel
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
