from typing import Optional, Literal, Union, List, Dict, Any
from pydantic import BaseModel, Field
from engine.contracts.model_configs import ModelConfig
from engine.contracts.feature_configs import FeaturesModel
from engine.contracts.types import ScaleName, MetricName
#can also do 'from shared_schemas import FeaturesModel, ScaleModel, SplitCVModel, EvalModel, ModelConfig, DataModel, RunConfig'

# ---- Request models (mirror config knobs for a dry-fit) ----

class ScalePreview(BaseModel):
    method: ScaleName = "standard"

class EvalPreview(BaseModel):
    metric: MetricName = "accuracy"
    n_shuffles: int = 0
    seed: Optional[int] = None

class PipelinePreviewRequest(BaseModel):
    scale: ScalePreview
    features: FeaturesModel
    model: ModelConfig
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
