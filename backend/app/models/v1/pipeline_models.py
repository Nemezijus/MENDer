from typing import Optional, Literal, Union, List, Dict, Any
from pydantic import BaseModel, Field

# ---- Request models (mirror config knobs for a dry-fit) ----

ScaleName = Literal["standard", "robust", "minmax", "maxabs", "quantile", "none"]

class ScalePreview(BaseModel):
    method: ScaleName = "standard"

FeatureName = Literal["none", "pca", "lda", "sfs"]

class FeaturePreview(BaseModel):
    method: FeatureName = "none"
    # PCA
    pca_n: Optional[int] = None
    pca_var: float = 0.95
    pca_whiten: bool = False
    # LDA
    lda_n: Optional[int] = None
    lda_solver: Literal["svd", "lsqr", "eigen"] = "svd"
    lda_shrinkage: Optional[Union[float, Literal["auto"]]] = None
    lda_tol: float = 1e-4
    # (lda_priors omitted for MVP; add if you need it)
    # SFS
    sfs_k: Union[int, Literal["auto"]] = "auto"
    sfs_direction: Literal["forward", "backward"] = "backward"
    sfs_cv: int = 5
    sfs_n_jobs: Optional[int] = None

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
    features: FeaturePreview
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
