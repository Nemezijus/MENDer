from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime

class PipelineStep(BaseModel):
    name: str
    class_path: Optional[str] = None          # e.g. "sklearn.preprocessing.StandardScaler"
    params: Dict[str, Any] = Field(default_factory=dict)

class ModelArtifactMeta(BaseModel):
    # Identity
    uid: str                                   # e.g. uuid4
    created_at: datetime
    mender_version: Optional[str] = None
    # Modelling task family for this artifact.
    # NOTE: backend/UI treat this as the authoritative task for filtering/routing.
    kind: Literal["classification", "regression", "unsupervised"] = "classification"

    # Data summary
    n_samples_train: Optional[int] = None
    n_samples_test: Optional[int] = None
    n_features_in: Optional[int] = None
    classes: Optional[List[Any]] = None        # label set if known

    # Config summary (as the user ran it)
    split: Dict[str, Any]                      # {'mode': 'holdout'|'kfold', ...}
    scale: Optional[Dict[str, Any]] = None
    features: Optional[Dict[str, Any]] = None
    model: Dict[str, Any]                      # {'algo': 'logreg'|... , hyperparams...}
    eval: Dict[str, Any]                       # metric, seed, etc.

    # Pipeline topology (opaque but helpful for inspection)
    pipeline: List[PipelineStep]               # ordered steps with params

    # Training result highlights
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    mean_score: Optional[float] = None
    std_score: Optional[float] = None
    n_splits: Optional[int] = None
    notes: List[str] = Field(default_factory=list)

    # Model complexity / statistics
    n_parameters: Optional[int] = None
    extra_stats: Dict[str, Any] = Field(default_factory=dict)

class ModelArtifact(BaseModel):
    meta: ModelArtifactMeta
