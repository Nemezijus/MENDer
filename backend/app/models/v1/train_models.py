from typing import Optional, List, Union, Dict, Any, Literal
from pydantic import BaseModel, Field
from shared_schemas.types import MetricName
from shared_schemas.model_configs import ModelConfig
from shared_schemas.run_config import DataModel
from shared_schemas.split_configs import SplitCVModel, SplitHoldoutModel
from shared_schemas.scale_configs import ScaleModel
from shared_schemas.feature_configs import FeaturesModel
from shared_schemas.eval_configs import EvalModel
from shared_schemas.unsupervised_configs import UnsupervisedEvalModel, FitScopeName

from .model_artifact import ModelArtifactMeta
from .metrics_models import ConfusionMatrix, RocMetrics, RegressionDiagnostics
from .decoder_models import DecoderOutputs


class TrainRequest(BaseModel):
    data: DataModel
    # Support both hold-out and k-fold in a single request model
    split: Union[SplitHoldoutModel, SplitCVModel]
    scale: ScaleModel
    features: FeaturesModel
    model: ModelConfig
    eval: EvalModel


class TrainResponse(BaseModel):
    # Common / legacy train fields
    metric_name: MetricName
    metric_value: float
    confusion: ConfusionMatrix
    n_train: int
    n_test: int
    notes: List[str] = Field(default_factory=list)

    # New: ROC metrics (classification only; None for regression)
    roc: Optional[RocMetrics] = None

    # New: regression diagnostics (regression only; None for classification)
    regression: Optional[RegressionDiagnostics] = None

    # Shuffle-baseline fields (used by both train & cv use-cases)
    shuffled_scores: Optional[List[float]] = None
    p_value: Optional[float] = None

    # Cross-validation-style fields (k-fold case; may be None for pure hold-out)
    fold_scores: Optional[List[float]] = None
    mean_score: Optional[float] = None
    std_score: Optional[float] = None
    n_splits: Optional[int] = None

    # Model description
    artifact: Optional[ModelArtifactMeta] = None

    # Optional: per-sample decoder outputs (classification or regression)
    decoder_outputs: Optional[DecoderOutputs] = None


# ---------------------------------------------------------------------------
# Unsupervised (clustering) request/response models
#
# NOTE: We keep the existing TrainRequest/TrainResponse models untouched so the
# current supervised /train endpoint remains backward compatible. A future patch
# will extend the router to accept and return these unsupervised variants.
# ---------------------------------------------------------------------------


class UnsupervisedTrainRequest(BaseModel):
    """Request payload for unsupervised training.

    - `data`: training dataset (X required; y may exist but is ignored)
    - `apply`: optional unseen/production dataset to assign to fitted clusters
    - `fit_scope`: whether to only train, or train + apply (UI-gated by algo capabilities)
    """

    task: Literal["unsupervised"] = "unsupervised"

    data: DataModel
    apply: Optional[DataModel] = None
    fit_scope: FitScopeName = "train_only"

    scale: ScaleModel
    features: FeaturesModel
    model: ModelConfig
    eval: UnsupervisedEvalModel = Field(default_factory=UnsupervisedEvalModel)

    use_y_for_external_metrics: bool = False
    external_metrics: List[str] = Field(default_factory=list)


class UnsupervisedOutputRow(BaseModel):
    """One per-sample unsupervised output row for preview tables."""

    # allow extra per-sample columns depending on estimator capabilities
    model_config = {"extra": "allow"}

    index: int
    cluster_id: int


class UnsupervisedOutputs(BaseModel):
    """Compact unsupervised per-sample outputs payload for Results UI."""

    notes: List[str] = Field(default_factory=list)
    preview_rows: List[UnsupervisedOutputRow] = Field(default_factory=list)
    n_rows_total: Optional[int] = None
    summary: Optional[Dict[str, Any]] = None


class UnsupervisedTrainResponse(BaseModel):
    task: Literal["unsupervised"] = "unsupervised"

    n_train: int
    n_features: int

    # Optional apply dataset summary
    n_apply: Optional[int] = None

    # Global post-fit diagnostics
    metrics: Dict[str, Optional[float]] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)

    cluster_summary: Dict[str, Any] = Field(default_factory=dict)
    diagnostics: Dict[str, Any] = Field(default_factory=dict)

    # Model description
    artifact: Optional[ModelArtifactMeta] = None

    # Optional: per-sample outputs (cluster id + optional columns)
    unsupervised_outputs: Optional[UnsupervisedOutputs] = None

    notes: List[str] = Field(default_factory=list)
