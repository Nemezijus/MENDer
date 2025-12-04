from typing import Optional, List, Union
from pydantic import BaseModel, Field
from shared_schemas.types import MetricName
from shared_schemas.model_configs import ModelConfig
from shared_schemas.run_config import DataModel
from shared_schemas.split_configs import SplitCVModel, SplitHoldoutModel
from shared_schemas.scale_configs import ScaleModel
from shared_schemas.feature_configs import FeaturesModel
from shared_schemas.eval_configs import EvalModel

from .model_artifact import ModelArtifactMeta
from .metrics_models import ConfusionMatrix, RocMetrics


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
