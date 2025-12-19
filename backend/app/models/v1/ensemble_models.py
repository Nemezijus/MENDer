from typing import Optional, List, Union

from pydantic import BaseModel, Field

from shared_schemas.types import MetricName
from shared_schemas.ensemble_configs import EnsembleConfig
from shared_schemas.run_config import DataModel
from shared_schemas.split_configs import SplitCVModel, SplitHoldoutModel
from shared_schemas.scale_configs import ScaleModel
from shared_schemas.feature_configs import FeaturesModel
from shared_schemas.eval_configs import EvalModel

from .model_artifact import ModelArtifactMeta
from .metrics_models import ConfusionMatrix, RocMetrics


class EnsembleTrainRequest(BaseModel):
    data: DataModel
    split: Union[SplitHoldoutModel, SplitCVModel]
    scale: ScaleModel
    features: FeaturesModel
    ensemble: EnsembleConfig
    eval: EvalModel


class EnsembleTrainResponse(BaseModel):
    metric_name: MetricName
    metric_value: float
    confusion: ConfusionMatrix
    n_train: int
    n_test: int
    notes: List[str] = Field(default_factory=list)

    # Classification only (None for regression)
    roc: Optional[RocMetrics] = None

    # Shuffle-baseline fields (optional; parity with single-model train)
    shuffled_scores: Optional[List[float]] = None
    p_value: Optional[float] = None

    # K-fold summary fields (optional; parity with single-model train)
    fold_scores: Optional[List[float]] = None
    mean_score: Optional[float] = None
    std_score: Optional[float] = None
    n_splits: Optional[int] = None

    # Optional artifact metadata (same shape as single-model for now)
    artifact: Optional[ModelArtifactMeta] = None
