from typing import List, Optional, Union
from pydantic import BaseModel, Field

from shared_schemas.model_configs import ModelConfig
from shared_schemas.run_config import DataModel
from shared_schemas.split_configs import SplitCVModel
from shared_schemas.scale_configs import ScaleModel
from shared_schemas.feature_configs import FeaturesModel
from shared_schemas.eval_configs import EvalModel

Number = Union[int, float]

class LearningCurveRequest(BaseModel):
    data: DataModel
    split: SplitCVModel              # requires mode="kfold"
    scale: ScaleModel
    features: FeaturesModel
    model: ModelConfig
    eval: EvalModel

    # LC-specific knobs
    train_sizes: Optional[List[Number]] = Field(
        default=None,
        description="Absolute integers or relative fractions in (0,1]. If None, use n_steps."
    )
    n_steps: int = Field(default=5, ge=2, description="If train_sizes is None, use linspace(0.1..1.0, n_steps).")
    n_jobs: int = 1                  # passed to sklearn.learning_curve

class LearningCurveResponse(BaseModel):
    train_sizes: List[Optional[int]]
    train_scores_mean: List[Optional[float]]
    train_scores_std: List[Optional[float]]
    val_scores_mean: List[Optional[float]]
    val_scores_std: List[Optional[float]]
