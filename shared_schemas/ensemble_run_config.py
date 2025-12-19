from __future__ import annotations

from typing import Union

from pydantic import BaseModel

from .ensemble_configs import EnsembleConfig
from .eval_configs import EvalModel
from .feature_configs import FeaturesModel
from .run_config import DataModel
from .scale_configs import ScaleModel
from .split_configs import SplitCVModel, SplitHoldoutModel


class EnsembleRunConfig(BaseModel):
    """
    End-to-end run configuration for training an ensemble.

    Mirrors RunConfig, but replaces `model` with `ensemble`.
    """
    data: DataModel
    split: Union[SplitHoldoutModel, SplitCVModel]
    scale: ScaleModel
    features: FeaturesModel
    ensemble: EnsembleConfig
    eval: EvalModel
