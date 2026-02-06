from __future__ import annotations

from typing import Optional

from shared_schemas.eval_configs import EvalModel
from shared_schemas.feature_configs import FeaturesModel
from shared_schemas.model_configs import ModelConfig

from engine.registries.features import make_feature_extractor

from utils.strategies.interfaces import FeatureExtractor


def make_features(
    cfg: FeaturesModel,
    *,
    seed: Optional[int],
    model_cfg: Optional[ModelConfig] = None,
    eval_cfg: Optional[EvalModel] = None,
) -> FeatureExtractor:
    """Backwards-compatible wrapper around engine registries."""

    return make_feature_extractor(cfg, seed=seed, model_cfg=model_cfg, eval_cfg=eval_cfg)
