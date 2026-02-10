from __future__ import annotations

from typing import Optional

from engine.contracts.eval_configs import EvalModel
from engine.contracts.feature_configs import FeaturesModel
from engine.contracts.model_configs import ModelConfig

from engine.registries.features import make_feature_extractor

from engine.components.interfaces import FeatureExtractor


def make_features(
    cfg: FeaturesModel,
    *,
    seed: Optional[int],
    model_cfg: Optional[ModelConfig] = None,
    eval_cfg: Optional[EvalModel] = None,
) -> FeatureExtractor:
    """Backwards-compatible wrapper around engine registries."""

    return make_feature_extractor(cfg, seed=seed, model_cfg=model_cfg, eval_cfg=eval_cfg)
