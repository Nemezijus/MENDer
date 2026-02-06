"""Built-in feature extractor registrations."""

from __future__ import annotations

from typing import Optional

from engine.registries.features import FeatureFactory, register_feature_extractor

from shared_schemas.eval_configs import EvalModel
from shared_schemas.feature_configs import FeaturesModel
from shared_schemas.model_configs import ModelConfig

from utils.strategies.features import NoOpFeatures, PCAFeatures, LDAFeatures, SFSFeatures


@register_feature_extractor("none")
def _none(cfg: FeaturesModel, seed: Optional[int], model_cfg: Optional[ModelConfig], eval_cfg: Optional[EvalModel]):
    return NoOpFeatures()


@register_feature_extractor("pca")
def _pca(cfg: FeaturesModel, seed: Optional[int], model_cfg: Optional[ModelConfig], eval_cfg: Optional[EvalModel]):
    return PCAFeatures(cfg=cfg, seed=seed)


@register_feature_extractor("lda")
def _lda(cfg: FeaturesModel, seed: Optional[int], model_cfg: Optional[ModelConfig], eval_cfg: Optional[EvalModel]):
    return LDAFeatures(cfg=cfg)


@register_feature_extractor("sfs")
def _sfs(cfg: FeaturesModel, seed: Optional[int], model_cfg: Optional[ModelConfig], eval_cfg: Optional[EvalModel]):
    if model_cfg is None or eval_cfg is None:
        raise ValueError("SFSFeatures requires model_cfg and eval_cfg.")
    return SFSFeatures(cfg=cfg, model_cfg=model_cfg, eval_cfg=eval_cfg, seed=seed)

