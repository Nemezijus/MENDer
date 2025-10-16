from __future__ import annotations
from typing import Optional

from utils.configs.configs import FeatureConfig, ModelConfig, EvalConfig
from utils.strategies.interfaces import FeatureExtractor
from utils.strategies.features import NoOpFeatures, PCAFeatures, LDAFeatures, SFSFeatures

def make_features(
    cfg: FeatureConfig,
    *,
    seed: Optional[int],
    model_cfg: Optional[ModelConfig] = None,
    eval_cfg: Optional[EvalConfig] = None,
) -> FeatureExtractor:
    method = (cfg.method or "none").lower()
    if method == "none":
        return NoOpFeatures()
    if method == "pca":
        return PCAFeatures(cfg=cfg, seed=seed)
    if method == "lda":
        return LDAFeatures(cfg=cfg)
    if method == "sfs":
        if model_cfg is None or eval_cfg is None:
            raise ValueError("SFSFeatures requires model_cfg and eval_cfg.")
        return SFSFeatures(cfg=cfg, model_cfg=model_cfg, eval_cfg=eval_cfg, seed=seed)
    raise ValueError(f"Unknown feature extractor method: {cfg.method}")
