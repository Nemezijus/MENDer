# utils/factories/feature_factory.py
from __future__ import annotations
from typing import Optional

from utils.configs.configs import FeatureConfig
from utils.strategies.interfaces import FeatureExtractor
from utils.strategies.features import NoOpFeatures, PCAFeatures

def make_features(cfg: FeatureConfig, *, seed: Optional[int]) -> FeatureExtractor:
    method = (cfg.method or "none").lower()
    if method == "none":
        return NoOpFeatures()
    if method == "pca":
        return PCAFeatures(cfg=cfg, seed=seed)
    raise ValueError(f"Unknown feature extractor method: {cfg.method}")
