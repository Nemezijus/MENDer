from __future__ import annotations
from typing import Optional

from utils.configs.configs import FeatureConfig
from utils.strategies.interfaces import FeatureExtractor
from utils.strategies.features import NoOpFeatures, PCAFeatures, LDAFeatures

def make_features(cfg: FeatureConfig, *, seed: Optional[int]) -> FeatureExtractor:
    """
    Map FeatureConfig → concrete FeatureExtractor strategy.
    - PCA: uses `seed` for deterministic behavior when randomized solvers are used.
    - LDA: no RNG; ignores `seed`.
    """
    method = (cfg.method or "none").lower()
    if method == "none":
        return NoOpFeatures()
    if method == "pca":
        return PCAFeatures(cfg=cfg, seed=seed)
    if method == "lda":
        return LDAFeatures(cfg=cfg)
    raise ValueError(f"Unknown feature extractor method: {cfg.method}")
