# utils/strategies/features.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Any
import numpy as np

from utils.configs.configs import FeatureConfig
from utils.strategies.interfaces import FeatureExtractor

# Your existing PCA helper (no change to its code)
from utils.preprocessing.general.feature_extraction.pca import (
    pca_fit_transform_train_test,
)

@dataclass
class NoOpFeatures(FeatureExtractor):
    """Pass-through feature extractor."""
    def fit_transform_train_test(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: Optional[np.ndarray] = None,
    ) -> Tuple[Any, np.ndarray, np.ndarray]:
        return None, X_train, X_test


@dataclass
class PCAFeatures(FeatureExtractor):
    """PCA-based feature extractor wrapping your existing helper."""
    cfg: FeatureConfig
    seed: Optional[int] = None  # int seed (derived earlier via RngManager)

    def fit_transform_train_test(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: Optional[np.ndarray] = None,
    ) -> Tuple[Any, np.ndarray, np.ndarray]:
        pca, Xtr_pca, Xte_pca = pca_fit_transform_train_test(
            X_train,
            X_test,
            n_components=self.cfg.pca_n,
            variance_threshold=self.cfg.pca_var,
            whiten=self.cfg.pca_whiten,
            random_state=self.seed,  # <- deterministic int or None
        )
        return pca, Xtr_pca, Xte_pca
