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
from utils.preprocessing.general.feature_extraction.lda import (
    lda_fit_transform_train_test,
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
    """PCA-based feature extractor wrapping the existing helper."""
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

@dataclass
class LDAFeatures(FeatureExtractor):
    """LDA-based feature extractor wrapping the existing helper."""
    cfg: FeatureConfig

    def fit_transform_train_test(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: Optional[np.ndarray] = None,
    ) -> Tuple[Any, np.ndarray, np.ndarray]:
        if y_train is None:
            raise ValueError("LDAFeatures requires y_train (supervised).")
        y_train = np.asarray(y_train).ravel()
        
        lda, Xtr_lda, Xte_lda = lda_fit_transform_train_test(
            X_train,
            X_test,
            y_train,
            n_components=self.cfg.lda_n,
            solver=self.cfg.lda_solver,
            shrinkage=self.cfg.lda_shrinkage,
            priors=self.cfg.lda_priors,
            tol=self.cfg.lda_tol
        )
        return lda, Xtr_lda, Xte_lda
