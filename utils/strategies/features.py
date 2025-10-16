# utils/strategies/features.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Any
import numpy as np

from utils.configs.configs import FeatureConfig, ModelConfig, EvalConfig
from utils.strategies.interfaces import FeatureExtractor

# Your existing PCA helper (no change to its code)
from utils.preprocessing.general.feature_extraction.pca import (
    pca_fit_transform_train_test,
)
from utils.preprocessing.general.feature_extraction.lda import (
    lda_fit_transform_train_test,
)
from utils.preprocessing.general.feature_selection.sfs import (
    sfs_fit_transform_train_test,
)
from utils.factories.model_factory import make_model

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


@dataclass
class SFSFeatures(FeatureExtractor):
    """
    Sequential Feature Selection wrapper.
    Uses the current model config to build the estimator used during selection
    (so selection matches your downstream classifier), unless you prefer the simple
    LogisticRegression default below.
    """
    cfg: FeatureConfig
    model_cfg: ModelConfig
    eval_cfg: EvalConfig
    seed: Optional[int] = None  # for CV shuffle determinism

    def fit_transform_train_test(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: Optional[np.ndarray] = None,
    ) -> Tuple[Any, np.ndarray, np.ndarray]:
        if y_train is None:
            raise ValueError("SFSFeatures requires y_train (supervised).")
        y_train = np.asarray(y_train).ravel()

        # Build the estimator used inside SFS to evaluate subsets.
        # Option A (recommended): use your model factory so selection matches the downstream model
        sel_model = make_model(self.model_cfg).build()

        # Option B (simple): uncomment to always use vanilla LR during selection
        # sel_model = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=1000)

        selector, Xtr_sel, Xte_sel = sfs_fit_transform_train_test(
            X_train,
            X_test,
            y_train,
            estimator=sel_model,
            n_features_to_select=self.cfg.sfs_k,
            direction=self.cfg.sfs_direction,
            scoring=self.eval_cfg.metric,   # keep consistent with your EvalConfig
            cv=self.cfg.sfs_cv,
            shuffle=True,
            random_state=self.seed,
            n_jobs=self.cfg.sfs_n_jobs,
        )
        return selector, Xtr_sel, Xte_sel