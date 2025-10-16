# utils/strategies/interfaces.py
from __future__ import annotations
from typing import Protocol, Tuple, Any, Optional, Literal, Sequence

import numpy as np

class DataLoader(Protocol):
    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (X, y) with shapes (n_samples, n_features) and (n_samples,)."""
        ...

class SanityChecker(Protocol):
    def check(self, X: np.ndarray, y: np.ndarray) -> None:
        """Raise/print warnings for basic dataset sanity (classes, sizes, etc.)."""
        ...
        
class Splitter(Protocol):
    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return X_train, X_test, y_train, y_test."""
        ...

class Scaler(Protocol):
    def fit_transform(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit on train, transform both; return (X_train_scaled, X_test_scaled)."""
        ...

class FeatureExtractor(Protocol):
    def fit_transform_train_test(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: Optional[np.ndarray] = None,
    ) -> Tuple[Any, np.ndarray, np.ndarray]:
        """
        Fit on train (optionally using labels) and transform train+test.
        Return (artifact, X_train_fx, X_test_fx), where artifact can be a PCA/LDA object or None.
        """
        ...

class ModelBuilder(Protocol):
    def build(self) -> Any:
        """Return a configured, unfitted estimator (e.g., LogisticRegression)."""
        ...

class Trainer(Protocol):
    def fit(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Any:
        """Fit the given model on (X_train, y_train); returns the fitted model."""
        ...

class Predictor(Protocol):
    def predict(self, model: Any, X_test: np.ndarray) -> np.ndarray:
        """Return hard labels via estimator.predict(X)."""
        ...

    def predict_scores(
        self,
        model: Any,
        X_test: np.ndarray,
        *,
        kind: Literal["auto", "proba", "decision"] = "auto",
    ) -> Tuple[np.ndarray, str]:
        """Return per-sample scores and which method was used ('proba'/'decision'/'predict')."""
        ...

class Evaluator(Protocol):
    def score(
        self,
        y_true: np.ndarray,
        y_pred: Optional[np.ndarray] = None,
        *,
        y_proba: Optional[np.ndarray] = None,
        y_score: Optional[np.ndarray] = None,
        labels: Optional[Sequence] = None,
    ) -> float:
        """
        Return a scalar metric. Hard-label metrics use y_pred; probabilistic metrics
        use y_proba (preferred) or y_score.
        """
        ...

    def quick_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict:
        """Optional convenience: common classification scores (accuracy, f1, confusion, ...)."""
        ...

class BaselineRunner(Protocol):
    def run(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """Return an array of baseline (e.g., shuffle) scores."""
        ...
