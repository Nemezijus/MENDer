from __future__ import annotations
from typing import Protocol, Tuple, Any, Optional, Literal, Sequence, Iterator, Union, Dict

import numpy as np

from engine.components.splitters.types import Split
from engine.contracts.results.decoder import DecoderOutputs
from engine.core.progress import ProgressCallback

class DataLoader(Protocol):
    def load(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Return (X, y). For unsupervised/X-only data, y may be None."""
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
    ) -> Iterator[Split]:
        """Yield a sequence of train/test splits.

        Implementations must yield :class:`engine.components.splitters.types.Split`.
        """
        ...

class Scaler(Protocol):
    def fit_transform(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit on train, transform both; return (X_train_scaled, X_test_scaled)."""
        ...
    # expose the configured sklearn transformer so a Pipeline can use it directly
    def make_transformer(self) -> Any:
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
    # expose the configured sklearn transformer so a Pipeline can use it directly
    def make_transformer(self) -> Any:
        ...

class ModelBuilder(Protocol):
    def make_estimator(self) -> Any:
        """Return a configured, unfitted estimator (classifier/regressor/clusterer)."""
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

    def predict_decoder_outputs(
        self,
        model: Any,
        X_test: np.ndarray,
        *,
        positive_class_label: Optional[Any] = None,
        include_decision_scores: bool = True,
        include_probabilities: bool = True,
    ) -> DecoderOutputs:
        """
        Return decoder-style per-sample outputs for classification:

        - y_pred: hard predictions
        - classes: class ordering (if available)
        - decision_scores: decision_function(X) (if available and include_decision_scores)
        - proba: predict_proba(X) (if available and include_probabilities)
        - positive_class_index / positive_score / positive_proba when positive_class_label is provided
        - margin: simple confidence proxy

        Implementations must be robust to Pipelines and gracefully return None fields
        if a capability is not available.
        """
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


class UnsupervisedEvaluator(Protocol):
    """Evaluate unsupervised (clustering) results.

    Unsupervised evaluation is post-fit diagnostics computed from X and cluster labels.
    The returned payload should be JSON-friendly (dict/list/scalars) so it can be
    serialized by the backend.
    """

    def evaluate(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        *,
        model: Optional[Any] = None,
    ) -> Dict[str, Any]:
        ...

class BaselineRunner(Protocol):
    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        n_shuffles: Optional[int] = None,
        progress: Optional[ProgressCallback] = None,
    ) -> np.ndarray:
        """Return an array of baseline (e.g., shuffle) scores."""
        ...

class TuningStrategy(Protocol):
    """
    Generic 'tuning' strategy: learning curves, validation curves, searches, etc.
    It is responsible for loading data (via factories) and returning a structured
    result for the backend/service to serialize.
    """
    def run(self) -> Any:
        ...

class MetricsComputer(Protocol):
    """
    Compute structured evaluation metrics (confusion-matrix-based metrics,
    ROC curves, etc.) from true labels and predictions.

    Implementations may assume this is only called for classification problems
    when kind="classification"; for regression, they can return empty metrics.
    """
    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        *,
        y_proba: Optional[np.ndarray] = None,
        y_score: Optional[np.ndarray] = None,
        labels: Optional[Sequence] = None,
    ) -> "MetricsPayload":
        ...

class EnsembleBuilder(Protocol):
    """
    Builder/strategy for an ensemble estimator (Voting/Bagging/AdaBoost/XGBoost).

    - `make_estimator()` returns an unfitted ensemble estimator.
    - `fit()` trains it on the provided data and returns the fitted estimator.
    - `expected_kind()` is derived from config (authoritative), and can be used
      by orchestrators/services to route evaluation logic.
    """
    def expected_kind(self) -> Literal["classification", "regression"]:
        ...

    def make_estimator(self, *, rngm: Optional[Any] = None, stream: str = "ensemble") -> Any:
        ...

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        rngm: Optional[Any] = None,
        stream: str = "ensemble",
    ) -> Any:
        ...