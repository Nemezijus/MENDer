"""Public BL façade entry points.

This module is the *only sanctioned invocation surface* for Engine business
logic (BL) once Segment 12 is complete.

Patch 12A1 establishes stable imports and function signatures. Concrete
implementations are added in later patches (12A3+).
"""

from __future__ import annotations

from typing import Optional

from engine.contracts.run_config import RunConfig
from engine.contracts.results.ensemble import EnsembleResult
from engine.contracts.results.prediction import PredictionResult
from engine.contracts.results.training import TrainResult
from engine.contracts.results.tuning import (
    GridSearchResult,
    LearningCurveResult,
    RandomSearchResult,
    ValidationCurveResult,
)
from engine.contracts.results.unsupervised import UnsupervisedResult
from engine.io.artifacts.store import ArtifactStore


def train_supervised(
    run_config: RunConfig,
    *,
    store: Optional[ArtifactStore] = None,
    rng: Optional[int] = None,
) -> TrainResult:
    """Train a supervised model and return a typed :class:`TrainResult`.

    Parameters
    ----------
    run_config:
        Full run configuration (data + split + features + model + eval).
    store:
        Optional :class:`~engine.io.artifacts.store.ArtifactStore`.
        When provided (or when defaulted in later patches), trained artifacts
        can be persisted.
    rng:
        Optional deterministic seed override.
    """

    raise NotImplementedError(
        "train_supervised is introduced in Segment 12 Patch 12A3 (or later)."
    )


def train_unsupervised(
    run_config: RunConfig,
    *,
    store: Optional[ArtifactStore] = None,
    rng: Optional[int] = None,
) -> UnsupervisedResult:
    """Train an unsupervised model and return a typed :class:`UnsupervisedResult`."""

    raise NotImplementedError(
        "train_unsupervised is introduced in Segment 12 Patch 12A3 (or later)."
    )


def train_ensemble(
    run_config: RunConfig,
    *,
    store: Optional[ArtifactStore] = None,
    rng: Optional[int] = None,
) -> EnsembleResult:
    """Train a supervised ensemble and return :class:`EnsembleResult`."""

    raise NotImplementedError(
        "train_ensemble is introduced in Segment 12 Patch 12A4 (or later)."
    )


def predict(
    run_config: RunConfig,
    *,
    store: Optional[ArtifactStore] = None,
    rng: Optional[int] = None,
) -> PredictionResult:
    """Apply a trained artifact to data and return :class:`PredictionResult`."""

    raise NotImplementedError(
        "predict is introduced in Segment 12 Patch 12A5 (or later)."
    )


def tune_learning_curve(
    run_config: RunConfig,
    *,
    store: Optional[ArtifactStore] = None,
    rng: Optional[int] = None,
) -> LearningCurveResult:
    """Run learning curve and return a typed :class:`LearningCurveResult`."""

    raise NotImplementedError(
        "tune_learning_curve is introduced in Segment 12 Patch 12A5 (or later)."
    )


def tune_validation_curve(
    run_config: RunConfig,
    *,
    store: Optional[ArtifactStore] = None,
    rng: Optional[int] = None,
) -> ValidationCurveResult:
    """Run validation curve and return :class:`ValidationCurveResult`."""

    raise NotImplementedError(
        "tune_validation_curve is introduced in Segment 12 Patch 12A5 (or later)."
    )


def grid_search(
    run_config: RunConfig,
    *,
    store: Optional[ArtifactStore] = None,
    rng: Optional[int] = None,
) -> GridSearchResult:
    """Run grid search and return :class:`GridSearchResult`."""

    raise NotImplementedError(
        "grid_search is introduced in Segment 12 Patch 12A5 (or later)."
    )


def random_search(
    run_config: RunConfig,
    *,
    store: Optional[ArtifactStore] = None,
    rng: Optional[int] = None,
) -> RandomSearchResult:
    """Run random search and return :class:`RandomSearchResult`."""

    raise NotImplementedError(
        "random_search is introduced in Segment 12 Patch 12A5 (or later)."
    )
