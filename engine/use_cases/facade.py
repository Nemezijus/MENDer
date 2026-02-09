"""Public BL façade entry points.

This module is the *only sanctioned invocation surface* for Engine business
logic (BL) once Segment 12 is complete.

Patch 12A1 establishes stable imports and function signatures. Concrete
implementations are added in later patches (12A3+).
"""

from __future__ import annotations

from typing import Optional

from engine.contracts.run_config import RunConfig
from engine.contracts.unsupervised_configs import UnsupervisedRunConfig
from engine.contracts.ensemble_run_config import EnsembleRunConfig
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
    """Train a supervised model and return a typed :class:`TrainResult`."""

    from engine.use_cases.supervised_training import train_supervised as _train

    return _train(run_config, store=store, rng=rng)


def train_unsupervised(
    run_config: UnsupervisedRunConfig,
    *,
    store: Optional[ArtifactStore] = None,
    rng: Optional[int] = None,
) -> UnsupervisedResult:
    """Train an unsupervised model and return a typed :class:`UnsupervisedResult`."""

    from engine.use_cases.unsupervised_training import train_unsupervised as _train

    return _train(run_config, store=store, rng=rng)


def train_ensemble(
    run_config: EnsembleRunConfig,
    *,
    store: Optional[ArtifactStore] = None,
    rng: Optional[int] = None,
) -> EnsembleResult:
    """Train a supervised ensemble and return :class:`EnsembleResult`."""

    from engine.use_cases.ensembles import train_ensemble as _train

    return _train(run_config, store=store, rng=rng)


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
