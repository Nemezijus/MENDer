"""Public BL façade entry points.

This module is the *only sanctioned invocation surface* for Engine business
logic (BL) once Segment 12 is complete.

Patch 12A1 establishes stable imports and function signatures. Concrete
implementations are added in later patches (12A3+).
"""

from __future__ import annotations

from typing import Any, Optional, Union

from engine.contracts.run_config import RunConfig
from engine.contracts.unsupervised_configs import UnsupervisedRunConfig
from engine.contracts.ensemble_run_config import EnsembleRunConfig
from engine.contracts.eval_configs import EvalModel
from engine.contracts.tuning_configs import (
    LearningCurveConfig,
    ValidationCurveConfig,
    GridSearchConfig,
    RandomizedSearchConfig,
)
from engine.contracts.results.ensemble import EnsembleResult
from engine.contracts.results.prediction import PredictionResult, UnsupervisedApplyResult
from engine.contracts.results.training import TrainResult
from engine.contracts.results.tuning import (
    GridSearchResult,
    LearningCurveResult,
    RandomSearchResult,
    ValidationCurveResult,
)
from engine.contracts.results.unsupervised import UnsupervisedResult
from engine.io.artifacts.store import ArtifactStore
from engine.io.artifacts.meta_models import ModelArtifactMetaDict


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
    *,
    artifact_uid: str,
    artifact_meta: ModelArtifactMetaDict,
    X: Any,
    y: Optional[Any] = None,
    eval_override: Optional[EvalModel] = None,
    max_preview_rows: int = 100,
    store: Optional[ArtifactStore] = None,
) -> Union[PredictionResult, UnsupervisedApplyResult]:
    """Apply a cached/persisted artifact to arrays.

    Parameters
    ----------
    artifact_uid:
        UID of a previously trained artifact.
    artifact_meta:
        Artifact meta (typically from the training response). Used to infer
        task kind, feature counts, and stored eval/decoder defaults.
    X, y:
        Arrays to apply the model to. ``y`` is optional and used only for
        scoring/preview columns.
    eval_override:
        Optional EvalModel used to override stored eval/decoder settings
        (useful for enabling/disabling decoder outputs on apply).
    store:
        Optional ArtifactStore. If the pipeline is not in the runtime cache,
        the use-case will attempt to load it from the provided store.
    """

    from engine.use_cases.prediction import apply_model_to_arrays as _apply

    return _apply(
        artifact_uid=artifact_uid,
        artifact_meta=artifact_meta,
        X=X,
        y=y,
        eval_override=eval_override,
        max_preview_rows=max_preview_rows,
        store=store,
    )


def tune_learning_curve(
    run_config: RunConfig,
    lc_cfg: LearningCurveConfig,
    *,
    store: Optional[ArtifactStore] = None,
    rng: Optional[int] = None,
) -> LearningCurveResult:
    """Run learning curve and return a typed :class:`LearningCurveResult`."""

    from engine.use_cases.tuning import tune_learning_curve as _run

    return _run(run_config, lc_cfg, store=store, rng=rng)


def tune_validation_curve(
    run_config: RunConfig,
    vc_cfg: ValidationCurveConfig,
    *,
    store: Optional[ArtifactStore] = None,
    rng: Optional[int] = None,
) -> ValidationCurveResult:
    """Run validation curve and return :class:`ValidationCurveResult`."""

    from engine.use_cases.tuning import tune_validation_curve as _run

    return _run(run_config, vc_cfg, store=store, rng=rng)


def grid_search(
    run_config: RunConfig,
    gs_cfg: GridSearchConfig,
    *,
    store: Optional[ArtifactStore] = None,
    rng: Optional[int] = None,
) -> GridSearchResult:
    """Run grid search and return :class:`GridSearchResult`."""

    from engine.use_cases.tuning import grid_search as _run

    return _run(run_config, gs_cfg, store=store, rng=rng)


def random_search(
    run_config: RunConfig,
    rs_cfg: RandomizedSearchConfig,
    *,
    store: Optional[ArtifactStore] = None,
    rng: Optional[int] = None,
) -> RandomSearchResult:
    """Run random search and return :class:`RandomSearchResult`."""

    from engine.use_cases.tuning import random_search as _run

    return _run(run_config, rs_cfg, store=store, rng=rng)
