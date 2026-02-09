"""Tuning use-cases (learning/validation curves, grid/random search).

This module provides backend- and script-friendly wrappers around the
existing tuning strategies in :mod:`utils.strategies.tuning`.

Segment 12 goal
---------------
Expose a single public BL entry point (the façade) while keeping orchestration
out of backend services.
"""

from __future__ import annotations

from typing import Optional

from engine.contracts.run_config import RunConfig
from engine.contracts.tuning_configs import (
    LearningCurveConfig,
    ValidationCurveConfig,
    GridSearchConfig,
    RandomizedSearchConfig,
)

from engine.contracts.results.tuning import (
    LearningCurveResult,
    ValidationCurveResult,
    GridSearchResult,
    RandomSearchResult,
)

from engine.io.artifacts.store import ArtifactStore

from engine.use_cases._deps import resolve_seed

from utils.factories.tuning_factory import (
    make_learning_curve_runner,
    make_validation_curve_runner,
    make_grid_search_runner,
    make_random_search_runner,
)


def _with_overridden_seed(cfg: RunConfig, seed: Optional[int]) -> RunConfig:
    """Best-effort override of eval.seed without mutating the input config."""

    if seed is None:
        return cfg

    try:
        ev = cfg.eval.model_copy(update={"seed": int(seed)})
        return cfg.model_copy(update={"eval": ev})
    except Exception:
        # If configs are not Pydantic v2, fall back to returning original.
        return cfg


def tune_learning_curve(
    run_config: RunConfig,
    lc_cfg: LearningCurveConfig,
    *,
    store: Optional[ArtifactStore] = None,  # kept for signature symmetry
    rng: Optional[int] = None,
) -> LearningCurveResult:
    _ = store  # tuning currently does not persist artifacts
    cfg = _with_overridden_seed(run_config, resolve_seed(rng))
    runner = make_learning_curve_runner(cfg, lc_cfg)
    res = runner.run()
    return LearningCurveResult.model_validate(res)


def tune_validation_curve(
    run_config: RunConfig,
    vc_cfg: ValidationCurveConfig,
    *,
    store: Optional[ArtifactStore] = None,
    rng: Optional[int] = None,
) -> ValidationCurveResult:
    _ = store
    cfg = _with_overridden_seed(run_config, resolve_seed(rng))
    runner = make_validation_curve_runner(cfg, vc_cfg)
    res = runner.run()
    return ValidationCurveResult.model_validate(res)


def grid_search(
    run_config: RunConfig,
    gs_cfg: GridSearchConfig,
    *,
    store: Optional[ArtifactStore] = None,
    rng: Optional[int] = None,
) -> GridSearchResult:
    _ = store
    cfg = _with_overridden_seed(run_config, resolve_seed(rng))
    runner = make_grid_search_runner(cfg, gs_cfg)
    res = runner.run()
    return GridSearchResult.model_validate(res)


def random_search(
    run_config: RunConfig,
    rs_cfg: RandomizedSearchConfig,
    *,
    store: Optional[ArtifactStore] = None,
    rng: Optional[int] = None,
) -> RandomSearchResult:
    _ = store
    cfg = _with_overridden_seed(run_config, resolve_seed(rng))
    runner = make_random_search_runner(cfg, rs_cfg)
    res = runner.run()
    return RandomSearchResult.model_validate(res)
