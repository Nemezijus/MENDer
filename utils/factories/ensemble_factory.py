from __future__ import annotations

from typing import Any

from shared_schemas.ensemble_run_config import EnsembleRunConfig

from utils.strategies.interfaces import EnsembleBuilder
from utils.strategies.ensembles import (
    VotingEnsembleStrategy,
    BaggingEnsembleStrategy,
    AdaBoostEnsembleStrategy,
    XGBoostEnsembleStrategy,
)


def make_ensemble_strategy(cfg: EnsembleRunConfig) -> EnsembleBuilder:
    """
    Return an ensemble strategy (builder) based on cfg.ensemble.kind.

    This is BL-only and must remain independent of backend/frontend.
    """
    kind = cfg.ensemble.kind

    if kind == "voting":
        return VotingEnsembleStrategy(cfg)
    if kind == "bagging":
        return BaggingEnsembleStrategy(cfg)
    if kind == "adaboost":
        return AdaBoostEnsembleStrategy(cfg)
    if kind == "xgboost":
        return XGBoostEnsembleStrategy(cfg)

    raise ValueError(f"Unknown ensemble kind: {kind!r}")


def make_ensemble_estimator(cfg: EnsembleRunConfig, **kwargs: Any) -> Any:
    """
    Convenience helper: build an unfitted ensemble estimator.
    kwargs are forwarded to strategy.make_estimator (e.g. rngm=..., stream=...).
    """
    return make_ensemble_strategy(cfg).make_estimator(**kwargs)
