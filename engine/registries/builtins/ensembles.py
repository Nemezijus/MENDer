"""Built-in ensemble strategy registrations."""

from __future__ import annotations

from engine.registries.ensembles import register_ensemble_kind

from shared_schemas.ensemble_run_config import EnsembleRunConfig

from utils.strategies.ensembles import (
    VotingEnsembleStrategy,
    BaggingEnsembleStrategy,
    AdaBoostEnsembleStrategy,
    XGBoostEnsembleStrategy,
)


@register_ensemble_kind("voting")
def _voting(cfg: EnsembleRunConfig):
    return VotingEnsembleStrategy(cfg)


@register_ensemble_kind("bagging")
def _bagging(cfg: EnsembleRunConfig):
    return BaggingEnsembleStrategy(cfg)


@register_ensemble_kind("adaboost")
def _adaboost(cfg: EnsembleRunConfig):
    return AdaBoostEnsembleStrategy(cfg)


@register_ensemble_kind("xgboost")
def _xgboost(cfg: EnsembleRunConfig):
    return XGBoostEnsembleStrategy(cfg)
