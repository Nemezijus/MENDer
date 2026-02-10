from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, cast

import numpy as np

from engine.contracts.ensemble_configs import (
    VotingEnsembleConfig,
    BaggingEnsembleConfig,
    AdaBoostEnsembleConfig,
    XGBoostEnsembleConfig,
)
from engine.contracts.ensemble_run_config import EnsembleRunConfig

from engine.components.interfaces import EnsembleBuilder
from engine.runtime.random.rng import RngManager
from engine.core.task_kind import EvalKind, ensure_uniform_model_task

from engine.components.ensembles.builders.voting import build_voting_ensemble, fit_voting_ensemble
from engine.components.ensembles.builders.bagging import (
    build_bagging_ensemble,
    fit_bagging_ensemble,
    build_balanced_bagging_ensemble,
    fit_balanced_bagging_ensemble,
)
from engine.components.ensembles.builders.adaboost import build_adaboost_ensemble, fit_adaboost_ensemble
from engine.components.ensembles.builders.xgboost import build_xgboost_ensemble, fit_xgboost_ensemble


@dataclass
class VotingEnsembleStrategy(EnsembleBuilder):
    cfg: EnsembleRunConfig

    def expected_kind(self) -> EvalKind:
        if not isinstance(self.cfg.ensemble, VotingEnsembleConfig):
            raise TypeError("VotingEnsembleStrategy requires VotingEnsembleConfig.")
        vcfg = cast(VotingEnsembleConfig, self.cfg.ensemble)
        return ensure_uniform_model_task([s.model for s in vcfg.estimators])

    def make_estimator(self, *, rngm: Optional[RngManager] = None, stream: str = "voting") -> Any:
        est, _kind = build_voting_ensemble(self.cfg, rngm=rngm, stream=stream, kind="auto")
        return est

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        rngm: Optional[RngManager] = None,
        stream: str = "voting",
    ) -> Any:
        return fit_voting_ensemble(self.cfg, X_train, y_train, rngm=rngm, stream=stream, kind="auto")


@dataclass
class BaggingEnsembleStrategy(EnsembleBuilder):
    cfg: EnsembleRunConfig

    def expected_kind(self) -> EvalKind:
        if not isinstance(self.cfg.ensemble, BaggingEnsembleConfig):
            raise TypeError("BaggingEnsembleStrategy requires BaggingEnsembleConfig.")
        bcfg = cast(BaggingEnsembleConfig, self.cfg.ensemble)

        if bcfg.balanced and bcfg.problem_kind != "classification":
            raise ValueError("Balanced bagging is only supported for classification (problem_kind='classification').")

        return "classification" if bcfg.problem_kind == "classification" else "regression"

    def make_estimator(self, *, rngm: Optional[RngManager] = None, stream: str = "bagging") -> Any:
        bcfg = cast(BaggingEnsembleConfig, self.cfg.ensemble)

        if bcfg.balanced:
            est, _kind = build_balanced_bagging_ensemble(self.cfg, rngm=rngm, stream=stream, kind="auto")
            return est

        est, _kind = build_bagging_ensemble(self.cfg, rngm=rngm, stream=stream, kind="auto")
        return est

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        rngm: Optional[RngManager] = None,
        stream: str = "bagging",
    ) -> Any:
        bcfg = cast(BaggingEnsembleConfig, self.cfg.ensemble)

        if bcfg.balanced:
            return fit_balanced_bagging_ensemble(self.cfg, X_train, y_train, rngm=rngm, stream=stream, kind="auto")

        return fit_bagging_ensemble(self.cfg, X_train, y_train, rngm=rngm, stream=stream, kind="auto")




@dataclass
class AdaBoostEnsembleStrategy(EnsembleBuilder):
    cfg: EnsembleRunConfig

    def expected_kind(self) -> EvalKind:
        if not isinstance(self.cfg.ensemble, AdaBoostEnsembleConfig):
            raise TypeError("AdaBoostEnsembleStrategy requires AdaBoostEnsembleConfig.")
        acfg = cast(AdaBoostEnsembleConfig, self.cfg.ensemble)
        return "classification" if acfg.problem_kind == "classification" else "regression"

    def make_estimator(self, *, rngm: Optional[RngManager] = None, stream: str = "adaboost") -> Any:
        est, _kind = build_adaboost_ensemble(self.cfg, rngm=rngm, stream=stream, kind="auto")
        return est

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        rngm: Optional[RngManager] = None,
        stream: str = "adaboost",
    ) -> Any:
        return fit_adaboost_ensemble(self.cfg, X_train, y_train, rngm=rngm, stream=stream, kind="auto")


@dataclass
class XGBoostEnsembleStrategy(EnsembleBuilder):
    cfg: EnsembleRunConfig

    def expected_kind(self) -> EvalKind:
        if not isinstance(self.cfg.ensemble, XGBoostEnsembleConfig):
            raise TypeError("XGBoostEnsembleStrategy requires XGBoostEnsembleConfig.")
        xcfg = cast(XGBoostEnsembleConfig, self.cfg.ensemble)
        return "classification" if xcfg.problem_kind == "classification" else "regression"

    def make_estimator(self, *, rngm: Optional[RngManager] = None, stream: str = "xgboost") -> Any:
        est, _kind = build_xgboost_ensemble(self.cfg, rngm=rngm, stream=stream, kind="auto")
        return est

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        rngm: Optional[RngManager] = None,
        stream: str = "xgboost",
    ) -> Any:
        return fit_xgboost_ensemble(self.cfg, X_train, y_train, rngm=rngm, stream=stream, kind="auto")
