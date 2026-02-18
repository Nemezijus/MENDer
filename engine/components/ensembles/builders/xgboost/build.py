from __future__ import annotations

from typing import Any, Optional, Tuple, Literal, cast

from engine.contracts.ensemble_configs import XGBoostEnsembleConfig
from engine.contracts.ensemble_run_config import EnsembleRunConfig
from engine.core.task_kind import EvalKind
from engine.core.random.rng import RngManager

from .vendor import import_xgboost


def build_xgboost_ensemble(
    run_cfg: EnsembleRunConfig,
    *,
    rngm: Optional[RngManager] = None,
    stream: str = "xgboost",
    kind: Literal["auto", "classification", "regression"] = "auto",
) -> Tuple[Any, EvalKind]:
    """Build an unfitted xgboost.XGBClassifier / xgboost.XGBRegressor.

    For classification, label encoding is handled in fit_xgboost_ensemble.
    """

    if not isinstance(run_cfg.ensemble, XGBoostEnsembleConfig):
        raise TypeError(
            f"build_xgboost_ensemble requires XGBoostEnsembleConfig, got {type(run_cfg.ensemble).__name__}"
        )

    cfg = cast(XGBoostEnsembleConfig, run_cfg.ensemble)

    expected_kind: EvalKind = "classification" if cfg.problem_kind == "classification" else "regression"
    if kind != "auto":
        requested_kind = cast(EvalKind, kind)
        if requested_kind != expected_kind:
            raise ValueError(
                f"Ensemble kind override ({requested_kind}) conflicts with xgboost problem_kind ({expected_kind})."
            )

    rngm = rngm or RngManager(None if run_cfg.eval.seed is None else int(run_cfg.eval.seed))
    rs = cfg.random_state if cfg.random_state is not None else rngm.child_seed(f"{stream}/xgb")

    xgb = import_xgboost()

    base_params: dict[str, Any] = dict(
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        max_depth=cfg.max_depth,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        reg_lambda=cfg.reg_lambda,
        reg_alpha=cfg.reg_alpha,
        min_child_weight=cfg.min_child_weight,
        gamma=cfg.gamma,
        n_jobs=cfg.n_jobs,
        random_state=rs,
    )

    if expected_kind == "classification":
        est = xgb.XGBClassifier(**base_params)
        return est, expected_kind

    base_params.setdefault("objective", "reg:squarederror")
    est = xgb.XGBRegressor(**base_params)
    return est, expected_kind
