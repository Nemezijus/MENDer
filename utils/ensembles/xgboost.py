# utils/ensembles/xgboost.py
from __future__ import annotations

import warnings
from typing import Any, Optional, Tuple, Literal, cast

import numpy as np

from shared_schemas.ensemble_configs import XGBoostEnsembleConfig
from shared_schemas.ensemble_run_config import EnsembleRunConfig

from utils.permutations.rng import RngManager
from utils.preprocessing.general.task_kind import EvalKind, infer_kind_from_y


def _import_xgboost():
    try:
        import xgboost as xgb  # type: ignore
    except Exception as e:
        raise ImportError(
            "XGBoost is not installed. Install it with `pip install xgboost` "
            "and ensure it's included in requirements.txt."
        ) from e
    return xgb


def build_xgboost_ensemble(
    run_cfg: EnsembleRunConfig,
    *,
    rngm: Optional[RngManager] = None,
    stream: str = "xgboost",
    kind: Literal["auto", "classification", "regression"] = "auto",
) -> Tuple[Any, EvalKind]:
    """
    Build an unfitted xgboost.XGBClassifier / xgboost.XGBRegressor based on EnsembleRunConfig.

    Returns: (estimator, expected_kind)
      - expected_kind is derived from cfg.problem_kind
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

    xgb = _import_xgboost()

    # Common params
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

    # Keep behavior stable; XGBoost uses an evaluation metric internally for some objectives.
    # We do NOT wire it to MENDer metrics here; evaluation stays in existing scoring pipeline.
    # Users can later expose eval_metric/objective if desired.
    if expected_kind == "classification":
        # Prefer a sensible default objective
        # (binary vs multiclass will be inferred on fit; XGBoost can handle both.)
        base_params.setdefault("objective", "binary:logistic")
        est = xgb.XGBClassifier(**base_params)
        return est, expected_kind

    base_params.setdefault("objective", "reg:squarederror")
    est = xgb.XGBRegressor(**base_params)
    return est, expected_kind


def fit_xgboost_ensemble(
    run_cfg: EnsembleRunConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    rngm: Optional[RngManager] = None,
    stream: str = "xgboost",
    kind: Literal["auto", "classification", "regression"] = "auto",
) -> Any:
    """
    Convenience: build + fit. Emits a warning if y looks like the opposite task type
    than cfg.problem_kind (when kind="auto").
    """
    model, expected_kind = build_xgboost_ensemble(run_cfg, rngm=rngm, stream=stream, kind=kind)

    if kind == "auto":
        y_kind = infer_kind_from_y(y_train)
        if y_kind != expected_kind:
            warnings.warn(
                f"Target y looks like '{y_kind}', but xgboost problem_kind is '{expected_kind}'. "
                f"Continuing with '{expected_kind}' based on config.",
                UserWarning,
            )

    model.fit(np.asarray(X_train), np.asarray(y_train).ravel())
    return model


__all__ = ["build_xgboost_ensemble", "fit_xgboost_ensemble"]
