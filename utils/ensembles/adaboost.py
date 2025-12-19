# utils/ensembles/adaboost.py
from __future__ import annotations

import warnings
from typing import Any, Optional, Tuple, Literal, cast

import numpy as np
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from shared_schemas.ensemble_configs import AdaBoostEnsembleConfig
from shared_schemas.ensemble_run_config import EnsembleRunConfig
from shared_schemas.model_configs import get_model_task

from utils.permutations.rng import RngManager
from utils.factories.pipeline_factory import make_pipeline_for_model_cfg
from utils.factories.scale_factory import make_scaler
from utils.factories.feature_factory import make_features
from utils.preprocessing.general.task_kind import EvalKind, infer_kind_from_y


def _make_default_stump_pipeline(
    *,
    run_cfg: EnsembleRunConfig,
    rngm: RngManager,
    stream: str,
    kind: EvalKind,
) -> Pipeline:
    """
    Build a (scale -> features -> decision stump) pipeline when cfg.base_estimator is None.

    Classic AdaBoost default is a decision stump (max_depth=1).
    We keep preprocessing consistent with MENDer by wrapping it in the same pipeline style.
    """
    features_seed = rngm.child_seed(f"{stream}/features")

    scaler_strategy = make_scaler(run_cfg.scale)
    feature_strategy = make_features(
        run_cfg.features,
        seed=features_seed,
        model_cfg=None,  # type: ignore[arg-type]
        eval_cfg=run_cfg.eval,
    )

    rs = rngm.child_seed(f"{stream}/default_stump")

    if kind == "classification":
        stump = DecisionTreeClassifier(max_depth=1, random_state=rs)
    else:
        stump = DecisionTreeRegressor(max_depth=1, random_state=rs)

    return Pipeline(
        steps=[
            ("scale", scaler_strategy.make_transformer()),
            ("feat", feature_strategy.make_transformer()),
            ("clf", stump),
        ]
    )


def build_adaboost_ensemble(
    run_cfg: EnsembleRunConfig,
    *,
    rngm: Optional[RngManager] = None,
    stream: str = "adaboost",
    kind: Literal["auto", "classification", "regression"] = "auto",
) -> Tuple[Any, EvalKind]:
    """
    Build an unfitted sklearn AdaBoostClassifier/AdaBoostRegressor based on EnsembleRunConfig.

    Returns: (estimator, expected_kind)
      - expected_kind is derived from cfg.problem_kind (and validated against base_estimator if provided)
    """
    if not isinstance(run_cfg.ensemble, AdaBoostEnsembleConfig):
        raise TypeError(
            f"build_adaboost_ensemble requires AdaBoostEnsembleConfig, got {type(run_cfg.ensemble).__name__}"
        )

    cfg = cast(AdaBoostEnsembleConfig, run_cfg.ensemble)

    rngm = rngm or RngManager(None if run_cfg.eval.seed is None else int(run_cfg.eval.seed))

    expected_kind: EvalKind = "classification" if cfg.problem_kind == "classification" else "regression"

    if kind != "auto":
        requested_kind = cast(EvalKind, kind)
        if requested_kind != expected_kind:
            raise ValueError(
                f"Ensemble kind override ({requested_kind}) conflicts with adaboost problem_kind ({expected_kind})."
            )

    # Build base estimator (AdaBoost calls it `estimator` in sklearn)
    if cfg.base_estimator is not None:
        task = get_model_task(cfg.base_estimator)
        if expected_kind == "classification" and task != "classification":
            raise ValueError(
                f"AdaBoost (classification) cannot use a regression base estimator (task={task})."
            )
        if expected_kind == "regression" and task != "regression":
            raise ValueError(
                f"AdaBoost (regression) cannot use a classification base estimator (task={task})."
            )

        base_est = make_pipeline_for_model_cfg(
            scale=run_cfg.scale,
            features=run_cfg.features,
            model_cfg=cfg.base_estimator,
            eval_cfg=run_cfg.eval,
            rngm=rngm,
            stream=f"{stream}/base",
        )
    else:
        base_est = _make_default_stump_pipeline(
            run_cfg=run_cfg,
            rngm=rngm,
            stream=f"{stream}/base",
            kind=expected_kind,
        )

    rs = cfg.random_state if cfg.random_state is not None else rngm.child_seed(f"{stream}/adaboost")

    # NOTE: algorithm only applies to classifier historically; keep it optional and guarded.
    if expected_kind == "classification":
        kwargs: dict[str, Any] = {}
        if cfg.algorithm is not None:
            kwargs["algorithm"] = cfg.algorithm

        est = AdaBoostClassifier(
            estimator=base_est,
            n_estimators=cfg.n_estimators,
            learning_rate=cfg.learning_rate,
            random_state=rs,
            **kwargs,
        )
        return est, expected_kind

    # Regression variant does not accept algorithm
    est = AdaBoostRegressor(
        estimator=base_est,
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        random_state=rs,
    )
    return est, expected_kind


def fit_adaboost_ensemble(
    run_cfg: EnsembleRunConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    rngm: Optional[RngManager] = None,
    stream: str = "adaboost",
    kind: Literal["auto", "classification", "regression"] = "auto",
) -> Any:
    """
    Convenience: build + fit. Emits a warning if y looks like the opposite task type
    than cfg.problem_kind (when kind="auto").
    """
    model, expected_kind = build_adaboost_ensemble(run_cfg, rngm=rngm, stream=stream, kind=kind)

    if kind == "auto":
        y_kind = infer_kind_from_y(y_train)
        if y_kind != expected_kind:
            warnings.warn(
                f"Target y looks like '{y_kind}', but adaboost problem_kind is '{expected_kind}'. "
                f"Continuing with '{expected_kind}' based on config.",
                UserWarning,
            )

    model.fit(np.asarray(X_train), np.asarray(y_train).ravel())
    return model


__all__ = ["build_adaboost_ensemble", "fit_adaboost_ensemble"]
