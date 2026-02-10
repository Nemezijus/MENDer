from __future__ import annotations

import warnings
from typing import Any, Optional, Tuple, Literal, cast

import numpy as np
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from engine.contracts.ensemble_configs import AdaBoostEnsembleConfig
from engine.contracts.ensemble_run_config import EnsembleRunConfig
from engine.contracts.model_configs import get_model_task

from engine.runtime.random.rng import RngManager
from engine.factories.pipeline_factory import make_preproc_pipeline_for_model_cfg
from engine.factories.model_factory import make_model
from engine.core.task_kind import EvalKind, infer_kind_from_y


def _make_default_stump(
    *,
    rngm: RngManager,
    stream: str,
    kind: EvalKind,
) -> Any:
    """
    Classic AdaBoost default is a decision stump (max_depth=1).
    We return the *bare* estimator (no preprocessing) so AdaBoost can pass sample_weight.
    """
    rs = rngm.child_seed(f"{stream}/default_stump")
    if kind == "classification":
        return DecisionTreeClassifier(max_depth=1, random_state=rs)
    return DecisionTreeRegressor(max_depth=1, random_state=rs)


def build_adaboost_ensemble(
    run_cfg: EnsembleRunConfig,
    *,
    rngm: Optional[RngManager] = None,
    stream: str = "adaboost",
    kind: Literal["auto", "classification", "regression"] = "auto",
) -> Tuple[Any, EvalKind]:
    """
    Build an unfitted AdaBoost ensemble, wrapped in an outer preprocessing pipeline:

        (scale -> features) -> AdaBoost(estimator=<bare base estimator>)

    This avoids the common sklearn error:
        "Pipeline doesn't support sample_weight"

    because AdaBoost passes sample_weight to the *base estimator*, not to a Pipeline.
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

    # ----- Build preprocessing (shared for the whole ensemble) -----
    # We pass model_cfg when available so feature strategies (e.g., SFS) can configure consistently.
    model_cfg_for_features = cfg.base_estimator
    preproc = make_preproc_pipeline_for_model_cfg(
        scale=run_cfg.scale,
        features=run_cfg.features,
        model_cfg=model_cfg_for_features,
        eval_cfg=run_cfg.eval,
        rngm=rngm,
        stream=f"{stream}/pre",
    )

    # ----- Build bare base estimator (AdaBoost will pass sample_weight here) -----
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

        model_seed = rngm.child_seed(f"{stream}/base/model")
        base_builder = make_model(cfg.base_estimator, seed=model_seed)
        base_est = base_builder.make_estimator()
    else:
        base_est = _make_default_stump(rngm=rngm, stream=f"{stream}/base", kind=expected_kind)

    rs = cfg.random_state if cfg.random_state is not None else rngm.child_seed(f"{stream}/adaboost")

    # NOTE: algorithm only applies to classifier historically; keep it optional and guarded.
    if expected_kind == "classification":
        # Default to SAMME (SAMME.R is deprecated in sklearn and also requires predict_proba)
        algo = cfg.algorithm if cfg.algorithm is not None else "SAMME"

        boost = AdaBoostClassifier(
            estimator=base_est,
            n_estimators=cfg.n_estimators,
            learning_rate=cfg.learning_rate,
            random_state=rs,
            algorithm=algo,
        )
        # Outer pipeline handles preprocessing once; AdaBoost sees transformed X.
        return Pipeline([("pre", preproc), ("clf", boost)]), expected_kind

    boost = AdaBoostRegressor(
        estimator=base_est,
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        random_state=rs,
    )
    return Pipeline([("pre", preproc), ("clf", boost)]), expected_kind


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
