from __future__ import annotations

import warnings
from typing import Any, Optional, Tuple, Literal, cast

import numpy as np
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from engine.contracts.ensemble_configs import BaggingEnsembleConfig
from engine.contracts.ensemble_run_config import EnsembleRunConfig
from engine.contracts.model_configs import get_model_task

from engine.core.random.rng import RngManager
from engine.factories.pipeline_factory import (
    make_pipeline_for_model_cfg,
    make_preproc_pipeline_for_model_cfg,
)
from engine.core.task_kind import EvalKind, infer_kind_from_y


def _import_balanced_bagging_classifier():
    try:
        from imblearn.ensemble import BalancedBaggingClassifier  # type: ignore
    except Exception as e:
        raise ImportError(
            "Balanced bagging requires `imbalanced-learn`. "
            "Please install it (e.g. imbalanced-learn==0.12.0) and try again."
        ) from e
    return BalancedBaggingClassifier


def _make_default_tree_pipeline(
    *,
    run_cfg: EnsembleRunConfig,
    rngm: RngManager,
    stream: str,
    kind: EvalKind,
) -> Pipeline:
    """
    Build a (scale -> features -> DecisionTree) pipeline when cfg.base_estimator is None.
    This keeps preprocessing consistent with other MENDer runs.
    """
    # Build preprocessing via the shared pipeline factory to keep behavior aligned
    # with the rest of the Engine (scale -> features).
    preproc = make_preproc_pipeline_for_model_cfg(
        scale=run_cfg.scale,
        features=run_cfg.features,
        model_cfg=None,
        eval_cfg=run_cfg.eval,
        rngm=rngm,
        stream=stream,
    )

    rs = rngm.child_seed(f"{stream}/default_tree")
    if kind == "classification":
        estimator = DecisionTreeClassifier(random_state=rs)
    else:
        estimator = DecisionTreeRegressor(random_state=rs)

    return Pipeline(steps=list(preproc.steps) + [("clf", estimator)])


def _build_base_estimator(
    *,
    run_cfg: EnsembleRunConfig,
    cfg: BaggingEnsembleConfig,
    rngm: RngManager,
    stream: str,
    expected_kind: EvalKind,
) -> Any:
    """Shared logic for building the base estimator pipeline (or default tree)."""
    if cfg.base_estimator is not None:
        task = get_model_task(cfg.base_estimator)
        if expected_kind == "classification" and task != "classification":
            raise ValueError(
                f"Bagging (classification) cannot use a regression base estimator (task={task})."
            )
        if expected_kind == "regression" and task != "regression":
            raise ValueError(
                f"Bagging (regression) cannot use a classification base estimator (task={task})."
            )

        return make_pipeline_for_model_cfg(
            scale=run_cfg.scale,
            features=run_cfg.features,
            model_cfg=cfg.base_estimator,
            eval_cfg=run_cfg.eval,
            rngm=rngm,
            stream=f"{stream}/base",
        )

    return _make_default_tree_pipeline(
        run_cfg=run_cfg,
        rngm=rngm,
        stream=f"{stream}/base",
        kind=expected_kind,
    )


def build_bagging_ensemble(
    run_cfg: EnsembleRunConfig,
    *,
    rngm: Optional[RngManager] = None,
    stream: str = "bagging",
    kind: Literal["auto", "classification", "regression"] = "auto",
) -> Tuple[Any, EvalKind]:
    """
    Build an unfitted sklearn BaggingClassifier/BaggingRegressor based on EnsembleRunConfig.

    Returns: (estimator, expected_kind)
      - expected_kind is derived from cfg.problem_kind (and validated against base_estimator if provided)
    """
    if not isinstance(run_cfg.ensemble, BaggingEnsembleConfig):
        raise TypeError(
            f"build_bagging_ensemble requires BaggingEnsembleConfig, got {type(run_cfg.ensemble).__name__}"
        )

    cfg = cast(BaggingEnsembleConfig, run_cfg.ensemble)

    rngm = rngm or RngManager(None if run_cfg.eval.seed is None else int(run_cfg.eval.seed))

    expected_kind: EvalKind = "classification" if cfg.problem_kind == "classification" else "regression"

    if kind != "auto":
        requested_kind = cast(EvalKind, kind)
        if requested_kind != expected_kind:
            raise ValueError(
                f"Ensemble kind override ({requested_kind}) conflicts with bagging problem_kind ({expected_kind})."
            )

    base_est = _build_base_estimator(
        run_cfg=run_cfg,
        cfg=cfg,
        rngm=rngm,
        stream=stream,
        expected_kind=expected_kind,
    )

    rs = cfg.random_state if cfg.random_state is not None else rngm.child_seed(f"{stream}/bagging")

    if expected_kind == "classification":
        est = BaggingClassifier(
            estimator=base_est,
            n_estimators=cfg.n_estimators,
            max_samples=cfg.max_samples,
            max_features=cfg.max_features,
            bootstrap=cfg.bootstrap,
            bootstrap_features=cfg.bootstrap_features,
            oob_score=cfg.oob_score,
            warm_start=cfg.warm_start,
            n_jobs=cfg.n_jobs,
            random_state=rs,
        )
        return est, expected_kind

    est = BaggingRegressor(
        estimator=base_est,
        n_estimators=cfg.n_estimators,
        max_samples=cfg.max_samples,
        max_features=cfg.max_features,
        bootstrap=cfg.bootstrap,
        bootstrap_features=cfg.bootstrap_features,
        oob_score=cfg.oob_score,
        warm_start=cfg.warm_start,
        n_jobs=cfg.n_jobs,
        random_state=rs,
    )
    return est, expected_kind


def build_balanced_bagging_ensemble(
    run_cfg: EnsembleRunConfig,
    *,
    rngm: Optional[RngManager] = None,
    stream: str = "bagging",
    kind: Literal["auto", "classification", "regression"] = "auto",
) -> Tuple[Any, EvalKind]:
    """
    Build an unfitted imbalanced-learn BalancedBaggingClassifier based on EnsembleRunConfig.

    Classification-only:
      - If cfg.problem_kind == "regression", raise ValueError.
    """
    if not isinstance(run_cfg.ensemble, BaggingEnsembleConfig):
        raise TypeError(
            f"build_balanced_bagging_ensemble requires BaggingEnsembleConfig, got {type(run_cfg.ensemble).__name__}"
        )

    cfg = cast(BaggingEnsembleConfig, run_cfg.ensemble)

    rngm = rngm or RngManager(None if run_cfg.eval.seed is None else int(run_cfg.eval.seed))

    expected_kind: EvalKind = "classification" if cfg.problem_kind == "classification" else "regression"

    if expected_kind != "classification":
        raise ValueError("Balanced bagging is only supported for classification (problem_kind='classification').")

    if kind != "auto":
        requested_kind = cast(EvalKind, kind)
        if requested_kind != expected_kind:
            raise ValueError(
                f"Ensemble kind override ({requested_kind}) conflicts with bagging problem_kind ({expected_kind})."
            )

    BalancedBaggingClassifier = _import_balanced_bagging_classifier()

    base_est = _build_base_estimator(
        run_cfg=run_cfg,
        cfg=cfg,
        rngm=rngm,
        stream=stream,
        expected_kind=expected_kind,
    )

    rs = cfg.random_state if cfg.random_state is not None else rngm.child_seed(f"{stream}/balanced_bagging")

    est = BalancedBaggingClassifier(
        estimator=base_est,
        n_estimators=cfg.n_estimators,
        max_samples=cfg.max_samples,
        max_features=cfg.max_features,
        bootstrap=cfg.bootstrap,
        bootstrap_features=cfg.bootstrap_features,
        oob_score=cfg.oob_score,
        warm_start=cfg.warm_start,
        n_jobs=cfg.n_jobs,
        random_state=rs,
        sampling_strategy=cfg.sampling_strategy,
        replacement=cfg.replacement,
    )
    return est, expected_kind


def fit_bagging_ensemble(
    run_cfg: EnsembleRunConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    rngm: Optional[RngManager] = None,
    stream: str = "bagging",
    kind: Literal["auto", "classification", "regression"] = "auto",
) -> Any:
    """
    Convenience: build + fit. Emits a warning if y looks like the opposite task type
    than cfg.problem_kind (when kind="auto").
    """
    model, expected_kind = build_bagging_ensemble(run_cfg, rngm=rngm, stream=stream, kind=kind)

    if kind == "auto":
        y_kind = infer_kind_from_y(y_train)
        if y_kind != expected_kind:
            warnings.warn(
                f"Target y looks like '{y_kind}', but bagging problem_kind is '{expected_kind}'. "
                f"Continuing with '{expected_kind}' based on config.",
                UserWarning,
            )

    model.fit(np.asarray(X_train), np.asarray(y_train).ravel())
    return model


def fit_balanced_bagging_ensemble(
    run_cfg: EnsembleRunConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    rngm: Optional[RngManager] = None,
    stream: str = "bagging",
    kind: Literal["auto", "classification", "regression"] = "auto",
) -> Any:
    """
    Convenience: build + fit for BalancedBaggingClassifier.
    Emits a warning if y looks like the opposite task type than cfg.problem_kind (when kind="auto").
    """
    model, expected_kind = build_balanced_bagging_ensemble(run_cfg, rngm=rngm, stream=stream, kind=kind)

    if kind == "auto":
        y_kind = infer_kind_from_y(y_train)
        if y_kind != expected_kind:
            warnings.warn(
                f"Target y looks like '{y_kind}', but bagging problem_kind is '{expected_kind}'. "
                f"Continuing with '{expected_kind}' based on config.",
                UserWarning,
            )

    model.fit(np.asarray(X_train), np.asarray(y_train).ravel())
    return model


__all__ = [
    "build_bagging_ensemble",
    "fit_bagging_ensemble",
    "build_balanced_bagging_ensemble",
    "fit_balanced_bagging_ensemble",
]
