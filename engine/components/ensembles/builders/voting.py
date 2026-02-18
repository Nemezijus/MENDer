from __future__ import annotations

import warnings
from typing import Any, Optional, Tuple, Literal, cast

import numpy as np
from sklearn.ensemble import VotingClassifier, VotingRegressor

from engine.contracts.ensemble_configs import VotingEnsembleConfig
from engine.contracts.ensemble_run_config import EnsembleRunConfig

from engine.core.random.rng import RngManager
from engine.factories.pipeline_factory import make_pipeline_for_model_cfg
from engine.core.task_kind import (
    EvalKind,
    infer_kind_from_y,
    ensure_uniform_model_task,
)


def _unique_name(base: str, used: set[str]) -> str:
    if base not in used:
        used.add(base)
        return base
    i = 2
    while f"{base}_{i}" in used:
        i += 1
    name = f"{base}_{i}"
    used.add(name)
    return name


def _build_estimators_and_weights(
    *,
    run_cfg: EnsembleRunConfig,
    cfg: VotingEnsembleConfig,
    rngm: RngManager,
    stream: str,
) -> Tuple[list[Tuple[str, Any]], Optional[list[float]]]:
    if len(cfg.estimators) < 2:
        raise ValueError("VotingEnsembleConfig.estimators must contain at least 2 estimators.")

    used_names: set[str] = set()
    estimators: list[Tuple[str, Any]] = []
    weights_raw: list[Optional[float]] = []

    for i, spec in enumerate(cfg.estimators, start=1):
        est_stream = f"{stream}/est{i}"

        algo = getattr(spec.model, "algo", "model")
        base_name = spec.name or f"{algo}_{i}"
        name = _unique_name(base_name, used_names)

        pipe = make_pipeline_for_model_cfg(
            scale=run_cfg.scale,
            features=run_cfg.features,
            model_cfg=spec.model,
            eval_cfg=run_cfg.eval,
            rngm=rngm,
            stream=est_stream,
        )

        estimators.append((name, pipe))
        weights_raw.append(spec.weight)

    if any(w is not None for w in weights_raw):
        weights = [1.0 if w is None else float(w) for w in weights_raw]
    else:
        weights = None

    return estimators, weights


def _validate_soft_voting(estimators: list[Tuple[str, Any]]) -> None:
    missing = []
    for name, est in estimators:
        if not hasattr(est, "predict_proba"):
            missing.append(name)
    if missing:
        raise ValueError(
            "Soft voting requires all estimators to support predict_proba, but these do not: "
            + ", ".join(missing)
        )


def build_voting_ensemble(
    run_cfg: EnsembleRunConfig,
    *,
    rngm: Optional[RngManager] = None,
    stream: str = "voting",
    kind: Literal["auto", "classification", "regression"] = "auto",
) -> Tuple[Any, EvalKind]:
    """
    Build an unfitted sklearn VotingClassifier/VotingRegressor based on EnsembleRunConfig.

    Returns: (estimator, expected_kind)
      - expected_kind is derived from base estimator configs (authoritative for MENDer)
    """
    if not isinstance(run_cfg.ensemble, VotingEnsembleConfig):
        raise TypeError(
            f"build_voting_ensemble requires VotingEnsembleConfig, got {type(run_cfg.ensemble).__name__}"
        )

    cfg = cast(VotingEnsembleConfig, run_cfg.ensemble)

    rngm = rngm or RngManager(None if run_cfg.eval.seed is None else int(run_cfg.eval.seed))

    expected_kind: EvalKind = ensure_uniform_model_task([s.model for s in cfg.estimators])

    if kind != "auto":
        requested_kind = cast(EvalKind, kind)
        if requested_kind != expected_kind:
            raise ValueError(
                f"Ensemble kind override ({requested_kind}) conflicts with model task ({expected_kind})."
            )

    estimators, weights = _build_estimators_and_weights(
        run_cfg=run_cfg,
        cfg=cfg,
        rngm=rngm,
        stream=stream,
    )

    if expected_kind == "classification":
        if cfg.voting == "soft":
            _validate_soft_voting(estimators)

        est = VotingClassifier(
            estimators=estimators,
            voting=cfg.voting,
            weights=weights,
            flatten_transform=cfg.flatten_transform,
            n_jobs=None,
        )
        return est, expected_kind

    # regression
    est = VotingRegressor(
        estimators=estimators,
        weights=weights,
        n_jobs=None,
    )
    return est, expected_kind


def fit_voting_ensemble(
    run_cfg: EnsembleRunConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    rngm: Optional[RngManager] = None,
    stream: str = "voting",
    kind: Literal["auto", "classification", "regression"] = "auto",
) -> Any:
    """
    Convenience: build + fit. Emits a warning if y looks like the opposite task type
    than what the selected base estimators imply (when kind="auto").
    """
    model, expected_kind = build_voting_ensemble(run_cfg, rngm=rngm, stream=stream, kind=kind)

    if kind == "auto":
        y_kind = infer_kind_from_y(y_train)
        if y_kind != expected_kind:
            warnings.warn(
                f"Target y looks like '{y_kind}', but voting base estimators are '{expected_kind}'. "
                f"Continuing with '{expected_kind}' based on model configs.",
                UserWarning,
            )

    model.fit(np.asarray(X_train), np.asarray(y_train).ravel())
    return model


__all__ = ["build_voting_ensemble", "fit_voting_ensemble"]
