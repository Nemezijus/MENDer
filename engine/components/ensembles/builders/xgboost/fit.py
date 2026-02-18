from __future__ import annotations

import warnings
from typing import Any, Optional, Literal, cast

import numpy as np

from engine.contracts.ensemble_configs import XGBoostEnsembleConfig
from engine.contracts.ensemble_run_config import EnsembleRunConfig
from engine.core.task_kind import infer_kind_from_y
from engine.core.random.rng import RngManager

from .build import build_xgboost_ensemble
from .labels import encode_labels, XGBClassifierLabelAdapter
from .utils import (
    clamp_fraction,
    default_patience,
    choose_eval_metric,
    train_val_split,
    compute_val_n,
)


def _set_params_best_effort(model: Any, **params: Any) -> None:
    try:
        model.set_params(**params)
    except Exception as e:
        # Keep behavior stable across xgboost versions
        warnings.warn(
            f"xgboost: failed to set params {sorted(params.keys())} ({type(e).__name__}: {e}). Continuing.",
            UserWarning,
        )


def _fit_with_eval_set(model: Any, X_tr: np.ndarray, y_tr: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> None:
    """Fit with eval_set; tolerate older xgboost sklearn-wrapper APIs."""

    try:
        model.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_val, y_val)], verbose=False)
    except TypeError as e:
        if "unexpected keyword argument" in str(e) and "verbose" in str(e):
            model.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_val, y_val)])
        else:
            raise


def fit_xgboost_ensemble(
    run_cfg: EnsembleRunConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    rngm: Optional[RngManager] = None,
    stream: str = "xgboost",
    kind: Literal["auto", "classification", "regression"] = "auto",
) -> Any:
    """Convenience: build + fit.

    Classification:
      - Encode y to 0..K-1 (required by XGBoost for multiclass).
      - Return an adapter so predict() yields original labels.

    Internal training eval_set:
      - We carve out a validation split from the TRAIN set to enable evals_result_ learning curves.
      - Optional early stopping uses that validation curve to stop training automatically.
    """

    if not isinstance(run_cfg.ensemble, XGBoostEnsembleConfig):
        raise TypeError(f"fit_xgboost_ensemble requires XGBoostEnsembleConfig, got {type(run_cfg.ensemble).__name__}")

    cfg = cast(XGBoostEnsembleConfig, run_cfg.ensemble)
    model, expected_kind = build_xgboost_ensemble(run_cfg, rngm=rngm, stream=stream, kind=kind)

    y_arr = np.asarray(y_train).ravel()

    if kind == "auto":
        y_kind = infer_kind_from_y(y_arr)
        if y_kind != expected_kind:
            warnings.warn(
                f"Target y looks like '{y_kind}', but xgboost problem_kind is '{expected_kind}'. "
                f"Continuing with '{expected_kind}' based on config.",
                UserWarning,
            )

    rngm = rngm or RngManager(None if run_cfg.eval.seed is None else int(run_cfg.eval.seed))
    split_rs = rngm.child_seed(f"{stream}/xgb/valsplit")

    frac = clamp_fraction(getattr(cfg, "eval_set_fraction", 0.2))
    use_es = bool(getattr(cfg, "use_early_stopping", False))
    patience_cfg = getattr(cfg, "early_stopping_rounds", None)

    if use_es:
        patience = int(patience_cfg) if patience_cfg is not None else default_patience(int(cfg.n_estimators))
        # Set on estimator for broad version compatibility (instead of passing to fit()).
        try:
            model.set_params(early_stopping_rounds=int(patience))
        except Exception as e:
            warnings.warn(
                f"xgboost: early_stopping_rounds not supported by this xgboost version ({type(e).__name__}: {e}). "
                "Disabling early stopping.",
                UserWarning,
            )

    X_np = np.asarray(X_train)
    n = int(X_np.shape[0])
    val_n = compute_val_n(n_samples=n, frac=frac)

    if expected_kind == "classification":
        classes, y_enc = encode_labels(y_arr)
        n_classes = int(classes.shape[0])

        # Ensure objective matches number of classes
        if n_classes <= 2:
            _set_params_best_effort(model, objective="binary:logistic")
        else:
            _set_params_best_effort(model, objective="multi:softprob", num_class=n_classes)

        eval_metric = choose_eval_metric("classification", n_classes=n_classes)
        _set_params_best_effort(model, eval_metric=eval_metric)

        split = train_val_split(
            X_np,
            y_enc,
            val_n=val_n,
            random_state=split_rs,
            stratify=(n_classes > 1),
        )

        _fit_with_eval_set(model, split.X_tr, split.y_tr, split.X_val, split.y_val)
        return XGBClassifierLabelAdapter(model=model, classes_=classes)

    # Regression
    eval_metric = choose_eval_metric("regression")
    try:
        model.set_params(eval_metric=eval_metric)
    except Exception:
        pass

    split = train_val_split(
        X_np,
        y_arr,
        val_n=val_n,
        random_state=split_rs,
        stratify=False,
    )

    _fit_with_eval_set(model, split.X_tr, split.y_tr, split.X_val, split.y_val)
    return model


__all__ = ["fit_xgboost_ensemble"]
