# utils/ensembles/xgboost.py
from __future__ import annotations

import warnings
from dataclasses import dataclass
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


@dataclass
class XGBClassifierLabelAdapter:
    """Wrap XGBClassifier so that callers see original class labels.

    XGBoost expects multiclass labels to be 0..K-1. We encode y during fit and
    decode predictions back to original labels.

    This adapter is intentionally minimal: it supports predict/predict_proba and
    delegates everything else to the underlying model.
    """

    model: Any
    classes_: np.ndarray  # original label values (sorted unique)

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_hat = self.model.predict(np.asarray(X))
        # XGBoost may return floats for class indices; coerce to int indices.
        idx = np.asarray(y_hat).astype(int, copy=False).ravel()
        idx = np.clip(idx, 0, len(self.classes_) - 1)
        return self.classes_[idx]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Probabilities are already in encoded class order 0..K-1, which matches classes_.
        return self.model.predict_proba(np.asarray(X))

    def __getattr__(self, name: str) -> Any:
        # Delegate to the wrapped xgboost model (params, boosters, etc.)
        return getattr(self.model, name)


def build_xgboost_ensemble(
    run_cfg: EnsembleRunConfig,
    *,
    rngm: Optional[RngManager] = None,
    stream: str = "xgboost",
    kind: Literal["auto", "classification", "regression"] = "auto",
) -> Tuple[Any, EvalKind]:
    """
    Build an unfitted xgboost.XGBClassifier / xgboost.XGBRegressor based on EnsembleRunConfig.

    NOTE: For classification, label encoding to 0..K-1 is handled in fit_xgboost_ensemble.
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

    # We intentionally do not bind MENDer scoring metrics here; evaluation is handled elsewhere.
    if expected_kind == "classification":
        # Objective/num_class are set in fit() after we know number of classes.
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
    Convenience: build + fit.

    For classification:
      - Encode y to 0..K-1 (required by XGBoost for multiclass).
      - Return an adapter so predict() yields original labels.

    Emits a warning if y looks like the opposite task type than cfg.problem_kind
    (when kind="auto").
    """
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

    if expected_kind == "classification":
        # Always encode (identity if already 0..K-1). This avoids XGBoost label constraints.
        classes, y_enc = np.unique(y_arr, return_inverse=True)
        n_classes = int(classes.shape[0])

        # Ensure objective matches number of classes
        # - binary:logistic for 2 classes
        # - multi:softprob (+num_class) for >2
        try:
            if n_classes <= 2:
                model.set_params(objective="binary:logistic")
            else:
                model.set_params(objective="multi:softprob", num_class=n_classes)
        except Exception:
            # If set_params fails for any reason, proceed; fit may still work depending on defaults.
            pass

        model.fit(np.asarray(X_train), np.asarray(y_enc).ravel())
        return XGBClassifierLabelAdapter(model=model, classes_=classes)

    # Regression: no label encoding needed
    model.fit(np.asarray(X_train), y_arr)
    return model


__all__ = ["build_xgboost_ensemble", "fit_xgboost_ensemble", "XGBClassifierLabelAdapter"]
