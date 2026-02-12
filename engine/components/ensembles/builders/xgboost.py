from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Literal, cast

import numpy as np

from engine.contracts.ensemble_configs import XGBoostEnsembleConfig
from engine.contracts.ensemble_run_config import EnsembleRunConfig

from engine.runtime.random.rng import RngManager
from engine.core.task_kind import EvalKind, infer_kind_from_y


def _import_xgboost():
    try:
        import xgboost as xgb  # type: ignore
    except Exception as e:
        raise ImportError(
            "XGBoost is not installed. Install it with `pip install xgboost` "
            "and ensure it's included in requirements.txt."
        ) from e
    return xgb


def _clamp(val: float, lo: float, hi: float) -> float:
    try:
        v = float(val)
    except Exception:
        return lo
    return max(lo, min(hi, v))


def _default_patience(n_estimators: int) -> int:
    """
    Reasonable default early stopping patience.
    - Small enough to stop when plateauing
    - Large enough to avoid stopping too early due to noisy validation on small datasets
    """
    try:
        n = int(n_estimators)
    except Exception:
        n = 300
    return int(min(50, max(10, n // 10)))


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


def _choose_eval_metric(expected_kind: EvalKind, n_classes: Optional[int] = None) -> str:
    """
    Choose a stable XGBoost eval_metric for logging learning curves.
    This is ONLY for internal eval_set logging (not the final metric).
    """
    if expected_kind == "regression":
        return "rmse"
    if n_classes is not None and n_classes > 2:
        return "mlogloss"
    return "logloss"


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

    Internal training eval_set:
      - We carve out a validation split from the TRAIN set to enable evals_result_ learning curves.
      - Optional early stopping uses that validation curve to stop training automatically.
      - This does NOT replace your reserved external test evaluation.
    """
    if not isinstance(run_cfg.ensemble, XGBoostEnsembleConfig):
        # should not happen, but keep defensive
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

    # local import to keep xgboost business logic lightweight unless used
    from sklearn.model_selection import train_test_split

    # ---- internal eval_set fraction (clamped) ----
    frac = _clamp(getattr(cfg, "eval_set_fraction", 0.2), 0.05, 0.5)

    # ---- optional early stopping (patience) ----
    use_es = bool(getattr(cfg, "use_early_stopping", False))
    patience_cfg = getattr(cfg, "early_stopping_rounds", None)
    patience: Optional[int] = None
    if use_es:
        patience = int(patience_cfg) if patience_cfg is not None else _default_patience(int(cfg.n_estimators))

        # Set on estimator for broad version compatibility (instead of passing to fit()).
        try:
            model.set_params(early_stopping_rounds=patience)
        except Exception as e:
            warnings.warn(
                f"xgboost: early_stopping_rounds not supported by this xgboost version ({type(e).__name__}: {e}). Disabling early stopping.",
                UserWarning,
            )
            patience = None
            use_es = False

    if expected_kind == "classification":
        # Always encode (identity if already 0..K-1). This avoids XGBoost label constraints.
        classes, y_enc = np.unique(y_arr, return_inverse=True)
        n_classes = int(classes.shape[0])

        # Ensure objective matches number of classes
        try:
            if n_classes <= 2:
                model.set_params(objective="binary:logistic")
            else:
                model.set_params(objective="multi:softprob", num_class=n_classes)
        except Exception as e:
            warnings.warn(
                f"xgboost: failed to set objective/num_class params ({type(e).__name__}: {e}). Continuing with estimator defaults.",
                UserWarning,
            )

        X_np = np.asarray(X_train)
        y_np = np.asarray(y_enc).ravel()

        # --- internal eval_set split (deterministic) ---
        n = int(X_np.shape[0])
        val_n = max(1, int(round(frac * n)))
        if val_n >= n:
            val_n = n - 1  # keep at least 1 training sample

        try:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_np,
                y_np,
                test_size=val_n,
                random_state=split_rs,
                stratify=y_np if n_classes > 1 else None,
                shuffle=True,
            )
        except Exception:
            # If stratified split fails (rare classes), fall back to non-stratified.
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_np,
                y_np,
                test_size=val_n,
                random_state=split_rs,
                shuffle=True,
            )

        eval_metric = _choose_eval_metric(expected_kind, n_classes=n_classes)

        # Set eval_metric on the estimator (older sklearn API compatibility)
        try:
            model.set_params(eval_metric=eval_metric)
        except Exception as e:
            warnings.warn(
                f"xgboost: failed to set eval_metric ({type(e).__name__}: {e}). Continuing with estimator defaults.",
                UserWarning,
            )

        # Fit with eval_set (try verbose=False, fall back if unsupported)
        try:
            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_tr, y_tr), (X_val, y_val)],
                verbose=False,
            )
        except TypeError as e:
            # some versions don't accept verbose kwarg either
            if "unexpected keyword argument" in str(e) and "verbose" in str(e):
                model.fit(
                    X_tr,
                    y_tr,
                    eval_set=[(X_tr, y_tr), (X_val, y_val)],
                )
            else:
                raise

        return XGBClassifierLabelAdapter(model=model, classes_=classes)

    # Regression: no label encoding needed
    X_np = np.asarray(X_train)
    y_np = y_arr

    n = int(X_np.shape[0])
    val_n = max(1, int(round(frac * n)))
    if val_n >= n:
        val_n = n - 1

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_np,
        y_np,
        test_size=val_n,
        random_state=split_rs,
        shuffle=True,
    )

    eval_metric = _choose_eval_metric(expected_kind)

    # Set eval_metric on the estimator (older sklearn API compatibility)
    try:
        model.set_params(eval_metric=eval_metric)
    except Exception:
        pass

    # Fit with eval_set (try verbose=False, fall back if unsupported)
    try:
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_tr, y_tr), (X_val, y_val)],
            verbose=False,
        )
    except TypeError as e:
        # some versions don't accept verbose kwarg either
        if "unexpected keyword argument" in str(e) and "verbose" in str(e):
            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_tr, y_tr), (X_val, y_val)],
            )
        else:
            raise

    return model


__all__ = ["build_xgboost_ensemble", "fit_xgboost_ensemble", "XGBClassifierLabelAdapter"]
