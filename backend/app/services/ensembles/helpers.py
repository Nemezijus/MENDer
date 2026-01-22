import numpy as np
from typing import Optional

from shared_schemas.ensemble_run_config import EnsembleRunConfig


def _unwrap_final_estimator(model):
    """
    If model is a sklearn Pipeline, return its last step estimator.
    Otherwise return model unchanged.
    """
    try:
        if hasattr(model, "steps") and isinstance(getattr(model, "steps"), list) and len(model.steps) > 0:
            return model.steps[-1][1]
    except Exception:
        pass
    return model


def _transform_through_pipeline(model, X):
    """
    If model is a fitted sklearn Pipeline, transform X through all steps except the last estimator.
    If any step cannot transform, return X unchanged.
    """
    try:
        if not (hasattr(model, "steps") and isinstance(getattr(model, "steps"), list) and len(model.steps) > 1):
            return X

        Xt = X
        for _, step in model.steps[:-1]:
            if step is None or step == "passthrough":
                continue
            if hasattr(step, "transform"):
                Xt = step.transform(Xt)
            else:
                # Can't safely transform (no refit). Fall back to raw X.
                return X
        return Xt
    except Exception:
        return X


def _slice_X_by_features(X, feat_idx):
    # Best-effort column slicing that works for numpy arrays, pandas DataFrames, and scipy sparse.
    if feat_idx is None:
        return X
    try:
        idx = feat_idx
        # sklearn uses numpy arrays of indices; ensure list-like works
        if not isinstance(idx, (slice, list, tuple)):
            idx = list(idx)
    except Exception:
        idx = feat_idx
    try:
        return X[:, idx]
    except Exception:
        try:
            import numpy as _np
            Xa = _np.asarray(X)
            return Xa[:, idx]
        except Exception:
            return X


def _get_classes_arr(model) -> Optional[np.ndarray]:
    classes = getattr(model, "classes_", None)
    if classes is None:
        return None
    try:
        arr = np.asarray(classes)
        return arr if arr.size > 0 else None
    except Exception:
        return None


def _should_decode_from_index_space(y_true: np.ndarray, y_pred_raw: np.ndarray, classes_arr: Optional[np.ndarray]) -> bool:
    """
    Deterministic decode decision:
      - we only decode if predictions look like integer indices in [0, K-1]
      - AND those indices are not already valid class labels (e.g. classes are 0..K-1)

    This avoids decoding when user classes are already 0..K-1.
    """
    if classes_arr is None:
        return False

    K = int(len(classes_arr))
    if K <= 0:
        return False

    a = np.asarray(y_pred_raw)
    if a.size == 0:
        return False

    # must be integer-ish indices
    if not np.issubdtype(a.dtype, np.integer):
        return False

    mn = int(a.min())
    mx = int(a.max())
    if mn < 0 or mx >= K:
        return False

    # If predicted values are already among classes_ values (e.g. classes_ are [0,1,2]),
    # then decoding would be wrong.
    try:
        pred_set = set(np.unique(a).tolist())
        class_set = set(np.unique(classes_arr).tolist())
        if pred_set.issubset(class_set):
            return False
    except Exception:
        pass

    # Also check: y_true values match the declared classes (common case)
    # If they do, decoding indices via classes_ is meaningful.
    try:
        yt_set = set(np.unique(y_true).tolist())
        class_set = set(np.unique(classes_arr).tolist())
        if not yt_set.issubset(class_set):
            # y_true doesn't match model.classes_ (unexpected) -> don't decode
            return False
    except Exception:
        return False

    return True


def _encode_y_true_to_index(y_true: np.ndarray, classes_arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    Deterministically map y_true values to 0..K-1 using model.classes_ ordering.
    Returns None if mapping not possible.
    """
    if classes_arr is None:
        return None
    try:
        idx_map = {cls: i for i, cls in enumerate(classes_arr.tolist())}
        return np.asarray([idx_map[v] for v in y_true], dtype=int)
    except Exception:
        return None


def _friendly_ensemble_training_error(
    e: Exception, cfg: EnsembleRunConfig, *, fold_id: int | None = None
) -> ValueError:
    """
    Convert common low-level sklearn/3rd-party errors into actionable, user-friendly messages.

    We return ValueError so the router can surface a 400 with a clear explanation.
    """
    kind = getattr(cfg.ensemble, "kind", "ensemble")
    msg = (str(e) or e.__class__.__name__).strip()
    fold_txt = f" (fold {fold_id})" if fold_id is not None else ""

    # --- Bagging / bootstrap: single-class resample -------------------------------
    if "needs samples of at least 2 classes" in msg and "only one class" in msg:
        return ValueError(
            "Bagging training failed because at least one bootstrap sample contained only a single class"
            f"{fold_txt}.\n"
            "Some base estimators (e.g., Logistic Regression, SVM) cannot be fit on single-class data.\n\n"
            "Try one of the following:\n"
            "• Enable Balanced bagging (recommended)\n"
            "• Increase max_samples (use larger bags)\n"
            "• Disable bootstrap (bootstrap=false)\n"
            "• Use a tree-based base estimator\n\n"
            f"Raw error: {msg}"
        )

    # --- Balanced bagging dependency missing --------------------------------------
    if isinstance(e, ImportError) or "imbalanced-learn" in msg.lower():
        return ValueError(
            "Balanced bagging requires the optional dependency `imbalanced-learn`.\n"
            "Install it in your environment (and rebuild Docker if applicable), then retry.\n\n"
            f"Raw error: {msg}"
        )

    # --- AdaBoost + pipeline sample_weight routing --------------------------------
    if "Pipeline doesn't support sample_weight" in msg:
        return ValueError(
            "AdaBoost training failed because the current preprocessing pipeline cannot accept sample weights.\n"
            "AdaBoost reweights samples at each boosting stage and requires the estimator fit() to support "
            "sample_weight.\n"
            "In MENDer, we address this by fitting preprocessing once, transforming X, then boosting on "
            "transformed features.\n\n"
            f"Raw error: {msg}"
        )

    # --- XGBoost label encoding ----------------------------------------------------
    if "Invalid classes inferred from unique values of `y`" in msg:
        return ValueError(
            "XGBoost classification requires class labels encoded as 0..K-1.\n"
            "MENDer encodes labels internally for training and decodes predictions back to original labels.\n\n"
            f"Raw error: {msg}"
        )

    # Fallback: keep original message but add context.
    return ValueError(f"{kind} training failed{fold_txt}: {msg}")


def _extract_base_estimator_algo_from_cfg(cfg: EnsembleRunConfig, default: str = "default") -> str:
    """Best-effort: get algo name for Bagging base estimator from config."""
    try:
        be = getattr(cfg.ensemble, "base_estimator", None)
        if be is None:
            return default
        return str(getattr(be, "algo", default) or default)
    except Exception:
        return default
