from __future__ import annotations

from typing import Any, Literal, Optional, Tuple
import numpy as np

# Canonical decoder API (Segment 6)
from .decoder.api import predict_decoder_outputs  # noqa: F401


def predict_labels(model: Any, X_test: np.ndarray) -> np.ndarray:
    """
    Predict hard labels with a scikit-learn–style estimator.

    Parameters
    ----------
    model : Any
        Fitted estimator exposing `predict(X)`.
    X_test : array-like of shape (n_samples, n_features)

    Returns
    -------
    y_pred : ndarray of shape (n_samples,)

    Raises
    ------
    AttributeError
        If `model` does not have `.predict(...)`.
    ValueError
        If X_test is not 2D.
    """
    if not hasattr(model, "predict"):
        raise AttributeError("`model` has no `.predict(...)` method.")

    X_test = np.asarray(X_test)
    if X_test.ndim != 2:
        raise ValueError(f"X_test must be 2D; got {X_test.shape}.")

    y_pred = model.predict(X_test)
    return np.asarray(y_pred)


def predict_scores(
    model: Any,
    X_test: np.ndarray,
    *,
    kind: Literal["auto", "proba", "decision"] = "auto",
) -> Tuple[np.ndarray, str]:
    """
    Get per-sample scores for evaluation/thresholding.

    Behavior
    --------
    - kind='proba'     → uses `predict_proba` (if available), returns class probabilities.
    - kind='decision'  → uses `decision_function` (if available), returns margins/scores.
    - kind='auto'      → prefer `predict_proba`; if unavailable, fall back to `decision_function`.

    Parameters
    ----------
    model : Any
        Fitted estimator.
    X_test : array-like, shape (n_samples, n_features)
    kind : {'auto','proba','decision'}, default='auto'

    Returns
    -------
    scores : ndarray
        For classifiers:
          - predict_proba → shape (n_samples, n_classes)
          - decision_function → shape (n_samples,) or (n_samples, n_classes)
        For regressors:
          - returns predictions with shape (n_samples,) (uses `.predict` as last resort).
    used : str
        Which method was used: 'proba', 'decision', or 'predict' (fallback).

    Notes
    -----
    - Some estimators (e.g., SVC with default settings) may not expose `predict_proba`.
    - For regressors, probabilities are not defined; this falls back to `.predict`.
    """
    X_test = np.asarray(X_test)
    if X_test.ndim != 2:
        raise ValueError(f"X_test must be 2D; got {X_test.shape}.")

    if kind == "proba":
        if hasattr(model, "predict_proba"):
            return np.asarray(model.predict_proba(X_test)), "proba"
        raise AttributeError("Requested kind='proba' but model has no predict_proba.")

    if kind == "decision":
        if hasattr(model, "decision_function"):
            return np.asarray(model.decision_function(X_test)), "decision"
        raise AttributeError("Requested kind='decision' but model has no decision_function.")

    # kind == 'auto'
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X_test)), "proba"
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X_test)), "decision"

    # Fallback (e.g., regressors or minimal classifiers)
    if hasattr(model, "predict"):
        return np.asarray(model.predict(X_test)), "predict"

    raise AttributeError("Model exposes none of predict_proba / decision_function / predict.")
