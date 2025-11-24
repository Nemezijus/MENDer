from __future__ import annotations

from typing import Any, Optional, Tuple

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - numpy may not be installed in all environments
    np = None  # type: ignore[assignment]

from utils.factories.predict_factory import make_predictor  # noqa: E402


class SimplePredictionError(RuntimeError):
    """Raised when a model cannot be used for prediction in a simple, generic way."""


def _normalize_input_X(X: Any) -> Any:
    """
    Normalize input X into something that is friendly to scikit-learn-style estimators.

    Intentionally conservative:
    - If X is already an array-like or DataFrame, we just return it.
    - For plain Python sequences, we optionally convert to numpy array (if available).
    """
    if np is None:
        # Without numpy, just pass through and rely on downstream estimator to complain if needed.
        return X

    # If it already looks like a numpy array, just return it.
    if isinstance(X, np.ndarray):
        return X

    # For simple sequences, coerce to array.
    # We avoid touching DataFrame-like objects (they will have .values / .to_numpy).
    if isinstance(X, (list, tuple)):
        try:
            return np.asarray(X)
        except Exception:
            return X

    return X


def simple_predict(
    model: Any,
    X: Any,
    *,
    want_scores: bool = False,
) -> Tuple[Any, Optional[Any], Optional[str]]:
    """
    Run a generic prediction using `model` on input `X`, leveraging the existing
    prediction strategy (Predictor) from `predict_factory`.

    Parameters
    ----------
    model:
        Fitted model or pipeline with a `predict` method. May optionally expose
        `predict_proba` and/or `decision_function` for classification tasks.
    X:
        Input features for which predictions should be made.
    want_scores:
        If True, also request scores from the predictor (probabilities or decision
        function, depending on the model and task). The exact kind is returned as
        the third element of the tuple.

    Returns
    -------
    (y_pred, y_score, score_kind)
        - y_pred: main predictions (labels or numeric predictions).
        - y_score: optional scores array (probabilities or decision scores).
        - score_kind: one of {"proba", "decision"} or None if scores not requested.

    Raises
    ------
    SimplePredictionError
        If the model does not define a usable `predict` method or prediction fails.
    """
    if not hasattr(model, "predict"):
        raise SimplePredictionError("Model object must define a 'predict' method.")

    X_norm = _normalize_input_X(X)

    predictor = make_predictor()

    try:
        y_pred = predictor.predict(model, X_norm)
    except Exception as exc:  # pragma: no cover - model-specific failure
        raise SimplePredictionError(f"Model prediction failed: {exc}") from exc

    y_score: Optional[Any] = None
    score_kind: Optional[str] = None

    if want_scores:
        try:
            # Let the underlying predictor choose the best representation ("auto").
            y_score, score_kind_literal = predictor.predict_scores(
                model, X_norm, kind="auto"
            )
            score_kind = str(score_kind_literal)
        except Exception as exc:  # pragma: no cover
            raise SimplePredictionError(f"Model score prediction failed: {exc}") from exc

    # Optionally normalize outputs to numpy arrays if available.
    if np is not None:
        try:
            y_pred = np.asarray(y_pred)
        except Exception:
            pass
        if y_score is not None:
            try:
                y_score = np.asarray(y_score)
            except Exception:
                pass

    return y_pred, y_score, score_kind
