from __future__ import annotations

from typing import Any, Optional, Tuple

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]

from engine.components.prediction.predicting import predict_labels, predict_scores


class SimplePredictionError(RuntimeError):
    """Raised when a model cannot be used for prediction in a simple, generic way."""


def _normalize_input_X(X: Any) -> Any:
    """Best-effort normalization of X for sklearn-style estimators."""
    if np is None:
        return X

    if isinstance(X, np.ndarray):
        return X

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
    """Generic prediction helper.

    Returns (y_pred, y_score, score_kind) where score_kind is one of
    {"proba", "decision", "predict"} or None.
    """

    if not hasattr(model, "predict"):
        raise SimplePredictionError("Model object must define a 'predict' method.")

    X_norm = _normalize_input_X(X)

    try:
        y_pred = predict_labels(model, X_norm)
    except Exception as exc:  # pragma: no cover
        raise SimplePredictionError(f"Model prediction failed: {exc}") from exc

    y_score: Optional[Any] = None
    score_kind: Optional[str] = None

    if want_scores:
        try:
            y_score, used = predict_scores(model, X_norm, kind="auto")
            score_kind = str(used)
        except Exception as exc:  # pragma: no cover
            raise SimplePredictionError(f"Model score prediction failed: {exc}") from exc

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
