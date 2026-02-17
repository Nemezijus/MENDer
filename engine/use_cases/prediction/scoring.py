from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

from engine.components.evaluation.scoring import PROBA_METRICS
from engine.components.prediction.predicting import predict_scores
from engine.factories.eval_factory import make_evaluator
from engine.use_cases.prediction.utils import safe_float_optional


def score_if_possible(
    *,
    task: str,
    eval_model: Any,
    pipeline: Any,
    X_arr: np.ndarray,
    y_true: Optional[np.ndarray],
    y_pred: np.ndarray,
    notes: list[str],
) -> Tuple[Optional[str], Optional[float]]:
    """Compute the configured metric if labels + eval config are available."""

    if y_true is None or eval_model is None:
        return None, None

    eval_kind = "regression" if task == "regression" else "classification"
    evaluator = make_evaluator(eval_model, kind=eval_kind)

    metric_name: Optional[str] = str(getattr(eval_model, "metric", None) or "") or None
    if metric_name is None:
        return None, None

    y_proba: Optional[np.ndarray] = None
    y_score: Optional[np.ndarray] = None

    if eval_kind == "classification" and metric_name in PROBA_METRICS:
        try:
            scores, used = predict_scores(pipeline, X_arr, kind="auto")
            if used == "proba":
                y_proba = np.asarray(scores)
            elif used == "decision":
                y_score = np.asarray(scores)
        except Exception as e:
            notes.append(
                f"Could not compute proba/decision scores for metric '{metric_name}': {e}"
            )

    try:
        metric_value = safe_float_optional(
            evaluator.score(y_true, y_pred, y_proba=y_proba, y_score=y_score)
        )
    except Exception as e:
        notes.append(f"Metric computation failed ({type(e).__name__}: {e})")
        metric_value = None

    return metric_name, metric_value
