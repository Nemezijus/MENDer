from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from engine.components.evaluation.scoring import PROBA_METRICS
from engine.reporting.ensembles.helpers import (
    _encode_y_true_to_index,
    _get_classes_arr,
    _should_decode_from_index_space,
)


SliceFn = Callable[[Any, Any], Any]


@dataclass(frozen=True)
class BasePredictionResult:
    base_preds: Optional[np.ndarray]
    base_scores: Optional[List[float]]


def _get_feature_subset(feature_indices_list: Any, i: int) -> Any:
    try:
        if feature_indices_list is not None and i < len(feature_indices_list):
            return feature_indices_list[i]
    except Exception:
        return None
    return None


def collect_base_predictions_classification(
    *,
    estimators: Sequence[Any],
    X: Any,
    y_true: Any,
    evaluator: Any,
    metric_name: str,
    classes_arr: Any = None,
    feature_indices_list: Any = None,
    slice_X: Optional[SliceFn] = None,
) -> BasePredictionResult:
    """Collect base estimator predictions (and optional per-estimator scores) for classification.

    Handles:
      - feature-subset slicing (bagging)
      - index-space decoding heuristics (xgboost/label-encoded estimators)
      - proba metrics evaluated deterministically in encoded space when needed
    """

    y_true_arr = np.asarray(y_true)
    if classes_arr is None:
        classes_arr = _get_classes_arr(estimators[0]) if len(estimators) > 0 else None

    y_true_enc = _encode_y_true_to_index(y_true_arr, classes_arr) if classes_arr is not None else None

    pred_cols: List[np.ndarray] = []
    score_list: List[float] = []

    for i, est in enumerate(estimators):
        if est is None:
            continue

        X_i = X
        feat_idx = _get_feature_subset(feature_indices_list, i)
        if slice_X is not None and feat_idx is not None:
            X_i = slice_X(X, feat_idx)

        yp_raw = np.asarray(est.predict(X_i))

        # Determine the values that should go into the report table/plots.
        if _should_decode_from_index_space(y_true_arr, yp_raw, classes_arr):
            yp_report = classes_arr[yp_raw.astype(int, copy=False)]
        else:
            yp_report = yp_raw

        pred_cols.append(np.asarray(yp_report))

        # Optional per-estimator score (best-effort)
        try:
            y_proba_i = None
            y_score_i = None

            if metric_name in PROBA_METRICS:
                if hasattr(est, "predict_proba"):
                    try:
                        y_proba_i = est.predict_proba(X_i)
                    except Exception:
                        y_proba_i = None
                if y_proba_i is None and hasattr(est, "decision_function"):
                    try:
                        y_score_i = est.decision_function(X_i)
                    except Exception:
                        y_score_i = None

                if y_proba_i is None and y_score_i is None:
                    s = None
                elif y_true_enc is not None and _should_decode_from_index_space(y_true_arr, yp_raw, classes_arr):
                    # Evaluate in encoded space deterministically.
                    s = evaluator.score(
                        y_true_enc,
                        y_pred=yp_raw,
                        y_proba=y_proba_i,
                        y_score=y_score_i,
                    )
                else:
                    s = evaluator.score(
                        y_true_arr,
                        y_pred=yp_report,
                        y_proba=y_proba_i,
                        y_score=y_score_i,
                    )
            else:
                s = evaluator.score(
                    y_true_arr,
                    y_pred=yp_report,
                    y_proba=None,
                    y_score=None,
                )

            if s is not None:
                score_list.append(float(s))
        except Exception:
            pass

    if not pred_cols:
        return BasePredictionResult(base_preds=None, base_scores=None)

    base_preds_mat = np.column_stack(pred_cols)
    return BasePredictionResult(
        base_preds=base_preds_mat,
        base_scores=score_list if score_list else None,
    )


def collect_base_predictions_regression(
    *,
    estimators: Sequence[Any],
    X: Any,
    y_true: Any,
    evaluator: Any,
    feature_indices_list: Any = None,
    slice_X: Optional[SliceFn] = None,
) -> BasePredictionResult:
    """Collect base estimator predictions (and optional per-estimator scores) for regression."""

    y_true_arr = np.asarray(y_true, dtype=float)
    pred_cols: List[np.ndarray] = []
    score_list: List[float] = []

    for i, est in enumerate(estimators):
        if est is None:
            continue

        X_i = X
        feat_idx = _get_feature_subset(feature_indices_list, i)
        if slice_X is not None and feat_idx is not None:
            X_i = slice_X(X, feat_idx)

        yp = np.asarray(est.predict(X_i), dtype=float)
        pred_cols.append(yp)

        try:
            s = evaluator.score(y_true_arr, y_pred=yp, y_proba=None, y_score=None)
            score_list.append(float(s))
        except Exception:
            pass

    if not pred_cols:
        return BasePredictionResult(base_preds=None, base_scores=None)

    base_preds_mat = np.column_stack(pred_cols)
    return BasePredictionResult(
        base_preds=base_preds_mat,
        base_scores=score_list if score_list else None,
    )
