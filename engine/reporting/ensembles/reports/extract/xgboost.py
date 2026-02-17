from __future__ import annotations

from typing import Any, Dict, Optional


def extract_xgboost_fold_info(*, xgb_model: Any) -> Dict[str, Any]:
    """Extract report-relevant information from a fitted XGBoost estimator.

    This is intentionally tolerant: missing attributes are returned as None.
    """

    params: Dict[str, Any] = {}
    try:
        params = dict(getattr(xgb_model, "get_params", lambda: {})() or {})
    except Exception:
        params = {}

    train_eval_metric: Optional[str] = None
    try:
        tem = params.get("eval_metric", None)
        if isinstance(tem, (list, tuple)) and len(tem) > 0:
            train_eval_metric = str(tem[0])
        elif tem is not None:
            train_eval_metric = str(tem)
    except Exception:
        train_eval_metric = None

    best_iteration = getattr(xgb_model, "best_iteration", None) or getattr(xgb_model, "best_iteration_", None)
    best_score = getattr(xgb_model, "best_score", None) or getattr(xgb_model, "best_score_", None)

    evals_result = getattr(xgb_model, "evals_result_", None)
    if callable(evals_result):
        evals_result = None
    if not evals_result:
        best_iteration = None
        best_score = None

    feat_imps = getattr(xgb_model, "feature_importances_", None)
    feature_names = None

    return {
        "params": params,
        "train_eval_metric": train_eval_metric,
        "best_iteration": best_iteration,
        "best_score": best_score,
        "evals_result": evals_result,
        "feature_importances": feat_imps,
        "feature_names": feature_names,
    }
