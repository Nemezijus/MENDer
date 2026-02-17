from typing import Any

from engine.contracts.ensemble_run_config import EnsembleRunConfig

from engine.reporting.ensembles.xgboost import XGBoostEnsembleReportAccumulator

from .extract import extract_xgboost_fold_info, resolve_estimator_and_X
from ..common import attach_report_error

def update_xgboost_report(
    *,
    cfg: EnsembleRunConfig,
    model: Any,
    Xtr: Any,
    Xte: Any,
    ytr: Any,
    yte: Any,
    y_pred: Any,
    fold_id: int,
    xgb_acc: XGBoostEnsembleReportAccumulator | None,
) -> XGBoostEnsembleReportAccumulator | None:
    """Update XGBoost report accumulator for the current fold (best-effort, never raises)."""
    try:
        inner, _ = resolve_estimator_and_X(model, Xte)

        info = extract_xgboost_fold_info(xgb_model=inner)
        if xgb_acc is None:
            xgb_acc = XGBoostEnsembleReportAccumulator.create(
                metric_name=str(cfg.eval.metric),
                task=str(getattr(cfg.ensemble, "problem_kind", None) or "classification"),
                train_eval_metric=info.get("train_eval_metric", None),
                params=info.get("params", {}) or {},
            )

        xgb_acc.add_fold(
            best_iteration=info.get("best_iteration", None),
            best_score=info.get("best_score", None),
            evals_result=info.get("evals_result", None),
            feature_importances=info.get("feature_importances", None),
            feature_names=info.get("feature_names", None),
        )

    except Exception as e:
        if xgb_acc is not None:
            attach_report_error(xgb_acc, where="ensembles.reports.xgboost", exc=e, context={"fold_id": fold_id})
        return xgb_acc

    return xgb_acc
