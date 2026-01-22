from typing import Any

from shared_schemas.ensemble_run_config import EnsembleRunConfig

from utils.postprocessing.ensembles.xgboost_ensemble_reporting import XGBoostEnsembleReportAccumulator

from ..helpers import _unwrap_final_estimator


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
        inner = _unwrap_final_estimator(model)

        # If pipeline-wrapped, prefer named_steps['clf'] (consistent with how you build pipelines)
        if hasattr(model, "named_steps") and isinstance(getattr(model, "named_steps"), dict):
            clf_step = model.named_steps.get("clf", None)
            if clf_step is not None:
                inner = clf_step

        # Unwrap XGB label adapter: XGBClassifierLabelAdapter(model=..., ...)
        inner = getattr(inner, "model", inner)

        if xgb_acc is None:
            params = {}
            try:
                params = dict(getattr(inner, "get_params", lambda: {})() or {})
            except Exception:
                params = {}

            # XGBoost training eval metric used for eval_set curves (often "mlogloss"/"logloss"/"rmse")
            train_eval_metric = None
            try:
                tem = params.get("eval_metric", None)
                if isinstance(tem, (list, tuple)) and len(tem) > 0:
                    train_eval_metric = str(tem[0])
                elif tem is not None:
                    train_eval_metric = str(tem)
            except Exception:
                train_eval_metric = None

            xgb_acc = XGBoostEnsembleReportAccumulator.create(
                metric_name=str(cfg.eval.metric),          # final metric (MENDer)
                train_eval_metric=train_eval_metric,       # training metric (XGB)
                params=params,
            )

        best_iteration = (
            getattr(inner, "best_iteration", None)
            or getattr(inner, "best_iteration_", None)
        )
        best_score = (
            getattr(inner, "best_score", None)
            or getattr(inner, "best_score_", None)
        )

        evals_result = getattr(inner, "evals_result_", None)
        if callable(evals_result):
            evals_result = None
        if not evals_result:
            best_iteration = None
            best_score = None
        feat_imps = getattr(inner, "feature_importances_", None)

        feature_names = None
        try:
            feature_names = None
        except Exception:
            feature_names = None

        xgb_acc.add_fold(
            best_iteration=best_iteration,
            best_score=best_score,
            evals_result=evals_result,
            feature_importances=feat_imps,
            feature_names=feature_names,
        )

    except Exception:
        return xgb_acc

    return xgb_acc
