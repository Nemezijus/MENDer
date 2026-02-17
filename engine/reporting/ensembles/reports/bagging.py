import numpy as np
from typing import Any, Tuple

from engine.contracts.ensemble_run_config import EnsembleRunConfig

from engine.reporting.ensembles.bagging import (
    BaggingEnsembleReportAccumulator,
    BaggingEnsembleRegressorReportAccumulator,
)

from ..helpers import _extract_base_estimator_algo_from_cfg, _get_classes_arr, _slice_X_by_features

from .extract import (
    collect_base_predictions_classification,
    collect_base_predictions_regression,
)

from ..common import attach_report_error

def update_bagging_report(
    *,
    cfg: EnsembleRunConfig,
    eval_kind: str,
    model: Any,
    Xte: Any,
    yte: Any,
    y_pred: Any,
    fold_id: int,
    evaluator: Any,
    bagging_cls_acc: BaggingEnsembleReportAccumulator | None,
    bagging_reg_acc: BaggingEnsembleRegressorReportAccumulator | None,
) -> Tuple[BaggingEnsembleReportAccumulator | None, BaggingEnsembleRegressorReportAccumulator | None]:
    """Update Bagging ensemble report accumulators for the current fold (best-effort, never raises)."""
    try:
        ests = getattr(model, "estimators_", None)
        feats_list = getattr(model, "estimators_features_", None)

        if ests is not None and len(ests) > 0:
            metric_name = str(cfg.eval.metric)
            base_algo = _extract_base_estimator_algo_from_cfg(cfg, default="default")

            # --- classification bagging report ---
            if eval_kind == "classification":
                if bagging_cls_acc is None:
                    bagging_cls_acc = BaggingEnsembleReportAccumulator.create(
                        metric_name=str(cfg.eval.metric),
                        base_algo=base_algo,
                        n_estimators=int(getattr(cfg.ensemble, "n_estimators", len(ests)) or len(ests)),
                        max_samples=getattr(cfg.ensemble, "max_samples", None),
                        max_features=getattr(cfg.ensemble, "max_features", None),
                        bootstrap=bool(getattr(cfg.ensemble, "bootstrap", True)),
                        bootstrap_features=bool(getattr(cfg.ensemble, "bootstrap_features", False)),
                        oob_score_enabled=bool(getattr(cfg.ensemble, "oob_score", False)),
                        balanced=bool(getattr(cfg.ensemble, "balanced", False)),
                        sampling_strategy=getattr(cfg.ensemble, "sampling_strategy", None),
                        replacement=getattr(cfg.ensemble, "replacement", None),
                    )

                classes_arr = _get_classes_arr(model)
                base_res = collect_base_predictions_classification(
                    estimators=ests,
                    X=Xte,
                    y_true=yte,
                    evaluator=evaluator,
                    metric_name=metric_name,
                    classes_arr=classes_arr,
                    feature_indices_list=feats_list,
                    slice_X=_slice_X_by_features,
                )

                if base_res.base_preds is not None and bagging_cls_acc is not None:

                    oob_score = getattr(model, "oob_score_", None)
                    oob_decision = getattr(model, "oob_decision_function_", None)

                    bagging_cls_acc.add_fold(
                        base_preds=base_res.base_preds,
                        oob_score=oob_score if oob_score is not None else None,
                        oob_decision_function=oob_decision,
                        base_estimator_scores=base_res.base_scores,
                    )

            # --- regression bagging report ---
            elif eval_kind == "regression":
                if bagging_reg_acc is None:
                    bagging_reg_acc = BaggingEnsembleRegressorReportAccumulator.create(
                        metric_name=str(cfg.eval.metric),
                        base_algo=base_algo,
                        n_estimators=int(getattr(cfg.ensemble, "n_estimators", len(ests)) or len(ests)),
                        max_samples=getattr(cfg.ensemble, "max_samples", None),
                        max_features=getattr(cfg.ensemble, "max_features", None),
                        bootstrap=bool(getattr(cfg.ensemble, "bootstrap", True)),
                        bootstrap_features=bool(getattr(cfg.ensemble, "bootstrap_features", False)),
                        oob_score_enabled=bool(getattr(cfg.ensemble, "oob_score", False)),
                    )

                yte_arr = np.asarray(yte, dtype=float)
                yens = np.asarray(y_pred, dtype=float)

                base_res = collect_base_predictions_regression(
                    estimators=ests,
                    X=Xte,
                    y_true=yte_arr,
                    evaluator=evaluator,
                    feature_indices_list=feats_list,
                    slice_X=_slice_X_by_features,
                )

                if base_res.base_preds is not None and bagging_reg_acc is not None:

                    oob_score = getattr(model, "oob_score_", None)
                    oob_pred = getattr(model, "oob_prediction_", None)

                    bagging_reg_acc.add_fold(
                        y_true=yte_arr,
                        ensemble_pred=yens,
                        base_preds=base_res.base_preds,
                        oob_score=oob_score if oob_score is not None else None,
                        oob_prediction=oob_pred,
                        base_estimator_scores=base_res.base_scores,
                    )
    except Exception as e:
        if bagging_cls_acc is not None:
            attach_report_error(bagging_cls_acc, where="ensembles.reports.bagging", exc=e, context={"fold_id": fold_id, "eval_kind": eval_kind})
        if bagging_reg_acc is not None:
            attach_report_error(bagging_reg_acc, where="ensembles.reports.bagging", exc=e, context={"fold_id": fold_id, "eval_kind": eval_kind})
        return bagging_cls_acc, bagging_reg_acc

    return bagging_cls_acc, bagging_reg_acc
