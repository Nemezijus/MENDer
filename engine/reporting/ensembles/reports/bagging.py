import numpy as np
from typing import Any, Tuple

from engine.contracts.ensemble_run_config import EnsembleRunConfig

from engine.components.evaluation.scoring import PROBA_METRICS
from engine.reporting.ensembles.bagging import (
    BaggingEnsembleReportAccumulator,
    BaggingEnsembleRegressorReportAccumulator,
)

from ..helpers import (
    _slice_X_by_features,
    _get_classes_arr,
    _should_decode_from_index_space,
    _encode_y_true_to_index,
    _extract_base_estimator_algo_from_cfg,
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

                base_pred_cols = []
                base_scores_list: list[float] = []

                classes_arr = _get_classes_arr(model)
                yte_arr = np.asarray(yte)
                yte_enc = _encode_y_true_to_index(yte_arr, classes_arr)

                for i, est in enumerate(ests):
                    if est is None:
                        continue

                    feat_idx = None
                    try:
                        if feats_list is not None and i < len(feats_list):
                            feat_idx = feats_list[i]
                    except Exception:
                        feat_idx = None

                    Xte_i = _slice_X_by_features(Xte, feat_idx)

                    # IMPORTANT: if max_features < 1.0, each base estimator was trained on a subset.
                    # We must apply the SAME subset at predict/score time.
                    yp_raw = np.asarray(est.predict(Xte_i))

                    if _should_decode_from_index_space(yte_arr, yp_raw, classes_arr):
                        yp_dec = classes_arr[yp_raw.astype(int, copy=False)]
                    else:
                        yp_dec = yp_raw

                    base_pred_cols.append(np.asarray(yp_dec))

                    # score distribution (handle PROBA metrics deterministically in encoded space)
                    try:
                        y_proba_i = None
                        y_score_i = None

                        if metric_name in PROBA_METRICS:
                            if hasattr(est, "predict_proba"):
                                try:
                                    y_proba_i = est.predict_proba(Xte_i)
                                except Exception:
                                    y_proba_i = None
                            if y_proba_i is None and hasattr(est, "decision_function"):
                                try:
                                    y_score_i = est.decision_function(Xte_i)
                                except Exception:
                                    y_score_i = None

                            if y_proba_i is None and y_score_i is None:
                                s = None
                            elif yte_enc is not None and _should_decode_from_index_space(yte_arr, yp_raw, classes_arr):
                                s = evaluator.score(
                                    yte_enc,
                                    y_pred=yp_raw,
                                    y_proba=y_proba_i,
                                    y_score=y_score_i,
                                )
                            else:
                                s = evaluator.score(
                                    yte_arr,
                                    y_pred=yp_dec,
                                    y_proba=y_proba_i,
                                    y_score=y_score_i,
                                )
                        else:
                            s = evaluator.score(
                                yte_arr,
                                y_pred=yp_dec,
                                y_proba=None,
                                y_score=None,
                            )

                        if s is not None:
                            base_scores_list.append(float(s))
                    except Exception:
                        pass

                if base_pred_cols and bagging_cls_acc is not None:
                    base_preds_mat = np.column_stack(base_pred_cols)

                    oob_score = getattr(model, "oob_score_", None)
                    oob_decision = getattr(model, "oob_decision_function_", None)

                    bagging_cls_acc.add_fold(
                        base_preds=base_preds_mat,
                        oob_score=oob_score if oob_score is not None else None,
                        oob_decision_function=oob_decision,
                        base_estimator_scores=base_scores_list if base_scores_list else None,
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

                base_pred_cols = []
                base_scores_list: list[float] = []

                yte_arr = np.asarray(yte, dtype=float)
                yens = np.asarray(y_pred, dtype=float)

                for i, est in enumerate(ests):
                    if est is None:
                        continue

                    feat_idx = None
                    try:
                        if feats_list is not None and i < len(feats_list):
                            feat_idx = feats_list[i]
                    except Exception:
                        feat_idx = None

                    Xte_i = _slice_X_by_features(Xte, feat_idx)

                    yp = np.asarray(est.predict(Xte_i), dtype=float)
                    base_pred_cols.append(yp)

                    try:
                        s = evaluator.score(
                            yte_arr,
                            y_pred=yp,
                            y_proba=None,
                            y_score=None,
                        )
                        base_scores_list.append(float(s))
                    except Exception:
                        pass

                if base_pred_cols and bagging_reg_acc is not None:
                    base_preds_mat = np.column_stack(base_pred_cols)

                    oob_score = getattr(model, "oob_score_", None)
                    oob_pred = getattr(model, "oob_prediction_", None)

                    bagging_reg_acc.add_fold(
                        y_true=yte_arr,
                        ensemble_pred=yens,
                        base_preds=base_preds_mat,
                        oob_score=oob_score if oob_score is not None else None,
                        oob_prediction=oob_pred,
                        base_estimator_scores=base_scores_list if base_scores_list else None,
                    )
    except Exception:
        return bagging_cls_acc, bagging_reg_acc

    return bagging_cls_acc, bagging_reg_acc
