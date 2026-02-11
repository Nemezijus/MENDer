import numpy as np
from typing import Any, Tuple

from engine.contracts.ensemble_run_config import EnsembleRunConfig

from engine.components.evaluation.scoring import PROBA_METRICS
from engine.reporting.ensembles.adaboost import (
    AdaBoostEnsembleReportAccumulator,
    AdaBoostEnsembleRegressorReportAccumulator,
)

from ..helpers import (
    _unwrap_final_estimator,
    _transform_through_pipeline,
    _get_classes_arr,
    _should_decode_from_index_space,
    _encode_y_true_to_index,
    _extract_base_estimator_algo_from_cfg,
)


def update_adaboost_report(
    *,
    cfg: EnsembleRunConfig,
    eval_kind: str,
    model: Any,
    Xte: Any,
    yte: Any,
    y_pred: Any,
    fold_id: int,
    evaluator: Any,
    adaboost_cls_acc: AdaBoostEnsembleReportAccumulator | None,
    adaboost_reg_acc: AdaBoostEnsembleRegressorReportAccumulator | None,
) -> Tuple[AdaBoostEnsembleReportAccumulator | None, AdaBoostEnsembleRegressorReportAccumulator | None]:
    """Update AdaBoost ensemble report accumulators for the current fold (best-effort, never raises)."""
    try:
        # AdaBoost often comes wrapped as a Pipeline(pre -> clf). In this repo, step names
        # may be ('pre','clf') or ('scale','feat','clf'). We handle both.
        boost = None
        Xte_boost = Xte

        # Prefer named_steps if present (most robust for your setup)
        if hasattr(model, "named_steps") and isinstance(getattr(model, "named_steps"), dict):
            # try "clf" for the boosting estimator
            clf_step = model.named_steps.get("clf", None)
            if clf_step is not None:
                boost = clf_step

            # try "pre" as the preprocessor (if you used that naming)
            pre_step = model.named_steps.get("pre", None)
            if pre_step is not None and hasattr(pre_step, "transform"):
                try:
                    Xte_boost = pre_step.transform(Xte)
                except Exception:
                    Xte_boost = Xte
            else:
                # otherwise transform through all non-final steps
                Xte_boost = _transform_through_pipeline(model, Xte)

        if boost is None:
            # Fall back: last estimator of pipeline OR model itself
            boost = _unwrap_final_estimator(model)
            if boost is not model:
                Xte_boost = _transform_through_pipeline(model, Xte)

        # If you ever wrap AdaBoost (similar to XGB label adapter), unwrap it
        boost = getattr(boost, "model", boost)

        ests = getattr(boost, "estimators_", None)
        w = getattr(boost, "estimator_weights_", None)
        errs = getattr(boost, "estimator_errors_", None)

        m = len(ests) if ests is not None else 0
        if w is not None and m > 0:
            w = np.asarray(w, dtype=float)[:m]
        if errs is not None and m > 0:
            errs = np.asarray(errs, dtype=float)[:m]

        if ests is not None and len(ests) > 0 and w is not None:
            base_algo = _extract_base_estimator_algo_from_cfg(cfg, default="default")
            metric_name = str(cfg.eval.metric)

            # --- classification adaboost report ---
            if eval_kind == "classification":
                if adaboost_cls_acc is None:
                    adaboost_cls_acc = AdaBoostEnsembleReportAccumulator.create(
                        metric_name=str(cfg.eval.metric),
                        base_algo=base_algo,
                        n_estimators=int(getattr(cfg.ensemble, "n_estimators", len(ests)) or len(ests)),
                        learning_rate=float(getattr(cfg.ensemble, "learning_rate", 1.0) or 1.0),
                        algorithm=str(getattr(cfg.ensemble, "algorithm", None) or None),
                    )

                classes_arr = _get_classes_arr(boost)
                if classes_arr is None:
                    classes_arr = _get_classes_arr(model)
                yte_arr = np.asarray(yte)
                yte_enc = _encode_y_true_to_index(yte_arr, classes_arr) if classes_arr is not None else None

                base_pred_cols = []
                base_scores_list: list[float] = []

                for est in ests:
                    if est is None:
                        continue

                    yp_raw = np.asarray(est.predict(Xte_boost))

                    # Deterministic decode protection (same as bagging)
                    if _should_decode_from_index_space(yte_arr, yp_raw, classes_arr):
                        yp_dec = classes_arr[yp_raw.astype(int, copy=False)]
                    else:
                        yp_dec = yp_raw

                    base_pred_cols.append(np.asarray(yp_dec))

                    # Optional per-stage score distribution (best-effort)
                    try:
                        y_proba_i = None
                        y_score_i = None

                        if metric_name in PROBA_METRICS:
                            if hasattr(est, "predict_proba"):
                                try:
                                    y_proba_i = est.predict_proba(Xte_boost)
                                except Exception:
                                    y_proba_i = None
                            if y_proba_i is None and hasattr(est, "decision_function"):
                                try:
                                    y_score_i = est.decision_function(Xte_boost)
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

                if base_pred_cols and adaboost_cls_acc is not None:
                    base_preds_mat = np.column_stack(base_pred_cols)

                    # ---- stage/weight diagnostics (configured vs fitted vs effective) ----
                    w_arr = np.asarray(w, dtype=float)
                    weight_eps = 1e-6

                    n_estimators_fitted = int(len(ests)) if ests is not None else int(base_preds_mat.shape[1])
                    n_nonzero_weights = int(np.sum(w_arr > 0))
                    n_nontrivial_weights = int(np.sum(w_arr > weight_eps))

                    weight_mass_topk = None
                    try:
                        ssum = float(np.sum(w_arr))
                        if ssum > 0:
                            w_sorted = np.sort(w_arr)[::-1]
                            c = np.cumsum(w_sorted) / ssum
                            topks = [5, 10, 20]
                            weight_mass_topk = {k: float(c[min(k, len(c)) - 1]) for k in topks if len(c) > 0}
                    except Exception:
                        weight_mass_topk = None

                    adaboost_cls_acc.add_fold(
                        base_preds=base_preds_mat,
                        estimator_weights=np.asarray(w, dtype=float)[: base_preds_mat.shape[1]],
                        estimator_errors=np.asarray(errs, dtype=float)[: base_preds_mat.shape[1]] if errs is not None else None,
                        base_estimator_scores=base_scores_list if base_scores_list else None,

                        # diagnostics
                        n_estimators_fitted=n_estimators_fitted,
                        n_nonzero_weights=n_nonzero_weights,
                        n_nontrivial_weights=n_nontrivial_weights,
                        weight_eps=weight_eps,
                        weight_mass_topk=weight_mass_topk,
                    )

            # --- regression adaboost report ---
            elif eval_kind == "regression":
                if adaboost_reg_acc is None:
                    adaboost_reg_acc = AdaBoostEnsembleRegressorReportAccumulator.create(
                        metric_name=str(cfg.eval.metric),
                        base_algo=base_algo,
                        n_estimators=int(getattr(cfg.ensemble, "n_estimators", len(ests)) or len(ests)),
                        learning_rate=float(getattr(cfg.ensemble, "learning_rate", 1.0) or 1.0),
                        loss=None,
                    )

                yte_arr = np.asarray(yte, dtype=float)
                yens = np.asarray(y_pred, dtype=float)

                base_pred_cols = []
                base_scores_list: list[float] = []

                for est in ests:
                    if est is None:
                        continue

                    yp = np.asarray(est.predict(Xte_boost), dtype=float)
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

                if base_pred_cols and adaboost_reg_acc is not None:
                    base_preds_mat = np.column_stack(base_pred_cols)

                    w_arr = np.asarray(w, dtype=float)
                    weight_eps = 1e-6

                    n_estimators_fitted = int(len(ests)) if ests is not None else int(base_preds_mat.shape[1])
                    n_nonzero_weights = int(np.sum(w_arr > 0))
                    n_nontrivial_weights = int(np.sum(w_arr > weight_eps))

                    weight_mass_topk = None
                    try:
                        ssum = float(np.sum(w_arr))
                        if ssum > 0:
                            w_sorted = np.sort(w_arr)[::-1]
                            c = np.cumsum(w_sorted) / ssum
                            topks = [5, 10, 20]
                            weight_mass_topk = {k: float(c[min(k, len(c)) - 1]) for k in topks if len(c) > 0}
                    except Exception:
                        weight_mass_topk = None

                    adaboost_reg_acc.add_fold(
                        y_true=yte_arr,
                        ensemble_pred=yens,
                        base_preds=base_preds_mat,
                        estimator_weights=np.asarray(w, dtype=float)[: base_preds_mat.shape[1]],
                        estimator_errors=np.asarray(errs, dtype=float)[: base_preds_mat.shape[1]] if errs is not None else None,
                        base_estimator_scores=base_scores_list if base_scores_list else None,
                        n_estimators_fitted=n_estimators_fitted,
                        n_nonzero_weights=n_nonzero_weights,
                        n_nontrivial_weights=n_nontrivial_weights,
                        weight_eps=weight_eps,
                        weight_mass_topk=weight_mass_topk,
                    )

    except Exception:
        return adaboost_cls_acc, adaboost_reg_acc

    return adaboost_cls_acc, adaboost_reg_acc
