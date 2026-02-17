import numpy as np
from typing import Any, Tuple

from engine.contracts.ensemble_run_config import EnsembleRunConfig

from engine.reporting.ensembles.adaboost import (
    AdaBoostEnsembleReportAccumulator,
    AdaBoostEnsembleRegressorReportAccumulator,
)

from ..helpers import _extract_base_estimator_algo_from_cfg, _get_classes_arr

from .extract import (
    collect_base_predictions_classification,
    collect_base_predictions_regression,
    resolve_estimator_and_X,
)

from ..common import attach_report_error


def _coerce_base_preds_2d(
    base_preds: Any,
    *,
    n_samples: int,
    n_classes: int | None = None,
) -> np.ndarray | None:
    """Coerce base-estimator predictions into a 2D (n_samples, n_estimators) array.

    The AdaBoost accumulator expects per-estimator *label* predictions for each sample.
    During refactors, helper functions may return probabilities with shape like:

        (n_samples, n_estimators, n_classes)
        (n_estimators, n_samples, n_classes)

    This helper normalizes those to a 2D array of predicted labels.
    """
    if base_preds is None:
        return None

    P = np.asarray(base_preds)
    if P.size == 0:
        return None

    # If probabilities were provided, reduce to labels.
    if P.ndim == 3:
        class_axis = None
        if n_classes is not None and n_classes in P.shape:
            for ax in (2, 1, 0):
                if P.shape[ax] == n_classes:
                    class_axis = ax
                    break
        if class_axis is None:
            # Fall back to smallest-dimension heuristic (classes are usually small).
            class_axis = int(np.argmin(P.shape))
        P = np.argmax(P, axis=class_axis)

    # Handle degenerate cases.
    if P.ndim == 1:
        if int(P.shape[0]) == int(n_samples):
            P = P.reshape(n_samples, 1)
        else:
            P = P.reshape(1, -1)

    if P.ndim != 2:
        return None

    # Align to (n_samples, n_estimators)
    if int(P.shape[0]) == int(n_samples):
        return P
    if int(P.shape[1]) == int(n_samples):
        return P.T

    # Unknown orientation; best-effort return.
    return P


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
        boost, Xte_boost = resolve_estimator_and_X(model, Xte)

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

                classes_arr = _get_classes_arr(boost) or _get_classes_arr(model)

                base_res = collect_base_predictions_classification(
                    estimators=ests,
                    X=Xte_boost,
                    y_true=yte,
                    evaluator=evaluator,
                    metric_name=metric_name,
                    classes_arr=classes_arr,
                )

                base_preds_2d = _coerce_base_preds_2d(
                    base_res.base_preds,
                    n_samples=int(np.asarray(yte).shape[0]),
                    n_classes=int(len(classes_arr)) if classes_arr is not None else None,
                )

                if base_preds_2d is not None and adaboost_cls_acc is not None:

                    # ---- stage/weight diagnostics (configured vs fitted vs effective) ----
                    w_arr = np.asarray(w, dtype=float)
                    weight_eps = 1e-6

                    n_estimators_fitted = int(len(ests)) if ests is not None else int(base_preds_2d.shape[1])
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
                        base_preds=base_preds_2d,
                        estimator_weights=np.asarray(w, dtype=float)[: base_preds_2d.shape[1]],
                        estimator_errors=np.asarray(errs, dtype=float)[: base_preds_2d.shape[1]] if errs is not None else None,
                        base_estimator_scores=base_res.base_scores,

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

                base_res = collect_base_predictions_regression(
                    estimators=ests,
                    X=Xte_boost,
                    y_true=yte_arr,
                    evaluator=evaluator,
                )

                base_preds_2d = _coerce_base_preds_2d(
                    base_res.base_preds,
                    n_samples=int(yte_arr.shape[0]),
                    n_classes=None,
                )

                if base_preds_2d is not None and adaboost_reg_acc is not None:

                    w_arr = np.asarray(w, dtype=float)
                    weight_eps = 1e-6

                    n_estimators_fitted = int(len(ests)) if ests is not None else int(base_preds_2d.shape[1])
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
                        base_preds=base_preds_2d,
                        estimator_weights=np.asarray(w, dtype=float)[: base_preds_2d.shape[1]],
                        estimator_errors=np.asarray(errs, dtype=float)[: base_preds_2d.shape[1]] if errs is not None else None,
                        base_estimator_scores=base_res.base_scores,
                        n_estimators_fitted=n_estimators_fitted,
                        n_nonzero_weights=n_nonzero_weights,
                        n_nontrivial_weights=n_nontrivial_weights,
                        weight_eps=weight_eps,
                        weight_mass_topk=weight_mass_topk,
                    )

    except Exception as e:
        if adaboost_cls_acc is not None:
            attach_report_error(adaboost_cls_acc, where="ensembles.reports.adaboost", exc=e, context={"fold_id": fold_id, "eval_kind": eval_kind})
        if adaboost_reg_acc is not None:
            attach_report_error(adaboost_reg_acc, where="ensembles.reports.adaboost", exc=e, context={"fold_id": fold_id, "eval_kind": eval_kind})
        return adaboost_cls_acc, adaboost_reg_acc

    return adaboost_cls_acc, adaboost_reg_acc
