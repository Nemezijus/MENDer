from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from engine.reporting.training.metrics_payloads import normalize_confusion, normalize_roc
from engine.reporting.training.regression_payloads import build_regression_diagnostics_payload


@dataclass
class PooledEvalOutputs:
    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: Optional[np.ndarray]
    y_score: Optional[np.ndarray]
    row_indices: np.ndarray
    order: Optional[np.ndarray]
    fold_ids: Optional[np.ndarray]


def pool_eval_outputs(
    *,
    y_true_parts: list[np.ndarray],
    y_pred_parts: list[np.ndarray],
    y_proba_parts: list[np.ndarray],
    y_score_parts: list[np.ndarray],
    test_indices_parts: list[Optional[np.ndarray]],
    eval_fold_ids_parts: list[np.ndarray],
) -> PooledEvalOutputs:
    """Concatenate fold-wise arrays and reorder to original index order if possible."""

    y_true_all = np.concatenate(y_true_parts) if y_true_parts else np.array([])
    y_pred_all = np.concatenate(y_pred_parts) if y_pred_parts else np.array([])
    fold_ids_all = np.concatenate(eval_fold_ids_parts) if eval_fold_ids_parts else None

    y_proba_all = np.concatenate(y_proba_parts, axis=0) if y_proba_parts else None
    y_score_all = np.concatenate(y_score_parts, axis=0) if y_score_parts else None

    order = None
    row_indices = None

    if test_indices_parts and all(p is not None for p in test_indices_parts):
        idx_all = np.concatenate([np.asarray(p) for p in test_indices_parts], axis=0)
        if idx_all.shape[0] == y_pred_all.shape[0]:
            order = np.argsort(idx_all, kind="stable")
            row_indices = idx_all[order]
            y_true_all = y_true_all[order]
            y_pred_all = y_pred_all[order]
            if fold_ids_all is not None and fold_ids_all.shape[0] == order.shape[0]:
                fold_ids_all = fold_ids_all[order]
            if y_proba_all is not None and y_proba_all.shape[0] == order.shape[0]:
                y_proba_all = y_proba_all[order]
            if y_score_all is not None and y_score_all.shape[0] == order.shape[0]:
                y_score_all = y_score_all[order]

    if row_indices is None:
        row_indices = np.arange(int(np.asarray(y_pred_all).shape[0]), dtype=int)

    return PooledEvalOutputs(
        y_true=y_true_all,
        y_pred=y_pred_all,
        y_proba=y_proba_all,
        y_score=y_score_all,
        row_indices=row_indices,
        order=order,
        fold_ids=fold_ids_all,
    )


def compute_eval_payloads(
    *,
    eval_kind: str,
    pooled: PooledEvalOutputs,
    metrics_computer,
    cfg,
) -> tuple[dict[str, Any], Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    """Compute normalized confusion/roc and regression diagnostics payloads."""

    y_true = pooled.y_true
    y_pred = pooled.y_pred

    confusion_payload_raw = None
    roc_raw = None

    if eval_kind == "classification" and y_true.size and y_pred.size:
        metrics_result = metrics_computer.compute(
            y_true,
            y_pred,
            y_proba=pooled.y_proba,
            y_score=pooled.y_score,
        )
        confusion_payload_raw = metrics_result.get("confusion")
        roc_raw = metrics_result.get("roc")

    labels_out, cm_mat, per_class, overall, macro_avg, weighted_avg = normalize_confusion(
        confusion_payload_raw
    )
    roc_payload = normalize_roc(roc_raw)

    confusion_out = {
        "labels": labels_out,
        "matrix": cm_mat,
        "per_class": per_class,
        "overall": overall,
        "macro_avg": macro_avg,
        "weighted_avg": weighted_avg,
    }

    regression_payload = None
    if eval_kind == "regression" and y_true.size and y_pred.size:
        try:
            regression_payload = build_regression_diagnostics_payload(
                y_true=y_true,
                y_pred=y_pred,
                seed=int(getattr(cfg.eval, "seed", 0) or 0),
            )
        except Exception:
            regression_payload = None

    return confusion_out, roc_payload, regression_payload
