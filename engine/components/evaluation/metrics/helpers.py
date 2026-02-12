from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from engine.core.shapes import coerce_1d
from engine.components.evaluation.types import (
    ConfusionPayload,
    BinaryRocPayload,
    MulticlassRocPayload,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def _as_1d(a: np.ndarray) -> np.ndarray:
    """Back-compat wrapper; use engine.core.shapes.coerce_1d."""

    return coerce_1d(a)


def _check_len(y_true: np.ndarray, y_pred_like: np.ndarray, name: str):
    if y_true.shape[0] != y_pred_like.shape[0]:
        raise ValueError(
            f"Length mismatch: y_true({y_true.shape[0]}) vs {name}({y_pred_like.shape[0]})."
        )


def confusion_matrix_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[Sequence] = None,
) -> ConfusionPayload:
    """Compute a structured set of confusion-matrix-based metrics.

    Returns a dict with:
    - labels: np.ndarray of class labels (in the order used for the matrix)
    - matrix: np.ndarray of shape (n_classes, n_classes)
    - per_class: list of dicts with TP/FP/TN/FN and derived rates per class (incl. MCC)
    - global: dict with overall accuracy and balanced_accuracy
    - macro_avg: macro-averaged precision/recall/f1/mcc
    - weighted_avg: weighted-averaged precision/recall/f1/mcc
    """
    y_true = _as_1d(y_true)
    y_pred = _as_1d(y_pred)
    _check_len(y_true, y_pred, "y_pred")

    if labels is None:
        labels_arr = np.unique(np.concatenate([y_true, y_pred]))
    else:
        labels_arr = np.asarray(labels)

    cm = confusion_matrix(y_true, y_pred, labels=labels_arr)
    total = cm.sum()

    per_class = []
    per_class_mcc = []
    supports = []

    for idx, label in enumerate(labels_arr):
        tp = int(cm[idx, idx])
        fn = int(cm[idx, :].sum() - tp)
        fp = int(cm[:, idx].sum() - tp)
        tn = int(total - (tp + fp + fn))
        support = int(cm[idx, :].sum())
        supports.append(support)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # = TPR
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        # Per-class MCC via sklearn (one-vs-rest)
        y_true_bin = (y_true == label).astype(int)
        y_pred_bin = (y_pred == label).astype(int)
        mcc_val = matthews_corrcoef(y_true_bin, y_pred_bin)
        if not np.isfinite(mcc_val):
            mcc_val = 0.0
        mcc_val = float(mcc_val)
        per_class_mcc.append(mcc_val)

        per_class.append(
            {
                "label": label,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "support": support,
                "tpr": recall,
                "fpr": fpr,
                "tnr": tnr,
                "fnr": fnr,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "mcc": mcc_val,
            }
        )

    supports_arr = np.asarray(supports, dtype=float)
    per_class_mcc_arr = np.asarray(per_class_mcc, dtype=float)

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    prec_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    macro_mcc = float(per_class_mcc_arr.mean()) if per_class_mcc_arr.size > 0 else 0.0
    if supports_arr.sum() > 0:
        weighted_mcc = float(np.average(per_class_mcc_arr, weights=supports_arr))
    else:
        weighted_mcc = macro_mcc

    return {
        "labels": labels_arr,
        "matrix": cm,
        "per_class": per_class,
        "global": {
            "accuracy": float(acc),
            "balanced_accuracy": float(bal_acc),
        },
        "macro_avg": {
            "precision": float(prec_macro),
            "recall": float(rec_macro),
            "f1": float(f1_macro),
            "mcc": float(macro_mcc),
        },
        "weighted_avg": {
            "precision": float(prec_weighted),
            "recall": float(rec_weighted),
            "f1": float(f1_weighted),
            "mcc": float(weighted_mcc),
        },
    }


def binary_roc_curve_from_scores(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    pos_label: Optional[float] = None,
) -> BinaryRocPayload:
    """Compute a ROC curve for binary classification from a 1D score/proba array."""
    y_true = _as_1d(y_true)
    y_score = _as_1d(y_score)
    _check_len(y_true, y_score, "y_score")

    labels = np.unique(y_true)
    if pos_label is None:
        if labels.size != 2:
            raise ValueError(
                "binary_roc_curve_from_scores requires exactly two classes when "
                f"pos_label is not given; got {labels.size} classes."
            )
        pos_label = labels[1]

    fpr, tpr, thresholds = roc_curve(
        y_true,
        y_score,
        pos_label=pos_label,
        drop_intermediate=False,
    )
    auc_val = roc_auc_score(y_true, y_score)

    return {
        "pos_label": pos_label,
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "auc": float(auc_val),
    }


def multiclass_roc_curves_from_scores(
    y_true: np.ndarray,
    y_score: np.ndarray,
    labels: Optional[Sequence] = None,
) -> MulticlassRocPayload:
    """Compute one-vs-rest ROC curves for multiclass classification from a 2D score/proba array."""
    y_true = _as_1d(y_true)
    Y = np.asarray(y_score)
    if Y.ndim != 2:
        raise ValueError(
            f"multiclass_roc_curves_from_scores expects a 2D array of scores, got shape {Y.shape}."
        )
    _check_len(y_true, Y, "y_score")

    if labels is None:
        labels_arr = np.unique(y_true)
    else:
        labels_arr = np.asarray(labels)

    if Y.shape[1] != labels_arr.size:
        raise ValueError(
            "Mismatch between number of columns in y_score "
            f"({Y.shape[1]}) and number of labels ({labels_arr.size})."
        )

    per_class: list[dict] = []
    aucs: list[float] = []

    y_true_indicator = (y_true[:, None] == labels_arr[None, :]).astype(int)

    for idx, label in enumerate(labels_arr):
        y_true_bin = y_true_indicator[:, idx]
        scores_k = Y[:, idx]

        fpr_k, tpr_k, thresholds_k = roc_curve(
            y_true_bin,
            scores_k,
            pos_label=1,
            drop_intermediate=False,
        )
        auc_k = roc_auc_score(y_true_bin, scores_k)

        per_class.append(
            {
                "label": label,
                "fpr": fpr_k,
                "tpr": tpr_k,
                "thresholds": thresholds_k,
                "auc": float(auc_k),
            }
        )
        aucs.append(auc_k)

    if per_class:
        all_fpr = np.unique(np.concatenate([cls_curve["fpr"] for cls_curve in per_class]))
        mean_tpr = np.zeros_like(all_fpr)
        for cls_curve in per_class:
            mean_tpr += np.interp(all_fpr, cls_curve["fpr"], cls_curve["tpr"])
        mean_tpr /= len(per_class)

        macro_fpr = all_fpr
        macro_tpr = mean_tpr
        macro_auc = float(np.mean(aucs)) if aucs else float("nan")
    else:
        macro_fpr = np.array([0.0, 1.0])
        macro_tpr = np.array([0.0, 1.0])
        macro_auc = float("nan")

    y_true_flat = y_true_indicator.ravel()
    y_score_flat = Y.ravel()

    micro_fpr, micro_tpr, micro_thresholds = roc_curve(
        y_true_flat,
        y_score_flat,
        pos_label=1,
        drop_intermediate=False,
    )
    micro_auc = roc_auc_score(y_true_indicator, Y, average="micro")

    return {
        "labels": labels_arr,
        "per_class": per_class,
        "macro_avg": {
            "auc": macro_auc,
            "fpr": macro_fpr,
            "tpr": macro_tpr,
            "thresholds": None,
        },
        "micro_avg": {
            "fpr": micro_fpr,
            "tpr": micro_tpr,
            "thresholds": micro_thresholds,
            "auc": float(micro_auc),
        },
    }
