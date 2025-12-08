from __future__ import annotations

from typing import Literal, Optional, Sequence
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    log_loss,
    roc_auc_score,
    roc_curve,
    average_precision_score,
    confusion_matrix,
    r2_score,
    explained_variance_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    matthews_corrcoef,
)


# ------------------------- Helpers -------------------------

def _as_1d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    return a.ravel()


def _check_len(y_true: np.ndarray, y_pred_like: np.ndarray, name: str):
    if y_true.shape[0] != y_pred_like.shape[0]:
        raise ValueError(
            f"Length mismatch: y_true({y_true.shape[0]}) vs {name}({y_pred_like.shape[0]})."
        )


def _infer_kind(y_true: np.ndarray) -> Literal["classification", "regression"]:
    """
    Heuristic:
      - If y_true has <= 20 unique values and looks discrete → classification
      - Else → regression
    Prefer passing `kind` explicitly in critical code.
    """
    y_true = _as_1d(y_true)
    uniq = np.unique(y_true)
    if y_true.dtype.kind in "iu" or (len(uniq) <= 20):
        return "classification"
    return "regression"


PROBA_METRICS = {"log_loss", "roc_auc_ovr", "roc_auc_ovo", "avg_precision_macro"}


def make_estimator_scorer(kind: str, metric: str):
    """
    Return a callable(estimator, X, y) suitable for sklearn's learning_curve / cross_val_score.
    It uses YOUR unified metric names and YOUR score(...) implementation.
    """

    def _scorer(estimator, X, y):
        y_true = _as_1d(y)

        if kind == "classification":
            if metric in PROBA_METRICS:
                # Metrics that need probabilities or decision scores
                if hasattr(estimator, "predict_proba"):
                    y_proba = estimator.predict_proba(X)
                    return score(
                        y_true,
                        kind="classification",
                        metric=metric,
                        y_proba=y_proba,
                    )
                elif hasattr(estimator, "decision_function"):
                    y_score = estimator.decision_function(X)
                    return score(
                        y_true,
                        kind="classification",
                        metric=metric,
                        y_score=y_score,
                    )
                else:
                    raise ValueError(
                        f"Metric '{metric}' requires predict_proba or decision_function "
                        f"but estimator {type(estimator).__name__} has neither."
                    )
            else:
                # Hard-label metrics
                y_pred = estimator.predict(X)
                return score(
                    y_true,
                    y_pred,
                    kind="classification",
                    metric=metric,
                )

        elif kind == "regression":
            y_pred = estimator.predict(X)
            return score(
                y_true,
                y_pred,
                kind="regression",
                metric=metric,
            )

        else:
            raise ValueError(f"Unsupported kind in make_estimator_scorer: {kind!r}")

    return _scorer


# ------------------------- Classification -------------------------

_CLASS_METRICS = {
    # hard-label metrics (need y_pred)
    "accuracy": lambda y, yhat, **kw: accuracy_score(y, yhat),
    "balanced_accuracy": lambda y, yhat, **kw: balanced_accuracy_score(y, yhat),
    "f1_macro": lambda y, yhat, **kw: f1_score(y, yhat, average="macro"),
    "f1_micro": lambda y, yhat, **kw: f1_score(y, yhat, average="micro"),
    "f1_weighted": lambda y, yhat, **kw: f1_score(y, yhat, average="weighted"),
    "precision_macro": lambda y, yhat, **kw: precision_score(
        y, yhat, average="macro", zero_division=0
    ),
    "recall_macro": lambda y, yhat, **kw: recall_score(
        y, yhat, average="macro", zero_division=0
    ),
    # probability / score-based metrics (need y_proba or y_score)
    "log_loss": "log_loss",                # needs y_proba
    "roc_auc_ovr": "roc_auc_ovr",          # needs y_score or y_proba
    "roc_auc_ovo": "roc_auc_ovo",          # needs y_score or y_proba
    "avg_precision_macro": "avg_precision_macro",  # needs y_score or y_proba (one-vs-rest)
}


def _classification_score(
    y_true: np.ndarray,
    *,
    metric: str = "accuracy",
    y_pred: Optional[np.ndarray] = None,
    y_proba: Optional[np.ndarray] = None,
    y_score: Optional[np.ndarray] = None,
    labels: Optional[Sequence] = None,
) -> float:
    """
    Compute a classification metric. Use y_pred for hard-label metrics.
    For probabilistic metrics, pass y_proba (preferred) or y_score.
    """
    y_true = _as_1d(y_true)

    if metric not in _CLASS_METRICS:
        raise ValueError(
            f"Unknown classification metric '{metric}'. Supported: {list(_CLASS_METRICS)}"
        )

    fn = _CLASS_METRICS[metric]

    # Hard-label metrics
    if callable(fn):
        if y_pred is None:
            raise ValueError(f"Metric '{metric}' requires y_pred (hard labels).")
        y_pred = _as_1d(y_pred)
        _check_len(y_true, y_pred, "y_pred")
        return float(fn(y_true, y_pred))

    # Probabilistic / score-based metrics
    # Prefer probabilities if available; else decision scores.
    Z = None
    if y_proba is not None:
        Z = np.asarray(y_proba)
        _check_len(y_true, Z, "y_proba")
    elif y_score is not None:
        Z = np.asarray(y_score)
        _check_len(y_true, Z, "y_score")
    else:
        raise ValueError(f"Metric '{metric}' requires y_proba or y_score.")

    # Align labels for multi-class cases
    # For binary with probabilities, expect shape (n_samples, 2) or (n_samples,) for positive class.
    # We convert to a consistent form as needed below.
    if metric == "log_loss":
        # log_loss expects class probabilities (n_samples, n_classes)
        return float(log_loss(y_true, Z, labels=labels))
    elif metric in ("roc_auc_ovr", "roc_auc_ovo"):
        multi_class = "ovr" if metric.endswith("ovr") else "ovo"
        return float(
            roc_auc_score(
                y_true,
                Z,
                multi_class=multi_class,
                labels=labels,
                average="macro",
            )
        )
    elif metric == "avg_precision_macro":
        # One-vs-rest average precision (macro)
        return float(average_precision_score(y_true, Z, average="macro"))

    raise RuntimeError("Unreachable branch in _classification_score.")


# ------------------------- Regression -------------------------

_REG_METRICS = {
    "r2": lambda y, yhat, **kw: r2_score(y, yhat),
    "explained_variance": lambda y, yhat, **kw: explained_variance_score(y, yhat),
    "mse": lambda y, yhat, **kw: mean_squared_error(y, yhat),
    "rmse": lambda y, yhat, **kw: np.sqrt(mean_squared_error(y, yhat)),
    "mae": lambda y, yhat, **kw: mean_absolute_error(y, yhat),
    "mape": lambda y, yhat, **kw: mean_absolute_percentage_error(y, yhat),
}


def _regression_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    metric: str = "r2",
) -> float:
    y_true = _as_1d(y_true)
    y_pred = _as_1d(y_pred)
    _check_len(y_true, y_pred, "y_pred")

    if metric not in _REG_METRICS:
        raise ValueError(
            f"Unknown regression metric '{metric}'. Supported: {list(_REG_METRICS)}"
        )
    return float(_REG_METRICS[metric](y_true, y_pred))


# ------------------------- Public API -------------------------

def score(
    y_true: np.ndarray,
    y_pred: Optional[np.ndarray] = None,
    *,
    kind: Literal["auto", "classification", "regression"] = "auto",
    metric: str = "accuracy",
    y_proba: Optional[np.ndarray] = None,
    y_score: Optional[np.ndarray] = None,
    labels: Optional[Sequence] = None,
) -> float:
    """
    Universal scorer for classification and regression.
    """
    y_true = _as_1d(y_true)
    if kind == "auto":
        kind = _infer_kind(y_true)

    if kind == "classification":
        return _classification_score(
            y_true,
            metric=metric,
            y_pred=y_pred,
            y_proba=y_proba,
            y_score=y_score,
            labels=labels,
        )
    elif kind == "regression":
        if y_pred is None:
            raise ValueError("Regression scoring requires y_pred.")
        return _regression_score(y_true, y_pred, metric=metric)
    else:
        raise ValueError("kind must be one of {'auto','classification','regression'}.")


def confusion_matrix_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[Sequence] = None,
) -> dict:
    """
    Compute a structured set of confusion-matrix-based metrics.

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
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0      # specificity
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

    # Aggregate metrics using sklearn (for consistency)
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    prec_macro = precision_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    rec_macro = recall_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    f1_macro = f1_score(
        y_true, y_pred, average="macro", zero_division=0
    )

    prec_weighted = precision_score(
        y_true, y_pred, average="weighted", zero_division=0
    )
    rec_weighted = recall_score(
        y_true, y_pred, average="weighted", zero_division=0
    )
    f1_weighted = f1_score(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # Macro / weighted MCC from per-class MCC
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
) -> dict:
    """
    Compute a ROC curve for binary classification from a 1D score/proba array.

    Returns dict with pos_label, fpr, tpr, thresholds, auc.
    """
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

    # Keep all points for smoother-looking curves
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
) -> dict:
    """
    Compute one-vs-rest ROC curves for multiclass classification from a 2D score/proba array.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        True labels.
    y_score : array-like, shape (n_samples, n_classes)
        Continuous scores or probabilities for each class.
    labels : sequence, optional
        Label ordering. If None, the sorted unique labels from y_true are used.

    Returns
    -------
    dict with:
        - labels: np.ndarray of class labels (order matches columns in y_score)
        - per_class: list of dicts {label, fpr, tpr, thresholds, auc}
        - macro_avg: dict with:
            - auc: macro-averaged AUC across classes  (unchanged, for compatibility)
            - fpr, tpr, thresholds (ROC curve for macro-average)
        - micro_avg: dict with:
            - fpr, tpr, thresholds, auc for micro-averaged ROC
    """
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

    # For micro-average, build a one-vs-rest indicator matrix: (n_samples, n_classes)
    y_true_indicator = (y_true[:, None] == labels_arr[None, :]).astype(int)

    # ---- Per-class one-vs-rest ROC curves ----
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

    # ---- Macro-average ROC curve ----
    if per_class:
        # Pool all unique FPRs from all per-class curves
        all_fpr = np.unique(
            np.concatenate([cls_curve["fpr"] for cls_curve in per_class])
        )

        # Interpolate TPRs at these FPR points, then average across classes
        mean_tpr = np.zeros_like(all_fpr)
        for cls_curve in per_class:
            mean_tpr += np.interp(all_fpr, cls_curve["fpr"], cls_curve["tpr"])
        mean_tpr /= len(per_class)

        macro_fpr = all_fpr
        macro_tpr = mean_tpr
        # Keep original semantic of macro_auc as mean of per-class AUCs
        macro_auc = float(np.mean(aucs)) if aucs else float("nan")
    else:
        macro_fpr = np.array([0.0, 1.0])
        macro_tpr = np.array([0.0, 1.0])
        macro_auc = float("nan")

    # ---- Micro-average ROC curve ----
    # Flatten one-vs-rest indicator and scores
    y_true_flat = y_true_indicator.ravel()
    y_score_flat = Y.ravel()

    micro_fpr, micro_tpr, micro_thresholds = roc_curve(
        y_true_flat,
        y_score_flat,
        pos_label=1,
        drop_intermediate=False,
    )
    # For indicator format, average="micro" equals ROC AUC on flattened arrays
    micro_auc = roc_auc_score(y_true_indicator, Y, average="micro")

    return {
        "labels": labels_arr,
        "per_class": per_class,
        "macro_avg": {
            "auc": macro_auc,          # existing field, unchanged semantics
            "fpr": macro_fpr,
            "tpr": macro_tpr,
            "thresholds": None,        # not well-defined for macro-averaged curve
        },
        "micro_avg": {
            "fpr": micro_fpr,
            "tpr": micro_tpr,
            "thresholds": micro_thresholds,
            "auc": float(micro_auc),
        },
    }
