from __future__ import annotations

from typing import Optional, List, Union, Literal
from pydantic import BaseModel


# ---------------- Confusion-matrix-based metrics -----------------


class PerClassConfusionMetrics(BaseModel):
    label: Union[int, float, str]
    tp: int
    fp: int
    tn: int
    fn: int
    support: int
    tpr: float
    fpr: float
    tnr: float
    fnr: float
    precision: float
    recall: float
    f1: float
    mcc: float


class ConfusionOverallMetrics(BaseModel):
    accuracy: float
    balanced_accuracy: float


class ConfusionAveragedMetrics(BaseModel):
    precision: float
    recall: float
    f1: float
    mcc: float


class ConfusionMatrix(BaseModel):
    labels: List[Union[int, float, str]]
    matrix: List[List[int]]

    # Rich metrics (classification only; optional to keep backward compatibility)
    per_class: Optional[List[PerClassConfusionMetrics]] = None
    overall: Optional[ConfusionOverallMetrics] = None
    macro_avg: Optional[ConfusionAveragedMetrics] = None
    weighted_avg: Optional[ConfusionAveragedMetrics] = None


# ---------------- ROC curves -------------------------------------


class RocCurve(BaseModel):
    """
    One ROC curve.

    For binary problems, `label` is usually the positive class and `curves`
    will have length 1. For multiclass, each entry corresponds to a
    one-vs-rest curve for that label.
    """
    label: Optional[Union[int, float, str]] = None
    fpr: List[float]
    tpr: List[float]
    thresholds: List[float]
    auc: float


class RocMetrics(BaseModel):
    kind: Literal["binary", "multiclass"]
    curves: List[RocCurve]

    # For multiclass, labels list the classes; for binary, it can be None
    labels: Optional[List[Union[int, float, str]]] = None
    positive_label: Optional[Union[int, float, str]] = None
    # For binary, macro_auc == curves[0].auc; for multiclass, macro_auc is
    # the macro-averaged AUC across classes.
    macro_auc: Optional[float] = None
    micro_auc: Optional[float] = None


# ---------------- Regression diagnostics ------------------------


class RegressionSummary(BaseModel):
    """Compact regression summary metrics computed on the evaluation set."""

    n: int

    rmse: Optional[float] = None
    mae: Optional[float] = None
    median_ae: Optional[float] = None
    r2: Optional[float] = None
    explained_variance: Optional[float] = None

    bias: Optional[float] = None
    residual_std: Optional[float] = None

    pearson_r: Optional[float] = None
    spearman_r: Optional[float] = None

    y_true_min: Optional[float] = None
    y_true_max: Optional[float] = None
    y_pred_min: Optional[float] = None
    y_pred_max: Optional[float] = None

    # Optional normalization (e.g., rmse/std(y_true) or rmse/(max-min))
    nrmse: Optional[float] = None


class RegressionXY(BaseModel):
    """Downsampled x/y vectors for scatter-like plots."""

    x: List[float]
    y: List[float]


class RegressionHistogram(BaseModel):
    edges: List[float]
    counts: List[float]


class RegressionBinnedError(BaseModel):
    edges: List[float]
    mae: List[float]
    rmse: List[float]
    n: List[float]


class RegressionDiagnostics(BaseModel):
    """Regression diagnostics payload used by Results UI."""

    summary: RegressionSummary
    pred_vs_true: Optional[RegressionXY] = None
    residuals_vs_pred: Optional[RegressionXY] = None
    residual_hist: Optional[RegressionHistogram] = None
    error_by_true_bin: Optional[RegressionBinnedError] = None

    # Endpoints for the ideal y=x line in the pred-vs-true plot.
    ideal_line: Optional[RegressionXY] = None
