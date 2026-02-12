from __future__ import annotations

"""Internal evaluation payload contracts.

These are *not* backend API schemas; they are internal, typed payload shapes used
inside the BL. The goal is to reduce pervasive `Dict[str, Any]` while keeping
flexibility where needed.
"""

from typing import Any, List, Optional, Sequence, Union

import numpy as np

try:
    from typing import TypedDict
except ImportError:  # pragma: no cover
    from typing_extensions import TypedDict  # type: ignore


class ConfusionPerClass(TypedDict):
    label: Any
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


class ConfusionGlobal(TypedDict):
    accuracy: float
    balanced_accuracy: float


class ConfusionAverages(TypedDict):
    precision: float
    recall: float
    f1: float
    mcc: float


ConfusionPayload = TypedDict(
    "ConfusionPayload",
    {
        "labels": np.ndarray,
        "matrix": np.ndarray,
        "per_class": List[ConfusionPerClass],
        "global": ConfusionGlobal,
        "macro_avg": ConfusionAverages,
        "weighted_avg": ConfusionAverages,
    },
)


class BinaryRocPayload(TypedDict):
    pos_label: Any
    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray
    auc: float


class MultiRocPerClass(TypedDict):
    label: Any
    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray
    auc: float


class MultiRocAverage(TypedDict):
    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: Optional[np.ndarray]
    auc: float


class MulticlassRocPayload(TypedDict):
    labels: np.ndarray
    per_class: List[MultiRocPerClass]
    macro_avg: MultiRocAverage
    micro_avg: MultiRocAverage


RocPayload = Union[BinaryRocPayload, MulticlassRocPayload]


class MetricsPayload(TypedDict):
    confusion: Optional[ConfusionPayload]
    roc: Optional[RocPayload]


# Regression diagnostics payloads are currently produced by reporting helpers.
# They remain intentionally flexible (plot-ready arrays, histograms, etc.).
RegressionPayload = dict
