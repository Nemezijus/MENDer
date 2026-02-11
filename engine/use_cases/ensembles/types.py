from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np


@dataclass
class DecoderState:
    enabled: bool
    max_preview_rows: int
    positive_label: Any = None
    include_scores: bool = True
    include_probabilities: bool = True
    calibrate_probabilities: bool = False
    include_margin: bool = False

    classes: Optional[np.ndarray] = None
    positive_index: Optional[int] = None
    proba_source: Optional[str] = None

    scores_all: List[np.ndarray] = field(default_factory=list)
    proba_all: List[np.ndarray] = field(default_factory=list)
    margin_all: List[np.ndarray] = field(default_factory=list)
    fold_ids_parts: List[np.ndarray] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


@dataclass
class ReportState:
    is_voting: bool = False
    is_bagging: bool = False
    is_adaboost: bool = False
    is_xgboost: bool = False

    voting_cls_acc: Any = None
    voting_reg_acc: Any = None
    bagging_cls_acc: Any = None
    bagging_reg_acc: Any = None
    adaboost_cls_acc: Any = None
    adaboost_reg_acc: Any = None
    xgb_acc: Any = None


@dataclass
class FoldState:
    fold_scores: List[float] = field(default_factory=list)
    y_true_parts: List[np.ndarray] = field(default_factory=list)
    y_pred_parts: List[np.ndarray] = field(default_factory=list)
    y_proba_parts: List[np.ndarray] = field(default_factory=list)
    y_score_parts: List[np.ndarray] = field(default_factory=list)

    test_indices_parts: List[Optional[np.ndarray]] = field(default_factory=list)
    eval_fold_ids_parts: List[np.ndarray] = field(default_factory=list)

    n_train_sizes: List[int] = field(default_factory=list)
    n_test_sizes: List[int] = field(default_factory=list)
