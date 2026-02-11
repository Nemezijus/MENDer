from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class DecoderParts:
    scores_parts: List[Optional[np.ndarray]] = field(default_factory=list)
    proba_parts: List[Optional[np.ndarray]] = field(default_factory=list)
    margin_parts: List[Optional[np.ndarray]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    classes: Optional[np.ndarray] = None
    positive_index: Optional[int] = None
    fold_ids_parts: List[np.ndarray] = field(default_factory=list)


@dataclass
class FoldRunOutputs:
    fold_scores: List[float] = field(default_factory=list)

    y_true_parts: List[np.ndarray] = field(default_factory=list)
    y_pred_parts: List[np.ndarray] = field(default_factory=list)
    y_proba_parts: List[np.ndarray] = field(default_factory=list)
    y_score_parts: List[np.ndarray] = field(default_factory=list)

    test_indices_parts: List[Optional[np.ndarray]] = field(default_factory=list)
    eval_fold_ids_parts: List[np.ndarray] = field(default_factory=list)

    n_train_sizes: List[int] = field(default_factory=list)
    n_test_sizes: List[int] = field(default_factory=list)

    decoder: DecoderParts = field(default_factory=DecoderParts)

    last_pipeline: Optional[object] = None
