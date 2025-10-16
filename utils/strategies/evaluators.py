# utils/strategies/evaluators.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence
import numpy as np

from utils.configs.configs import EvalConfig
from utils.strategies.interfaces import Evaluator
from utils.postprocessing.scoring import score as score_fn, classification_report_quick

@dataclass
class SklearnEvaluator(Evaluator):
    """
    Wraps your existing scoring functions.
    - Uses EvalConfig.metric by default.
    - Supports both hard-label and probabilistic metrics via optional args.
    """
    cfg: EvalConfig
    kind: str = "classification"   # or "regression" (you can override per use-case)

    def score(
        self,
        y_true: np.ndarray,
        y_pred: Optional[np.ndarray] = None,
        *,
        y_proba: Optional[np.ndarray] = None,
        y_score: Optional[np.ndarray] = None,
        labels: Optional[Sequence] = None,
    ) -> float:
        return score_fn(
            y_true,
            y_pred,
            kind=self.kind,                 # explicit kind to avoid heuristic surprises
            metric=self.cfg.metric,         # default metric from config
            y_proba=y_proba,
            y_score=y_score,
            labels=labels,
        )

    def quick_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict:
        # classification convenience; safe to call only when kind == "classification"
        return classification_report_quick(y_true, y_pred)
