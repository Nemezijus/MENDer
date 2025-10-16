from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Tuple, Literal
import numpy as np

from utils.strategies.interfaces import Predictor
from utils.postprocessing.predicting import predict_labels, predict_scores

@dataclass
class SklearnPredictor(Predictor):
    """
    Thin adapter around your existing prediction helpers.
    No RNG; just calls predict_labels / predict_scores.
    """

    def predict(self, model: Any, X_test: np.ndarray) -> np.ndarray:
        return predict_labels(model, X_test)

    def predict_scores(
        self,
        model: Any,
        X_test: np.ndarray,
        *,
        kind: Literal["auto", "proba", "decision"] = "auto",
    ) -> Tuple[np.ndarray, str]:
        return predict_scores(model, X_test, kind=kind)
