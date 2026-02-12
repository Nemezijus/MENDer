from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Tuple, Literal, Optional
import numpy as np

from engine.components.interfaces import Predictor
from engine.components.prediction import predict_labels, predict_scores, predict_decoder_outputs
from engine.contracts.results.decoder import DecoderOutputs

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

    def predict_decoder_outputs(
        self,
        model: Any,
        X_test: np.ndarray,
        *,
        positive_class_label: Optional[Any] = None,
        include_decision_scores: bool = True,
        include_probabilities: bool = True,
    ) -> DecoderOutputs:
        return predict_decoder_outputs(
            model,
            X_test,
            positive_class_label=positive_class_label,
            include_decision_scores=include_decision_scores,
            include_probabilities=include_probabilities,
            include_summary=False,
            max_preview_rows=200,
        )