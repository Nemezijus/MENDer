from __future__ import annotations
from engine.components.interfaces import Predictor
from engine.components.prediction.predictors import SklearnPredictor

def make_predictor() -> Predictor:
    """Create a prediction strategy. Today: sklearn-style predictor."""
    return SklearnPredictor()
