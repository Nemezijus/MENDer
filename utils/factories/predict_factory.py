# utils/factories/predict_factory.py
from __future__ import annotations
from utils.strategies.interfaces import Predictor
from utils.strategies.predictors import SklearnPredictor

def make_predictor() -> Predictor:
    """Create a prediction strategy. Today: sklearn-style predictor."""
    return SklearnPredictor()
