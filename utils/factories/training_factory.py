from __future__ import annotations
from utils.strategies.interfaces import Trainer
from utils.strategies.trainers import SklearnTrainer

def make_trainer() -> Trainer:
    """Create a training strategy. Today: sklearn-style trainer."""
    return SklearnTrainer()
