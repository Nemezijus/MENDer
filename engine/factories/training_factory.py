from __future__ import annotations
from engine.components.interfaces import Trainer
from engine.components.trainers.trainers import SklearnTrainer

def make_trainer() -> Trainer:
    """Create a training strategy. Today: sklearn-style trainer."""
    return SklearnTrainer()
