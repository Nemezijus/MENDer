from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional
import numpy as np

from engine.components.interfaces import Trainer
from engine.components.trainers.fitting import fit_model  # your existing function

@dataclass
class SklearnTrainer(Trainer):
    """Thin adapter around your fit_model; no RNG or state."""
    def fit(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Any:
        return fit_model(model, X_train, y_train, sample_weight=sample_weight)
