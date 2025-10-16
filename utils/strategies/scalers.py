from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np

from utils.configs.configs import ScaleConfig
from utils.strategies.interfaces import Scaler
from utils.preprocessing.general.feature_scaling import scale_train_test

@dataclass
class PairScaler(Scaler):
    """
    Wraps the existing scale_train_test(X_train, X_test, method=...).
    No RNG, no side effects. Just fits on train, transforms both.
    """
    cfg: ScaleConfig

    def fit_transform(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return scale_train_test(
            X_train, X_test,
            method=self.cfg.method,
            # If you later expose more options (e.g., quantile output distribution),
            # add them to ScaleConfig and pass through here.
        )
