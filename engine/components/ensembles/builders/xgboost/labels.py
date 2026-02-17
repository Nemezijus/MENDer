from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np


def encode_labels(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Encode labels to 0..K-1 and return (classes, y_encoded)."""

    y_arr = np.asarray(y).ravel()
    classes, y_enc = np.unique(y_arr, return_inverse=True)
    return classes, y_enc.astype(int)


@dataclass
class XGBClassifierLabelAdapter:
    """Adapter that converts numeric XGBoost predictions back to original labels."""

    model: Any
    classes_: np.ndarray

    def predict(self, X: Any) -> Any:
        y_pred = self.model.predict(X)
        y_pred = np.asarray(y_pred).astype(int)
        return self.classes_[y_pred]

    def predict_proba(self, X: Any) -> Any:
        return self.model.predict_proba(X)

    def get_booster(self) -> Any:
        return self.model.get_booster()

    def set_params(self, **params: Any) -> Any:
        return self.model.set_params(**params)

    def get_params(self, deep: bool = True) -> Any:
        return self.model.get_params(deep=deep)
