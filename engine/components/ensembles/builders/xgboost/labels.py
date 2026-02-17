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

    def fit(self, X: Any, y: Any, **fit_params: Any) -> "XGBClassifierLabelAdapter":
        """Fit the wrapped classifier while encoding labels to 0..K-1.

        XGBoost's sklearn-wrapper expects integer class indices for multiclass.
        We encode the provided labels, fit the underlying model, and store the
        original label values in ``classes_``.
        """

        classes, y_enc = encode_labels(np.asarray(y))
        self.classes_ = classes
        self.model.fit(X, y_enc, **fit_params)
        return self

    def predict(self, X: Any) -> Any:
        y_pred = self.model.predict(X)
        # XGBoost may return floats for class indices; coerce to int indices.
        idx = np.asarray(y_pred).astype(int, copy=False).ravel()
        if idx.size:
            idx = np.clip(idx, 0, int(self.classes_.shape[0]) - 1)
        return self.classes_[idx]

    def predict_proba(self, X: Any) -> Any:
        return self.model.predict_proba(X)

    def get_booster(self) -> Any:
        return self.model.get_booster()

    def set_params(self, **params: Any) -> Any:
        return self.model.set_params(**params)

    def get_params(self, deep: bool = True) -> Any:
        return self.model.get_params(deep=deep)

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the wrapped xgboost model."""

        return getattr(self.model, name)
