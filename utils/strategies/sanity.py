from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class BasicClassificationSanity:
    warn_on_few_per_class: bool = True
    min_per_class_factor: float = 2.0  # warn if n_samples < factor * n_classes

    def check(self, X: np.ndarray, y: np.ndarray) -> None:
        y = np.asarray(y).ravel()
        classes = np.unique(y)
        if classes.size < 2:
            raise ValueError("y must contain at least two classes.")
        if self.warn_on_few_per_class and X.shape[0] < self.min_per_class_factor * classes.size:
            print("[WARN] Few trials per class; results may be unstable.")
