from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal

import numpy as np


def clamp_fraction(x: float, *, lo: float = 0.0, hi: float = 0.5, default: float = 0.2) -> float:
    try:
        v = float(x)
    except Exception:
        return float(default)
    if not np.isfinite(v):
        return float(default)
    return float(min(max(v, lo), hi))


def default_patience(n_estimators: int) -> int:
    # Reasonable heuristic: 10% of estimators, capped.
    n = int(n_estimators)
    return int(min(max(10, n // 10), 200))


def compute_val_n(*, n_samples: int, frac: float) -> int:
    n = int(n_samples)
    val_n = int(round(n * float(frac)))
    return int(min(max(1, val_n), max(1, n - 1)))


@dataclass(frozen=True)
class Split:
    X_tr: np.ndarray
    y_tr: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray


def train_val_split(
    X: np.ndarray,
    y: np.ndarray,
    *,
    val_n: int,
    random_state: int,
    stratify: bool,
) -> Split:
    """Deterministic train/val split used for xgboost eval_set learning curves."""

    n = int(X.shape[0])
    rs = np.random.RandomState(int(random_state))

    idx = np.arange(n)
    if stratify:
        # stratify by y values (assumes integer-encoded labels)
        y_flat = np.asarray(y).ravel()
        uniq = np.unique(y_flat)
        parts: list[np.ndarray] = []
        for u in uniq:
            ii = idx[y_flat == u]
            rs.shuffle(ii)
            parts.append(ii)
        idx = np.concatenate(parts)
    else:
        rs.shuffle(idx)

    val_n = int(min(max(1, val_n), max(1, n - 1)))
    val_idx = idx[:val_n]
    tr_idx = idx[val_n:]

    X_tr = np.asarray(X)[tr_idx]
    y_tr = np.asarray(y)[tr_idx]
    X_val = np.asarray(X)[val_idx]
    y_val = np.asarray(y)[val_idx]
    return Split(X_tr=X_tr, y_tr=y_tr, X_val=X_val, y_val=y_val)


def choose_eval_metric(kind: Literal["classification", "regression"], *, n_classes: Optional[int] = None) -> str:
    if kind == "regression":
        return "rmse"
    if n_classes is None or int(n_classes) <= 2:
        return "logloss"
    return "mlogloss"
