# utils/preprocessing/trial_split.py
from __future__ import annotations

from typing import Tuple, Union
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split


def _to_child_seed(rng: Union[None, int, np.random.Generator]) -> Union[int, None]:
    """
    Convert rng to a single integer seed for sklearn APIs.

    - If rng is a Generator: draw one child int (order-sensitive by design).
    - If rng is an int/np.integer: use it directly (no extra draw).
    - If rng is None: return None (sklearn will be non-deterministic or use its default).
    - If rng is any other hashable type: create a local generator and draw one int.
    """
    if isinstance(rng, np.random.Generator):
        return int(rng.integers(1 << 32))
    if isinstance(rng, (int, np.integer)) or rng is None:
        return None if rng is None else int(rng)
    # Fallback for unusual but hashable seeds (e.g., str/tuple)
    return int(np.random.default_rng(rng).integers(1 << 32))


def _split_trials(
    X: np.ndarray,
    y: np.ndarray,
    train_frac: float,
    rng: Union[None, int, np.random.Generator],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Single stratified shuffle-split using StratifiedShuffleSplit.
    """
    if not (0.0 < train_frac < 1.0):
        raise ValueError("train_frac must be in (0, 1).")

    X = np.asarray(X)
    y = np.asarray(y).ravel()
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y length mismatch: {X.shape[0]} vs {y.shape[0]}.")

    random_state = _to_child_seed(rng)

    sss = StratifiedShuffleSplit(n_splits=1, train_size=train_frac, random_state=random_state)
    (train_idx, test_idx), = sss.split(X, y)
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def split(
    X: np.ndarray,
    y: np.ndarray,
    train_frac: float = 0.8,
    *,
    custom: bool = False,
    rng: Union[None, int, np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Public splitter: stratified train/test split on `y`.

    Modes
    -----
    - custom=False (default): uses sklearn.train_test_split(stratify=y).
    - custom=True: uses a single StratifiedShuffleSplit via _split_trials.
    """
    X = np.asarray(X)
    y = np.asarray(y).ravel()

    if not (0.0 < train_frac < 1.0):
        raise ValueError("train_frac must be in (0, 1).")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y length mismatch: {X.shape[0]} vs {y.shape[0]}.")

    random_state = _to_child_seed(rng)

    if custom:
        return _split_trials(X, y, train_frac, rng=random_state)

    return train_test_split(
        X, y,
        train_size=train_frac,
        stratify=y,
        random_state=random_state,
    )
