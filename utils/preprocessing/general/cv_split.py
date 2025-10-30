from __future__ import annotations
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from typing import Iterator, Tuple, Union


def generate_folds(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    stratified: bool = True,
    shuffle: bool = True,
    random_state: Union[int, None] = None,
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Generator yielding (X_train, X_test, y_train, y_test) for each fold.
    Mirrors the style of utils/preprocessing/general/trial_split.py.
    """
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y length mismatch: {X.shape[0]} vs {y.shape[0]}")

    splitter_cls = StratifiedKFold if stratified else KFold
    splitter = splitter_cls(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
    )

    for train_idx, test_idx in splitter.split(X, y):
        yield X[train_idx], X[test_idx], y[train_idx], y[test_idx]
