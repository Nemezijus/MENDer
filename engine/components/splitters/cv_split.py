from __future__ import annotations
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from typing import Iterator, Union

from engine.components.splitters.types import Split


def generate_folds(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    stratified: bool = True,
    shuffle: bool = True,
    random_state: Union[int, None] = None,
) -> Iterator[Split]:
    """Yield :class:`Split` for each fold."""
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
        yield Split(
            Xtr=X[train_idx],
            Xte=X[test_idx],
            ytr=y[train_idx],
            yte=y[test_idx],
            idx_tr=np.asarray(train_idx, dtype=int),
            idx_te=np.asarray(test_idx, dtype=int),
        )
