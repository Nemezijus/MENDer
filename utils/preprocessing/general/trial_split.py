# utils/preprocessing/trial_split.py
from __future__ import annotations

from typing import Tuple, Union
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

def _to_child_seed(rng: Union[None, int, np.random.Generator]) -> Union[int, None]:
    """
    Convert rng to a single integer child seed, mirroring pre-refactor behavior.
    - If rng is a Generator: draw one child int.
    - If rng is an int: use it directly (do NOT draw again).
    - If rng is None: return None.
    """
    if isinstance(rng, np.random.Generator):
        return int(rng.integers(1 << 32))
    if isinstance(rng, (int, np.integer)) or rng is None:
        return None if rng is None else int(rng)
    # If someone passed a hashable seed (e.g., tuple/str), mimic old pattern:
    return int(np.random.default_rng(rng).integers(1 << 32))


def _split_trials(
    X: np.ndarray,
    y: np.ndarray,
    train_frac: float,
    rng: Union[None, int, np.random.Generator],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Internal helper: single stratified shuffle-split.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    y : array-like of shape (n_samples,)
    train_frac : float
        Fraction of samples in the train set; must be in (0, 1).
    rng : int | numpy.random.Generator | None
        Seed or Generator for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    if not (0.0 < train_frac < 1.0):
        raise ValueError("train_frac must be in (0, 1).")

    X = np.asarray(X)
    y = np.asarray(y).ravel()

    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y length mismatch: {X.shape[0]} vs {y.shape[0]}.")

    # StratifiedShuffleSplit expects an int seed for deterministic behavior
    # gen = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    # seed = int(gen.integers(2**32 - 1))
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
    Public splitter: returns train/test stratified on `y`.

    Two modes:
    - custom=False (default): uses sklearn's `train_test_split` with stratify=y.
    - custom=True: uses our own single `StratifiedShuffleSplit` helper.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    y : array-like of shape (n_samples,)
    train_frac : float, default=0.8
        Fraction of samples to place in the train set; must be in (0, 1).
    custom : bool, default=False
        If True, use `_split_trials` (StratifiedShuffleSplit). Otherwise, use
        `train_test_split(stratify=y)`. Results are similar; `custom=True`
        gives you a single explicit split object.
    rng : int | numpy.random.Generator | None, default=None
        Seed or Generator for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test

    Notes
    -----
    - Always uses stratification by `y` to preserve class proportions.
    - Raises if X and y lengths differ or if `train_frac` is invalid.
    """
    X = np.asarray(X)
    y = np.asarray(y).ravel()

    if not (0.0 < train_frac < 1.0):
        raise ValueError("train_frac must be in (0, 1).")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y length mismatch: {X.shape[0]} vs {y.shape[0]}.")

    # gen = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    # seed = int(gen.integers(2**32 - 1))
    random_state = _to_child_seed(rng)

    if custom:
        return _split_trials(X, y, train_frac, rng=random_state)

    return train_test_split(
        X, y,
        train_size=train_frac,
        stratify=y,
        random_state=random_state,
    )
