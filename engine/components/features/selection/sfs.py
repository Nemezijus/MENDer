from __future__ import annotations

from typing import Optional, Tuple, Union, Literal, Any
import numpy as np

from sklearn.base import clone
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import StratifiedKFold

from engine.core.shapes import ensure_xy_aligned


def perform_sfs(
    X: np.ndarray,
    y: np.ndarray,
    estimator: Any,
    *,
    n_features_to_select: Union[int, Literal["auto"]],
    direction: Literal["forward", "backward"] = "backward",
    scoring: str = "accuracy",
    cv: int = 5,
    shuffle: bool = True,
    random_state: Optional[int] = None,
    n_jobs: Optional[int] = None,
) -> tuple[SequentialFeatureSelector, np.ndarray]:
    """
    Fit Sequential Feature Selector (SFS) on all data (X,y); return (selector, X_selected).
    """
    X, y = ensure_xy_aligned(X, y)
    base = clone(estimator)

    skf = StratifiedKFold(
        n_splits=int(cv),
        shuffle=bool(shuffle),
        random_state=None if not shuffle else (None if random_state is None else int(random_state)),
    )

    sfs = SequentialFeatureSelector(
        base,
        n_features_to_select=n_features_to_select,
        direction=direction,
        scoring=scoring,
        cv=skf,
        n_jobs=n_jobs,
    )
    X_sel = sfs.fit_transform(X, y)
    return sfs, X_sel


def sfs_fit_transform_train_test(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    estimator: Any,
    *,
    n_features_to_select: Union[int, Literal["auto"]],
    direction: Literal["forward", "backward"] = "backward",
    scoring: str = "accuracy",
    cv: int = 5,
    shuffle: bool = True,
    random_state: Optional[int] = None,
    n_jobs: Optional[int] = None,
) -> tuple[SequentialFeatureSelector, np.ndarray, np.ndarray]:
    """
    Fit SFS on TRAIN only, then transform train and test.
    """
    X_train, y_train = ensure_xy_aligned(X_train, y_train)
    X_test = np.asarray(X_test)
    if X_test.ndim != 2:
        raise ValueError(f"X_test must be 2D. Got {X_test.shape}.")
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("X_train and X_test must have the same number of features.")

    base = clone(estimator)
    skf = StratifiedKFold(
        n_splits=int(cv),
        shuffle=bool(shuffle),
        random_state=None if not shuffle else (None if random_state is None else int(random_state)),
    )

    sfs = SequentialFeatureSelector(
        base,
        n_features_to_select=n_features_to_select,
        direction=direction,
        scoring=scoring,
        cv=skf,
        n_jobs=n_jobs,
    )
    X_train_sel = sfs.fit_transform(X_train, y_train)
    X_test_sel = sfs.transform(X_test)
    return sfs, X_train_sel, X_test_sel
