from __future__ import annotations

"""SearchCV-like utilities for unsupervised (clustering) tuning.

Scikit-learn's GridSearchCV / RandomizedSearchCV assume that scoring on the
validation fold is always possible. For many clustering estimators, predicting
labels for unseen samples is not supported (no ``predict``), which makes
``mean_test_score`` undefined.

These utilities implement a small, sklearn-ish subset for unsupervised tuning:

* Always compute *train-side* intrinsic scores on each fold.
* Compute *validation-side* intrinsic scores only when predicting labels for
  unseen samples is possible.
* When validation scores are unavailable, pick best parameters using the
  train-side score, and emit a human-friendly note.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import math
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid, ParameterSampler

from .adapters import build_cv_results
from .param_space import coerce_metric, make_kfold, prefers_higher
from .scoring import labels_for_training, score_from_labels, supports_predict, transform_features


@dataclass
class _BaseUnsupervisedSearchCV:
    estimator: Any
    metric: str
    cv: Any
    refit: bool = True
    return_train_score: bool = False
    shuffle: bool = False
    random_state: Optional[int] = None
    # When shuffle=True, allow an independent random state for CV splitting so
    # parameter sampling RNG can remain stable.
    cv_random_state: Optional[int] = None

    # public sklearn-ish attributes
    best_params_: Dict[str, Any] | None = None
    best_score_: Optional[float] = None
    best_index_: Optional[int] = None
    best_estimator_: Any | None = None
    cv_results_: Dict[str, Any] | None = None
    note_: Optional[str] = None

    def _cv_splitter(self):
        if isinstance(self.cv, int):
            rs = self.cv_random_state if self.cv_random_state is not None else self.random_state
            return make_kfold(self.cv, shuffle=self.shuffle, random_state=rs)
        return self.cv

    def _evaluate_params(self, X: np.ndarray, params: Dict[str, Any], predict_supported: bool) -> tuple[float, float, float, float]:
        """Return (mean_train, std_train, mean_val, std_val) for one parameter setting."""

        splitter = self._cv_splitter()
        fold_train: List[float] = []
        fold_val: List[float] = []

        for tr_idx, va_idx in splitter.split(X):
            Xtr = X[tr_idx]
            Xva = X[va_idx]

            pipe = clone(self.estimator)
            if params:
                pipe.set_params(**params)
            pipe.fit(Xtr)

            # train-side
            Ztr = transform_features(pipe, Xtr)
            ytr = labels_for_training(pipe, Xtr)
            fold_train.append(score_from_labels(Ztr, ytr, self.metric) if ytr is not None else float("nan"))

            # val-side
            if predict_supported:
                try:
                    yva = np.asarray(pipe.predict(Xva))
                    Zva = transform_features(pipe, Xva)
                    fold_val.append(score_from_labels(Zva, yva, self.metric))
                except Exception:
                    fold_val.append(float("nan"))
            else:
                fold_val.append(float("nan"))

        mtr = float(np.nanmean(fold_train))
        str_ = float(np.nanstd(fold_train))
        mva = float(np.nanmean(fold_val))
        sva = float(np.nanstd(fold_val))
        return mtr, str_, mva, sva

    def _select_candidate(self, mean_train: float, mean_val: float, predict_supported: bool) -> Optional[float]:
        # Prefer validation when it exists and is finite.
        if predict_supported and math.isfinite(mean_val):
            return mean_val
        if math.isfinite(mean_train):
            return mean_train
        return None

    def _refit_best(self, X: np.ndarray):
        if not self.refit or not self.best_params_:
            return
        est = clone(self.estimator)
        est.set_params(**self.best_params_)
        est.fit(X)
        self.best_estimator_ = est


class UnsupervisedGridSearchCV(_BaseUnsupervisedSearchCV):
    """A minimal GridSearchCV analogue for clustering metrics."""

    def __init__(
        self,
        *,
        estimator: Any,
        param_grid: Dict[str, Sequence[Any]] | None,
        metric: str,
        cv: Any = 5,
        refit: bool = True,
        return_train_score: bool = False,
        shuffle: bool = False,
        random_state: Optional[int] = None,
        cv_random_state: Optional[int] = None,
    ):
        super().__init__(
            estimator=estimator,
            metric=coerce_metric(metric),
            cv=cv,
            refit=refit,
            return_train_score=return_train_score,
            shuffle=shuffle,
            random_state=random_state,
            cv_random_state=cv_random_state,
        )
        self.param_grid = param_grid or {}

    def fit(self, X: np.ndarray):
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[0] < 2:
            raise ValueError("X must be a 2D array with at least 2 samples.")

        predict_supported = supports_predict(self.estimator)
        higher_is_better = prefers_higher(self.metric)

        grid = list(ParameterGrid(self.param_grid)) if self.param_grid else [{}]

        mean_train: List[float] = []
        std_train: List[float] = []
        mean_val: List[float] = []
        std_val: List[float] = []

        best_score: Optional[float] = None
        best_index: Optional[int] = None

        for i, params in enumerate(grid):
            mtr, str_, mva, sva = self._evaluate_params(X, params, predict_supported)
            mean_train.append(mtr)
            std_train.append(str_)
            mean_val.append(mva)
            std_val.append(sva)

            candidate = self._select_candidate(mtr, mva, predict_supported)
            if candidate is None:
                continue

            if best_score is None:
                best_score, best_index = candidate, i
            else:
                if higher_is_better:
                    if candidate > best_score:
                        best_score, best_index = candidate, i
                else:
                    if candidate < best_score:
                        best_score, best_index = candidate, i

        self.cv_results_ = build_cv_results(
            param_keys=list(self.param_grid.keys()),
            params_list=grid,
            mean_train=mean_train,
            std_train=std_train,
            mean_val=mean_val,
            std_val=std_val,
        )

        self.best_index_ = int(best_index) if best_index is not None else None
        self.best_score_ = float(best_score) if best_score is not None else None
        self.best_params_ = grid[best_index] if best_index is not None else {}

        if not predict_supported:
            self.note_ = (
                "Validation scores are unavailable for this model because it does not support predicting labels for unseen samples "
                "(no predict()). Grid search is optimized using train-side scores."
            )

        self._refit_best(X)
        return self


class UnsupervisedRandomizedSearchCV(_BaseUnsupervisedSearchCV):
    """A minimal RandomizedSearchCV analogue for clustering metrics."""

    def __init__(
        self,
        *,
        estimator: Any,
        param_distributions: Dict[str, Any] | None,
        n_iter: int = 20,
        metric: str,
        cv: Any = 5,
        refit: bool = True,
        random_state: Optional[int] = None,
        cv_random_state: Optional[int] = None,
        return_train_score: bool = False,
        shuffle: bool = False,
    ):
        super().__init__(
            estimator=estimator,
            metric=coerce_metric(metric),
            cv=cv,
            refit=refit,
            return_train_score=return_train_score,
            shuffle=shuffle,
            random_state=random_state,
            cv_random_state=cv_random_state,
        )
        self.param_distributions = param_distributions or {}
        self.n_iter = int(n_iter)

    def fit(self, X: np.ndarray):
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[0] < 2:
            raise ValueError("X must be a 2D array with at least 2 samples.")

        predict_supported = supports_predict(self.estimator)
        higher_is_better = prefers_higher(self.metric)

        if self.param_distributions:
            sampler = list(
                ParameterSampler(
                    self.param_distributions,
                    n_iter=self.n_iter,
                    random_state=self.random_state,
                )
            )
        else:
            sampler = [{}]

        mean_train: List[float] = []
        std_train: List[float] = []
        mean_val: List[float] = []
        std_val: List[float] = []

        best_score: Optional[float] = None
        best_index: Optional[int] = None

        for i, params in enumerate(sampler):
            mtr, str_, mva, sva = self._evaluate_params(X, params, predict_supported)
            mean_train.append(mtr)
            std_train.append(str_)
            mean_val.append(mva)
            std_val.append(sva)

            candidate = self._select_candidate(mtr, mva, predict_supported)
            if candidate is None:
                continue

            if best_score is None:
                best_score, best_index = candidate, i
            else:
                if higher_is_better:
                    if candidate > best_score:
                        best_score, best_index = candidate, i
                else:
                    if candidate < best_score:
                        best_score, best_index = candidate, i

        self.cv_results_ = build_cv_results(
            param_keys=list(self.param_distributions.keys()),
            params_list=sampler,
            mean_train=mean_train,
            std_train=std_train,
            mean_val=mean_val,
            std_val=std_val,
        )

        self.best_index_ = int(best_index) if best_index is not None else None
        self.best_score_ = float(best_score) if best_score is not None else None
        self.best_params_ = sampler[best_index] if best_index is not None else {}

        if not predict_supported:
            self.note_ = (
                "Validation scores are unavailable for this model because it does not support predicting labels for unseen samples "
                "(no predict()). Random search is optimized using train-side scores."
            )

        self._refit_best(X)
        return self
