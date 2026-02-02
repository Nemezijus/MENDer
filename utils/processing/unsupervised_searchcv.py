from __future__ import annotations

"""SearchCV-like utilities for unsupervised (clustering) tuning.

Why this exists
--------------
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

The strategy layer (utils/strategies/tuning.py) should stay orchestration-only
and delegate the implementation details here.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import math
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold, ParameterGrid, ParameterSampler

from utils.postprocessing.unsupervised_scoring import compute_unsupervised_metrics


UNSUPERVISED_METRICS = {"silhouette", "davies_bouldin", "calinski_harabasz"}


def _coerce_metric(metric: str) -> str:
    return metric if metric in UNSUPERVISED_METRICS else "silhouette"


def _prefers_higher(metric: str) -> bool:
    # Davies–Bouldin: lower is better; the others: higher is better.
    return metric != "davies_bouldin"


def _supports_predict(pipe) -> bool:
    """Return True iff the *final estimator* supports predicting labels."""
    try:
        if hasattr(pipe, "steps") and pipe.steps:
            return hasattr(pipe.steps[-1][1], "predict")
    except Exception:
        pass
    return hasattr(pipe, "predict")


def _transform_features(pipe, X: np.ndarray) -> np.ndarray:
    """Transform X using Pipeline preprocessing if available."""
    try:
        return np.asarray(pipe[:-1].transform(X))
    except Exception:
        return np.asarray(X)


def _labels_for_training(pipe, X_train: np.ndarray) -> Optional[np.ndarray]:
    """Get labels for the training set after fitting.

    For many clustering estimators, labels are available via ``labels_``.
    If not, try predicting labels on the training data.
    """
    est = None
    try:
        if hasattr(pipe, "named_steps"):
            est = pipe.named_steps.get("clf")
    except Exception:
        est = None

    if est is None and hasattr(pipe, "steps") and pipe.steps:
        est = pipe.steps[-1][1]

    labels = getattr(est, "labels_", None) if est is not None else None
    if labels is not None:
        labels = np.asarray(labels)
        if labels.shape[0] == X_train.shape[0]:
            return labels

    if _supports_predict(pipe):
        try:
            return np.asarray(pipe.predict(X_train))
        except Exception:
            return None

    return None


def _score_from_labels(Z: np.ndarray, labels: np.ndarray, metric: str) -> float:
    metric = _coerce_metric(metric)
    m, _warnings = compute_unsupervised_metrics(Z, labels, [metric])
    v = m.get(metric)
    return float(v) if v is not None else float("nan")


def _sanitize_float_list(values: Sequence[float]) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    for v in values:
        try:
            fv = float(v)
            out.append(fv if math.isfinite(fv) else None)
        except Exception:
            out.append(None)
    return out


def _to_py(v: Any) -> Any:
    """Convert numpy scalars/arrays into pure python types for JSON."""
    if isinstance(v, (np.generic,)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def _make_kfold(cv: int, *, shuffle: bool, random_state: Optional[int]):
    return KFold(n_splits=int(cv), shuffle=bool(shuffle), random_state=random_state)


@dataclass
class _BaseUnsupervisedSearchCV:
    estimator: Any
    metric: str
    cv: Any
    refit: bool = True
    return_train_score: bool = False
    shuffle: bool = False
    random_state: Optional[int] = None

    # public sklearn-ish attributes
    best_params_: Dict[str, Any] | None = None
    best_score_: Optional[float] = None
    best_index_: Optional[int] = None
    best_estimator_: Any | None = None
    cv_results_: Dict[str, Any] | None = None
    note_: Optional[str] = None

    def _cv_splitter(self):
        if isinstance(self.cv, int):
            return _make_kfold(self.cv, shuffle=self.shuffle, random_state=self.random_state)
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
            Ztr = _transform_features(pipe, Xtr)
            ytr = _labels_for_training(pipe, Xtr)
            fold_train.append(_score_from_labels(Ztr, ytr, self.metric) if ytr is not None else float("nan"))

            # val-side
            if predict_supported:
                try:
                    yva = np.asarray(pipe.predict(Xva))
                    Zva = _transform_features(pipe, Xva)
                    fold_val.append(_score_from_labels(Zva, yva, self.metric))
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
    ):
        super().__init__(
            estimator=estimator,
            metric=_coerce_metric(metric),
            cv=cv,
            refit=refit,
            return_train_score=return_train_score,
            shuffle=shuffle,
            random_state=random_state,
        )
        self.param_grid = param_grid or {}

    def fit(self, X: np.ndarray):
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[0] < 2:
            raise ValueError("X must be a 2D array with at least 2 samples.")

        predict_supported = _supports_predict(self.estimator)
        prefers_higher = _prefers_higher(self.metric)

        grid = list(ParameterGrid(self.param_grid)) if self.param_grid else [{}]

        cv_results: Dict[str, List[Any]] = {}
        for k in self.param_grid.keys():
            cv_results[f"param_{k}"] = []

        mean_train: List[float] = []
        std_train: List[float] = []
        mean_val: List[float] = []
        std_val: List[float] = []

        best_score: Optional[float] = None
        best_index: Optional[int] = None

        for i, params in enumerate(grid):
            for k in self.param_grid.keys():
                cv_results[f"param_{k}"].append(_to_py(params.get(k)))

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
                if prefers_higher:
                    if candidate > best_score:
                        best_score, best_index = candidate, i
                else:
                    if candidate < best_score:
                        best_score, best_index = candidate, i

        # cv_results fields expected by the frontend
        cv_results["mean_score"] = _sanitize_float_list(mean_train)
        cv_results["std_score"] = _sanitize_float_list(std_train)
        cv_results["mean_test_score"] = _sanitize_float_list(mean_val)
        cv_results["std_test_score"] = _sanitize_float_list(std_val)

        self.cv_results_ = cv_results
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
        return_train_score: bool = False,
        shuffle: bool = False,
    ):
        super().__init__(
            estimator=estimator,
            metric=_coerce_metric(metric),
            cv=cv,
            refit=refit,
            return_train_score=return_train_score,
            shuffle=shuffle,
            random_state=random_state,
        )
        self.param_distributions = param_distributions or {}
        self.n_iter = int(n_iter)

    def fit(self, X: np.ndarray):
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[0] < 2:
            raise ValueError("X must be a 2D array with at least 2 samples.")

        predict_supported = _supports_predict(self.estimator)
        prefers_higher = _prefers_higher(self.metric)

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

        cv_results: Dict[str, List[Any]] = {}
        for k in self.param_distributions.keys():
            cv_results[f"param_{k}"] = []

        mean_train: List[float] = []
        std_train: List[float] = []
        mean_val: List[float] = []
        std_val: List[float] = []

        best_score: Optional[float] = None
        best_index: Optional[int] = None

        for i, params in enumerate(sampler):
            for k in self.param_distributions.keys():
                cv_results[f"param_{k}"].append(_to_py(params.get(k)))

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
                if prefers_higher:
                    if candidate > best_score:
                        best_score, best_index = candidate, i
                else:
                    if candidate < best_score:
                        best_score, best_index = candidate, i

        cv_results["mean_score"] = _sanitize_float_list(mean_train)
        cv_results["std_score"] = _sanitize_float_list(std_train)
        cv_results["mean_test_score"] = _sanitize_float_list(mean_val)
        cv_results["std_test_score"] = _sanitize_float_list(std_val)

        self.cv_results_ = cv_results
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
