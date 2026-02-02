from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import math
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import (
    StratifiedKFold,
    KFold,
    learning_curve as sk_learning_curve,
    validation_curve as sk_validation_curve,
    GridSearchCV,
    RandomizedSearchCV,
    ParameterGrid,
    ParameterSampler,
)

from shared_schemas.run_config import RunConfig
from shared_schemas.model_configs import get_model_task
from shared_schemas.tuning_configs import (
    LearningCurveConfig,
    ValidationCurveConfig,
    GridSearchConfig,
    RandomizedSearchConfig,
)
from shared_schemas.unsupervised_configs import UnsupervisedRunConfig, UnsupervisedEvalModel

from utils.factories.data_loading_factory import make_data_loader
from utils.factories.sanity_factory import make_sanity_checker
from utils.factories.pipeline_factory import make_pipeline, make_unsupervised_pipeline
from utils.permutations.rng import RngManager
from utils.postprocessing.scoring import make_estimator_scorer
from utils.postprocessing.unsupervised_scoring import compute_unsupervised_metrics

from .interfaces import TuningStrategy


UNSUPERVISED_METRICS = {"silhouette", "davies_bouldin", "calinski_harabasz"}


def _resolve_param_name_for_pipeline(pipe, raw_name: str) -> str:
    """Map a logical model parameter name (e.g. 'C') to a Pipeline parameter name.

    - If raw_name already exists in pipe.get_params(), it is returned unchanged.
    - Otherwise we try <last_step_name>__<raw_name>.
    - If that also doesn't exist, we return raw_name and let sklearn raise.
    """
    if not raw_name:
        return raw_name

    params = pipe.get_params(deep=True)
    if raw_name in params:
        return raw_name

    if pipe.steps:
        last_step_name = pipe.steps[-1][0]
        candidate = f"{last_step_name}__{raw_name}"
        if candidate in params:
            return candidate

    return raw_name


def _sanitize_floats(values: List[float]) -> List[float | None]:
    out: List[float | None] = []
    for v in values:
        try:
            fv = float(v)
            out.append(fv if math.isfinite(fv) else None)
        except Exception:
            out.append(None)
    return out


def _coerce_unsupervised_metric(metric: str) -> str:
    return metric if metric in UNSUPERVISED_METRICS else "silhouette"


def _metric_prefers_higher(metric: str) -> bool:
    # Davies–Bouldin: lower is better; the others: higher is better.
    return metric != "davies_bouldin"


def _unsupervised_score_from_labels(Z: np.ndarray, labels: np.ndarray, metric: str) -> float:
    metric = _coerce_unsupervised_metric(metric)
    m, _warnings = compute_unsupervised_metrics(Z, labels, [metric])
    v = m.get(metric)
    return float(v) if v is not None else float("nan")


def _fit_labels_for_training(pipe, X_train: np.ndarray) -> np.ndarray | None:
    """Return labels for the *training* set after fitting.

    For many clustering estimators, training labels are available via labels_.
    If labels_ is not present, fall back to predict when available.
    """
    try:
        est = pipe.named_steps.get("clf")
    except Exception:
        est = None

    labels = getattr(est, "labels_", None) if est is not None else None
    if labels is not None:
        labels = np.asarray(labels)
        if labels.shape[0] == X_train.shape[0]:
            return labels

    if hasattr(pipe, "predict"):
        try:
            return np.asarray(pipe.predict(X_train))
        except Exception:
            return None

    return None


def _transform_features(pipe, X: np.ndarray) -> np.ndarray:
    """Transform X using the preprocessing part of a Pipeline if possible."""
    try:
        return np.asarray(pipe[:-1].transform(X))
    except Exception:
        return np.asarray(X)


def _cv_for_cfg(cfg: RunConfig, *, rngm: RngManager, stream: str, force_unstratified: bool = False):
    cv_seed = rngm.child_seed(f"{stream}/split") if cfg.split.shuffle else None

    if force_unstratified:
        return KFold(
            n_splits=cfg.split.n_splits,
            shuffle=cfg.split.shuffle,
            random_state=cv_seed,
        )

    stratified_flag = getattr(cfg.split, "stratified", True)
    if stratified_flag:
        return StratifiedKFold(
            n_splits=cfg.split.n_splits,
            shuffle=cfg.split.shuffle,
            random_state=cv_seed,
        )

    return KFold(
        n_splits=cfg.split.n_splits,
        shuffle=cfg.split.shuffle,
        random_state=cv_seed,
    )


# -------------------------------
# Learning curve
# -------------------------------

@dataclass
class LearningCurveRunner(TuningStrategy):
    cfg: RunConfig
    lc: LearningCurveConfig

    def run(self) -> Dict[str, Any]:
        cfg = self.cfg

        task = get_model_task(cfg.model)
        if task == "unsupervised":
            return self._run_unsupervised()

        eval_kind = "regression" if task == "regression" else "classification"
        scorer = make_estimator_scorer(eval_kind, cfg.eval.metric)

        loader = make_data_loader(cfg.data)
        X, y = loader.load()
        make_sanity_checker().check(X, y)

        rngm = RngManager(None if cfg.eval.seed is None else int(cfg.eval.seed))
        cv = _cv_for_cfg(cfg, rngm=rngm, stream="tuning/lc", force_unstratified=(eval_kind != "classification"))

        pipe = make_pipeline(cfg, rngm, stream="tuning/lc")

        if self.lc.train_sizes is not None:
            train_sizes = np.asarray(self.lc.train_sizes)
        else:
            train_sizes = np.linspace(0.1, 1.0, self.lc.n_steps)

        sizes_abs, train_scores, val_scores = sk_learning_curve(
            estimator=pipe,
            X=X,
            y=y,
            train_sizes=train_sizes,
            cv=cv,
            scoring=scorer,
            n_jobs=self.lc.n_jobs,
            shuffle=False,
            return_times=False,
        )

        train_mean = np.mean(train_scores, axis=1).tolist()
        train_std = np.std(train_scores, axis=1).tolist()
        val_mean = np.mean(val_scores, axis=1).tolist()
        val_std = np.std(val_scores, axis=1).tolist()

        return {
            "train_sizes": sizes_abs.tolist(),
            "train_scores_mean": _sanitize_floats(train_mean),
            "train_scores_std": _sanitize_floats(train_std),
            "val_scores_mean": _sanitize_floats(val_mean),
            "val_scores_std": _sanitize_floats(val_std),
        }

    def _run_unsupervised(self) -> Dict[str, Any]:
        cfg = self.cfg
        metric = _coerce_unsupervised_metric(str(cfg.eval.metric))

        loader = make_data_loader(cfg.data)
        X, _y = loader.load()
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[0] < 2:
            raise ValueError("X must be a 2D array with at least 2 samples for unsupervised tuning.")

        rngm = RngManager(None if cfg.eval.seed is None else int(cfg.eval.seed))
        cv = _cv_for_cfg(cfg, rngm=rngm, stream="tuning/lc", force_unstratified=True)

        uns_eval = UnsupervisedEvalModel(metrics=[metric], seed=cfg.eval.seed, compute_embedding_2d=False, per_sample_outputs=False)
        uns_cfg = UnsupervisedRunConfig(data=cfg.data, scale=cfg.scale, features=cfg.features, model=cfg.model, eval=uns_eval)
        base_pipe = make_unsupervised_pipeline(uns_cfg, rngm, stream="tuning/lc")

        # train sizes
        if self.lc.train_sizes is not None:
            train_sizes = np.asarray(self.lc.train_sizes)
        else:
            train_sizes = np.linspace(0.1, 1.0, self.lc.n_steps)

        # convert to absolute sample counts per fold
        # (use the *training fold* size as reference)
        train_means: List[float] = []
        train_stds: List[float] = []
        val_means: List[float] = []
        val_stds: List[float] = []
        sizes_abs: List[int] = []

        predict_supported = hasattr(base_pipe, "predict")

        for s_i, s in enumerate(train_sizes):
            fold_train_scores: List[float] = []
            fold_val_scores: List[float] = []

            for f_i, (tr_idx, va_idx) in enumerate(cv.split(X)):
                Xtr_full = X[tr_idx]
                Xva = X[va_idx]

                if isinstance(s, (int, np.integer)):
                    n_train = int(s)
                else:
                    n_train = int(math.ceil(float(s) * float(Xtr_full.shape[0])))

                n_train = max(2, min(n_train, int(Xtr_full.shape[0])))

                # stable subsampling within fold
                if cfg.split.shuffle:
                    seed = rngm.child_seed(f"tuning/lc/subsample/{s_i}/{f_i}")
                    rng = np.random.default_rng(seed)
                    perm = rng.permutation(Xtr_full.shape[0])
                else:
                    perm = np.arange(Xtr_full.shape[0])

                Xtr = Xtr_full[perm[:n_train]]

                pipe = clone(base_pipe)
                pipe.fit(Xtr)

                Ztr = _transform_features(pipe, Xtr)
                labels_tr = _fit_labels_for_training(pipe, Xtr)
                if labels_tr is None:
                    fold_train_scores.append(float("nan"))
                else:
                    fold_train_scores.append(_unsupervised_score_from_labels(Ztr, labels_tr, metric))

                if predict_supported:
                    try:
                        labels_va = np.asarray(pipe.predict(Xva))
                        Zva = _transform_features(pipe, Xva)
                        fold_val_scores.append(_unsupervised_score_from_labels(Zva, labels_va, metric))
                    except Exception:
                        fold_val_scores.append(float("nan"))
                else:
                    fold_val_scores.append(float("nan"))

            sizes_abs.append(n_train)  # representative absolute size
            train_means.append(float(np.nanmean(fold_train_scores)))
            train_stds.append(float(np.nanstd(fold_train_scores)))
            val_means.append(float(np.nanmean(fold_val_scores)))
            val_stds.append(float(np.nanstd(fold_val_scores)))

        note = None
        if not predict_supported:
            note = "Validation scores are unavailable for this model because it does not support predicting labels for unseen samples (no predict())."

        return {
            "metric_used": metric,
            "note": note,
            "train_sizes": sizes_abs,
            "train_scores_mean": _sanitize_floats(train_means),
            "train_scores_std": _sanitize_floats(train_stds),
            "val_scores_mean": _sanitize_floats(val_means),
            "val_scores_std": _sanitize_floats(val_stds),
        }


# -------------------------------
# Validation curve
# -------------------------------

@dataclass
class ValidationCurveRunner(TuningStrategy):
    cfg: RunConfig
    vc: ValidationCurveConfig

    def run(self) -> Dict[str, Any]:
        cfg = self.cfg

        task = get_model_task(cfg.model)
        if task == "unsupervised":
            return self._run_unsupervised()

        eval_kind = "regression" if task == "regression" else "classification"
        scorer = make_estimator_scorer(eval_kind, cfg.eval.metric)

        loader = make_data_loader(cfg.data)
        X, y = loader.load()
        make_sanity_checker().check(X, y)

        rngm = RngManager(None if cfg.eval.seed is None else int(cfg.eval.seed))

        pipe = make_pipeline(cfg, rngm, stream="tuning/vc")

        raw_name = self.vc.param_name
        param_name = _resolve_param_name_for_pipeline(pipe, raw_name)
        param_range = self.vc.param_range

        train_scores, val_scores = sk_validation_curve(
            estimator=pipe,
            X=X,
            y=y,
            param_name=param_name,
            param_range=param_range,
            scoring=scorer,
            n_jobs=self.vc.n_jobs,
            error_score="raise",
        )

        train_mean = np.mean(train_scores, axis=1).tolist()
        train_std = np.std(train_scores, axis=1).tolist()
        val_mean = np.mean(val_scores, axis=1).tolist()
        val_std = np.std(val_scores, axis=1).tolist()

        return {
            "param_name": raw_name,
            "param_range": list(param_range),
            "train_scores_mean": _sanitize_floats(train_mean),
            "train_scores_std": _sanitize_floats(train_std),
            "val_scores_mean": _sanitize_floats(val_mean),
            "val_scores_std": _sanitize_floats(val_std),
        }

    def _run_unsupervised(self) -> Dict[str, Any]:
        cfg = self.cfg
        metric = _coerce_unsupervised_metric(str(cfg.eval.metric))

        loader = make_data_loader(cfg.data)
        X, _y = loader.load()
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[0] < 2:
            raise ValueError("X must be a 2D array with at least 2 samples for unsupervised tuning.")

        rngm = RngManager(None if cfg.eval.seed is None else int(cfg.eval.seed))
        cv = _cv_for_cfg(cfg, rngm=rngm, stream="tuning/vc", force_unstratified=True)

        uns_eval = UnsupervisedEvalModel(metrics=[metric], seed=cfg.eval.seed, compute_embedding_2d=False, per_sample_outputs=False)
        uns_cfg = UnsupervisedRunConfig(data=cfg.data, scale=cfg.scale, features=cfg.features, model=cfg.model, eval=uns_eval)
        base_pipe = make_unsupervised_pipeline(uns_cfg, rngm, stream="tuning/vc")

        raw_name = self.vc.param_name
        param_name = _resolve_param_name_for_pipeline(base_pipe, raw_name)
        param_range = list(self.vc.param_range)

        predict_supported = hasattr(base_pipe, "predict")

        train_means: List[float] = []
        train_stds: List[float] = []
        val_means: List[float] = []
        val_stds: List[float] = []

        for p_i, p_val in enumerate(param_range):
            fold_train_scores: List[float] = []
            fold_val_scores: List[float] = []

            for f_i, (tr_idx, va_idx) in enumerate(cv.split(X)):
                Xtr = X[tr_idx]
                Xva = X[va_idx]

                pipe = clone(base_pipe)
                try:
                    pipe.set_params(**{param_name: p_val})
                except Exception:
                    # let sklearn-style error surface clearly
                    raise

                pipe.fit(Xtr)

                Ztr = _transform_features(pipe, Xtr)
                labels_tr = _fit_labels_for_training(pipe, Xtr)
                fold_train_scores.append(
                    _unsupervised_score_from_labels(Ztr, labels_tr, metric) if labels_tr is not None else float("nan")
                )

                if predict_supported:
                    try:
                        labels_va = np.asarray(pipe.predict(Xva))
                        Zva = _transform_features(pipe, Xva)
                        fold_val_scores.append(_unsupervised_score_from_labels(Zva, labels_va, metric))
                    except Exception:
                        fold_val_scores.append(float("nan"))
                else:
                    fold_val_scores.append(float("nan"))

            train_means.append(float(np.nanmean(fold_train_scores)))
            train_stds.append(float(np.nanstd(fold_train_scores)))
            val_means.append(float(np.nanmean(fold_val_scores)))
            val_stds.append(float(np.nanstd(fold_val_scores)))

        note = None
        if not predict_supported:
            note = "Validation scores are unavailable for this model because it does not support predicting labels for unseen samples (no predict())."

        return {
            "metric_used": metric,
            "note": note,
            "param_name": raw_name,
            "param_range": param_range,
            "train_scores_mean": _sanitize_floats(train_means),
            "train_scores_std": _sanitize_floats(train_stds),
            "val_scores_mean": _sanitize_floats(val_means),
            "val_scores_std": _sanitize_floats(val_stds),
        }


# -------------------------------
# Grid search
# -------------------------------

@dataclass
class GridSearchRunner(TuningStrategy):
    cfg: RunConfig
    gs: GridSearchConfig

    def run(self) -> Dict[str, Any]:
        cfg = self.cfg

        task = get_model_task(cfg.model)
        if task == "unsupervised":
            return self._run_unsupervised()

        eval_kind = "regression" if task == "regression" else "classification"
        scorer = make_estimator_scorer(eval_kind, cfg.eval.metric)

        loader = make_data_loader(cfg.data)
        X, y = loader.load()
        make_sanity_checker().check(X, y)

        rngm = RngManager(None if cfg.eval.seed is None else int(cfg.eval.seed))
        pipe = make_pipeline(cfg, rngm, stream="tuning/grid")

        raw_grid = self.gs.param_grid or {}
        resolved_param_grid: Dict[str, Any] = {}
        for raw_name, values in raw_grid.items():
            resolved_name = _resolve_param_name_for_pipeline(pipe, raw_name)
            resolved_param_grid[resolved_name] = values

        gs = GridSearchCV(
            estimator=pipe,
            param_grid=resolved_param_grid,
            scoring=scorer,
            cv=self.gs.cv,
            n_jobs=self.gs.n_jobs,
            refit=self.gs.refit,
            return_train_score=self.gs.return_train_score,
        )
        gs.fit(X, y)

        results = gs.cv_results_

        def _to_py(v: Any) -> Any:
            import numpy as _np
            if isinstance(v, (_np.generic,)):
                return v.item()
            if isinstance(v, _np.ndarray):
                return v.tolist()
            return v

        cv_results_sanitized = {k: [_to_py(v) for v in vs] for k, vs in results.items()}

        return {
            "metric_used": str(cfg.eval.metric),
            "best_params": gs.best_params_,
            "best_score": float(gs.best_score_),
            "best_index": int(gs.best_index_),
            "cv_results": cv_results_sanitized,
        }

    def _run_unsupervised(self) -> Dict[str, Any]:
        cfg = self.cfg
        metric = _coerce_unsupervised_metric(str(cfg.eval.metric))

        loader = make_data_loader(cfg.data)
        X, _y = loader.load()
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[0] < 2:
            raise ValueError("X must be a 2D array with at least 2 samples for unsupervised tuning.")

        rngm = RngManager(None if cfg.eval.seed is None else int(cfg.eval.seed))
        cv = _cv_for_cfg(cfg, rngm=rngm, stream="tuning/grid", force_unstratified=True)

        uns_eval = UnsupervisedEvalModel(metrics=[metric], seed=cfg.eval.seed, compute_embedding_2d=False, per_sample_outputs=False)
        uns_cfg = UnsupervisedRunConfig(data=cfg.data, scale=cfg.scale, features=cfg.features, model=cfg.model, eval=uns_eval)
        base_pipe = make_unsupervised_pipeline(uns_cfg, rngm, stream="tuning/grid")

        predict_supported = hasattr(base_pipe, "predict")

        raw_grid = self.gs.param_grid or {}
        resolved_param_grid: Dict[str, Any] = {}
        for raw_name, values in raw_grid.items():
            resolved_name = _resolve_param_name_for_pipeline(base_pipe, raw_name)
            resolved_param_grid[resolved_name] = values

        # Prepare cv_results-like dict used by the frontend panels
        cv_results: Dict[str, List[Any]] = {}
        for k in resolved_param_grid.keys():
            cv_results[f"param_{k}"] = []

        mean_train: List[float] = []
        std_train: List[float] = []
        mean_val: List[float] = []
        std_val: List[float] = []

        # Evaluate each param combination
        grid = list(ParameterGrid(resolved_param_grid)) if resolved_param_grid else [{}]

        prefers_higher = _metric_prefers_higher(metric)

        best_index = None
        best_score = None

        for i, params in enumerate(grid):
            # store param values
            for k in resolved_param_grid.keys():
                cv_results[f"param_{k}"].append(params.get(k))

            fold_train_scores: List[float] = []
            fold_val_scores: List[float] = []

            for tr_idx, va_idx in cv.split(X):
                Xtr = X[tr_idx]
                Xva = X[va_idx]

                pipe = clone(base_pipe)
                if params:
                    pipe.set_params(**params)

                pipe.fit(Xtr)

                Ztr = _transform_features(pipe, Xtr)
                labels_tr = _fit_labels_for_training(pipe, Xtr)
                fold_train_scores.append(
                    _unsupervised_score_from_labels(Ztr, labels_tr, metric) if labels_tr is not None else float("nan")
                )

                if predict_supported:
                    try:
                        labels_va = np.asarray(pipe.predict(Xva))
                        Zva = _transform_features(pipe, Xva)
                        fold_val_scores.append(_unsupervised_score_from_labels(Zva, labels_va, metric))
                    except Exception:
                        fold_val_scores.append(float("nan"))
                else:
                    fold_val_scores.append(float("nan"))

            mtr = float(np.nanmean(fold_train_scores))
            str_ = float(np.nanstd(fold_train_scores))
            mva = float(np.nanmean(fold_val_scores))
            sva = float(np.nanstd(fold_val_scores))

            mean_train.append(mtr)
            std_train.append(str_)
            mean_val.append(mva)
            std_val.append(sva)

            # choose selection score: validation when available, else train
            use_val = predict_supported and math.isfinite(mva)
            candidate = mva if use_val else mtr

            if not math.isfinite(candidate):
                continue

            if best_score is None:
                best_score = candidate
                best_index = i
            else:
                if prefers_higher:
                    if candidate > best_score:
                        best_score = candidate
                        best_index = i
                else:
                    if candidate < best_score:
                        best_score = candidate
                        best_index = i

        # cv_results arrays
        cv_results["mean_score"] = _sanitize_floats(mean_train)
        cv_results["std_score"] = _sanitize_floats(std_train)
        cv_results["mean_test_score"] = _sanitize_floats(mean_val)
        cv_results["std_test_score"] = _sanitize_floats(std_val)

        best_params = grid[best_index] if best_index is not None else {}

        note = None
        if not predict_supported:
            note = "Validation scores are unavailable for this model because it does not support predicting labels for unseen samples (no predict()). Grid search is optimized using train-side scores."

        return {
            "metric_used": metric,
            "note": note,
            "best_params": best_params,
            "best_score": float(best_score) if best_score is not None else None,
            "best_index": int(best_index) if best_index is not None else None,
            "cv_results": cv_results,
        }


# -------------------------------
# Randomized search
# -------------------------------

@dataclass
class RandomizedSearchRunner(TuningStrategy):
    cfg: RunConfig
    rs: RandomizedSearchConfig

    def run(self) -> Dict[str, Any]:
        cfg = self.cfg

        task = get_model_task(cfg.model)
        if task == "unsupervised":
            return self._run_unsupervised()

        eval_kind = "regression" if task == "regression" else "classification"
        scorer = make_estimator_scorer(eval_kind, cfg.eval.metric)

        loader = make_data_loader(cfg.data)
        X, y = loader.load()
        make_sanity_checker().check(X, y)

        rngm = RngManager(None if cfg.eval.seed is None else int(cfg.eval.seed))
        pipe = make_pipeline(cfg, rngm, stream="tuning/random")

        raw_dists = self.rs.param_distributions or {}
        resolved_param_distributions: Dict[str, Any] = {}
        for raw_name, dist in raw_dists.items():
            resolved_name = _resolve_param_name_for_pipeline(pipe, raw_name)
            resolved_param_distributions[resolved_name] = dist

        rs = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=resolved_param_distributions,
            n_iter=self.rs.n_iter,
            scoring=scorer,
            cv=self.rs.cv,
            n_jobs=self.rs.n_jobs,
            refit=self.rs.refit,
            random_state=self.rs.random_state,
            return_train_score=self.rs.return_train_score,
        )
        rs.fit(X, y)

        results = rs.cv_results_

        def _to_py(v: Any) -> Any:
            import numpy as _np
            if isinstance(v, (_np.generic,)):
                return v.item()
            if isinstance(v, _np.ndarray):
                return v.tolist()
            return v

        cv_results_sanitized = {k: [_to_py(v) for v in vs] for k, vs in results.items()}

        return {
            "metric_used": str(cfg.eval.metric),
            "best_params": rs.best_params_,
            "best_score": float(rs.best_score_),
            "best_index": int(rs.best_index_),
            "cv_results": cv_results_sanitized,
        }

    def _run_unsupervised(self) -> Dict[str, Any]:
        cfg = self.cfg
        metric = _coerce_unsupervised_metric(str(cfg.eval.metric))

        loader = make_data_loader(cfg.data)
        X, _y = loader.load()
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[0] < 2:
            raise ValueError("X must be a 2D array with at least 2 samples for unsupervised tuning.")

        rngm = RngManager(None if cfg.eval.seed is None else int(cfg.eval.seed))
        cv = _cv_for_cfg(cfg, rngm=rngm, stream="tuning/random", force_unstratified=True)

        uns_eval = UnsupervisedEvalModel(metrics=[metric], seed=cfg.eval.seed, compute_embedding_2d=False, per_sample_outputs=False)
        uns_cfg = UnsupervisedRunConfig(data=cfg.data, scale=cfg.scale, features=cfg.features, model=cfg.model, eval=uns_eval)
        base_pipe = make_unsupervised_pipeline(uns_cfg, rngm, stream="tuning/random")

        predict_supported = hasattr(base_pipe, "predict")
        prefers_higher = _metric_prefers_higher(metric)

        raw_dists = self.rs.param_distributions or {}
        resolved_param_distributions: Dict[str, Any] = {}
        for raw_name, dist in raw_dists.items():
            resolved_name = _resolve_param_name_for_pipeline(base_pipe, raw_name)
            resolved_param_distributions[resolved_name] = dist

        # sample parameter sets
        sampler = list(
            ParameterSampler(
                resolved_param_distributions,
                n_iter=self.rs.n_iter,
                random_state=self.rs.random_state if self.rs.random_state is not None else (cfg.eval.seed or None),
            )
        ) if resolved_param_distributions else [{}]

        cv_results: Dict[str, List[Any]] = {}
        for k in resolved_param_distributions.keys():
            cv_results[f"param_{k}"] = []

        mean_train: List[float] = []
        std_train: List[float] = []
        mean_val: List[float] = []
        std_val: List[float] = []

        best_index = None
        best_score = None

        for i, params in enumerate(sampler):
            for k in resolved_param_distributions.keys():
                cv_results[f"param_{k}"].append(params.get(k))

            fold_train_scores: List[float] = []
            fold_val_scores: List[float] = []

            for tr_idx, va_idx in cv.split(X):
                Xtr = X[tr_idx]
                Xva = X[va_idx]

                pipe = clone(base_pipe)
                if params:
                    pipe.set_params(**params)

                pipe.fit(Xtr)

                Ztr = _transform_features(pipe, Xtr)
                labels_tr = _fit_labels_for_training(pipe, Xtr)
                fold_train_scores.append(
                    _unsupervised_score_from_labels(Ztr, labels_tr, metric) if labels_tr is not None else float("nan")
                )

                if predict_supported:
                    try:
                        labels_va = np.asarray(pipe.predict(Xva))
                        Zva = _transform_features(pipe, Xva)
                        fold_val_scores.append(_unsupervised_score_from_labels(Zva, labels_va, metric))
                    except Exception:
                        fold_val_scores.append(float("nan"))
                else:
                    fold_val_scores.append(float("nan"))

            mtr = float(np.nanmean(fold_train_scores))
            str_ = float(np.nanstd(fold_train_scores))
            mva = float(np.nanmean(fold_val_scores))
            sva = float(np.nanstd(fold_val_scores))

            mean_train.append(mtr)
            std_train.append(str_)
            mean_val.append(mva)
            std_val.append(sva)

            use_val = predict_supported and math.isfinite(mva)
            candidate = mva if use_val else mtr
            if not math.isfinite(candidate):
                continue

            if best_score is None:
                best_score = candidate
                best_index = i
            else:
                if prefers_higher:
                    if candidate > best_score:
                        best_score = candidate
                        best_index = i
                else:
                    if candidate < best_score:
                        best_score = candidate
                        best_index = i

        cv_results["mean_score"] = _sanitize_floats(mean_train)
        cv_results["std_score"] = _sanitize_floats(std_train)
        cv_results["mean_test_score"] = _sanitize_floats(mean_val)
        cv_results["std_test_score"] = _sanitize_floats(std_val)

        best_params = sampler[best_index] if best_index is not None else {}

        note = None
        if not predict_supported:
            note = "Validation scores are unavailable for this model because it does not support predicting labels for unseen samples (no predict()). Random search is optimized using train-side scores."

        return {
            "metric_used": metric,
            "note": note,
            "best_params": best_params,
            "best_score": float(best_score) if best_score is not None else None,
            "best_index": int(best_index) if best_index is not None else None,
            "cv_results": cv_results,
        }
