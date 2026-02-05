from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
import math
import numpy as np
from sklearn.model_selection import (
    StratifiedKFold,
    KFold,
    learning_curve as sk_learning_curve,
    validation_curve as sk_validation_curve,
    GridSearchCV,
    RandomizedSearchCV,
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

from engine.contracts.results import (
    LearningCurveResult,
    ValidationCurveResult,
    GridSearchResult,
    RandomSearchResult,
)

from utils.factories.data_loading_factory import make_data_loader
from utils.factories.sanity_factory import make_sanity_checker
from utils.factories.pipeline_factory import make_pipeline, make_unsupervised_pipeline
from utils.permutations.rng import RngManager
from engine.components.evaluation.scoring import make_estimator_scorer
from utils.processing.unsupervised_curves import (
    compute_unsupervised_learning_curve,
    compute_unsupervised_validation_curve,
)
from utils.processing.unsupervised_searchcv import (
    UnsupervisedGridSearchCV,
    UnsupervisedRandomizedSearchCV,
)

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
            return LearningCurveResult.model_validate(self._run_unsupervised())

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

        payload = {
            "train_sizes": sizes_abs.tolist(),
            "train_scores_mean": _sanitize_floats(train_mean),
            "train_scores_std": _sanitize_floats(train_std),
            "val_scores_mean": _sanitize_floats(val_mean),
            "val_scores_std": _sanitize_floats(val_std),
        }
        return LearningCurveResult.model_validate(payload)

    def _run_unsupervised(self) -> Dict[str, Any]:
        cfg = self.cfg
        metric = _coerce_unsupervised_metric(str(cfg.eval.metric))

        loader = make_data_loader(cfg.data)
        X, _y = loader.load()
        X = np.asarray(X)

        rngm = RngManager(None if cfg.eval.seed is None else int(cfg.eval.seed))
        cv = _cv_for_cfg(cfg, rngm=rngm, stream="tuning/lc", force_unstratified=True)

        # Build the same unsupervised pipeline used elsewhere (preprocessing + clustering).
        uns_eval = UnsupervisedEvalModel(
            metrics=[metric],
            seed=cfg.eval.seed,
            compute_embedding_2d=False,
            per_sample_outputs=False,
        )
        uns_cfg = UnsupervisedRunConfig(
            data=cfg.data,
            scale=cfg.scale,
            features=cfg.features,
            model=cfg.model,
            eval=uns_eval,
        )
        pipe = make_unsupervised_pipeline(uns_cfg, rngm, stream="tuning/lc")

        # Delegate implementation to utils/processing.
        return compute_unsupervised_learning_curve(
            estimator=pipe,
            X=X,
            metric=metric,
            cv=cv,
            train_sizes=self.lc.train_sizes,
            n_steps=self.lc.n_steps,
            n_jobs=self.lc.n_jobs,
            shuffle=bool(cfg.split.shuffle),
            random_state=rngm.child_seed("tuning/lc/shuffle") if cfg.split.shuffle else None,
        )


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
            return ValidationCurveResult.model_validate(self._run_unsupervised())

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

        payload = {
            "param_name": raw_name,
            "param_range": list(param_range),
            "train_scores_mean": _sanitize_floats(train_mean),
            "train_scores_std": _sanitize_floats(train_std),
            "val_scores_mean": _sanitize_floats(val_mean),
            "val_scores_std": _sanitize_floats(val_std),
        }
        return ValidationCurveResult.model_validate(payload)

    def _run_unsupervised(self) -> Dict[str, Any]:
        cfg = self.cfg
        metric = _coerce_unsupervised_metric(str(cfg.eval.metric))

        loader = make_data_loader(cfg.data)
        X, _y = loader.load()
        X = np.asarray(X)

        rngm = RngManager(None if cfg.eval.seed is None else int(cfg.eval.seed))
        cv = _cv_for_cfg(cfg, rngm=rngm, stream="tuning/vc", force_unstratified=True)

        uns_eval = UnsupervisedEvalModel(
            metrics=[metric],
            seed=cfg.eval.seed,
            compute_embedding_2d=False,
            per_sample_outputs=False,
        )
        uns_cfg = UnsupervisedRunConfig(
            data=cfg.data,
            scale=cfg.scale,
            features=cfg.features,
            model=cfg.model,
            eval=uns_eval,
        )
        pipe = make_unsupervised_pipeline(uns_cfg, rngm, stream="tuning/vc")

        raw_name = self.vc.param_name
        param_name = _resolve_param_name_for_pipeline(pipe, raw_name)

        # Delegate heavy lifting to utils/processing.
        res = compute_unsupervised_validation_curve(
            estimator=pipe,
            X=X,
            metric=metric,
            cv=cv,
            param_name=param_name,
            param_range=list(self.vc.param_range),
            n_jobs=self.vc.n_jobs,
        )

        # Keep the external-facing param_name consistent with the request.
        res["param_name"] = raw_name
        res["param_range"] = list(self.vc.param_range)
        return res


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
            return GridSearchResult.model_validate(self._run_unsupervised())

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

            if isinstance(v, (np.generic,)):
                return v.item()
            if isinstance(v, np.ndarray):
                return v.tolist()
            return v

        cv_results_sanitized = {k: [_to_py(v) for v in vs] for k, vs in results.items()}

        payload = {
            "metric_used": str(cfg.eval.metric),
            "best_params": gs.best_params_,
            "best_score": float(gs.best_score_),
            "best_index": int(gs.best_index_),
            "cv_results": cv_results_sanitized,
        }
        return GridSearchResult.model_validate(payload)

    def _run_unsupervised(self) -> Dict[str, Any]:
        cfg = self.cfg
        metric = _coerce_unsupervised_metric(str(cfg.eval.metric))

        loader = make_data_loader(cfg.data)
        X, _y = loader.load()
        X = np.asarray(X)

        rngm = RngManager(None if cfg.eval.seed is None else int(cfg.eval.seed))

        # Build the unsupervised pipeline (preprocessing + clustering).
        uns_eval = UnsupervisedEvalModel(
            metrics=[metric],
            seed=cfg.eval.seed,
            compute_embedding_2d=False,
            per_sample_outputs=False,
        )
        uns_cfg = UnsupervisedRunConfig(
            data=cfg.data,
            scale=cfg.scale,
            features=cfg.features,
            model=cfg.model,
            eval=uns_eval,
        )
        pipe = make_unsupervised_pipeline(uns_cfg, rngm, stream="tuning/grid")

        raw_grid = self.gs.param_grid or {}
        resolved_param_grid: Dict[str, Any] = {}
        for raw_name, values in raw_grid.items():
            resolved_name = _resolve_param_name_for_pipeline(pipe, raw_name)
            resolved_param_grid[resolved_name] = values

        # Delegate the SearchCV-like implementation.
        search = UnsupervisedGridSearchCV(
            estimator=pipe,
            param_grid=resolved_param_grid,
            metric=metric,
            cv=self.gs.cv,
            refit=self.gs.refit,
            return_train_score=self.gs.return_train_score,
            shuffle=bool(cfg.split.shuffle),
            random_state=(rngm.child_seed("tuning/grid/cv") if cfg.split.shuffle else None),
        )
        search.fit(X)

        return {
            "metric_used": metric,
            "note": search.note_,
            "best_params": search.best_params_ or {},
            "best_score": search.best_score_,
            "best_index": search.best_index_,
            "cv_results": search.cv_results_ or {},
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
            return RandomSearchResult.model_validate(self._run_unsupervised())

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

        payload = {
            "metric_used": str(cfg.eval.metric),
            "best_params": rs.best_params_,
            "best_score": float(rs.best_score_),
            "best_index": int(rs.best_index_),
            "cv_results": cv_results_sanitized,
        }
        return RandomSearchResult.model_validate(payload)

    def _run_unsupervised(self) -> Dict[str, Any]:
        cfg = self.cfg
        metric = _coerce_unsupervised_metric(str(cfg.eval.metric))

        loader = make_data_loader(cfg.data)
        X, _y = loader.load()
        X = np.asarray(X)

        rngm = RngManager(None if cfg.eval.seed is None else int(cfg.eval.seed))

        # Build the unsupervised pipeline (preprocessing + clustering).
        uns_eval = UnsupervisedEvalModel(
            metrics=[metric],
            seed=cfg.eval.seed,
            compute_embedding_2d=False,
            per_sample_outputs=False,
        )
        uns_cfg = UnsupervisedRunConfig(
            data=cfg.data,
            scale=cfg.scale,
            features=cfg.features,
            model=cfg.model,
            eval=uns_eval,
        )
        pipe = make_unsupervised_pipeline(uns_cfg, rngm, stream="tuning/random")

        raw_dists = self.rs.param_distributions or {}
        resolved_param_distributions: Dict[str, Any] = {}
        for raw_name, dist in raw_dists.items():
            resolved_name = _resolve_param_name_for_pipeline(pipe, raw_name)
            resolved_param_distributions[resolved_name] = dist

        # Delegate the SearchCV-like implementation.
        search = UnsupervisedRandomizedSearchCV(
            estimator=pipe,
            param_distributions=resolved_param_distributions,
            n_iter=self.rs.n_iter,
            metric=metric,
            cv=self.rs.cv,
            refit=self.rs.refit,
            return_train_score=self.rs.return_train_score,
            random_state=(self.rs.random_state if self.rs.random_state is not None else cfg.eval.seed),
            shuffle=bool(cfg.split.shuffle),
            cv_random_state=(rngm.child_seed("tuning/random/cv") if cfg.split.shuffle else None),
        )
        search.fit(X)

        return {
            "metric_used": metric,
            "note": search.note_,
            "best_params": search.best_params_ or {},
            "best_score": search.best_score_,
            "best_index": search.best_index_,
            "cv_results": search.cv_results_ or {},
        }
