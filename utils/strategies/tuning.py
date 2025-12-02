from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

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
from utils.factories.data_loading_factory import make_data_loader
from utils.factories.sanity_factory import make_sanity_checker
from utils.factories.pipeline_factory import make_pipeline
from utils.permutations.rng import RngManager
from utils.postprocessing.scoring import make_estimator_scorer

from .interfaces import TuningStrategy

def _resolve_param_name_for_pipeline(pipe, raw_name: str) -> str:
  """
  Map a logical model parameter name (e.g. 'C') to the actual Pipeline
  parameter name (e.g. 'clf__C') if needed.

  - If raw_name already exists in pipe.get_params(), it is returned unchanged.
  - Otherwise we try <last_step_name>__<raw_name>.
  - If that also doesn't exist, we return raw_name and let sklearn raise.
  """
  if not raw_name:
      return raw_name

  params = pipe.get_params(deep=True)
  if raw_name in params:
      return raw_name

  # e.g. final step is 'clf', we map 'C' -> 'clf__C'
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

@dataclass
class LearningCurveRunner(TuningStrategy):
    """
    Business-logic wrapper around sklearn.model_selection.learning_curve.

    It takes a full RunConfig (data, split, scale, features, model, eval)
    plus a LearningCurveConfig and returns a dict with train/val means & stds.
    """
    cfg: RunConfig
    lc: LearningCurveConfig

    def run(self) -> Dict[str, Any]:
        cfg = self.cfg

        # Decide task kind from model (classification vs regression)
        task = get_model_task(cfg.model)
        eval_kind = "regression" if task == "regression" else "classification"
        scorer = make_estimator_scorer(eval_kind, cfg.eval.metric)

        # 1) Load & sanity-check
        loader = make_data_loader(cfg.data)
        X, y = loader.load()
        make_sanity_checker().check(X, y)

        # 2) RNG & CV
        rngm = RngManager(None if cfg.eval.seed is None else int(cfg.eval.seed))
        cv_seed = rngm.child_seed("tuning/lc/split") if cfg.split.shuffle else None
        stratified_flag = getattr(cfg.split, "stratified", True)

        if eval_kind == "classification" and stratified_flag:
            cv = StratifiedKFold(
                n_splits=cfg.split.n_splits,
                shuffle=cfg.split.shuffle,
                random_state=cv_seed,
            )
        else:
            cv = KFold(
                n_splits=cfg.split.n_splits,
                shuffle=cfg.split.shuffle,
                random_state=cv_seed,
            )

        # 3) Estimator pipeline (scale -> feat -> model)
        pipe = make_pipeline(cfg, rngm, stream="tuning/lc")

        # 4) Train sizes
        if self.lc.train_sizes is not None:
            train_sizes = np.asarray(self.lc.train_sizes)
        else:
            # default linspace in (0.1, 1.0]
            train_sizes = np.linspace(0.1, 1.0, self.lc.n_steps)

        # 5) sklearn.learning_curve
        sizes_abs, train_scores, val_scores = sk_learning_curve(
            estimator=pipe,
            X=X,
            y=y,
            train_sizes=train_sizes,
            cv=cv,
            scoring=scorer,
            n_jobs=self.lc.n_jobs,
            shuffle=False,  # shuffling handled in the CV splitter
            return_times=False,
        )

        # 6) Aggregate
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
    

@dataclass
class ValidationCurveRunner(TuningStrategy):
    """
    Wrapper around sklearn.model_selection.validation_curve.

    Varies one hyperparameter and returns train/val means & stds.
    """
    cfg: RunConfig
    vc: ValidationCurveConfig

    def run(self) -> Dict[str, Any]:
        cfg = self.cfg

        task = get_model_task(cfg.model)
        eval_kind = "regression" if task == "regression" else "classification"
        scorer = make_estimator_scorer(eval_kind, cfg.eval.metric)

        # Load & sanity-check
        loader = make_data_loader(cfg.data)
        X, y = loader.load()
        make_sanity_checker().check(X, y)

        rngm = RngManager(None if cfg.eval.seed is None else int(cfg.eval.seed))

        # We let sklearn build its internal CV from cv=int, or later you can
        # reuse your split config here as well if you want finer control.
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

@dataclass
class GridSearchRunner(TuningStrategy):
    """
    Wrap GridSearchCV over your Pipeline.

    It keeps all sklearn details inside business logic; the service just
    serializes the resulting dict.
    """
    cfg: RunConfig
    gs: GridSearchConfig

    def run(self) -> Dict[str, Any]:
        cfg = self.cfg

        task = get_model_task(cfg.model)
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

        # cv_results_ can be quite large; you can thin it later if needed.
        results = gs.cv_results_

        # Convert numpy types to plain Python types where possible
        # (services / Pydantic may handle this too).
        def _to_py(v: Any) -> Any:
            if isinstance(v, (np.generic,)):
                return v.item()
            if isinstance(v, np.ndarray):
                return v.tolist()
            return v

        cv_results_sanitized = {
            k: [_to_py(v) for v in vs] for k, vs in results.items()
        }

        return {
            "best_params": gs.best_params_,
            "best_score": float(gs.best_score_),
            "best_index": int(gs.best_index_),
            "cv_results": cv_results_sanitized,
        }

@dataclass
class RandomizedSearchRunner(TuningStrategy):
    """
    Wrap RandomizedSearchCV over your Pipeline.
    """
    cfg: RunConfig
    rs: RandomizedSearchConfig

    def run(self) -> Dict[str, Any]:
        cfg = self.cfg

        task = get_model_task(cfg.model)
        eval_kind = "regression" if task == "regression" else "classification"
        scorer = make_estimator_scorer(eval_kind, cfg.eval.metric)

        loader = make_data_loader(cfg.data)
        X, y = loader.load()
        make_sanity_checker().check(X, y)

        rngm = RngManager(None if cfg.eval.seed is None else int(cfg.eval.seed))
        pipe = make_pipeline(cfg, rngm, stream="tuning/random")

        # Resolve logical param names (e.g. "C") to pipeline names (e.g. "clf__C")
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
            if isinstance(v, (np.generic,)):
                return v.item()
            if isinstance(v, np.ndarray):
                return v.tolist()
            return v

        cv_results_sanitized = {
            k: [_to_py(v) for v in vs] for k, vs in results.items()
        }

        return {
            "best_params": rs.best_params_,
            "best_score": float(rs.best_score_),
            "best_index": int(rs.best_index_),
            "cv_results": cv_results_sanitized,
        }