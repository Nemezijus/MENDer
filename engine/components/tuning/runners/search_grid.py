from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from sklearn.model_selection import GridSearchCV

from engine.contracts.model_configs import get_model_task
from engine.contracts.run_config import RunConfig
from engine.contracts.tuning_configs import GridSearchConfig
from engine.contracts.unsupervised_configs import UnsupervisedRunConfig, UnsupervisedEvalModel
from engine.contracts.results import GridSearchResult

from engine.factories.data_loading_factory import make_data_loader
from engine.factories.pipeline_factory import make_pipeline, make_unsupervised_pipeline
from engine.factories.sanity_factory import make_sanity_checker
from engine.runtime.random.rng import RngManager
from engine.components.evaluation.scoring import make_estimator_scorer
from engine.components.tuning.unsupervised import UnsupervisedGridSearchCV

from ...interfaces import TuningStrategy
from .common import coerce_unsupervised_metric, resolve_param_name_for_pipeline


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
            resolved_name = resolve_param_name_for_pipeline(pipe, raw_name)
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
        metric = coerce_unsupervised_metric(str(cfg.eval.metric))

        loader = make_data_loader(cfg.data)
        X, _y = loader.load()
        X = np.asarray(X)

        rngm = RngManager(None if cfg.eval.seed is None else int(cfg.eval.seed))

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
            resolved_name = resolve_param_name_for_pipeline(pipe, raw_name)
            resolved_param_grid[resolved_name] = values

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
