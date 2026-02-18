from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from sklearn.model_selection import learning_curve as sk_learning_curve

from engine.contracts.model_configs import get_model_task
from engine.contracts.run_config import RunConfig
from engine.contracts.tuning_configs import LearningCurveConfig
from engine.contracts.unsupervised_configs import UnsupervisedRunConfig, UnsupervisedEvalModel
from engine.contracts.results import LearningCurveResult

from engine.factories.data_loading_factory import make_data_loader
from engine.factories.pipeline_factory import make_pipeline, make_unsupervised_pipeline
from engine.factories.sanity_factory import make_sanity_checker
from engine.core.random.rng import RngManager
from engine.components.evaluation.scoring import make_estimator_scorer
from engine.components.tuning.unsupervised import compute_unsupervised_learning_curve

from ...interfaces import TuningStrategy
from .common import coerce_unsupervised_metric, cv_for_cfg, sanitize_floats


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
        cv = cv_for_cfg(cfg, rngm=rngm, stream="tuning/lc", force_unstratified=(eval_kind != "classification"))

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
            "train_scores_mean": sanitize_floats(train_mean),
            "train_scores_std": sanitize_floats(train_std),
            "val_scores_mean": sanitize_floats(val_mean),
            "val_scores_std": sanitize_floats(val_std),
        }
        return LearningCurveResult.model_validate(payload)

    def _run_unsupervised(self) -> Dict[str, Any]:
        cfg = self.cfg
        metric = coerce_unsupervised_metric(str(cfg.eval.metric))

        loader = make_data_loader(cfg.data)
        X, _y = loader.load()
        X = np.asarray(X)

        rngm = RngManager(None if cfg.eval.seed is None else int(cfg.eval.seed))
        cv = cv_for_cfg(cfg, rngm=rngm, stream="tuning/lc", force_unstratified=True)

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
