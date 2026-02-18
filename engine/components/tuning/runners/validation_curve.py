from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from sklearn.model_selection import validation_curve as sk_validation_curve

from engine.contracts.model_configs import get_model_task
from engine.contracts.run_config import RunConfig
from engine.contracts.tuning_configs import ValidationCurveConfig
from engine.contracts.unsupervised_configs import UnsupervisedRunConfig, UnsupervisedEvalModel
from engine.contracts.results import ValidationCurveResult

from engine.factories.data_loading_factory import make_data_loader
from engine.factories.pipeline_factory import make_pipeline, make_unsupervised_pipeline
from engine.factories.sanity_factory import make_sanity_checker
from engine.core.random.rng import RngManager
from engine.components.evaluation.scoring import make_estimator_scorer
from engine.components.tuning.unsupervised import compute_unsupervised_validation_curve

from ...interfaces import TuningStrategy
from .common import coerce_unsupervised_metric, cv_for_cfg, resolve_param_name_for_pipeline, sanitize_floats


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
        param_name = resolve_param_name_for_pipeline(pipe, raw_name)
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
            "train_scores_mean": sanitize_floats(train_mean),
            "train_scores_std": sanitize_floats(train_std),
            "val_scores_mean": sanitize_floats(val_mean),
            "val_scores_std": sanitize_floats(val_std),
        }
        return ValidationCurveResult.model_validate(payload)

    def _run_unsupervised(self) -> Dict[str, Any]:
        cfg = self.cfg
        metric = coerce_unsupervised_metric(str(cfg.eval.metric))

        loader = make_data_loader(cfg.data)
        X, _y = loader.load()
        X = np.asarray(X)

        rngm = RngManager(None if cfg.eval.seed is None else int(cfg.eval.seed))
        cv = cv_for_cfg(cfg, rngm=rngm, stream="tuning/vc", force_unstratified=True)

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
        param_name = resolve_param_name_for_pipeline(pipe, raw_name)

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
