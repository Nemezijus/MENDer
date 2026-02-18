from __future__ import annotations

"""Label-shuffle baseline orchestration.

This module owns orchestration for the label-shuffle baseline:
- splitter construction (holdout vs k-fold)
- pipeline construction via pipeline_factory
- evaluation (including proba/decision-score handling)

The baseline *component* (:class:`engine.components.baselines.baselines.LabelShuffleBaseline`)
handles only the baseline-specific mechanics (shuffling labels N times).
"""

from typing import Optional

import numpy as np

from engine.components.baselines.baselines import LabelShuffleBaseline
from engine.components.evaluation.scoring import PROBA_METRICS
from engine.contracts.model_configs import get_model_task
from engine.contracts.run_config import RunConfig
from engine.core.progress import ProgressCallback
from engine.factories.eval_factory import make_evaluator
from engine.factories.pipeline_factory import make_pipeline
from engine.factories.split_factory import make_splitter
from engine.core.random.rng import RngManager


def _eval_kind(cfg: RunConfig) -> str:
    """Return "classification" or "regression" based on model task metadata."""
    task = get_model_task(cfg.model)
    return "regression" if task == "regression" else "classification"


def _score_holdout(
    *,
    cfg: RunConfig,
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    eval_kind: str,
) -> float:
    splitter = make_splitter(cfg.split, seed=seed)
    split = next(splitter.split(X, y))

    rngm = RngManager(int(seed))
    pipeline = make_pipeline(cfg, rngm, stream="baseline/holdout")
    pipeline.fit(split.Xtr, split.ytr)
    y_pred = pipeline.predict(split.Xte)

    evaluator = make_evaluator(cfg.eval, kind=eval_kind)

    y_proba = None
    y_score = None
    if eval_kind == "classification":
        if hasattr(pipeline, "predict_proba"):
            try:
                y_proba = pipeline.predict_proba(split.Xte)
            except Exception:
                y_proba = None
        if hasattr(pipeline, "decision_function"):
            try:
                y_score = pipeline.decision_function(split.Xte)
            except Exception:
                y_score = None

        if cfg.eval.metric in PROBA_METRICS and y_proba is None and y_score is None:
            raise ValueError(
                f"Metric '{cfg.eval.metric}' requires predict_proba or decision_function, "
                f"but estimator {type(pipeline).__name__} has neither."
            )

    return float(evaluator.score(split.yte, y_pred=y_pred, y_proba=y_proba, y_score=y_score))


def _score_kfold(
    *,
    cfg: RunConfig,
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    eval_kind: str,
) -> float:
    splitter = make_splitter(cfg.split, seed=seed)
    evaluator = make_evaluator(cfg.eval, kind=eval_kind)

    rngm = RngManager(int(seed))
    fold_scores: list[float] = []

    for fold_id, split in enumerate(splitter.split(X, y), start=1):
        pipeline = make_pipeline(cfg, rngm, stream=f"baseline/cv/fold{fold_id}")
        pipeline.fit(split.Xtr, split.ytr)
        y_pred = pipeline.predict(split.Xte)

        y_proba = None
        y_score = None
        if eval_kind == "classification":
            if hasattr(pipeline, "predict_proba"):
                try:
                    y_proba = pipeline.predict_proba(split.Xte)
                except Exception:
                    y_proba = None
            if hasattr(pipeline, "decision_function"):
                try:
                    y_score = pipeline.decision_function(split.Xte)
                except Exception:
                    y_score = None

            if cfg.eval.metric in PROBA_METRICS and y_proba is None and y_score is None:
                raise ValueError(
                    f"Metric '{cfg.eval.metric}' requires predict_proba or decision_function, "
                    f"but estimator {type(pipeline).__name__} has neither."
                )

        fold_scores.append(
            float(evaluator.score(split.yte, y_pred=y_pred, y_proba=y_proba, y_score=y_score))
        )

    return float(np.mean(fold_scores)) if fold_scores else float("nan")


def make_label_shuffle_baseline(cfg: RunConfig, rngm: RngManager) -> LabelShuffleBaseline:
    """Assemble a label-shuffle baseline runner.

    Returns a *component* baseline runner configured with a use-case-owned scorer.
    """

    eval_kind = _eval_kind(cfg)
    use_kfold = getattr(cfg.split, "mode", "").lower() == "kfold"

    def score_once(X: np.ndarray, y_shuf: np.ndarray, seed: int) -> float:
        if use_kfold:
            return _score_kfold(cfg=cfg, X=X, y=y_shuf, seed=seed, eval_kind=eval_kind)
        return _score_holdout(cfg=cfg, X=X, y=y_shuf, seed=seed, eval_kind=eval_kind)

    return LabelShuffleBaseline(
        rngm=rngm,
        score_once=score_once,
        n_shuffles_default=int(getattr(cfg.eval, "n_shuffles", 0) or 0),
    )


def run_label_shuffle_baseline(
    *,
    cfg: RunConfig,
    X: np.ndarray,
    y: np.ndarray,
    rngm: RngManager,
    n_shuffles: Optional[int] = None,
    progress: Optional[ProgressCallback] = None,
) -> np.ndarray:
    """Execute the label-shuffle baseline and return the score distribution."""

    runner = make_label_shuffle_baseline(cfg, rngm)
    return np.asarray(runner.run(X, y, n_shuffles=n_shuffles, progress=progress), dtype=float)
