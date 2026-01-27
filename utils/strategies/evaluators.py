from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence
import numpy as np

from shared_schemas.eval_configs import EvalModel
from shared_schemas.unsupervised_configs import UnsupervisedEvalModel
from utils.strategies.interfaces import Evaluator, UnsupervisedEvaluator
from utils.postprocessing.scoring import score as score_fn
from utils.postprocessing.unsupervised_scoring import compute_unsupervised_metrics
from utils.postprocessing.clustering_diagnostics import (
    cluster_summary,
    embedding_2d,
    model_diagnostics,
    per_sample_outputs,
)

@dataclass
class SklearnEvaluator(Evaluator):
    """
    Wraps your existing scoring functions.
    - Uses Evalmodel.metric by default.
    - Supports both hard-label and probabilistic metrics via optional args.
    """
    cfg: EvalModel
    kind: str = "classification"   # or "regression" (you can override per use-case)

    def score(
        self,
        y_true: np.ndarray,
        y_pred: Optional[np.ndarray] = None,
        *,
        y_proba: Optional[np.ndarray] = None,
        y_score: Optional[np.ndarray] = None,
        labels: Optional[Sequence] = None,
    ) -> float:
        return score_fn(
            y_true,
            y_pred,
            kind=self.kind,                 # explicit kind to avoid heuristic surprises
            metric=self.cfg.metric,         # default metric from config
            y_proba=y_proba,
            y_score=y_score,
            labels=labels,
        )


@dataclass
class SklearnUnsupervisedEvaluator(UnsupervisedEvaluator):
    """Compute unsupervised (clustering) diagnostics.

    This strategy deliberately does not depend on backend/frontend...
    """

    cfg: UnsupervisedEvalModel

    def evaluate(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        *,
        model: Optional[Any] = None,
    ) -> Dict[str, Any]:
        warnings: list[str] = []

        # Global (scalar) metrics
        metrics_arg = self.cfg.metrics if self.cfg.metrics else None
        metrics, w = compute_unsupervised_metrics(np.asarray(X), np.asarray(labels), metrics=metrics_arg)
        warnings.extend(w)

        # Basic summary
        summary = dict(cluster_summary(labels))

        # Model-specific diagnostics
        diag: Dict[str, Any] = {}
        if model is not None:
            d, w = model_diagnostics(model, X, labels)
            diag = dict(d)
            warnings.extend(w)

        # Optional 2D embedding
        emb = None
        if self.cfg.compute_embedding_2d:
            emb, w = embedding_2d(X, method=self.cfg.embedding_method, seed=int(self.cfg.seed or 0))
            warnings.extend(w)

        # Per-sample outputs (exportable)
        per_sample: Dict[str, Any] = {"cluster_id": [int(v) for v in np.asarray(labels).reshape(-1).tolist()]}
        if self.cfg.per_sample_outputs:
            if model is None:
                per_sample["is_noise"] = [bool(int(v) == -1) for v in np.asarray(labels).reshape(-1).tolist()]
            else:
                p, w = per_sample_outputs(
                    model,
                    X,
                    labels,
                    include_cluster_probabilities=bool(self.cfg.include_cluster_probabilities),
                )
                per_sample = dict(p)
                warnings.extend(w)

        return {
            "metrics": metrics,
            "cluster_summary": summary,
            "model_diagnostics": diag,
            "embedding_2d": emb,
            "per_sample": per_sample,
            "warnings": warnings,
        }
