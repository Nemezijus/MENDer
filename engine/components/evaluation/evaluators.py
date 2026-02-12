from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence
import numpy as np

from engine.contracts.eval_configs import EvalModel
from engine.contracts.unsupervised_configs import UnsupervisedEvalModel
from engine.components.interfaces import Evaluator, UnsupervisedEvaluator
from engine.components.evaluation.scoring import score as score_fn
from engine.components.evaluation.unsupervised_scoring import compute_unsupervised_metrics
from engine.reporting.diagnostics.clustering_diagnostics import (
    cluster_summary,
    embedding_2d,
    model_diagnostics,
    per_sample_outputs,
    build_plot_data,
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

        # Plot payloads (frontend visualizations)
        plot_data: Dict[str, Any] = {}
        try:
            pd, w = build_plot_data(
                model=model,
                X=X,
                labels=labels,
                per_sample=per_sample,
                embedding=emb,
                seed=int(self.cfg.seed or 0),
            )
            plot_data = dict(pd)
            warnings.extend(w)
            # Attach embedding labels for coloring (aligned to emb.idx)
            if emb is not None and plot_data.get("embedding_labels") is not None:
                emb = dict(emb)
                emb["label"] = [int(v) for v in plot_data.get("embedding_labels", [])]
        except Exception as e:
            plot_data = {}
            warnings.append(f"build_plot_data failed: {type(e).__name__}: {e}")

        return {
            "metrics": metrics,
            "cluster_summary": summary,
            "model_diagnostics": diag,
            "embedding_2d": emb,
            "plot_data": plot_data,
            "per_sample": per_sample,
            "warnings": warnings,
        }
