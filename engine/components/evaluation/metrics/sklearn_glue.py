from __future__ import annotations

from typing import Any

import numpy as np

from engine.components.evaluation.unsupervised_scoring import compute_unsupervised_metrics

from .helpers import _as_1d
from .registry import PROBA_METRICS, score


def make_estimator_scorer(kind: str, metric: str):
    """Return a callable(estimator, X, y) suitable for sklearn scoring APIs."""

    def _scorer(estimator: Any, X: Any, y=None):
        if y is None and kind in {"classification", "regression"}:
            raise ValueError(
                f"y is required for {kind} scoring, but got y=None (metric={metric!r})."
            )

        if kind == "classification":
            y_true = _as_1d(y)

            if metric in PROBA_METRICS:
                if hasattr(estimator, "predict_proba"):
                    y_proba = estimator.predict_proba(X)
                    return score(y_true, kind="classification", metric=metric, y_proba=y_proba)
                if hasattr(estimator, "decision_function"):
                    y_score = estimator.decision_function(X)
                    return score(y_true, kind="classification", metric=metric, y_score=y_score)
                raise ValueError(
                    f"Metric '{metric}' requires predict_proba or decision_function "
                    f"but estimator {type(estimator).__name__} has neither."
                )

            y_pred = estimator.predict(X)
            return score(y_true, y_pred, kind="classification", metric=metric)

        if kind == "unsupervised":
            # Unsupervised metrics ignore y.
            try:
                final_est = None
                try:
                    if hasattr(estimator, "steps") and estimator.steps:
                        final_est = estimator.steps[-1][1]
                except Exception:
                    final_est = None

                labels = (
                    getattr(final_est, "labels_", None)
                    if final_est is not None
                    else getattr(estimator, "labels_", None)
                )

                if labels is not None:
                    labels = np.asarray(labels)
                    if labels.shape[0] == np.asarray(X).shape[0]:
                        try:
                            Z = estimator[:-1].transform(X)
                        except Exception:
                            Z = np.asarray(X)
                        metrics, _warnings = compute_unsupervised_metrics(Z, labels, [metric])
                        v = metrics.get(metric)
                        return float(v) if v is not None else float("nan")

                if hasattr(estimator, "predict"):
                    labels = estimator.predict(X)
                    try:
                        Z = estimator[:-1].transform(X)
                    except Exception:
                        Z = np.asarray(X)
                    metrics, _warnings = compute_unsupervised_metrics(Z, labels, [metric])
                    v = metrics.get(metric)
                    return float(v) if v is not None else float("nan")

                return float("nan")
            except Exception:
                return float("nan")

        if kind == "regression":
            y_true = _as_1d(y)
            y_pred = estimator.predict(X)
            return score(y_true, y_pred, kind="regression", metric=metric)

        raise ValueError(f"Unsupported kind in make_estimator_scorer: {kind!r}")

    return _scorer
