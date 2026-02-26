from __future__ import annotations

"""Evaluation defaults that depend on task kind.

The UI should not invent contract defaults. The Engine owns default metrics and
exposes them through /schema/defaults.

This module is intentionally small and contract-owned.
"""

from typing import Dict, Optional

from .choices import MetricName


# NOTE: These defaults are used whenever eval.metric is omitted / None.
# They are also surfaced to the UI as eval.defaults.metric_by_task.
DEFAULT_METRIC_BY_TASK: Dict[str, MetricName] = {
    "classification": "accuracy",
    "regression": "r2",
    # Some tuning flows route unsupervised models through RunConfig + EvalModel.
    "unsupervised": "silhouette",
}


def default_metric_for_task(task: Optional[str]) -> MetricName:
    """Return the default metric for a task kind.

    Parameters
    ----------
    task:
        Expected values: 'classification' | 'regression' | 'unsupervised'.
        Unknown/None falls back to classification default.
    """

    t = str(task or "classification")
    return DEFAULT_METRIC_BY_TASK.get(t, DEFAULT_METRIC_BY_TASK["classification"])


def resolve_metric(metric: Optional[MetricName], *, task: Optional[str]) -> MetricName:
    """Return an effective metric.

    If an explicit metric override is provided, keep it.
    Otherwise, use the Engine-owned default for the given task.
    """

    return metric if metric is not None else default_metric_for_task(task)
