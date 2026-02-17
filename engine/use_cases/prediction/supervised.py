from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from engine.contracts.eval_configs import EvalModel
from engine.contracts.results.prediction import PredictionResult
from engine.reporting.prediction.prediction_results import build_prediction_table

from engine.use_cases.prediction.decoder import maybe_compute_decoder_outputs
from engine.use_cases.prediction.scoring import score_if_possible


def apply_supervised(
    *,
    task: str,
    pipeline: Any,
    X_arr: np.ndarray,
    y: Optional[Any],
    eval_model: Optional[EvalModel],
    max_preview_rows: int,
) -> PredictionResult:
    """Apply a supervised pipeline to X (optionally with labels y) and return PredictionResult."""

    n_samples = int(X_arr.shape[0])
    n_features = int(X_arr.shape[1])

    y_arr: Optional[np.ndarray] = None
    has_labels = y is not None
    if y is not None:
        y_arr = np.asarray(y).reshape(-1)
        if int(y_arr.shape[0]) != n_samples:
            raise ValueError(f"y has {int(y_arr.shape[0])} samples but X has {n_samples}.")

    notes: list[str] = []

    y_pred = np.asarray(pipeline.predict(X_arr)).reshape(-1)

    metric_name, metric_value = score_if_possible(
        task=task,
        eval_model=eval_model,
        pipeline=pipeline,
        X_arr=X_arr,
        y_true=y_arr,
        y_pred=y_pred,
        notes=notes,
    )

    # Preview table
    n_preview = min(int(max_preview_rows), n_samples)
    preview_rows = build_prediction_table(
        indices=range(n_samples),
        y_pred=y_pred,
        y_true=y_arr,
        task="regression" if task == "regression" else "classification",
        max_rows=n_preview,
    )

    decoder_outputs = maybe_compute_decoder_outputs(
        task=task,
        eval_model=eval_model,
        pipeline=pipeline,
        X_arr=X_arr,
        y_true=y_arr,
        n_samples=n_samples,
        preview_rows_cap=n_preview,
        notes=notes,
    )

    payload: Dict[str, Any] = {
        "n_samples": n_samples,
        "n_features": n_features,
        "task": task,
        "has_labels": bool(has_labels),
        "metric_name": metric_name,
        "metric_value": metric_value,
        "preview": preview_rows,
        "notes": notes,
        "decoder_outputs": decoder_outputs,
    }

    return PredictionResult.model_validate(payload)
