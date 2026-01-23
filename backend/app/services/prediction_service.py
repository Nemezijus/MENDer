from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from utils.factories.predict_factory import make_predictor
from utils.factories.eval_factory import make_evaluator
from utils.factories.export_factory import make_exporter
from utils.predicting.prediction_results import build_prediction_table
from utils.io.export.result_export import ExportResult
from utils.persistence.eval_outputs_cache import eval_outputs_cache

from .predictions.helpers import (
    build_preview_rows,
    safe_float_optional,
    setup_prediction,
)
from .predictions.decoder_payloads import (
    add_decoder_outputs_preview,
    maybe_merge_decoder_into_export_table,
)


def apply_model_to_arrays(
    *,
    artifact_uid: str,
    artifact_meta: Any,
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    max_preview_rows: int = 100,
) -> Dict[str, Any]:
    """Apply a cached model pipeline to (X, optional y) and return preview + metric."""
    pipeline, X_arr, y_arr, task, ev, eval_kind = setup_prediction(
        artifact_uid=artifact_uid,
        artifact_meta=artifact_meta,
        X=X,
        y=y,
    )

    predictor = make_predictor()
    y_pred = predictor.predict(pipeline, X_arr)
    y_pred = np.asarray(y_pred).ravel()

    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    if y_arr is not None:
        evaluator = make_evaluator(ev, kind=eval_kind)
        metric_name = ev.metric
        try:
            metric_value = safe_float_optional(evaluator.score(y_arr, y_pred))
        except Exception:
            metric_value = None

    n_samples = int(X_arr.shape[0])
    n_features = int(X_arr.shape[1])

    n_preview = min(max_preview_rows, n_samples)
    indices = list(range(n_preview))
    y_true_preview = y_arr[:n_preview] if y_arr is not None else None
    y_pred_preview = y_pred[:n_preview]

    preview_rows = build_preview_rows(
        indices=indices,
        y_pred=y_pred_preview,
        task=eval_kind,
        y_true=y_true_preview,
    )

    result: Dict[str, Any] = {
        "n_samples": n_samples,
        "n_features": n_features,
        "task": task,
        "has_labels": bool(y_arr is not None),
        "metric_name": metric_name,
        "metric_value": metric_value,
        "preview": preview_rows,
        "notes": [],
    }

    result["notes"].append(f"Task inferred from artifact kind: {task}.")
    if metric_name:
        result["notes"].append(
            f"Evaluation metric on this dataset: {metric_name} "
            "(only defined if labels were provided)."
        )

    # Optional decoder outputs (classification-only)
    add_decoder_outputs_preview(
        result=result,
        pipeline=pipeline,
        X_arr=X_arr,
        y_arr=y_arr,
        ev=ev,
        eval_kind=eval_kind,
        max_preview_rows=max_preview_rows,
    )

    return result


def export_predictions_to_csv(
    *,
    artifact_uid: str,
    artifact_meta: Any,
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    filename: Optional[str] = None,
) -> ExportResult:
    """Apply a cached model pipeline to X (and optional y) and export FULL prediction table as CSV."""
    pipeline, X_arr, y_arr, task, ev, eval_kind = setup_prediction(
        artifact_uid=artifact_uid,
        artifact_meta=artifact_meta,
        X=X,
        y=y,
    )

    predictor = make_predictor()
    y_pred = predictor.predict(pipeline, X_arr)
    y_pred = np.asarray(y_pred).ravel()

    n_samples = int(X_arr.shape[0])

    table = build_prediction_table(
        indices=range(n_samples),
        y_pred=y_pred,
        task=eval_kind,
        y_true=y_arr,
        max_rows=None,
    )

    table = maybe_merge_decoder_into_export_table(
        table=table,
        pipeline=pipeline,
        X_arr=X_arr,
        y_arr=y_arr,
        ev=ev,
        eval_kind=eval_kind,
    )

    exporter = make_exporter("csv")
    export_result = exporter.export(
        table,
        dest=None,
        filename=filename,
    )
    return export_result


def export_decoder_outputs_to_csv(
    *,
    artifact_uid: str,
    filename: Optional[str] = None,
) -> ExportResult:
    """Export cached evaluation outputs (decoder table) as CSV.

    Note: this exports what was cached at training time (OOF pooled for kfold, test split for holdout).
    """
    cached = eval_outputs_cache.get(artifact_uid)
    if cached is None:
        raise ValueError(
            "No cached evaluation outputs available for this artifact. "
            "Re-run training with decoder/regression results enabled."
        )

    y_pred = np.asarray(cached.y_pred).ravel()
    y_true = None if cached.y_true is None else np.asarray(cached.y_true).ravel()
    indices = cached.indices
    if indices is None:
        indices = np.arange(int(y_pred.shape[0]), dtype=int)

    table = build_prediction_table(
        indices=indices,
        y_pred=y_pred,
        task=cached.task,
        y_true=y_true,
        max_rows=None,
    )

    # Attach fold_id if available (useful for CV pooled exports)
    if cached.fold_ids is not None:
        fold_ids = np.asarray(cached.fold_ids).ravel()
        for i, r in enumerate(table):
            if i >= fold_ids.shape[0]:
                break
            try:
                r["fold_id"] = int(fold_ids[i])
            except Exception:
                pass

    exporter = make_exporter("csv")
    export_result = exporter.export(
        table,
        dest=None,
        filename=filename,
    )
    return export_result
