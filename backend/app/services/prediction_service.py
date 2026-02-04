from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from utils.factories.predict_factory import make_predictor
from utils.factories.eval_factory import make_evaluator
from utils.factories.export_factory import make_exporter
from utils.predicting.prediction_results import build_prediction_table
from utils.io.export.result_export import ExportResult
from utils.persistence.eval_outputs_cache import eval_outputs_cache
from utils.persistence.artifact_cache import artifact_cache

from .predictions.helpers import (
    build_preview_rows,
    safe_float_optional,
    setup_prediction_common,
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
    """Apply a cached model pipeline to (X, optional y) and return preview + metric.

    Notes
    -----
    - For supervised artifacts (classification/regression), this returns a PredictionRow preview.
    - For unsupervised artifacts, this returns an UnsupervisedPredictionRow preview (cluster_id per sample).
    """

    task_kind = getattr(artifact_meta, "kind", None) or "classification"
    if task_kind == "unsupervised":
        # Unsupervised apply: y is ignored; requires predict-capable estimator.
        pipeline, X_arr, _task = setup_prediction_common(
            artifact_uid=artifact_uid,
            artifact_meta=artifact_meta,
            X=X,
        )

        # For unsupervised apply we require predict().
        try:
            cluster_id = pipeline.predict(X_arr)
        except Exception as e:
            raise ValueError(
                "This unsupervised model does not support predicting cluster assignments for new data. "
                "Only predict-capable clustering models can be applied to unseen datasets."
            ) from e

        cluster_id = np.asarray(cluster_id).ravel()

        n_samples = int(X_arr.shape[0])
        n_features = int(X_arr.shape[1])
        n_preview = min(max_preview_rows, n_samples)

        preview = [
            {
                "index": int(i),
                "cluster_id": int(cluster_id[i]) if i < int(cluster_id.shape[0]) else -1,
            }
            for i in range(n_preview)
        ]

        out = {
            "n_samples": n_samples,
            "n_features": n_features,
            "task": "unsupervised",
            "preview": preview,
            "notes": [
                "Unsupervised apply ignores labels (y) even if provided.",
                "Cluster assignments were produced via pipeline.predict(X).",
            ],
        }

        from engine.contracts.results.prediction import UnsupervisedApplyResult

        return UnsupervisedApplyResult.model_validate(out).model_dump()

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

    from engine.contracts.results.prediction import PredictionResult

    return PredictionResult.model_validate(result).model_dump()


def export_predictions_to_csv(
    *,
    artifact_uid: str,
    artifact_meta: Any,
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    filename: Optional[str] = None,
) -> ExportResult:
    """Apply a cached model pipeline to X (and optional y) and export FULL prediction table as CSV."""

    task_kind = getattr(artifact_meta, "kind", None) or "classification"
    if task_kind == "unsupervised":
        pipeline, X_arr, _task = setup_prediction_common(
            artifact_uid=artifact_uid,
            artifact_meta=artifact_meta,
            X=X,
        )

        try:
            cluster_id = pipeline.predict(X_arr)
        except Exception as e:
            raise ValueError(
                "This unsupervised model does not support predicting cluster assignments for new data. "
                "Only predict-capable clustering models can be applied to unseen datasets."
            ) from e

        cluster_id = np.asarray(cluster_id).ravel()

        n_samples = int(X_arr.shape[0])
        table = [
            {"index": int(i), "cluster_id": int(cluster_id[i]) if i < int(cluster_id.shape[0]) else -1}
            for i in range(n_samples)
        ]

        exporter = make_exporter("csv")
        export_result = exporter.export(
            table,
            dest=None,
            filename=filename,
        )
        return export_result

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

    # Unsupervised cached outputs
    if (cached.task == "unsupervised") or (cached.cluster_id is not None) or (cached.per_sample is not None):
        # Determine length from available payload
        n = None
        if cached.cluster_id is not None:
            try:
                n = int(np.asarray(cached.cluster_id).ravel().shape[0])
            except Exception:
                n = None
        if n is None and cached.per_sample:
            # Use the first column as the length reference
            try:
                first_key = next(iter(cached.per_sample.keys()))
                n = int(np.asarray(cached.per_sample[first_key]).ravel().shape[0])
            except Exception:
                n = None
        if n is None:
            raise ValueError(
                "Cached outputs for this unsupervised artifact are incomplete. "
                "Re-run training to cache per-sample outputs."
            )

        indices = cached.indices
        if indices is None:
            indices = np.arange(n, dtype=int)
        else:
            indices = np.asarray(indices).ravel()
            if int(indices.shape[0]) != n:
                indices = np.arange(n, dtype=int)

        fold_ids = None
        if cached.fold_ids is not None:
            try:
                fold_ids = np.asarray(cached.fold_ids).ravel()
            except Exception:
                fold_ids = None

        cluster_id = None
        if cached.cluster_id is not None:
            try:
                cluster_id = np.asarray(cached.cluster_id).ravel()
            except Exception:
                cluster_id = None

        # Build base rows
        table = []
        for i in range(n):
            row: Dict[str, Any] = {"index": int(indices[i])}
            if fold_ids is not None and i < int(fold_ids.shape[0]):
                try:
                    row["fold_id"] = int(fold_ids[i])
                except Exception:
                    row["fold_id"] = fold_ids[i]
            if cluster_id is not None and i < int(cluster_id.shape[0]):
                try:
                    row["cluster_id"] = int(cluster_id[i])
                except Exception:
                    row["cluster_id"] = cluster_id[i]

            # Attach any additional per-sample columns
            if cached.per_sample:
                for k, v in cached.per_sample.items():
                    try:
                        col = np.asarray(v).ravel()
                        if i < int(col.shape[0]):
                            val = col[i]
                            row[k] = val.item() if hasattr(val, "item") else val
                    except Exception:
                        # Ignore malformed columns
                        continue

            table.append(row)

        exporter = make_exporter("csv")
        export_result = exporter.export(
            table,
            dest=None,
            filename=filename,
        )
        return export_result

    # Supervised cached outputs
    y_pred = np.asarray(cached.y_pred).ravel()
    y_true = None if cached.y_true is None else np.asarray(cached.y_true).ravel()

    indices = cached.indices
    if indices is None:
        indices = np.arange(int(y_pred.shape[0]), dtype=int)

    fold_ids = None
    if cached.fold_ids is not None:
        try:
            fold_ids = np.asarray(cached.fold_ids).ravel()
        except Exception:
            fold_ids = None

    table = build_prediction_table(
        indices=indices,
        y_pred=y_pred,
        task=cached.task,
        y_true=y_true,
        max_rows=None,
    )

    # Attach fold_id (useful for pooled CV exports and post-hoc analyses).
    # We rebuild rows so fold_id is guaranteed to be present in the header.
    if fold_ids is not None:
        rebuilt = []
        n_f = int(fold_ids.shape[0])
        for i, row in enumerate(table):
            fid = ''
            if i < n_f:
                fid = fold_ids[i]
                try:
                    fid = int(fid)
                except Exception:
                    # Keep as-is if it cannot be cast cleanly
                    pass

            if isinstance(row, dict):
                # Keep 'index' first (if present), then fold_id, then the rest
                if 'index' in row:
                    new_row = {'index': row.get('index', ''), 'fold_id': fid}
                    for k, v in row.items():
                        if k == 'index':
                            continue
                        new_row[k] = v
                else:
                    new_row = {'fold_id': fid, **row}
                rebuilt.append(new_row)
            else:
                rebuilt.append(row)

        table = rebuilt

    exporter = make_exporter("csv")
    export_result = exporter.export(
        table,
        dest=None,
        filename=filename,
    )
    return export_result
