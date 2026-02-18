from __future__ import annotations

"""Prediction export use-cases.

These functions were previously implemented in
``backend/app/services/prediction_service.py``.

They are *Engine-owned orchestration* because they coordinate:
- pipeline loading (cache / optional store)
- shape validation
- optional decoder computation and table merge
- exporter selection and file payload creation

The backend should only:
- load X/y from disk
- call ``engine.api.*``
- shape HTTP response.
"""

from typing import Any, Optional

import numpy as np

from engine.core.shapes import coerce_X_only, maybe_transpose_for_expected_n_features

from engine.contracts.eval_configs import EvalModel

from engine.factories.export_factory import make_exporter

from engine.io.artifacts.store import ArtifactStore
from engine.io.export.csv_export import ExportResult

from engine.reporting.prediction.prediction_results import (
    build_prediction_table,
    merge_prediction_and_decoder_tables,
)

from engine.components.prediction import predict_decoder_outputs

from engine.use_cases.prediction.loading import load_pipeline
from engine.use_cases.prediction.meta import resolve_eval_model
from engine.use_cases.prediction.utils import meta_get
from engine.use_cases.prediction.validation import maybe_validate_n_features


def _maybe_merge_decoder_into_export_table(
    *,
    table: Any,
    pipeline: Any,
    X_arr: np.ndarray,
    y_arr: Optional[np.ndarray],
    eval_model: Optional[EvalModel],
    eval_kind: str,
) -> Any:
    """Merge decoder columns into the prediction export table when enabled."""

    if eval_kind != "classification" or eval_model is None:
        return table

    decoder_cfg = getattr(eval_model, "decoder", None)
    decoder_enabled = bool(getattr(decoder_cfg, "enabled", False)) if decoder_cfg is not None else False
    decoder_export_enabled = bool(getattr(decoder_cfg, "enable_export", True)) if decoder_cfg is not None else True

    if not (decoder_enabled and decoder_export_enabled):
        return table

    decoder_positive_label = getattr(decoder_cfg, "positive_class_label", None) if decoder_cfg is not None else None
    decoder_include_scores = bool(getattr(decoder_cfg, "include_decision_scores", True)) if decoder_cfg is not None else True
    decoder_include_probabilities = bool(getattr(decoder_cfg, "include_probabilities", True)) if decoder_cfg is not None else True
    decoder_include_margin = bool(getattr(decoder_cfg, "include_margin", True)) if decoder_cfg is not None else True
    decoder_calibrate_probabilities = bool(getattr(decoder_cfg, "calibrate_probabilities", False)) if decoder_cfg is not None else False

    n_samples = int(X_arr.shape[0])

    try:
        dec = predict_decoder_outputs(
            pipeline,
            X_arr,
            y_true=y_arr,
            indices=range(n_samples),
            positive_class_label=decoder_positive_label,
            include_decision_scores=decoder_include_scores,
            include_probabilities=decoder_include_probabilities,
            include_margin=decoder_include_margin,
            calibrate_probabilities=decoder_calibrate_probabilities,
            max_preview_rows=None,  # export wants full table
            include_summary=False,
        )

        decoder_rows = [r.model_dump() for r in (dec.preview_rows or [])]
        return merge_prediction_and_decoder_tables(
            prediction_rows=table,
            decoder_rows=decoder_rows,
        )
    except Exception:
        return table


def export_predictions_to_csv(
    *,
    artifact_uid: str,
    artifact_meta: Any,
    X: Any,
    y: Optional[Any],
    filename: Optional[str] = None,
    eval_override: Optional[EvalModel] = None,
    store: Optional[ArtifactStore] = None,
) -> ExportResult:
    """Export predictions as CSV for a given artifact + dataset."""

    pipeline = load_pipeline(artifact_uid=artifact_uid, store=store)

    X_arr = coerce_X_only(X)

    n_features_expected = (
        meta_get(artifact_meta, "n_features_in", None) or meta_get(artifact_meta, "n_features", None)
    )
    X_arr = maybe_transpose_for_expected_n_features(X_arr, expected_n_features=n_features_expected)
    maybe_validate_n_features(X_arr, n_features_expected)

    task = str(meta_get(artifact_meta, "kind", "classification"))
    if task not in {"classification", "regression", "unsupervised"}:
        task = "classification"

    n_samples = int(np.asarray(X_arr).shape[0])

    # -----------------
    # Unsupervised export
    # -----------------
    if task == "unsupervised":
        try:
            cluster_ids = np.asarray(pipeline.predict(np.asarray(X_arr))).reshape(-1)
        except Exception as e:
            raise ValueError(f"Unsupervised pipeline does not support predict(...): {e}") from e

        table = [
            {"index": int(i), "cluster_id": int(cluster_ids[i])}
            for i in range(int(cluster_ids.shape[0]))
        ]

        exporter = make_exporter("csv")
        return exporter.export(table, dest=None, filename=filename)

    # ---------------
    # Supervised export
    # ---------------

    y_arr: Optional[np.ndarray] = None
    if y is not None:
        y_arr = np.asarray(y).reshape(-1)
        if int(y_arr.shape[0]) != n_samples:
            raise ValueError(f"y has {int(y_arr.shape[0])} samples but X has {n_samples}.")

    y_pred = np.asarray(pipeline.predict(np.asarray(X_arr))).reshape(-1)

    eval_model = resolve_eval_model(artifact_meta=artifact_meta, eval_override=eval_override)
    eval_kind = "regression" if task == "regression" else "classification"

    table = build_prediction_table(
        indices=range(n_samples),
        y_pred=y_pred,
        y_true=y_arr,
        task=eval_kind,
        max_rows=None,
    )

    table = _maybe_merge_decoder_into_export_table(
        table=table,
        pipeline=pipeline,
        X_arr=np.asarray(X_arr),
        y_arr=y_arr,
        eval_model=eval_model,
        eval_kind=eval_kind,
    )

    exporter = make_exporter("csv")
    return exporter.export(table, dest=None, filename=filename)
