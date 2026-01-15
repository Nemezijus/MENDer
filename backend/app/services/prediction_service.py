from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import math
import numpy as np

from shared_schemas.eval_configs import EvalModel

from utils.factories.predict_factory import make_predictor
from utils.factories.eval_factory import make_evaluator
from utils.factories.sanity_factory import make_sanity_checker
from utils.persistence.artifact_cache import artifact_cache
from utils.factories.export_factory import make_exporter
from utils.predicting.prediction_results import build_prediction_table, build_decoder_output_table, merge_prediction_and_decoder_tables
from utils.postprocessing.decoder_outputs import compute_decoder_outputs
from utils.io.export.result_export import ExportResult


def _safe_float(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except Exception:
        return None
    if not math.isfinite(f):
        return None
    return f


def _build_preview_rows(
    indices: Sequence[int],
    y_pred: np.ndarray,
    task: str,
    y_true: Optional[np.ndarray] = None,
) -> list[Dict[str, Any]]:
    """
    Build a compact list of dicts compatible with PredictionRow.

    For classification:
      - correct = (y_pred == y_true) when y_true is provided.

    For regression:
      - residual = y_true - y_pred
      - abs_error = |residual|
    """
    rows: list[Dict[str, Any]] = []

    y_true_arr: Optional[np.ndarray] = None
    if y_true is not None:
        y_true_arr = np.asarray(y_true).ravel()
        if y_true_arr.shape[0] != len(indices):
            # Just in case; we only slice consistently in the caller
            y_true_arr = None

    for i, idx in enumerate(indices):
        row: Dict[str, Any] = {
            "index": int(idx),
            "y_pred": y_pred[i].item() if hasattr(y_pred[i], "item") else y_pred[i],
        }

        if y_true_arr is not None:
            y_true_val = y_true_arr[i]
            row["y_true"] = (
                y_true_val.item() if hasattr(y_true_val, "item") else y_true_val
            )

            if task == "regression":
                try:
                    resid = float(y_true_val) - float(y_pred[i])
                    row["residual"] = resid
                    row["abs_error"] = abs(resid)
                except Exception:
                    # best-effort; leave them out on failure
                    pass
            else:
                # classification-style correctness
                row["correct"] = bool(y_pred[i] == y_true_val)

        rows.append(row)

    return rows


def _setup_prediction(
    artifact_uid: str,
    artifact_meta: Any,  # ModelArtifactMeta instance
    X: np.ndarray,
    y: Optional[np.ndarray],
) -> Tuple[Any, np.ndarray, Optional[np.ndarray], str, EvalModel, str]:
    """
    Shared preparation step for both 'apply' and 'export' flows:

      - Get pipeline from cache
      - Normalize shapes of X/y
      - Align X orientation using n_features_in from artifact
      - Infer task & eval config

    Returns:
        (pipeline, X_arr, y_arr, task, eval_model, eval_kind)
    """
    # Pipeline
    pipeline = artifact_cache.get(artifact_uid)
    if pipeline is None:
        raise ValueError(
            f"No cached model pipeline found for artifact_uid={artifact_uid!r}. "
            "Train a model or load an artifact first."
        )

    # Normalize X / y
    X_arr = np.asarray(X)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    if X_arr.ndim != 2:
        raise ValueError(f"Expected 2D X for prediction; got shape {X_arr.shape}.")

    y_arr: Optional[np.ndarray] = None
    if y is not None:
        y_arr = np.asarray(y).ravel()
        make_sanity_checker().check(X_arr, y_arr)

    # Feature-count sanity & orientation
    n_features_meta = getattr(artifact_meta, "n_features_in", None) or getattr(
        artifact_meta, "n_features", None
    )

    if n_features_meta is not None:
        n_features_meta = int(n_features_meta)

        if X_arr.ndim != 2:
            raise ValueError(f"X must be 2D; got shape {X_arr.shape}.")

        # If columns already match expected features, assume X is (n_samples, n_features)
        if X_arr.shape[1] == n_features_meta:
            pass
        # Else if rows match expected features, assume X is (n_features, n_samples) and transpose
        elif X_arr.shape[0] == n_features_meta:
            X_arr = X_arr.T

        if X_arr.shape[1] != n_features_meta:
            raise ValueError(
                f"Feature mismatch: model expects {n_features_meta} features, "
                f"but X has shape {X_arr.shape}."
            )

    # Task & eval config from artifact meta
    task = getattr(artifact_meta, "kind", None) or "classification"
    eval_dict = getattr(artifact_meta, "eval", None) or {}
    ev = EvalModel.parse_obj(eval_dict)
    eval_kind = "regression" if task == "regression" else "classification"

    return pipeline, X_arr, y_arr, task, ev, eval_kind


def apply_model_to_arrays(
    *,
    artifact_uid: str,
    artifact_meta: Any,
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    max_preview_rows: int = 100,
) -> Dict[str, Any]:
    """
    Apply a cached model pipeline to an already-loaded dataset (X, optional y)
    and return a preview plus metric.
    """
    pipeline, X_arr, y_arr, task, ev, eval_kind = _setup_prediction(
        artifact_uid, artifact_meta, X, y
    )

    # Predict
    predictor = make_predictor()
    y_pred = predictor.predict(pipeline, X_arr)
    y_pred = np.asarray(y_pred).ravel()

    # Metric (optional)
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    if y_arr is not None:
        evaluator = make_evaluator(ev, kind=eval_kind)
        metric_name = ev.metric
        try:
            metric_value = _safe_float(evaluator.score(y_arr, y_pred))
        except Exception:
            metric_value = None

    # Preview
    n_samples = int(X_arr.shape[0])
    n_features = int(X_arr.shape[1])

    n_preview = min(max_preview_rows, n_samples)
    indices = list(range(n_preview))
    y_true_preview = y_arr[:n_preview] if y_arr is not None else None
    y_pred_preview = y_pred[:n_preview]

    preview_rows = _build_preview_rows(
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


    # --- Optional decoder outputs (classification-only) -------------------------
    decoder_cfg = getattr(ev, "decoder", None)
    decoder_enabled = bool(getattr(decoder_cfg, "enabled", False)) if decoder_cfg is not None else False
    if decoder_enabled and eval_kind == "classification":
        decoder_positive_label = getattr(decoder_cfg, "positive_class_label", None) if decoder_cfg is not None else None
        decoder_include_scores = bool(getattr(decoder_cfg, "include_decision_scores", True)) if decoder_cfg is not None else True
        decoder_include_probabilities = bool(getattr(decoder_cfg, "include_probabilities", True)) if decoder_cfg is not None else True
        decoder_include_margin = bool(getattr(decoder_cfg, "include_margin", True)) if decoder_cfg is not None else True
        decoder_calibrate_probabilities = bool(getattr(decoder_cfg, "calibrate_probabilities", False)) if decoder_cfg is not None else False

        # Allow decoder-specific preview cap; fall back to this endpoint's preview cap
        decoder_preview_cap = int(getattr(decoder_cfg, "max_preview_rows", max_preview_rows) or max_preview_rows) if decoder_cfg is not None else max_preview_rows
        n_preview_dec = min(decoder_preview_cap, n_samples)

        try:
            dec = compute_decoder_outputs(
                pipeline,
                X_arr,
                positive_class_label=decoder_positive_label,
                include_decision_scores=decoder_include_scores,
                include_probabilities=decoder_include_probabilities,
                calibrate_probabilities=decoder_calibrate_probabilities,
            )

            ds_preview = None
            pr_preview = None
            mg_preview = None

            if dec.decision_scores is not None and decoder_include_scores:
                ds_preview = np.asarray(dec.decision_scores)[:n_preview_dec]
            if dec.proba is not None and decoder_include_probabilities:
                pr_preview = np.asarray(dec.proba)[:n_preview_dec]
            if decoder_include_margin and dec.margin is not None:
                mg_preview = np.asarray(dec.margin)[:n_preview_dec]

            y_true_dec_preview = y_arr[:n_preview_dec] if y_arr is not None else None
            y_pred_dec_preview = np.asarray(dec.y_pred).ravel()[:n_preview_dec]

            rows = build_decoder_output_table(
                indices=list(range(n_preview_dec)),
                y_pred=y_pred_dec_preview,
                y_true=y_true_dec_preview,
                classes=dec.classes,
                decision_scores=ds_preview,
                proba=pr_preview,
                margin=mg_preview,
                positive_class_label=dec.positive_class_label,
                positive_class_index=dec.positive_class_index,
            )

            result["decoder_outputs"] = {
                "classes": (dec.classes.tolist() if hasattr(dec.classes, "tolist") else dec.classes) if dec.classes is not None else None,
                "positive_class_label": dec.positive_class_label,
                "positive_class_index": dec.positive_class_index,
                "has_decision_scores": bool(dec.decision_scores is not None and decoder_include_scores),
                "has_proba": bool(dec.proba is not None and decoder_include_probabilities),
                "notes": list(dec.notes or []),
                "preview_rows": rows,
                "n_rows_total": int(n_samples),
            }
        except Exception as e:
            result["decoder_outputs"] = {
                "classes": None,
                "positive_class_label": decoder_positive_label,
                "positive_class_index": None,
                "has_decision_scores": False,
                "has_proba": False,
                "notes": [f"Decoder outputs could not be computed ({type(e).__name__}: {e})"],
                "preview_rows": [],
                "n_rows_total": None,
            }

    return result


def export_predictions_to_csv(
    *,
    artifact_uid: str,
    artifact_meta: Any,
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    filename: Optional[str] = None,
) -> ExportResult:
    """
    Apply a cached model pipeline to X (and optional y) and export
    FULL prediction table as CSV.

    This is used by the /models/apply/export endpoint and returns an
    ExportResult (content bytes, filename, mime_type, size, path).
    """
    pipeline, X_arr, y_arr, task, ev, eval_kind = _setup_prediction(
        artifact_uid, artifact_meta, X, y
    )

    # Predict
    predictor = make_predictor()
    y_pred = predictor.predict(pipeline, X_arr)
    y_pred = np.asarray(y_pred).ravel()

    n_samples = int(X_arr.shape[0])

    # Build full prediction table (one row per sample)
    table = build_prediction_table(
        indices=range(n_samples),
        y_pred=y_pred,
        task=eval_kind,
        y_true=y_arr,
        max_rows=None,  # no preview truncation here
    )


    # --- Optional decoder outputs columns in export (classification-only) ----
    decoder_cfg = getattr(ev, "decoder", None)
    decoder_enabled = bool(getattr(decoder_cfg, "enabled", False)) if decoder_cfg is not None else False
    decoder_export_enabled = (
        bool(getattr(decoder_cfg, "enable_export", True)) if decoder_cfg is not None else True
    )
    if decoder_enabled and decoder_export_enabled and eval_kind == "classification":
        decoder_positive_label = (
            getattr(decoder_cfg, "positive_class_label", None) if decoder_cfg is not None else None
        )
        decoder_include_scores = (
            bool(getattr(decoder_cfg, "include_decision_scores", True)) if decoder_cfg is not None else True
        )
        decoder_include_probabilities = (
            bool(getattr(decoder_cfg, "include_probabilities", True)) if decoder_cfg is not None else True
        )
        decoder_include_margin = (
            bool(getattr(decoder_cfg, "include_margin", True)) if decoder_cfg is not None else True
        )
        decoder_calibrate_probabilities = (
            bool(getattr(decoder_cfg, "calibrate_probabilities", False)) if decoder_cfg is not None else False
        )

        try:
            dec = compute_decoder_outputs(
                pipeline,
                X_arr,
                positive_class_label=decoder_positive_label,
                include_decision_scores=decoder_include_scores,
                include_probabilities=decoder_include_probabilities,
                calibrate_probabilities=decoder_calibrate_probabilities,
            )
            ds = dec.decision_scores if (dec.decision_scores is not None and decoder_include_scores) else None
            pr = dec.proba if (dec.proba is not None and decoder_include_probabilities) else None
            mg = dec.margin if (dec.margin is not None and decoder_include_margin) else None
            y_pred_dec = np.asarray(dec.y_pred).ravel()

            decoder_rows = build_decoder_output_table(
                indices=range(n_samples),
                y_pred=y_pred_dec,
                y_true=y_arr,
                classes=dec.classes,
                decision_scores=ds,
                proba=pr,
                margin=mg,
                positive_class_label=dec.positive_class_label,
                positive_class_index=dec.positive_class_index,
                max_rows=None,
            )
            table = merge_prediction_and_decoder_tables(
                prediction_rows=table,
                decoder_rows=decoder_rows,
            )
        except Exception:
            # Export should still succeed even if decoder outputs cannot be computed
            pass
    exporter = make_exporter("csv")
    export_result = exporter.export(
        table,
        dest=None,
        filename=filename,
    )
    return export_result