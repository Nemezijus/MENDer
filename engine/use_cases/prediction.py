"""Prediction use-case.

This module implements the orchestration previously living in
``backend/app/services/prediction_service.py``.

Design goals
------------
- Backend-free: no imports from backend.
- Script-friendly: can run with only the Engine
- Uses the canonical Segment-6 prediction/decoder API.
- Returns Segment-4 result contracts.

Notes
-----
- For persisted artifacts, callers may pass an :class:`~engine.io.artifacts.store.ArtifactStore`
  so the pipeline can be loaded if it is not present in the runtime cache.
"""

from __future__ import annotations

import warnings

from typing import Any, Dict, Optional, Sequence, Tuple, Union
import math

import numpy as np

from engine.core.shapes import coerce_X_only, maybe_transpose_for_expected_n_features

from engine.contracts.eval_configs import EvalModel
from engine.contracts.results.prediction import (
    PredictionResult,
    UnsupervisedApplyResult,
    UnsupervisedApplyRow,
)

from engine.components.prediction import predict_decoder_outputs
from engine.components.prediction.predicting import predict_scores
from engine.components.evaluation.scoring import PROBA_METRICS
from engine.reporting.prediction.prediction_results import build_prediction_table
from engine.runtime.caches.artifact_cache import artifact_cache

from engine.use_cases.artifacts import load_model_from_store
from engine.io.artifacts.store import ArtifactStore
from engine.io.artifacts.meta_models import ModelArtifactMetaDict

from engine.factories.eval_factory import make_evaluator


def _meta_get(meta: Any, key: str, default: Any = None) -> Any:
    if meta is None:
        return default
    if hasattr(meta, key):
        return getattr(meta, key)
    if isinstance(meta, dict):
        return meta.get(key, default)
    return default


def _safe_float_optional(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except Exception:
        return None
    if not math.isfinite(f):
        return None
    return f


def _maybe_validate_n_features(X_arr: np.ndarray, n_features_expected: Optional[int]) -> None:
    if n_features_expected is None:
        return
    exp = int(n_features_expected)
    if X_arr.ndim != 2:
        raise ValueError(f"Expected 2D X for prediction; got shape {X_arr.shape}.")
    if X_arr.shape[1] != exp:
        raise ValueError(
            f"Feature mismatch: model expects {exp} features, but X has shape {X_arr.shape}."
        )


def _load_pipeline(
    *,
    artifact_uid: str,
    store: Optional[ArtifactStore],
) -> Any:
    pipeline = artifact_cache.get(artifact_uid)
    if pipeline is not None:
        return pipeline

    if store is None:
        raise ValueError(
            f"No cached model pipeline found for artifact_uid={artifact_uid!r}. "
            "Train a model in this process, or provide an ArtifactStore to load it."
        )

    pipeline, _meta = load_model_from_store(store, artifact_uid)
    try:
        artifact_cache.put(artifact_uid, pipeline)
    except Exception as e:
        warnings.warn(
            f"artifact_cache.put failed for artifact_uid={artifact_uid!r}: {type(e).__name__}: {e}",
            RuntimeWarning,
        )
    return pipeline


def apply_model_to_arrays(
    *,
    artifact_uid: str,
    artifact_meta: ModelArtifactMetaDict,
    X: Any,
    y: Optional[Any] = None,
    eval_override: Optional[EvalModel] = None,
    max_preview_rows: int = 100,
    store: Optional[ArtifactStore] = None,
) -> Union[PredictionResult, UnsupervisedApplyResult]:
    """Apply an artifact to numpy-like arrays.

    Returns
    -------
    PredictionResult | UnsupervisedApplyResult
        Typed result contracts.
    """

    pipeline = _load_pipeline(artifact_uid=artifact_uid, store=store)

    X_arr = coerce_X_only(X)
    n_features_expected = _meta_get(artifact_meta, "n_features_in", None) or _meta_get(artifact_meta, "n_features", None)
    X_arr = maybe_transpose_for_expected_n_features(X_arr, expected_n_features=n_features_expected)
    _maybe_validate_n_features(X_arr, n_features_expected)

    task = str(_meta_get(artifact_meta, "kind", "classification"))
    if task not in {"classification", "regression", "unsupervised"}:
        task = "classification"

    n_samples = int(X_arr.shape[0])
    n_features = int(X_arr.shape[1])

    # -----------------
    # Unsupervised apply
    # -----------------
    if task == "unsupervised":
        try:
            cluster_ids = np.asarray(pipeline.predict(X_arr)).reshape(-1)
        except Exception as e:
            raise ValueError(f"Unsupervised pipeline does not support predict(...): {e}") from e

        n_preview = min(int(max_preview_rows), n_samples)
        rows = [
            UnsupervisedApplyRow(index=int(i), cluster_id=int(cluster_ids[i]))
            for i in range(n_preview)
        ]
        return UnsupervisedApplyResult(
            n_samples=n_samples,
            n_features=n_features,
            preview=rows,
            notes=[],
        )

    # ---------------
    # Supervised apply
    # ---------------

    y_arr: Optional[np.ndarray] = None
    has_labels = y is not None
    if y is not None:
        y_arr = np.asarray(y).reshape(-1)
        if y_arr.shape[0] != n_samples:
            raise ValueError(
                f"y has {int(y_arr.shape[0])} samples but X has {n_samples}."
            )

    # Eval config from meta, optionally overridden
    ev: Optional[EvalModel] = None
    if eval_override is not None:
        ev = eval_override
    else:
        eval_dict = _meta_get(artifact_meta, "eval", {}) or {}
        try:
            ev = EvalModel.model_validate(eval_dict)
        except Exception:
            try:
                ev = EvalModel.parse_obj(eval_dict)  # type: ignore[attr-defined]
            except Exception:
                ev = None

    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    notes: list[str] = []

    y_pred = np.asarray(pipeline.predict(X_arr)).reshape(-1)

    # Score if possible
    if y_arr is not None and ev is not None:
        eval_kind = "regression" if task == "regression" else "classification"
        evaluator = make_evaluator(ev, kind=eval_kind)
        metric_name = str(getattr(ev, "metric", None) or "") or None

        y_proba: Optional[np.ndarray] = None
        y_score: Optional[np.ndarray] = None
        if metric_name is not None and metric_name in PROBA_METRICS and eval_kind == "classification":
            try:
                scores, used = predict_scores(pipeline, X_arr, kind="auto")
                if used == "proba":
                    y_proba = np.asarray(scores)
                elif used == "decision":
                    y_score = np.asarray(scores)
            except Exception as e:
                notes.append(f"Could not compute proba/decision scores for metric '{metric_name}': {e}")

        try:
            metric_value = _safe_float_optional(
                evaluator.score(
                    y_arr,
                    y_pred,
                    y_proba=y_proba,
                    y_score=y_score,
                )
            )
        except Exception as e:
            notes.append(f"Metric computation failed ({type(e).__name__}: {e})")

    # Preview table
    n_preview = min(int(max_preview_rows), n_samples)
    preview_rows = build_prediction_table(
        indices=range(n_samples),
        y_pred=y_pred,
        y_true=y_arr,
        task="regression" if task == "regression" else "classification",
        max_rows=n_preview,
    )

    # Optional decoder outputs (classification only)
    decoder_outputs = None
    if ev is not None and task == "classification":
        decoder_cfg = getattr(ev, "decoder", None)
        decoder_enabled = bool(getattr(decoder_cfg, "enabled", False)) if decoder_cfg is not None else False
        if decoder_enabled:
            decoder_positive_label = getattr(decoder_cfg, "positive_class_label", None)
            decoder_include_scores = bool(getattr(decoder_cfg, "include_decision_scores", True))
            decoder_include_probabilities = bool(getattr(decoder_cfg, "include_probabilities", True))
            decoder_include_margin = bool(getattr(decoder_cfg, "include_margin", True))
            decoder_calibrate_probabilities = bool(getattr(decoder_cfg, "calibrate_probabilities", False))

            decoder_preview_cap = int(getattr(decoder_cfg, "max_preview_rows", n_preview) or n_preview)
            n_preview_dec = min(decoder_preview_cap, n_samples)

            try:
                decoder_outputs = predict_decoder_outputs(
                    pipeline,
                    X_arr,
                    y_true=y_arr,
                    indices=range(n_samples),
                    positive_class_label=decoder_positive_label,
                    include_decision_scores=decoder_include_scores,
                    include_probabilities=decoder_include_probabilities,
                    include_margin=decoder_include_margin,
                    calibrate_probabilities=decoder_calibrate_probabilities,
                    max_preview_rows=n_preview_dec,
                    include_summary=False,
                )
            except Exception as e:
                notes.append(f"Decoder outputs could not be computed ({type(e).__name__}: {e})")

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
