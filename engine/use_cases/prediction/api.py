"""Prediction use-case (orchestrator).

This package implements the orchestration previously living in
``backend/app/services/prediction_service.py``.

Design goals
------------
- Backend-free: no imports from backend.
- Script-friendly: can run with only the Engine.
- Returns Segment-4 result contracts.

Notes
-----
- For persisted artifacts, callers may pass an :class:`~engine.io.artifacts.store.ArtifactStore`
  so the pipeline can be loaded if it is not present in the runtime cache.
"""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np

from engine.core.shapes import coerce_X_only, maybe_transpose_for_expected_n_features

from engine.contracts.eval_configs import EvalModel
from engine.contracts.results.prediction import PredictionResult, UnsupervisedApplyResult

from engine.io.artifacts.store import ArtifactStore
from engine.io.artifacts.meta_models import ModelArtifactMetaDict

from engine.use_cases.prediction.loading import load_pipeline
from engine.use_cases.prediction.meta import resolve_eval_model
from engine.use_cases.prediction.unsupervised import apply_unsupervised
from engine.use_cases.prediction.supervised import apply_supervised
from engine.use_cases.prediction.utils import meta_get
from engine.use_cases.prediction.validation import maybe_validate_n_features


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
    """Apply a trained artifact to numpy-like arrays."""

    pipeline = load_pipeline(artifact_uid=artifact_uid, store=store)

    X_arr = coerce_X_only(X)

    n_features_expected = (
        meta_get(artifact_meta, "n_features_in", None) or meta_get(artifact_meta, "n_features", None)
    )
    X_arr = maybe_transpose_for_expected_n_features(
        X_arr, expected_n_features=n_features_expected
    )
    maybe_validate_n_features(X_arr, n_features_expected)

    task = str(meta_get(artifact_meta, "kind", "classification"))
    if task not in {"classification", "regression", "unsupervised"}:
        task = "classification"

    # -----------------
    # Unsupervised apply
    # -----------------
    if task == "unsupervised":
        return apply_unsupervised(
            pipeline=pipeline,
            X_arr=np.asarray(X_arr),
            max_preview_rows=max_preview_rows,
        )

    # ---------------
    # Supervised apply
    # ---------------

    eval_model = resolve_eval_model(artifact_meta=artifact_meta, eval_override=eval_override)

    return apply_supervised(
        task="regression" if task == "regression" else "classification",
        pipeline=pipeline,
        X_arr=np.asarray(X_arr),
        y=y,
        eval_model=eval_model,
        max_preview_rows=max_preview_rows,
    )
