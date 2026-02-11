from __future__ import annotations

from typing import Any, Optional

import numpy as np

from engine.io.artifacts.meta import ArtifactBuilderInput, build_model_artifact_meta
from engine.runtime.caches.artifact_cache import artifact_cache
from engine.runtime.caches.eval_outputs_cache import EvalOutputs, eval_outputs_cache
from engine.use_cases._deps import resolve_store
from engine.use_cases.artifacts import save_model_to_store


def attach_artifact_and_persist(
    *,
    cfg,
    pipeline,
    X,
    eval_kind: str,
    n_train: int,
    n_test: int,
    result: dict[str, Any],
    y_true: Optional[np.ndarray],
    y_pred: Optional[np.ndarray],
    row_indices: np.ndarray,
    fold_ids: Optional[np.ndarray],
    store: Any,
) -> dict[str, Any]:
    """Build artifact meta, attach to result, persist and cache best-effort."""

    n_features_in = (
        int(np.asarray(X).shape[1])
        if hasattr(X, "shape") and len(np.asarray(X).shape) > 1
        else None
    )

    classes_out = None
    try:
        confusion = result.get("confusion") or None
        if confusion and confusion.get("labels"):
            classes_out = confusion.get("labels")
    except Exception:
        classes_out = None

    artifact_input = ArtifactBuilderInput(
        cfg=cfg,
        pipeline=pipeline,
        n_train=n_train,
        n_test=n_test,
        n_features=n_features_in,
        classes=classes_out,
        kind=eval_kind,
        summary={
            "metric_name": result.get("metric_name"),
            "metric_value": result.get("metric_value"),
            "mean_score": result.get("mean_score"),
            "std_score": result.get("std_score"),
            "n_splits": result.get("n_splits"),
            "notes": result.get("notes", []),
        },
    )

    artifact_meta = build_model_artifact_meta(artifact_input)
    result["artifact"] = artifact_meta

    store_resolved = resolve_store(store)
    try:
        save_model_to_store(store_resolved, pipeline, artifact_meta)
    except Exception:
        pass

    # Cache fitted pipeline (process-local)
    try:
        uid = artifact_meta["uid"]
        artifact_cache.put(uid, pipeline)

        if y_pred is not None and y_true is not None and np.asarray(y_true).size:
            try:
                eval_outputs_cache.put(
                    uid,
                    EvalOutputs(
                        task=eval_kind,
                        indices=row_indices,
                        fold_ids=fold_ids,
                        y_true=y_true,
                        y_pred=y_pred,
                    ),
                )
            except Exception:
                pass
    except Exception:
        pass

    return artifact_meta
