from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from engine.io.artifacts.meta import ArtifactBuilderInput, build_model_artifact_meta
from engine.runtime.caches.artifact_cache import artifact_cache
from engine.runtime.caches.eval_outputs_cache import EvalOutputs, eval_outputs_cache
from engine.use_cases._deps import resolve_store
from engine.use_cases.artifacts import save_model_to_store


def extra_stats_from_ensemble_report(ensemble_report: Optional[dict]) -> Dict[str, Any]:
    """Derive a small set of scalar extra-stats for artifact meta.

    Best-effort and must never raise.

    Note
    ----
    These stats belong in the *artifact meta* (meta["extra_stats"]) rather than the
    top-level EnsembleResult payload (which is strict and forbids extra fields).
    """

    out: Dict[str, Any] = {}
    if not ensemble_report or not isinstance(ensemble_report, dict):
        return out

    try:
        kind = ensemble_report.get("kind")

        if kind == "xgboost":
            task = (ensemble_report.get("task") or "classification")
            if task == "classification":
                sim = ensemble_report.get("similarity") or {}
                errs = ensemble_report.get("errors") or {}
                out.update(
                    {
                        "ensemble_n_estimators": ensemble_report.get("n_estimators"),
                        "ensemble_best_estimator": (ensemble_report.get("best_estimator") or {}).get("name"),
                        "ensemble_pairwise_mean_similarity": sim.get("pairwise_mean_similarity"),
                        "ensemble_pairwise_min_similarity": sim.get("pairwise_min_similarity"),
                        "ensemble_pairwise_max_similarity": sim.get("pairwise_max_similarity"),
                        "ensemble_estimator_error_rate": errs.get("estimator_error_rate"),
                    }
                )
            else:
                out.update(
                    {
                        "ensemble_n_estimators": ensemble_report.get("n_estimators"),
                        "ensemble_best_estimator": (ensemble_report.get("best_estimator") or {}).get("name"),
                    }
                )

        elif kind == "voting":
            task = (ensemble_report.get("task") or "classification")
            if task == "classification":
                agreement = ensemble_report.get("agreement") or {}
                vote = ensemble_report.get("vote") or {}
                change = ensemble_report.get("change_vs_best") or {}
                out.update(
                    {
                        "ensemble_n_estimators": ensemble_report.get("n_estimators"),
                        "ensemble_all_agree_rate": agreement.get("all_agree_rate"),
                        "ensemble_pairwise_agreement": agreement.get("pairwise_mean_agreement"),
                        "ensemble_tie_rate": vote.get("tie_rate"),
                        "ensemble_mean_margin": vote.get("mean_margin"),
                        "ensemble_best_estimator": (ensemble_report.get("best_estimator") or {}).get("name"),
                        "ensemble_corrected_vs_best": change.get("corrected"),
                        "ensemble_harmed_vs_best": change.get("harmed"),
                    }
                )
            else:
                out.update(
                    {
                        "ensemble_n_estimators": ensemble_report.get("n_estimators"),
                        "ensemble_best_estimator": (ensemble_report.get("best_estimator") or {}).get("name"),
                    }
                )

        elif kind == "bagging":
            task = (ensemble_report.get("task") or "classification")
            bag = ensemble_report.get("bagging") or {}
            oob = ensemble_report.get("oob") or {}
            if task == "classification":
                div = ensemble_report.get("diversity") or {}
                errs = ensemble_report.get("errors") or {}
                out.update(
                    {
                        "ensemble_n_estimators": bag.get("n_estimators"),
                        "ensemble_base_algo": bag.get("base_algo"),
                        "ensemble_oob_score": oob.get("score"),
                        "ensemble_pairwise_mean_diversity": div.get("pairwise_mean_diversity"),
                        "ensemble_pairwise_min_diversity": div.get("pairwise_min_diversity"),
                        "ensemble_pairwise_max_diversity": div.get("pairwise_max_diversity"),
                        "ensemble_estimator_error_rate": errs.get("estimator_error_rate"),
                    }
                )
            else:
                out.update(
                    {
                        "ensemble_n_estimators": bag.get("n_estimators"),
                        "ensemble_base_algo": bag.get("base_algo"),
                        "ensemble_oob_score": oob.get("score"),
                    }
                )

        elif kind == "adaboost":
            task = (ensemble_report.get("task") or "classification")
            if task == "classification":
                div = ensemble_report.get("diversity") or {}
                errs = ensemble_report.get("errors") or {}
                out.update(
                    {
                        "ensemble_n_estimators": ensemble_report.get("n_estimators"),
                        "ensemble_base_algo": ensemble_report.get("base_algo"),
                        "ensemble_pairwise_mean_diversity": div.get("pairwise_mean_diversity"),
                        "ensemble_pairwise_min_diversity": div.get("pairwise_min_diversity"),
                        "ensemble_pairwise_max_diversity": div.get("pairwise_max_diversity"),
                        "ensemble_estimator_error_rate": errs.get("estimator_error_rate"),
                    }
                )
            else:
                out.update(
                    {
                        "ensemble_n_estimators": ensemble_report.get("n_estimators"),
                        "ensemble_base_algo": ensemble_report.get("base_algo"),
                    }
                )

    except Exception:
        return {}

    # Keep only JSON-safe scalars
    filtered: Dict[str, Any] = {}
    for k, v in out.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            filtered[k] = v
    return filtered


def attach_artifact_and_persist(
    *,
    cfg,
    model,
    X,
    eval_kind: str,
    n_train: int,
    n_test: int,
    result: Dict[str, Any],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    row_indices: np.ndarray,
    fold_ids: Optional[np.ndarray],
    ensemble_report: Optional[dict],
    store: Any,
) -> Dict[str, Any]:
    """Build artifact meta, cache outputs, and persist to store.

    Important: do NOT add extra fields to `result` that are not part of the
    strict `EnsembleResult` contract.
    """

    store_resolved = resolve_store(store)

    # Extra stats belong in artifact meta only
    extra_stats = extra_stats_from_ensemble_report(ensemble_report)

    classes_list = None
    try:
        classes = getattr(model, "classes_", None)
        if classes is not None:
            classes_list = [c.item() if hasattr(c, "item") else c for c in np.asarray(classes).tolist()]
    except Exception:
        classes_list = None

    artifact_inp = ArtifactBuilderInput(
        cfg=cfg,
        pipeline=model,
        n_train=n_train,
        n_test=n_test,
        n_features=(int(np.asarray(X).shape[1]) if np.asarray(X).ndim == 2 else None),
        classes=classes_list,
        summary={
            "metric_name": result.get("metric_name"),
            "metric_value": result.get("metric_value"),
            "mean_score": result.get("mean_score"),
            "std_score": result.get("std_score"),
            "n_splits": result.get("n_splits"),
            "notes": result.get("notes", []),
            "extra_stats": extra_stats,
        },
        kind=eval_kind,
    )

    meta = build_model_artifact_meta(artifact_inp)

    # Cache fitted pipeline for quick access
    try:
        artifact_cache.put(str(meta["uid"]), model)
    except Exception as e:
        # Cache failures should not abort training, but must be visible.
        try:
            notes = result.get("notes") if isinstance(result, dict) else None
            if isinstance(notes, list):
                notes.append(f"artifact_cache.put failed: {type(e).__name__}: {e}")
        except Exception:
            pass

    # Cache eval outputs (used for exports)
    try:
        eval_outputs_cache.put(
            str(meta["uid"]),
            EvalOutputs(
                task=eval_kind,
                indices=row_indices,
                fold_ids=fold_ids,
                y_true=y_true,
                y_pred=y_pred,
                classes=classes_list,
            ),
        )
    except Exception as e:
        # Cache failures should not abort training, but must be visible.
        try:
            notes = result.get("notes") if isinstance(result, dict) else None
            if isinstance(notes, list):
                notes.append(f"eval_outputs_cache.put failed: {type(e).__name__}: {e}")
        except Exception:
            pass

    # Persist artifact bytes + meta
    save_model_to_store(store_resolved, model, meta)

    result["artifact"] = meta
    return meta
