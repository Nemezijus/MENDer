from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import uuid

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from engine.contracts.ensemble_run_config import EnsembleRunConfig
from engine.contracts.run_config import RunConfig
from engine.contracts.unsupervised_configs import UnsupervisedRunConfig

from .meta_models import ArtifactSummary, ModelArtifactMetaDict


@dataclass
class ArtifactBuilderInput:
    """Inputs required to build a ModelArtifactMeta-like dict."""

    cfg: Union[RunConfig, EnsembleRunConfig, UnsupervisedRunConfig]
    pipeline: Union[Pipeline, BaseEstimator]  # fitted sklearn Pipeline or estimator
    n_train: Optional[int]
    n_test: Optional[int]
    n_features: Optional[int]
    classes: Optional[List[Any]]
    summary: ArtifactSummary  # scores, notes, n_splits, metric name/value
    kind: Optional[Literal["classification", "regression", "unsupervised"]] = None


def _safe_params_dict(step) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        params = step.get_params()
    except Exception:
        return out
    for k, v in params.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        elif k == "cv":
            n = getattr(v, "n_splits", None)
            out["cv_n_splits"] = int(n) if n is not None else None
    return out


def _estimate_param_stats(pipeline: Union[Pipeline, BaseEstimator]) -> Tuple[Optional[int], Dict[str, Any]]:
    """Best-effort model complexity stats from the fitted pipeline.

    Returns (n_parameters, extra_stats). Must never raise.
    """

    n_parameters: Optional[int] = None
    extra: Dict[str, Any] = {}

    try:
        est = pipeline
        if hasattr(pipeline, "steps"):
            try:
                est = pipeline.steps[-1][1]
            except Exception:
                est = pipeline

        # Linear-like models
        coef = getattr(est, "coef_", None)
        if coef is not None:
            c = np.asarray(coef)
            n_parameters = int(c.size)
            intercept = getattr(est, "intercept_", None)
            if intercept is not None:
                b = np.asarray(intercept)
                n_parameters += int(b.size)
            extra["has_coef_"] = True

        # SV models
        sv = getattr(est, "support_vectors_", None)
        if sv is not None:
            sv_arr = np.asarray(sv)
            extra["n_support_vectors"] = int(sv_arr.shape[0])

        # Tree ensembles
        estimators = getattr(est, "estimators_", None)
        if estimators is not None:
            flat: List[Any] = []
            try:
                for e in estimators.ravel():
                    flat.append(e)
            except Exception:
                try:
                    flat = list(estimators)
                except Exception:
                    flat = []

            n_trees = len(flat)
            if n_trees:
                extra["n_trees"] = int(n_trees)

            total_nodes = 0
            max_depth = None
            for tree in flat:
                t = getattr(tree, "tree_", None)
                if t is None:
                    continue
                total_nodes += int(getattr(t, "node_count", 0))
                depth = getattr(t, "max_depth", None)
                if depth is not None:
                    max_depth = depth if max_depth is None else max(max_depth, depth)

            if total_nodes:
                extra["total_tree_nodes"] = int(total_nodes)
            if max_depth is not None:
                extra["max_tree_depth"] = int(max_depth)

        # Single tree
        if "total_tree_nodes" not in extra:
            t = getattr(est, "tree_", None)
            if t is not None:
                extra["total_tree_nodes"] = int(getattr(t, "node_count", 0))
                extra["max_tree_depth"] = int(getattr(t, "max_depth", 0))

        # PCA step stats
        try:
            if hasattr(pipeline, "steps"):
                for _, step in pipeline.steps:
                    n_comp_attr = getattr(step, "n_components_", None)
                    if n_comp_attr is not None:
                        try:
                            extra["pca_n_components"] = int(n_comp_attr)
                        except Exception:
                            pass
                        break
        except Exception:
            pass

    except Exception:
        return None, {}

    return n_parameters, extra


def build_model_artifact_meta(inp: ArtifactBuilderInput) -> ModelArtifactMetaDict:
    """Return a dict compatible with backend ModelArtifactMeta(**dict)."""

    steps: List[Dict[str, Any]] = []
    try:
        for name, step in inp.pipeline.steps:
            steps.append(
                {
                    "name": name,
                    "class_path": f"{step.__class__.__module__}.{step.__class__.__name__}",
                    "params": _safe_params_dict(step),
                }
            )
    except Exception:
        steps = []

    split_dict = getattr(inp.cfg, "split", None)
    scale_dict = getattr(inp.cfg, "scale", None)
    features_dict = getattr(inp.cfg, "features", None)
    model_dict = getattr(inp.cfg, "model", None)
    eval_dict = getattr(inp.cfg, "eval", None)

    split_dict = split_dict.model_dump(exclude_none=True) if split_dict is not None else None
    scale_dict = scale_dict.model_dump(exclude_none=True) if scale_dict is not None else None
    features_dict = features_dict.model_dump(exclude_none=True) if features_dict is not None else None
    model_dict = model_dict.model_dump(exclude_none=True) if model_dict is not None else None
    eval_dict = eval_dict.model_dump(exclude_none=True) if eval_dict is not None else None

    ensemble_cfg = getattr(inp.cfg, "ensemble", None)
    ensemble_dict = (
        ensemble_cfg.model_dump(exclude_none=True) if ensemble_cfg is not None else None
    )

    if model_dict is None and ensemble_dict is not None:
        model_dict = {
            "algo": "ensemble",
            "ensemble_kind": ensemble_dict.get("kind"),
            "ensemble": ensemble_dict,
        }

    n_parameters, extra_stats = _estimate_param_stats(inp.pipeline)

    extra_from_summary = inp.summary.get("extra_stats")
    if isinstance(extra_from_summary, dict):
        for k, v in extra_from_summary.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                extra_stats[k] = v

    kind = inp.kind
    if kind is None:
        model_cfg = getattr(inp.cfg, "model", None)
        task_attr = None
        if model_cfg is not None:
            task_attr = (
                getattr(model_cfg.__class__, "task", None)
                or getattr(model_cfg, "task", None)
            )
        if task_attr in ("classification", "regression"):
            kind = task_attr
        elif task_attr in ("clustering", "unsupervised"):
            kind = "unsupervised"
        else:
            kind = "classification"

    return {
        "uid": str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc),
        "mender_version": None,
        "kind": kind,
        "n_samples_train": inp.n_train,
        "n_samples_test": inp.n_test,
        "n_features_in": inp.n_features,
        "classes": inp.classes,
        "split": split_dict if split_dict is not None else {},
        "scale": scale_dict,
        "features": features_dict,
        "model": model_dict,
        "eval": eval_dict,
        "pipeline": steps,
        "metric_name": inp.summary.get("metric_name"),
        "metric_value": inp.summary.get("metric_value"),
        "mean_score": inp.summary.get("mean_score"),
        "std_score": inp.summary.get("std_score"),
        "n_splits": inp.summary.get("n_splits"),
        "notes": inp.summary.get("notes", []),
        "n_parameters": n_parameters,
        "extra_stats": extra_stats,
    }
