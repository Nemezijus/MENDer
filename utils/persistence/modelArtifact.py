from typing import Any, Dict, List, Optional, Tuple, Literal
from dataclasses import dataclass
from datetime import datetime, timezone
import uuid

import numpy as np


@dataclass
class ArtifactBuilderInput:
    cfg: Any
    pipeline: Any            # fitted sklearn Pipeline (or compatible)
    n_train: Optional[int]
    n_test: Optional[int]
    n_features: Optional[int]
    classes: Optional[List[Any]]
    summary: Dict[str, Any]  # scores, notes, n_splits, metric name/value
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
        # else: drop non-serializables
    return out


def _estimate_param_stats(pipeline: Any) -> Tuple[Optional[int], Dict[str, Any]]:
    """Best-effort model complexity stats from the fitted pipeline.

    Returns (n_parameters, extra_stats) where:
      - n_parameters is primarily defined for linear models (coef_ + intercept_).
      - extra_stats may contain keys like:
          * n_support_vectors
          * n_trees, total_tree_nodes, max_tree_depth
          * pca_n_components (if a PCA step is present)

    This must never raise – on failure it returns (None, {}).
    """
    n_parameters: Optional[int] = None
    extra: Dict[str, Any] = {}

    try:
        est = pipeline
        # If this is a sklearn Pipeline, take the final step as the estimator.
        if hasattr(pipeline, "steps"):
            try:
                est = pipeline.steps[-1][1]
            except Exception:
                est = pipeline

        # 1) Linear-style models: coef_ (+ optional intercept_)
        coef = getattr(est, "coef_", None)
        if coef is not None:
            c = np.asarray(coef)
            n_parameters = int(c.size)
            intercept = getattr(est, "intercept_", None)
            if intercept is not None:
                b = np.asarray(intercept)
                n_parameters += int(b.size)
            extra["has_coef_"] = True

        # 2) Support-vector models
        sv = getattr(est, "support_vectors_", None)
        if sv is not None:
            sv_arr = np.asarray(sv)
            extra["n_support_vectors"] = int(sv_arr.shape[0])

        # 3) Tree ensembles (RandomForest*, GradientBoosting*, etc.)
        estimators = getattr(est, "estimators_", None)
        if estimators is not None:
            # estimators_ may be 1D list or array-like
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

        # 4) Single decision tree
        if "total_tree_nodes" not in extra:
            t = getattr(est, "tree_", None)
            if t is not None:
                extra["total_tree_nodes"] = int(getattr(t, "node_count", 0))
                extra["max_tree_depth"] = int(getattr(t, "max_depth", 0))

        # 5) PCA step: look through pipeline steps for a PCA-like transformer.
        #    We only need n_components_ from the fitted object.
        try:
            if hasattr(pipeline, "steps"):
                for name, step in pipeline.steps:
                    # Heuristic: PCA has attribute n_components_ after fit.
                    n_comp_attr = getattr(step, "n_components_", None)
                    if n_comp_attr is not None:
                        try:
                            extra["pca_n_components"] = int(n_comp_attr)
                        except Exception:
                            pass
                        break
        except Exception:
            # never let stats gathering break artifact creation
            pass

    except Exception:
        # Best-effort only; ignore any errors
        return None, {}

    return n_parameters, extra


def build_model_artifact_meta(inp: ArtifactBuilderInput) -> Dict[str, Any]:
    """Return a dict compatible with ModelArtifactMeta(**dict)."""
    
    steps: List[Dict[str, Any]] = []
    try:
        for name, step in inp.pipeline.steps:  # sklearn Pipeline API
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

    # If this is an ensemble run config, cfg.model doesn't exist -> make a valid dict
    if model_dict is None and ensemble_dict is not None:
        model_dict = {
            "algo": "ensemble",
            "ensemble_kind": ensemble_dict.get("kind"),
            "ensemble": ensemble_dict,
        }

    # Model complexity stats
    n_parameters, extra_stats = _estimate_param_stats(inp.pipeline)
    extra_from_summary = inp.summary.get("extra_stats") if isinstance(inp.summary, dict) else None
    if isinstance(extra_from_summary, dict):
        for k, v in extra_from_summary.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                extra_stats[k] = v
    kind = inp.kind
    if kind is None:
        # Try to infer from model config's `task` attribute (ClassVar or instance)
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
            # Codebase convention: treat all unsupervised clustering runs under a single kind.
            kind = "unsupervised"
        else:
            # Defensive fallback – should not normally happen
            kind = "classification"
    # -------------------------------------------------

    return {
        "uid": str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc),
        "mender_version": None,
        "kind": kind,
        "n_samples_train": inp.n_train,
        "n_samples_test": inp.n_test,
        "n_features_in": inp.n_features,
        "classes": inp.classes,
        # Backend schema historically required split to be a dict.
        # For unsupervised runs there is no split; keep an empty dict for compatibility.
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
