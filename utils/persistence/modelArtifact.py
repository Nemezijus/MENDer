from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timezone
import uuid

@dataclass
class ArtifactBuilderInput:
    cfg: Any                 
    pipeline: Any            # fitted sklearn Pipeline (or compatible)
    n_train: Optional[int]
    n_test: Optional[int]
    n_features: Optional[int]
    classes: Optional[List[Any]]
    summary: Dict[str, Any]  # scores, notes, n_splits, metric name/value

def _safe_params_dict(step) -> Dict[str, Any]:
    out = {}
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

def build_model_artifact_meta(inp: ArtifactBuilderInput) -> Dict[str, Any]:
    """Return a dict compatible with ModelArtifactMeta(**dict)."""
    steps: List[Dict[str, Any]] = []
    try:
        for name, step in inp.pipeline.steps:  # sklearn Pipeline API
            steps.append({
                "name": name,
                "class_path": f"{step.__class__.__module__}.{step.__class__.__name__}",
                "params": _safe_params_dict(step)
            })
    except Exception:
        steps = []

    split_dict    = getattr(inp.cfg, "split", None)
    scale_dict    = getattr(inp.cfg, "scale", None)
    features_dict = getattr(inp.cfg, "features", None)
    model_dict    = getattr(inp.cfg, "model", None)
    eval_dict     = getattr(inp.cfg, "eval", None)

    split_dict    = split_dict.model_dump(exclude_none=True)    if split_dict    is not None else None
    scale_dict    = scale_dict.model_dump(exclude_none=True)    if scale_dict    is not None else None
    features_dict = features_dict.model_dump(exclude_none=True) if features_dict is not None else None
    model_dict    = model_dict.model_dump(exclude_none=True)    if model_dict    is not None else None
    eval_dict     = eval_dict.model_dump(exclude_none=True)     if eval_dict     is not None else None

    return {
        "uid": str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc),
        "mender_version": None,
        "kind": "classification",
        "n_samples_train": inp.n_train,
        "n_samples_test": inp.n_test,
        "n_features_in": inp.n_features,
        "classes": inp.classes,
        "split": split_dict,
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
    }