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

    return {
        "uid": str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc),
        "mender_version": None,
        "kind": "classification",
        "n_samples_train": inp.n_train,
        "n_samples_test": inp.n_test,
        "n_features_in": inp.n_features,
        "classes": inp.classes,
        "split": inp.cfg.split.__dict__,
        "scale": getattr(inp.cfg, "scale", None).__dict__ if getattr(inp.cfg, "scale", None) else None,
        "features": getattr(inp.cfg, "features", None).__dict__ if getattr(inp.cfg, "features", None) else None,
        "model": inp.cfg.model.__dict__,
        "eval": inp.cfg.eval.__dict__,
        "pipeline": steps,
        "metric_name": inp.summary.get("metric_name"),
        "metric_value": inp.summary.get("metric_value"),
        "mean_score": inp.summary.get("mean_score"),
        "std_score": inp.summary.get("std_score"),
        "n_splits": inp.summary.get("n_splits"),
        "notes": inp.summary.get("notes", []),
    }
