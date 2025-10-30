from __future__ import annotations
from typing import Dict, Any, List
from dataclasses import asdict

# Reuse your config dataclasses and pipeline factory
from utils.configs.configs import (
    RunConfig, DataConfig, SplitConfig, ScaleConfig, FeatureConfig, ModelConfig, EvalConfig
)
from utils.permutations.rng import RngManager
from utils.factories.pipeline_factory import make_pipeline

def _class_path(obj: Any) -> str:
    # sklearn transformers/estimators or literal "passthrough"
    if isinstance(obj, str):
        return obj
    cls = obj.__class__
    return f"{cls.__module__}.{cls.__name__}"

def _params_dict(obj: Any) -> Dict[str, Any]:
    if hasattr(obj, "get_params"):
        try:
            return obj.get_params(deep=False)
        except Exception:
            return {}
    return {}

def preview_pipeline(payload) -> Dict[str, Any]:
    """
    Build (but do not fit) the pipeline using your factory. If construction succeeds,
    return step class paths and params; otherwise surface the error in a friendly shape.
    """
    errors: List[str] = []
    notes: List[str] = []

    # Build full RunConfig (data/split are irrelevant for dry-fit; use defaults)
    cfg = RunConfig(
        data=DataConfig(),  # not used
        split=SplitConfig(),  # not used
        scale=ScaleConfig(**payload.scale.model_dump()),
        features=FeatureConfig(**payload.features.model_dump()),
        model=ModelConfig(**payload.model.model_dump()),
        eval=EvalConfig(**payload.eval.model_dump()),
    )

    # Deterministic seed manager (if provided)
    rngm = RngManager(None if cfg.eval.seed is None else int(cfg.eval.seed))

    # Try to construct pipeline using your factory
    try:
        pipe = make_pipeline(cfg, rngm, stream="real")
    except Exception as e:
        return {
            "ok": False,
            "steps": [],
            "notes": notes,
            "errors": [str(e)],
        }

    # Introspect the three steps
    steps_out = []
    for name, step in pipe.steps:
        steps_out.append({
            "name": name if name in ("scale", "feat", "clf") else name,
            "class_path": _class_path(step),
            "params": _params_dict(step),
        })

    # Optional notes (examples)
    if cfg.features.method == "lda":
        notes.append("LDA requires labels during fit; preview only instantiates the step.")
    if cfg.features.method == "sfs":
        notes.append("SFS will build an inner estimator from your ModelConfig during feature selection.")

    return {
        "ok": True,
        "steps": steps_out,
        "notes": notes,
        "errors": [],
    }
