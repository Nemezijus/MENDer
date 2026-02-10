from __future__ import annotations
from typing import Dict, Any, List

from engine.contracts.run_config import RunConfig, DataModel
from engine.contracts.split_configs import SplitHoldoutModel
from engine.contracts.scale_configs import ScaleModel
from engine.contracts.feature_configs import FeaturesModel
from engine.contracts.model_configs import ModelConfig
from engine.contracts.eval_configs import EvalModel

from engine.runtime.random.rng import RngManager
from engine.factories.pipeline_factory import make_pipeline


def _class_path(obj: Any) -> str:
    if isinstance(obj, str):  # e.g., "passthrough"
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
    Build (but do not fit) the pipeline using your factory.
    Returns step class paths and params; surfaces construction errors.
    """
    errors: List[str] = []
    notes: List[str] = []

    # Build a minimal, valid RunConfig.
    # Data/split are irrelevant for preview; use a dummy holdout split.
    cfg = RunConfig(
        data=DataModel(),                          # not used in preview
        split=SplitHoldoutModel(),                 # satisfies schema; not used for construction
        scale=ScaleModel(**payload.scale.model_dump()),
        features=FeaturesModel(**payload.features.model_dump()),
        model=ModelConfig(**payload.model.model_dump()),
        eval=EvalModel(**payload.eval.model_dump()),
    )

    # Deterministic seed manager (if provided)
    rngm = RngManager(None if cfg.eval.seed is None else int(cfg.eval.seed))

    # Try to construct pipeline using the factory
    try:
        pipe = make_pipeline(cfg, rngm, stream="preview")
    except Exception as e:
        return {
            "ok": False,
            "steps": [],
            "notes": notes,
            "errors": [str(e)],
        }

    # Introspect the steps
    steps_out = []
    for name, step in pipe.steps:
        steps_out.append({
            "name": name,
            "class_path": _class_path(step),
            "params": _params_dict(step),
        })

    # Optional notes
    if cfg.features.method == "lda":
        notes.append("LDA requires labels during fit; preview only instantiates the step.")
    if cfg.features.method == "sfs":
        notes.append("SFS builds an inner estimator from your ModelConfig during feature selection.")

    return {
        "ok": True,
        "steps": steps_out,
        "notes": notes,
        "errors": [],
    }
