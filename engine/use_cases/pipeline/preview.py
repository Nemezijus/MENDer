from __future__ import annotations

from typing import Any, Dict, List

from engine.contracts.run_config import RunConfig
from engine.factories.pipeline_factory import make_pipeline
from engine.runtime.random.rng import RngManager


def _class_path(obj: Any) -> str:
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


def preview_pipeline(cfg: RunConfig) -> Dict[str, Any]:
    """Build (but do not fit) the sklearn pipeline and return a preview payload.

    Returns a backend-friendly dict:
        { ok: bool, steps: [...], notes: [...], errors: [...] }

    This is a *use-case* because it orchestrates multiple Engine parts:
    - RNG manager
    - pipeline factory
    - best-effort introspection/serialization
    """

    errors: List[str] = []
    notes: List[str] = []

    rngm = RngManager(None if cfg.eval.seed is None else int(cfg.eval.seed))

    try:
        pipe = make_pipeline(cfg, rngm, stream="preview")
    except Exception as e:
        return {"ok": False, "steps": [], "notes": notes, "errors": [str(e)]}

    steps_out = []
    for name, step in pipe.steps:
        steps_out.append(
            {
                "name": name,
                "class_path": _class_path(step),
                "params": _params_dict(step),
            }
        )

    # Optional notes (best-effort)
    try:
        if getattr(cfg.features, "method", None) == "lda":
            notes.append("LDA requires labels during fit; preview only instantiates the step.")
        if getattr(cfg.features, "method", None) == "sfs":
            notes.append("SFS builds an inner estimator from your ModelConfig during feature selection.")
    except Exception:
        pass

    return {"ok": True, "steps": steps_out, "notes": notes, "errors": errors}
