from __future__ import annotations

"""Small sklearn-centric helpers used across BL.

These helpers are used in multiple places (prediction extraction, diagnostics,
and ensemble reporting). Keeping them here avoids subtle drift.

Implementation notes
--------------------
* Uses duck-typing instead of importing sklearn at import time.
* For pipelines, we only transform through steps that expose `.transform`.
  If an unknown step is encountered, we conservatively return the original X.
"""

from typing import Any


def unwrap_final_estimator(model: Any) -> Any:
    """Return the final estimator for a Pipeline-like model, else the model itself."""
    try:
        steps = getattr(model, "steps", None)
        if isinstance(steps, list) and len(steps) > 0:
            return steps[-1][1]
    except Exception:
        pass
    return model


def transform_through_pipeline(model: Any, X: Any) -> Any:
    """Transform X through a fitted sklearn Pipeline excluding its final estimator.

    If the model is not a pipeline (or has no transformers), returns X unchanged.
    If a step does not support `.transform`, returns X unchanged.
    """
    try:
        steps = getattr(model, "steps", None)
        if not (isinstance(steps, list) and len(steps) > 1):
            return X

        Xt = X
        for _, step in steps[:-1]:
            if step is None or step == "passthrough":
                continue
            if hasattr(step, "transform"):
                Xt = step.transform(Xt)
            else:
                return X
        return Xt
    except Exception:
        return X


__all__ = ["unwrap_final_estimator", "transform_through_pipeline"]
