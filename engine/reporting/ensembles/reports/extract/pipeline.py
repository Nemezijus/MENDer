from __future__ import annotations

from typing import Any, Tuple

from engine.types.sklearn import SkModel, SkEstimator

from engine.reporting.ensembles.helpers import _transform_through_pipeline, _unwrap_final_estimator


def resolve_estimator_and_X(model: SkModel, X: Any) -> Tuple[SkEstimator | Any, Any]:
    """Resolve the reporting estimator and the feature matrix it should see.

    Ensembles in MENDer are sometimes returned as:
      - a bare estimator (e.g. BaggingClassifier)
      - a Pipeline(..., clf=<estimator>)
      - a Pipeline(..., clf=<adapter(model=<estimator>)>)

    For reporting we often need both:
      1) the *inner* estimator (for attributes like estimators_, weights_, etc.)
      2) the appropriately transformed X (when the estimator lived inside a pipeline)

    This function is best-effort and never raises.
    """

    est = model
    X_out = X

    # Preferred: named_steps['clf'] for your pipeline convention.
    try:
        if hasattr(model, "named_steps") and isinstance(getattr(model, "named_steps"), dict):
            clf_step = model.named_steps.get("clf", None)
            if clf_step is not None:
                est = clf_step

            # If a dedicated preprocessor step exists (legacy naming), use it.
            pre_step = model.named_steps.get("pre", None)
            if pre_step is not None and hasattr(pre_step, "transform"):
                try:
                    X_out = pre_step.transform(X)
                except Exception:
                    X_out = X
            else:
                # Otherwise transform through all non-final steps.
                try:
                    X_out = _transform_through_pipeline(model, X)
                except Exception:
                    X_out = X
        else:
            # Fall back: unwrap the final estimator, and transform through pipeline
            # only if the model was actually a pipeline.
            inner = _unwrap_final_estimator(model)
            if inner is not model:
                est = inner
                try:
                    X_out = _transform_through_pipeline(model, X)
                except Exception:
                    X_out = X
            else:
                est = model
                X_out = X
    except Exception:
        est = model
        X_out = X

    # Unwrap common adapter shape: adapter.model
    est = getattr(est, "model", est)
    return est, X_out
