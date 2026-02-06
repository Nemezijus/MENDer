from .artifacts import (
    load_model_bytes,
    load_model_from_store,
    save_model_bytes,
    save_model_to_store,
)

# Segment 12 (Patch 12A): façade entry points.
# Implementations will be added incrementally; the public surface is
# established here so scripts/backend can depend on stable imports.
from .facade import (
    grid_search,
    predict,
    random_search,
    train_ensemble,
    train_supervised,
    train_unsupervised,
    tune_learning_curve,
    tune_validation_curve,
)

__all__ = [
    "save_model_bytes",
    "load_model_bytes",
    "save_model_to_store",
    "load_model_from_store",

    # Segment 12 façade
    "train_supervised",
    "train_unsupervised",
    "train_ensemble",
    "predict",
    "tune_learning_curve",
    "tune_validation_curve",
    "grid_search",
    "random_search",
]
