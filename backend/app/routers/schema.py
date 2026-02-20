from __future__ import annotations

"""Schema / defaults endpoints.

The backend must not hardcode engine contract inventories.
All schema + defaults + enum bundles are generated inside the Engine BL and
accessed only via :mod:`engine.api`.
"""

from typing import Any

from fastapi import APIRouter

import engine.api as api

from ..models.v1.tuning_models import (
    LearningCurveRequest,
    ValidationCurveRequest,
    GridSearchRequest,
    RandomSearchRequest,
)


# NOTE: Versioning (/api/v1) is applied in backend/app/main.py.
router = APIRouter(prefix="/schema")


def _bundle() -> dict[str, Any]:
    # Keep as a tiny helper to avoid recomputing slices inconsistently.
    return api.get_ui_schema_bundle()


def _field_default(field_info: Any) -> Any:
    """Return a JSON-serializable default for a Pydantic field."""
    # Pydantic v2 FieldInfo exposes `default_factory` and `default`.
    df = getattr(field_info, "default_factory", None)
    if df is not None:
        return df()
    return getattr(field_info, "default", None)


@router.get("/defaults")
def get_all_defaults() -> dict[str, Any]:
    """Consolidated defaults + enums for the UI."""

    return _bundle()


@router.get("/tuning-defaults")
def get_tuning_defaults() -> dict[str, Any]:
    """Backend-owned tuning request defaults.

    These defaults live in backend request models (e.g. cv=5 for GridSearch).
    They are **not** Engine contract defaults, so they are exposed here to keep
    the frontend from hardcoding request-level defaults.
    """

    lc = {
        "train_sizes": _field_default(LearningCurveRequest.model_fields["train_sizes"]),
        "n_steps": _field_default(LearningCurveRequest.model_fields["n_steps"]),
        "n_jobs": _field_default(LearningCurveRequest.model_fields["n_jobs"]),
    }

    vc = {
        "n_jobs": _field_default(ValidationCurveRequest.model_fields["n_jobs"]),
    }

    gs = {
        "param_grid": _field_default(GridSearchRequest.model_fields["param_grid"]),
        "cv": _field_default(GridSearchRequest.model_fields["cv"]),
        "n_jobs": _field_default(GridSearchRequest.model_fields["n_jobs"]),
        "refit": _field_default(GridSearchRequest.model_fields["refit"]),
        "return_train_score": _field_default(
            GridSearchRequest.model_fields["return_train_score"]
        ),
    }

    rs = {
        "param_distributions": _field_default(
            RandomSearchRequest.model_fields["param_distributions"]
        ),
        "n_iter": _field_default(RandomSearchRequest.model_fields["n_iter"]),
        "cv": _field_default(RandomSearchRequest.model_fields["cv"]),
        "n_jobs": _field_default(RandomSearchRequest.model_fields["n_jobs"]),
        "refit": _field_default(RandomSearchRequest.model_fields["refit"]),
        "random_state": _field_default(RandomSearchRequest.model_fields["random_state"]),
        "return_train_score": _field_default(
            RandomSearchRequest.model_fields["return_train_score"]
        ),
    }

    return {
        "learning_curve": lc,
        "validation_curve": vc,
        "grid_search": gs,
        "random_search": rs,
    }


@router.get("/enums")
def get_enums() -> dict[str, Any]:
    """Expose centralized enum values for dropdowns."""

    b = _bundle()
    return {"enums": b.get("enums", {})}


@router.get("/model")
def get_model_schema() -> dict[str, Any]:
    """Schema + per-algo defaults for :class:`engine.contracts.model_configs.ModelConfig`."""

    b = _bundle()
    return {
        "schema": b["models"]["schema"],
        "defaults": b["models"]["defaults"],
    }


@router.get("/ensemble")
def get_ensemble_schema() -> dict[str, Any]:
    """Schema + per-kind defaults for :class:`engine.contracts.ensemble_configs.EnsembleConfig`."""

    b = _bundle()
    return {
        "schema": b["ensembles"]["schema"],
        "defaults": b["ensembles"]["defaults"],
    }


@router.get("/features")
def get_features_schema() -> dict[str, Any]:
    b = _bundle()
    return b["features"]


@router.get("/split/holdout")
def get_split_holdout_schema() -> dict[str, Any]:
    b = _bundle()
    return b["split"]["holdout"]


@router.get("/split/kfold")
def get_split_kfold_schema() -> dict[str, Any]:
    b = _bundle()
    return b["split"]["kfold"]


@router.get("/scale")
def get_scale_schema() -> dict[str, Any]:
    b = _bundle()
    return b["scale"]


@router.get("/eval")
def get_eval_schema() -> dict[str, Any]:
    b = _bundle()
    return b["eval"]
