from __future__ import annotations

"""Schema / defaults endpoints.

The backend must not hardcode engine contract inventories.
All schema + defaults + enum bundles are generated inside the Engine BL and
accessed only via :mod:`engine.api`.
"""

from typing import Any

from fastapi import APIRouter

import engine.api as api


# NOTE: Versioning (/api/v1) is applied in backend/app/main.py.
router = APIRouter(prefix="/schema")


def _bundle() -> dict[str, Any]:
    # Keep as a tiny helper to avoid recomputing slices inconsistently.
    return api.get_ui_schema_bundle()


@router.get("/defaults")
def get_all_defaults() -> dict[str, Any]:
    """Consolidated defaults + enums for the UI."""

    return _bundle()


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
