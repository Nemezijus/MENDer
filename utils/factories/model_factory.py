# utils/factories/model_factory.py
from __future__ import annotations

from shared_schemas.model_configs import ModelConfig

from engine.registries.models import make_model_builder

# NOTE: interface currently in utils; will migrate to engine later.
from utils.strategies.interfaces import ModelBuilder


def make_model(cfg: ModelConfig, *, seed: int | None = None) -> ModelBuilder:
    """Thin wrapper around engine registries.

    Kept for backwards compatibility with existing call sites.
    """

    return make_model_builder(cfg, seed=seed)
