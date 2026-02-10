from __future__ import annotations

from engine.contracts.model_configs import ModelConfig

from engine.registries.models import make_model_builder

from engine.components.interfaces import ModelBuilder


def make_model(cfg: ModelConfig, *, seed: int | None = None) -> ModelBuilder:
    """Thin wrapper around engine registries.

    Kept for backwards compatibility with existing call sites.
    """

    return make_model_builder(cfg, seed=seed)
