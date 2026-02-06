from __future__ import annotations

from typing import Callable, Optional

from engine.registries.base import Registry

from shared_schemas.eval_configs import EvalModel
from shared_schemas.feature_configs import FeaturesModel
from shared_schemas.model_configs import ModelConfig

from utils.strategies.interfaces import FeatureExtractor


FeatureFactory = Callable[
    [FeaturesModel, Optional[int], Optional[ModelConfig], Optional[EvalModel]],
    FeatureExtractor,
]

_FEATURE_EXTRACTORS: Registry[str, FeatureFactory] = Registry(_name="feature_extractors")

_BUILTINS_LOADED = False


def register_feature_extractor(method: str) -> Callable[[FeatureFactory], FeatureFactory]:
    return _FEATURE_EXTRACTORS.register(method.lower())


def _ensure_builtins() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return

    from engine.registries.builtins import features as _  # noqa: F401

    _BUILTINS_LOADED = True


def make_feature_extractor(
    cfg: FeaturesModel,
    *,
    seed: Optional[int],
    model_cfg: Optional[ModelConfig] = None,
    eval_cfg: Optional[EvalModel] = None,
) -> FeatureExtractor:
    """Return a FeatureExtractor for the given FeaturesModel config."""

    _ensure_builtins()

    method = (cfg.method or "none").lower()
    factory = _FEATURE_EXTRACTORS.try_get(method)
    if factory is None:
        raise ValueError(f"Unknown feature extractor method: {cfg.method}")

    return factory(cfg, seed, model_cfg, eval_cfg)


def list_feature_methods() -> list[str]:
    _ensure_builtins()
    return sorted(list(_FEATURE_EXTRACTORS.keys()))
