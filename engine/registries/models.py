from __future__ import annotations

from typing import Callable, Optional, Type

from shared_schemas.model_configs import ModelConfig

from engine.registries.base import Registry

# NOTE: These interfaces currently live under utils. As we migrate fully to engine,
# these Protocols/classes will move as well. Keeping them here avoids changing
# large call sites during the refactor.
from utils.strategies.interfaces import ModelBuilder


# Factory takes (cfg, seed) and returns a ModelBuilder.
ModelBuilderFactory = Callable[[ModelConfig, Optional[int]], ModelBuilder]


_BUILDERS_BY_CONFIG: Registry[Type[ModelConfig], ModelBuilderFactory] = Registry(
    _name="model_builders_by_config"
)
_BUILDERS_BY_ALGO: Registry[str, ModelBuilderFactory] = Registry(_name="model_builders_by_algo")

_BUILTINS_LOADED = False


def register_model_builder(
    config_type: Type[ModelConfig],
    *,
    algo: Optional[str] = None,
) -> Callable[[ModelBuilderFactory], ModelBuilderFactory]:
    """Decorator to register a ModelBuilder factory.

    This enables "add a model = register it" without editing factory if/else chains.

    Parameters
    ----------
    config_type:
        Pydantic config type (e.g. LogRegConfig).
    algo:
        Optional algorithm key (e.g. "logreg"). If supplied, also registers by algo.
    """

    def deco(factory: ModelBuilderFactory) -> ModelBuilderFactory:
        _BUILDERS_BY_CONFIG.register(config_type)(factory)
        if algo is not None:
            _BUILDERS_BY_ALGO.register(str(algo))(factory)
        return factory

    return deco


def _ensure_builtins() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    # Import triggers registration side-effects.
    from engine.registries.builtins import models as _  # noqa: F401

    _BUILTINS_LOADED = True


def make_model_builder(cfg: ModelConfig, *, seed: Optional[int] = None) -> ModelBuilder:
    """Return a ModelBuilder for the provided config."""

    _ensure_builtins()

    t = type(cfg)
    factory = _BUILDERS_BY_CONFIG.try_get(t)

    # Allow config inheritance via MRO fallback.
    if factory is None:
        for base in t.mro()[1:]:
            factory = _BUILDERS_BY_CONFIG.try_get(base)
            if factory is not None:
                break

    if factory is None:
        algo = getattr(cfg, "algo", None)
        if algo is not None:
            factory = _BUILDERS_BY_ALGO.try_get(str(algo))

    if factory is None:
        raise ValueError(f"Unsupported algo: {getattr(cfg, 'algo', None)} ({t.__name__})")

    return factory(cfg, seed)


def list_model_algos() -> list[str]:
    _ensure_builtins()
    return sorted(list(_BUILDERS_BY_ALGO.keys()))
