from __future__ import annotations

from typing import Callable

from engine.registries.base import Registry

from engine.contracts.ensemble_run_config import EnsembleRunConfig

from engine.components.interfaces import EnsembleBuilder

EnsembleFactory = Callable[[EnsembleRunConfig], EnsembleBuilder]

_ENSEMBLES: Registry[str, EnsembleFactory] = Registry(_name="ensembles")

_BUILTINS_LOADED = False


def register_ensemble_kind(kind: str) -> Callable[[EnsembleFactory], EnsembleFactory]:
    return _ENSEMBLES.register(kind.lower())


def _ensure_builtins() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    from engine.registries.builtins import ensembles as _  # noqa: F401
    _BUILTINS_LOADED = True


def make_ensemble_strategy(cfg: EnsembleRunConfig) -> EnsembleBuilder:
    _ensure_builtins()
    kind = str(cfg.ensemble.kind).lower()
    factory = _ENSEMBLES.try_get(kind)
    if factory is None:
        raise ValueError(f"Unknown ensemble kind: {kind!r}")
    return factory(cfg)


def list_ensemble_kinds() -> list[str]:
    _ensure_builtins()
    return sorted(list(_ENSEMBLES.keys()))
