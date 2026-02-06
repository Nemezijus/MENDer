from __future__ import annotations

from typing import Callable, Optional, Union

from engine.registries.base import Registry

from shared_schemas.split_configs import SplitHoldoutModel, SplitCVModel

from utils.strategies.interfaces import Splitter

SplitConfig = Union[SplitHoldoutModel, SplitCVModel]

SplitterFactory = Callable[[SplitConfig, Optional[int]], Splitter]

_SPLITTERS: Registry[str, SplitterFactory] = Registry(_name="splitters")

_BUILTINS_LOADED = False


def register_splitter(mode: str) -> Callable[[SplitterFactory], SplitterFactory]:
    return _SPLITTERS.register(mode.lower())


def _ensure_builtins() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    from engine.registries.builtins import splitters as _  # noqa: F401
    _BUILTINS_LOADED = True


def make_splitter(cfg: SplitConfig, *, seed: Optional[int] = None) -> Splitter:
    _ensure_builtins()
    mode = getattr(cfg, "mode", "holdout")
    factory = _SPLITTERS.try_get(str(mode).lower())
    if factory is None:
        raise ValueError(f"Unknown split mode: {mode!r}")
    return factory(cfg, seed)


def list_split_modes() -> list[str]:
    _ensure_builtins()
    return sorted(list(_SPLITTERS.keys()))
