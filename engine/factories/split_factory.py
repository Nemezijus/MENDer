from __future__ import annotations

from typing import Optional, Union

from engine.contracts.split_configs import SplitHoldoutModel, SplitCVModel

from engine.registries.splitters import make_splitter as _make_splitter

from engine.components.interfaces import Splitter

SplitConfig = Union[SplitHoldoutModel, SplitCVModel]


def make_splitter(cfg: SplitConfig, seed: Optional[int] = None) -> Splitter:
    """Backwards-compatible wrapper around engine registries."""
    return _make_splitter(cfg, seed=seed)
