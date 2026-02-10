"""Built-in train/test split registrations."""

from __future__ import annotations

from typing import Optional

from engine.registries.splitters import SplitConfig, SplitterFactory, register_splitter

from engine.components.splitters.splitters import HoldOutSplitter, KFoldSplitter


@register_splitter("holdout")
def _holdout(cfg: SplitConfig, seed: Optional[int]):
    return HoldOutSplitter(cfg=cfg, seed=seed)


@register_splitter("kfold")
def _kfold(cfg: SplitConfig, seed: Optional[int]):
    return KFoldSplitter(cfg=cfg, seed=seed)
