from __future__ import annotations
from typing import Optional

from utils.configs.configs import SplitConfig
from utils.strategies.splitters import HoldOutSplitter, KFoldSplitter
from utils.strategies.interfaces import Splitter

def make_splitter(cfg: SplitConfig, seed: Optional[int] = None) -> Splitter:
    mode = getattr(cfg, "mode", "holdout")
    if mode == "kfold":
        return KFoldSplitter(cfg=cfg, seed=seed)
    # default: hold-out splitter
    return HoldOutSplitter(cfg=cfg, seed=seed)
