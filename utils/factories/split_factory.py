from __future__ import annotations
from typing import Optional

from utils.configs.configs import SplitConfig
from utils.strategies.splitters import StratifiedSplitter, KFoldSplitter
# You don’t actually need to import the Splitter interface for this file to work,
# but if you want the return type:
from utils.strategies.interfaces import Splitter

def make_splitter(cfg: SplitConfig, seed: Optional[int] = None) -> Splitter:
    mode = getattr(cfg, "mode", "holdout")
    if mode == "kfold":
        return KFoldSplitter(cfg=cfg, seed=seed)
    # default: your existing hold-out splitter
    return StratifiedSplitter(cfg=cfg, seed=seed)
