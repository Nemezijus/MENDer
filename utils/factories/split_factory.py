# utils/factories/split_factory.py
from __future__ import annotations
from typing import Optional

from utils.configs.configs import SplitConfig
from utils.strategies.splitters import StratifiedSplitter
from utils.strategies.interfaces import Splitter

def make_splitter(
    cfg: SplitConfig,
    *,
    seed: Optional[int],
    use_custom: bool = False,
) -> Splitter:
    """
    Create a splitter strategy from config.
    Currently returns a stratified splitter that wraps your existing trial_split.split.
    """
    return StratifiedSplitter(cfg=cfg, seed=seed, use_custom=use_custom)
