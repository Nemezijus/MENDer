# utils/strategies/splitters.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np

from utils.configs.configs import SplitConfig
from utils.preprocessing.general.trial_split import split as split_trials

@dataclass
class StratifiedSplitter:
    cfg: SplitConfig
    seed: Optional[int] = None
    use_custom: bool = False

    def split(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return split_trials(
            X, y,
            train_frac=self.cfg.train_frac,
            custom=self.use_custom,
            rng=self.seed,
        )
