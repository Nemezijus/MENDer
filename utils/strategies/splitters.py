from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, Iterator
import numpy as np

from shared_schemas.split_configs import SplitHoldoutModel, SplitCVModel
from utils.preprocessing.general.trial_split import split as split_trials
from utils.preprocessing.general.cv_split import generate_folds
from .interfaces import Splitter

@dataclass
class HoldOutSplitter(Splitter):
    cfg: SplitHoldoutModel
    seed: Optional[int] = None
    use_custom: bool = False

    def split(self, X: np.ndarray, y: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        Xtr, Xte, ytr, yte = split_trials(
            X, y,
            train_frac=self.cfg.train_frac,
            custom=self.use_custom,
            rng=self.seed,
        )
        yield Xtr, Xte, ytr, yte
    
@dataclass
class KFoldSplitter(Splitter):
    cfg: SplitCVModel
    seed: Optional[int] = None

    def split(self, X: np.ndarray, y: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        yield from generate_folds(
            X, y,
            n_splits=self.cfg.n_splits,
            stratified=self.cfg.stratified,
            shuffle=self.cfg.shuffle,
            random_state=(self.seed if self.cfg.shuffle else None),
        )