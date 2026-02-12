from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Iterator
import numpy as np

from engine.contracts.split_configs import SplitHoldoutModel, SplitCVModel
from engine.components.splitters.trial_split import split as split_trials
from engine.components.splitters.cv_split import generate_folds
from engine.components.splitters.types import Split
from ..interfaces import Splitter


@dataclass
class HoldOutSplitter(Splitter):
    cfg: SplitHoldoutModel
    seed: Optional[int] = None
    use_custom: bool = False

    def split(self, X: np.ndarray, y: np.ndarray) -> Iterator[Split]:
        Xtr, Xte, ytr, yte = split_trials(
            X,
            y,
            train_frac=self.cfg.train_frac,
            custom=self.use_custom,
            stratify=self.cfg.stratified,
            rng=self.seed,
        )
        yield Split(Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte, idx_tr=None, idx_te=None)


@dataclass
class KFoldSplitter(Splitter):
    cfg: SplitCVModel
    seed: Optional[int] = None

    def split(self, X: np.ndarray, y: np.ndarray) -> Iterator[Split]:
        yield from generate_folds(
            X,
            y,
            n_splits=self.cfg.n_splits,
            stratified=self.cfg.stratified,
            shuffle=self.cfg.shuffle,
            random_state=(self.seed if self.cfg.shuffle else None),
        )
