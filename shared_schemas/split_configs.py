from __future__ import annotations

from typing import Literal
from pydantic import BaseModel

class SplitHoldoutModel(BaseModel):
    mode: Literal["holdout"] = "holdout"
    train_frac: float = 0.75
    stratified: bool = True
    shuffle: bool = True

class SplitCVModel(BaseModel):
    mode: Literal["kfold"] = "kfold"
    n_splits: int = 5
    stratified: bool = True
    shuffle: bool = True
