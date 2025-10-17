from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional

from sklearn.linear_model import LogisticRegression

from utils.configs.configs import ModelConfig
from utils.strategies.interfaces import ModelBuilder


@dataclass
class LogRegBuilder(ModelBuilder):
    cfg: ModelConfig

    # NEW
    def make_estimator(self) -> Any:
        penalty = self.cfg.penalty
        solver = self.cfg.solver
        if penalty == "none" and solver in ("liblinear",):
            solver = "lbfgs"

        return LogisticRegression(
            C=self.cfg.C,
            penalty=penalty,
            solver=solver,
            max_iter=self.cfg.max_iter,
            class_weight=self.cfg.class_weight,   # None or "balanced"
            multi_class="auto",
        )

    # keep old name as an alias
    def build(self) -> Any:
        return self.make_estimator()
