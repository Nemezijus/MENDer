# utils/strategies/models.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional

from sklearn.linear_model import LogisticRegression

from utils.configs.configs import ModelConfig
from utils.strategies.interfaces import ModelBuilder


@dataclass
class LogRegBuilder(ModelBuilder):
    """Builds a configured LogisticRegression from ModelConfig."""
    cfg: ModelConfig

    def build(self) -> Any:
        # Basic validation + gentle guards for future extensibility
        penalty = self.cfg.penalty
        solver = self.cfg.solver

        # Map a few common “gotchas”
        if penalty == "none" and solver in ("liblinear",):
            # liblinear requires l1/l2; switch to lbfgs/newton-cg/saga for 'none'
            solver = "lbfgs"

        kwargs = dict(
            C=self.cfg.C,
            penalty=penalty,
            solver=solver,
            max_iter=self.cfg.max_iter,
            class_weight=self.cfg.class_weight,   # None or "balanced"
            multi_class="auto",
        )

        return LogisticRegression(**kwargs)
