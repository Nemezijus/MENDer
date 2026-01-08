from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator

from .model_configs import ModelConfig
from .types import EnsembleKind


# -----------------------------
# Voting
# -----------------------------

class VotingEstimatorSpec(BaseModel):
    """
    One base estimator inside a voting ensemble.

    Notes:
      - `weight` is preferred over duplicating identical estimators.
      - `name` is optional; BL can auto-generate stable names if omitted.
    """
    model: ModelConfig
    name: Optional[str] = None
    weight: Optional[float] = None


class VotingEnsembleConfig(BaseModel):
    kind: Literal["voting"] = "voting"

    voting: Literal["hard", "soft"] = "hard"
    estimators: list[VotingEstimatorSpec] = Field(default_factory=list, min_length=2)

    # sklearn VotingClassifier/VotingRegressor option (kept here for parity even if unused initially)
    flatten_transform: bool = True


# -----------------------------
# Bagging
# -----------------------------

class BaggingEnsembleConfig(BaseModel):
    kind: Literal["bagging"] = "bagging"

    # needed to choose BaggingClassifier vs BaggingRegressor when base_estimator is None
    problem_kind: Literal["classification", "regression"] = "classification"

    # optional override; if None, BL uses sklearn default base estimator for the chosen problem_kind
    base_estimator: Optional[ModelConfig] = None

    n_estimators: int = 10
    max_samples: Union[int, float] = 1.0
    max_features: Union[int, float] = 1.0
    bootstrap: bool = True
    bootstrap_features: bool = False
    oob_score: bool = False
    warm_start: bool = False
    n_jobs: Optional[int] = None
    random_state: Optional[int] = None

    # ---- Balanced bagging (classification-only) ----
    # Uses imbalanced-learn's BalancedBaggingClassifier, which applies a sampler
    # inside each bag to help mitigate class imbalance and reduce single-class bags.
    balanced: bool = False

    # imbalanced-learn accepts many forms (str, dict, callable). Keep as str for now.
    sampling_strategy: str = "auto"

    # Whether the sampler is allowed to sample with replacement.
    # (This is NOT the same as sklearn bagging's `bootstrap`.)
    replacement: bool = False

    @field_validator("max_samples", "max_features", mode="before")
    @classmethod
    def _coerce_int_one_to_float(cls, v):
        # UI uses fractions; avoid sklearn interpreting 1 as "1 sample"
        if isinstance(v, int) and v == 1:
            return 1.0
        return v


# -----------------------------
# AdaBoost
# -----------------------------

class AdaBoostEnsembleConfig(BaseModel):
    kind: Literal["adaboost"] = "adaboost"

    # needed to choose AdaBoostClassifier vs AdaBoostRegressor when base_estimator is None
    problem_kind: Literal["classification", "regression"] = "classification"

    # optional override; if None, BL uses sklearn default base estimator for the chosen problem_kind
    base_estimator: Optional[ModelConfig] = None

    n_estimators: int = 50
    learning_rate: float = 1.0

    # For compatibility across sklearn versions, keep this optional and let BL decide defaults.
    # (Historically classifier supports "SAMME"/"SAMME.R"; newer sklearn versions may deprecate)
    algorithm: Optional[Literal["SAMME", "SAMME.R"]] = None

    random_state: Optional[int] = None


# -----------------------------
# XGBoost (schema only; BL may use xgboost library if installed)
# -----------------------------

class XGBoostEnsembleConfig(BaseModel):
    kind: Literal["xgboost"] = "xgboost"

    problem_kind: Literal["classification", "regression"] = "classification"

    n_estimators: int = 300
    learning_rate: float = 0.1
    max_depth: int = 6
    subsample: float = 1.0
    colsample_bytree: float = 1.0

    reg_lambda: float = 1.0
    reg_alpha: float = 0.0

    min_child_weight: float = 1.0
    gamma: float = 0.0

    use_early_stopping: bool = True
    early_stopping_rounds: Optional[int] = None
    eval_set_fraction: float = 0.2

    n_jobs: Optional[int] = None
    random_state: Optional[int] = None


# -----------------------------
# Discriminated union
# -----------------------------

EnsembleConfig = Annotated[
    Union[
        VotingEnsembleConfig,
        BaggingEnsembleConfig,
        AdaBoostEnsembleConfig,
        XGBoostEnsembleConfig,
    ],
    Field(discriminator="kind"),
]


__all__ = [
    "EnsembleKind",
    "VotingEstimatorSpec",
    "VotingEnsembleConfig",
    "BaggingEnsembleConfig",
    "AdaBoostEnsembleConfig",
    "XGBoostEnsembleConfig",
    "EnsembleConfig",
]
