# utils/configs.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal, Union, Sequence

# ---- data I/O ----
@dataclass
class DataConfig:
    x_path: Optional[str] = None
    y_path: Optional[str] = None
    npz_path: Optional[str] = None
    x_key: str = "X"
    y_key: str = "y"

# ---- split ----
@dataclass
class SplitConfig:
    train_frac: float = 0.8
    stratified: bool = True
    # ---- for cross-validation ----
    mode: Literal["holdout", "kfold"] = "holdout"   # new
    n_splits: int = 5
    shuffle: bool = True

# ---- scaling ----
ScaleName = Literal["standard", "robust", "minmax", "maxabs", "quantile", "none"]
@dataclass
class ScaleConfig:
    method: ScaleName = "standard"

# ---- features ----
FeatureName = Literal["none", "pca", "lda", "sfs"]  # later: "pls", etc.
@dataclass
class FeatureConfig:
    method: FeatureName = "none"

    # PCA
    pca_n: Optional[int] = None
    pca_var: float = 0.95
    pca_whiten: bool = False

    # LDA
    lda_n: Optional[int] = None
    lda_solver: Literal["svd", "lsqr", "eigen"] = "svd"
    lda_shrinkage: Optional[Union[float, Literal["auto"]]] = None
    lda_tol: float = 1e-4
    lda_priors: Optional[Sequence[float]] = None

    # SFS
    sfs_k: Union[int, Literal["auto"]] = "auto"
    sfs_direction: Literal["forward", "backward"] = "backward"
    sfs_cv: int = 5
    sfs_n_jobs: Optional[int] = None


# ---- model ----
PenaltyName = Literal["l2", "none"]  # later: add "elasticnet"
@dataclass
class ModelConfig:
    algo: Literal["logreg"] = "logreg"  # future: "lda", "svm", "ridge", etc.
    C: float = 1.0
    penalty: PenaltyName = "l2"
    solver: str = "lbfgs"
    max_iter: int = 1000
    class_weight: Optional[Literal["balanced"]] = None
    # future: l1_ratio for elasticnet, etc.

# ---- evaluation ----
MetricName = Literal["accuracy", "balanced_accuracy", "f1_macro"]
@dataclass
class EvalConfig:
    metric: MetricName = "accuracy"
    n_shuffles: int = 200
    seed: Optional[int] = None

# ---- top-level run config ----
@dataclass
class RunConfig:
    data: DataConfig
    split: SplitConfig
    scale: ScaleConfig
    features: FeatureConfig
    model: ModelConfig
    eval: EvalConfig
