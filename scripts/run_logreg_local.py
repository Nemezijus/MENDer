# scripts/run_logreg_local.py
from __future__ import annotations

from utils.configs.configs import (
    RunConfig, DataConfig, SplitConfig, ScaleConfig, FeatureConfig,
    ModelConfig, EvalConfig
)
from instances.logreg_classify_with_shuffle import run_logreg_decoding

# ==== EDIT THESE VALUES AS YOU LIKE ==========================================
# Example A: NPZ bundle (Allen)
# DATA = DataConfig(
#     npz_path=r"./data/allen_vc/vc512326618_dg_orientation_mean.npz",
#     x_key="X",   # change if your npz keys differ
#     y_key="y",
#     x_path=None, y_path=None,
# )

# Example B: MAT pair (comment A out and uncomment this if you prefer)
DATA = DataConfig(
    x_path=r"./data/calcium/m67/2025_10_07/data_ensemble_mean.mat",
    y_path=r"./data/calcium/m67/2025_10_07/labels.mat",
)

SPLIT   = SplitConfig(train_frac=0.75, stratified=True)
SCALE   = ScaleConfig(method="standard")   # "standard","robust","minmax","maxabs","quantile","none"

# Choose one of: method="none" | "pca" | "lda" | "sfs"
FEATURE = FeatureConfig(
    method="pca",

    # PCA knobs (used when method="pca")
    pca_n=None,            # None => use variance threshold below
    pca_var=0.95,
    pca_whiten=False,

    # LDA knobs (used when method="lda")
    lda_n=None,
    lda_solver="svd",   # "svd"|"lsqr"|"eigen"
    lda_shrinkage=None, # None|"auto"|float (lsqr/eigen only)
    lda_tol=1e-4,

    # SFS knobs
    sfs_k=10,                  # integer or "auto"
    sfs_direction="forward",  # "backward" = SBS; "forward" also possible
    sfs_cv=5,
    sfs_n_jobs=None,           # or -1 for parallel if environment allows
)

MODEL   = ModelConfig(
    algo="logreg",
    C=1.0,
    penalty="l2",        # "l2"|"none"
    solver="lbfgs",
    max_iter=1000,
    class_weight=None,   # or "balanced"
)

EVAL    = EvalConfig(
    metric="accuracy",   # "accuracy"|"balanced_accuracy"|"f1_macro"
    n_shuffles=100,
    seed=42,
)
# ============================================================================

def main():
    cfg = RunConfig(
        data=DATA, split=SPLIT, scale=SCALE,
        features=FEATURE, model=MODEL, eval=EVAL,
    )
    run_logreg_decoding(cfg)

if __name__ == "__main__":
    main()
