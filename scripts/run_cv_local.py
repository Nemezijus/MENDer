# scripts/run_cv_local.py
from __future__ import annotations

from shared_schemas.run_config import RunConfig, DataModel
from shared_schemas.split_configs import SplitCVModel
from shared_schemas.scale_configs import ScaleModel
from shared_schemas.feature_configs import FeaturesModel
from shared_schemas.model_configs import ModelConfig
from shared_schemas.eval_configs import EvalModel
from engine.api import train_supervised

# ==== EDIT THESE AS YOU LIKE ==================================================
# Example A: NPZ bundle
# DATA = DataConfig(
#     npz_path=r"./data/allen_vc/vc512326618_dg_orientation_mean.npz",
#     x_key="X", y_key="y", x_path=None, y_path=None,
# )

# Example B: MAT pair
DATA = DataModel(
    x_path=r"./data/calcium/m67/2025_10_07/data_ensemble_mean.mat",
    y_path=r"./data/calcium/m67/2025_10_07/labels.mat",
)

SPLIT = SplitCVModel(
    mode="kfold",         # <- important
    n_splits=10,
    stratified=True,
    shuffle=True,         # set False for deterministic, seed ignored by sklearn in that case
    train_frac=0.75,      # unused in kfold, kept for compatibility with your dataclass
)

SCALE = ScaleModel(method="standard")

FEATURE = FeaturesModel(
    method="none",
    pca_n=None,
    pca_var=0.95,
    pca_whiten=False,

    # LDA controls (ignored unless method="lda")
    lda_n=None,
    lda_solver="svd",
    lda_shrinkage=None,
    lda_tol=1e-4,

    # SFS controls (ignored unless method="sfs")
    sfs_k="auto",
    sfs_direction="backward",
    sfs_cv=5,
    sfs_n_jobs=None,
)

MODEL = ModelConfig(
    algo="logreg",
    C=1.0,
    penalty="l2",
    solver="lbfgs",
    max_iter=1000,
    class_weight=None,
)

EVAL = EvalModel(
    metric="accuracy",
    n_shuffles=200,   # number of label permutations; set 0 to skip baseline
    seed=42,
)
# ============================================================================


def main():
    cfg = RunConfig(
        data=DATA, split=SPLIT, scale=SCALE,
        features=FEATURE, model=MODEL, eval=EVAL,
    )
    result = train_supervised(cfg)

    print("\n=== TRAIN RESULT ===")
    print(f"Task: {result.task}")
    print(f"Metric: {result.metric_name} = {result.metric_value}")
    if result.metric_mean is not None:
        print(f"Mean/Std (CV): {result.metric_mean} ± {result.metric_std}")
    if result.artifact_meta is not None:
        print(f"Artifact UID: {result.artifact_meta.uid}")
    if result.notes:
        print("Notes:")
        for n in result.notes:
            print(f"- {n}")


if __name__ == "__main__":
    main()
