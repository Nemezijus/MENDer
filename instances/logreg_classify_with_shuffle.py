from __future__ import annotations

import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SequentialFeatureSelector

from visualizations.general.plot_decision_regions import plot_decision_regions
from engine.core.random.rng import RngManager

from engine.contracts.run_config import RunConfig, DataModel
from engine.contracts.split_configs import SplitHoldoutModel
from engine.contracts.scale_configs import ScaleModel
from engine.contracts.feature_configs import FeaturesModel
from engine.contracts.model_configs import ModelConfig
from engine.contracts.eval_configs import EvalModel

from engine.factories.data_loading_factory import make_data_loader
from engine.factories.sanity_factory import make_sanity_checker
from engine.factories.split_factory import make_splitter
from engine.factories.eval_factory import make_evaluator
from engine.factories.baseline_factory import make_baseline
from engine.factories.pipeline_factory import make_pipeline, make_preproc_pipeline

warnings.filterwarnings("ignore", category=FutureWarning)


# ----------------------- Plotting helpers -----------------------
def _plot_confusion(y_true, y_pred, title="Confusion matrix"):
    cm = confusion_matrix(y_true, y_pred)
    with np.errstate(invalid="ignore"):
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm_norm, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    classes = np.unique(np.concatenate([y_true, y_pred]))
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels([str(c)[:18] for c in classes], rotation=45, ha="right")
    ax.set_yticklabels([str(c)[:18] for c in classes])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            ax.text(
                j, i, str(val), ha="center", va="center",
                color="k" if cm_norm[i, j] > 0.5 else "w", fontsize=12
            )

    plt.tight_layout()
    plt.show()


def plot_shuffle_results(real_score: float, shuffled_scores: np.ndarray, metric: str):
    plt.figure(figsize=(8, 5))
    plt.hist(shuffled_scores, bins=20, alpha=0.75, color="gray", edgecolor="black")
    plt.axvline(real_score, color="red", linewidth=2, label=f"Real {metric} = {real_score:.3f}")
    plt.xlabel(metric.capitalize())
    plt.ylabel("Count")
    plt.title(f"Shuffle-label baseline ({metric})")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ----------------------- Orchestrator -----------------------
def run_logreg_decoding(cfg: RunConfig):
    """
    Orchestrates: load → sanity → split → pipeline.fit → predict/score → baseline → plots.
    Uses factories for data, split, eval, baseline, and pipeline assembly.
    """

    # 1) Load data
    loader = make_data_loader(cfg.data)
    X, y = loader.load()

    # 2) Sanity checks (classification)
    sanity = make_sanity_checker()
    sanity.check(X, y)

    # 3) Central RNG
    rngm = RngManager(None if cfg.eval.seed is None else int(cfg.eval.seed))

    # 4) Split
    split_seed = rngm.child_seed("real/split")
    splitter = make_splitter(cfg.split, seed=split_seed)
    X_train, X_test, y_train, y_test = splitter.split(X, y)
    print(f"[INFO] Split: train={X_train.shape[0]}, test={X_test.shape[0]} (features={X_train.shape[1]})")

    # Optional class counts (keep quiet by default)
    # import numpy as _np
    # def _counts(arr):
    #     u, c = _np.unique(arr, return_counts=True)
    #     return dict(zip(u.tolist(), c.tolist()))
    # print(f"[INFO] Split classes: train={_counts(y_train)}, test={_counts(y_test)}")

    # 5) Build + fit pipeline
    pipeline = make_pipeline(cfg, rngm, stream="real")
    pipeline.fit(X_train, y_train)

    # 6) Predict + score
    evaluator = make_evaluator(cfg.eval, kind="classification")
    y_pred = pipeline.predict(X_test)
    real_score = evaluator.score(y_test, y_pred)

    # 6.1) If the feature step is SFS, print selection summary
    feat_step = pipeline.named_steps.get("feat", None)
    if isinstance(feat_step, SequentialFeatureSelector):
        support = feat_step.get_support()
        k = int(support.sum())
        p = int(X_train.shape[1])
        print(f"[INFO] SFS selected {k}/{p} features; test {cfg.eval.metric} = {real_score:.3f}")

    # 7) Plots
    _plot_confusion(y_test, y_pred, title=f"Confusion ({cfg.eval.metric}={real_score:.3f})")

    # Decision regions in feature space (scale+feat only)
    preproc = make_preproc_pipeline(cfg, rngm, stream="real")
    preproc.fit(X_train, y_train)
    X_train_fx = preproc.transform(X_train)
    plot_decision_regions(
        X_train_fx, y_train,
        classifier=pipeline.named_steps["clf"],
        resolution=0.02
    )

    # 8) Shuffle-label baseline
    baseline = make_baseline(cfg, rngm)
    shuffled_scores = baseline.run(X, y)

    # 9) Report
    print(f"Real {cfg.eval.metric}: {real_score:.3f}")
    print(f"Shuffled mean: {np.mean(shuffled_scores):.3f} ± {np.std(shuffled_scores):.3f}")
    plot_shuffle_results(real_score, shuffled_scores, metric=cfg.eval.metric)

    return real_score, shuffled_scores


# ----------------------- CLI -----------------------
if __name__ == "__main__":
    import argparse

    # helpers for optional values
    def parse_optional_int(s: str | None):
        if s is None:
            return None
        s = str(s).strip()
        if s.lower() in {"none", "null", ""}:
            return None
        try:
            return int(s)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"Invalid int or None: {s!r}. Use an integer or the word 'None'."
            )

    def parse_optional_int_or_auto(s: str | None):
        if s is None:
            return "auto"
        s = str(s).strip()
        if s.lower() in {"auto"}:
            return "auto"
        if s.lower() in {"none", "null", ""}:
            return "auto"
        try:
            return int(s)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"Invalid value: {s!r}. Use an integer or 'auto'."
            )

    parser = argparse.ArgumentParser(
        description="Trial-level decoding with label-shuffle baseline (Logistic Regression)."
    )

    # ---- data ----
    parser.add_argument("--x_path", type=str, default=None)
    parser.add_argument("--y_path", type=str, default=None)
    parser.add_argument("--npz_path", type=str, default=None)
    parser.add_argument("--x_key", type=str, default="X")
    parser.add_argument("--y_key", type=str, default="y")

    # ---- eval + split ----
    parser.add_argument("--n_shuffles", type=int, default=200)
    parser.add_argument("--train_frac", type=float, default=0.8)

    # ---- scaling ----
    parser.add_argument("--scale", type=str, default="standard",
                        choices=["standard", "robust", "minmax", "maxabs", "quantile", "none"])

    # ---- features (PCA / LDA / SFS) ----
    parser.add_argument("--features", type=str, default="none",
                        choices=["none", "pca", "lda", "sfs"])

    # PCA options
    parser.add_argument("--pca_n", type=parse_optional_int, default=None,
                        help="If None, use variance threshold (pca_var).")
    parser.add_argument("--pca_var", type=float, default=0.95,
                        help="Variance threshold for auto n_components when pca_n is None.")
    parser.add_argument("--pca_whiten", action="store_true",
                        help="Whether to whiten PCA outputs.")

    # LDA options
    parser.add_argument("--lda_n", type=parse_optional_int, default=None,
                        help="Desired LDA components; if None, use min(n_features, n_classes-1).")
    parser.add_argument("--lda_solver", type=str, default="svd",
                        choices=["svd", "lsqr", "eigen"],
                        help="LDA solver. Use 'lsqr' or 'eigen' to enable shrinkage.")
    parser.add_argument("--lda_shrinkage", type=str, default=None,
                        help="None|'auto'|<float>. Only used with solvers 'lsqr' or 'eigen'.")
    parser.add_argument("--lda_tol", type=float, default=1e-4,
                        help="Tolerance for LDA (svd solver).")

    # SFS options
    parser.add_argument("--sfs_k", type=parse_optional_int_or_auto, default="auto",
                        help="Number of features to select (int) or 'auto'.")
    parser.add_argument("--sfs_direction", type=str, default="backward",
                        choices=["forward", "backward"],
                        help="'backward' ≈ SBS; 'forward' ≈ SFS.")
    parser.add_argument("--sfs_cv", type=int, default=5,
                        help="Number of StratifiedKFold folds during selection.")
    parser.add_argument("--sfs_n_jobs", type=int, default=None,
                        help="Parallelism for SFS (e.g., -1 for all cores).")

    # ---- model (Logistic Regression) ----
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--penalty", type=str, default="l2", choices=["l2", "none"])
    parser.add_argument("--solver", type=str, default="lbfgs")
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--class_weight", type=str, default=None, choices=[None, "balanced"])

    # ---- metric + seed ----
    parser.add_argument("--metric", type=str, default="accuracy",
                        choices=["accuracy", "balanced_accuracy", "f1_macro"])
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    # Convert lda_shrinkage: None | 'auto' | float
    lda_shrinkage_val = None
    if args.lda_shrinkage is not None and str(args.lda_shrinkage).lower() != "none":
        if str(args.lda_shrinkage).lower() == "auto":
            lda_shrinkage_val = "auto"
        else:
            try:
                lda_shrinkage_val = float(args.lda_shrinkage)
            except ValueError:
                raise SystemExit(
                    "Invalid --lda_shrinkage. Use 'auto' or a float (e.g., 0.1), or omit for None."
                )

    cfg = RunConfig(
        data=DataModel(
            x_path=args.x_path, y_path=args.y_path,
            npz_path=args.npz_path, x_key=args.x_key, y_key=args.y_key
        ),
        split=SplitHoldoutModel(train_frac=args.train_frac, stratified=True),
        scale=ScaleModel(method=args.scale),
        features=FeaturesModel(
            method=args.features,
            # PCA
            pca_n=args.pca_n, pca_var=args.pca_var, pca_whiten=args.pca_whiten,
            # LDA
            lda_n=args.lda_n,
            lda_solver=args.lda_solver,
            lda_shrinkage=lda_shrinkage_val,
            lda_tol=args.lda_tol,
            # SFS
            sfs_k=args.sfs_k,
            sfs_direction=args.sfs_direction,
            sfs_cv=args.sfs_cv,
            sfs_n_jobs=args.sfs_n_jobs,
        ),
        model=ModelConfig(
            algo="logreg", C=args.C, penalty=args.penalty,
            solver=args.solver, max_iter=args.max_iter,
            class_weight=(None if args.class_weight in (None, "None") else "balanced")
        ),
        eval=EvalModel(metric=args.metric, n_shuffles=args.n_shuffles, seed=args.seed),
    )

    run_logreg_decoding(cfg)
