from __future__ import annotations

import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
)

from visualizations.general.plot_decision_regions import plot_decision_regions
from utils.permutations.rng import RngManager
from utils.configs.configs import RunConfig, DataConfig, SplitConfig, ScaleConfig, FeatureConfig, ModelConfig, EvalConfig

from utils.factories.data_loading_factory import make_data_loader
from utils.factories.sanity_factory import make_sanity_checker
from utils.factories.split_factory import make_splitter
from utils.factories.scale_factory import make_scaler
from utils.factories.feature_factory import make_features
from utils.factories.model_factory import make_model
from utils.factories.training_factory import make_trainer
from utils.factories.predict_factory import make_predictor
from utils.factories.eval_factory import make_evaluator
from utils.factories.baseline_factory import make_baseline

warnings.filterwarnings("ignore", category=FutureWarning)


# ----------------------- Plotting -----------------------
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

    # tick labels — keep them readable even if strings
    classes = np.unique(np.concatenate([y_true, y_pred]))
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels([str(c)[:18] for c in classes], rotation=45, ha="right")
    ax.set_yticklabels([str(c)[:18] for c in classes])

    # annotate cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            ax.text(j, i, str(val), ha="center", va="center",
                    color="k" if cm_norm[i, j] > 0.5 else "w", fontsize=12)

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
    Orchestrates: load → sanity → split → scale → features → model → train → predict → score → baseline → plots.
    Uses strategy/factory adapters so steps are swappable and configs stay small.
    """

    # 1) Load data
    loader = make_data_loader(cfg.data)
    X, y = loader.load()

    # 2) Sanity checks (classification)
    sanity = make_sanity_checker()
    sanity.check(X, y)

    # 3) Central RNG
    rngm = RngManager(None if cfg.eval.seed is None else int(cfg.eval.seed))

    # 4) Build strategies for the REAL run
    split_seed    = rngm.child_seed("real/split")
    features_seed = rngm.child_seed("real/features")

    splitter   = make_splitter(cfg.split, seed=split_seed)
    scaler     = make_scaler(cfg.scale)
    features   = make_features(cfg.features, seed=features_seed)
    model_bld  = make_model(cfg.model)
    trainer    = make_trainer()
    predictor  = make_predictor()
    evaluator  = make_evaluator(cfg.eval, kind="classification")

    # 5) Data flow: split → scale → features
    X_train, X_test, y_train, y_test = splitter.split(X, y)
    X_train, X_test = scaler.fit_transform(X_train, X_test)
    feat_artifact, X_train_fx, X_test_fx = features.fit_transform_train_test(X_train, X_test, y_train)

    # 6) Model: build, train, predict, score
    model = model_bld.build()
    model = trainer.fit(model, X_train_fx, y_train)
    y_pred = predictor.predict(model, X_test_fx)
    real_score = evaluator.score(y_test, y_pred)

    # 7) Plots
    _plot_confusion(y_test, y_pred, title=f"Confusion ({cfg.eval.metric}={real_score:.3f})")
    plot_decision_regions(
        X_train_fx, y_train,
        classifier=model,
        resolution=0.02,
        title="train (space used by the model)"
    )

    # 8) Shuffle-label baseline (pluggable)
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

    # ---- features (PCA / LDA) ----
    parser.add_argument("--features", type=str, default="none", choices=["none", "pca", "lda"])

    # PCA options
    parser.add_argument("--pca_n", type=int, default=None,
                        help="If None, use variance threshold (pca_var).")
    parser.add_argument("--pca_var", type=float, default=0.95,
                        help="Variance threshold for auto n_components when pca_n is None.")
    parser.add_argument("--pca_whiten", action="store_true",
                        help="Whether to whiten PCA outputs.")

    # LDA options
    parser.add_argument("--lda_n", type=int, default=None,
                        help="Desired LDA components; if None, use min(n_features, n_classes-1).")
    parser.add_argument("--lda_solver", type=str, default="svd",
                        choices=["svd", "lsqr", "eigen"],
                        help="LDA solver. Use 'lsqr' or 'eigen' to enable shrinkage.")
    parser.add_argument("--lda_shrinkage", type=str, default=None,
                        help="None|'auto'|<float>. Only used with solvers 'lsqr' or 'eigen'.")
    parser.add_argument("--lda_tol", type=float, default=1e-4,
                        help="Tolerance for LDA (svd solver).")

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
        data=DataConfig(
            x_path=args.x_path, y_path=args.y_path,
            npz_path=args.npz_path, x_key=args.x_key, y_key=args.y_key
        ),
        split=SplitConfig(train_frac=args.train_frac, stratified=True),
        scale=ScaleConfig(method=args.scale),
        features=FeatureConfig(
            method=args.features,
            # PCA
            pca_n=args.pca_n, pca_var=args.pca_var, pca_whiten=args.pca_whiten,
            # LDA
            lda_n=args.lda_n,
            lda_solver=args.lda_solver,
            lda_shrinkage=lda_shrinkage_val,
            lda_tol=args.lda_tol,
            # lda_priors left at default None (add a CLI list later if you need it)
        ),
        model=ModelConfig(
            algo="logreg", C=args.C, penalty=args.penalty,
            solver=args.solver, max_iter=args.max_iter,
            class_weight=(None if args.class_weight in (None, "None") else "balanced")
        ),
        eval=EvalConfig(metric=args.metric, n_shuffles=args.n_shuffles, seed=args.seed),
    )

    run_logreg_decoding(cfg)