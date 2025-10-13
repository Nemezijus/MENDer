from __future__ import annotations

import warnings
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Tuple, Optional, Any
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
)
from sklearn.base import clone

from utils.parse.data_read import load_mat_variable
from utils.permutations.shuffle import shuffle_simple_vector
from utils.pipelines.classification import train_and_score_classifier
from visualizations.general.plot_decision_regions import plot_decision_regions
from utils.permutations.rng import RngManager

warnings.filterwarnings("ignore", category=FutureWarning)

# ----------------------- I/O helpers -----------------------

def _load_dataset(
    x_path: Optional[str],
    y_path: Optional[str],
    npz_path: Optional[str],
    x_key: str = "X",
    y_key: str = "y",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load (X, y) from either:
      - separate .mat files via --x_path/--y_path
      - a single .npz file via --npz_path (expects keys X/y unless overridden)
      - (also works if --x_path points to .npz and --y_path is omitted)

    Returns
    -------
    X: (n_samples, n_features)
    y: (n_samples,)
    """
    if npz_path:
        data = np.load(npz_path, allow_pickle=True)
        if x_key not in data or y_key not in data:
            raise KeyError(
                f"Keys '{x_key}' and/or '{y_key}' not found in {npz_path}. "
                f"Available: {list(data.keys())}"
            )
        X = np.asarray(data[x_key])
        y = np.asarray(data[y_key]).ravel()
        return X, y

    if x_path and x_path.lower().endswith(".npz") and (not y_path):
        # convenience: allow --x_path to be a single .npz
        data = np.load(x_path, allow_pickle=True)
        if x_key not in data or y_key not in data:
            raise KeyError(
                f"Keys '{x_key}' and/or '{y_key}' not found in {x_path}. "
                f"Available: {list(data.keys())}"
            )
        X = np.asarray(data[x_key])
        y = np.asarray(data[y_key]).ravel()
        return X, y

    if not (x_path and y_path):
        raise ValueError(
            "Provide either (--x_path and --y_path) for .mat files, or --npz_path for a single NPZ."
        )

    # .mat route
    X = np.asarray(load_mat_variable(x_path))
    y = np.asarray(load_mat_variable(y_path)).ravel()
    return X, y


# ----------------------- Core helpers -----------------------

def _coerce_shapes(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Ensure X is (n_samples, n_features) and y is (n_samples,)."""
    X = np.asarray(X)
    y = np.asarray(y).ravel()

    # If X looks like (features, samples), transpose to (samples, features)
    if X.ndim == 1:
        X = X[:, None]
    if X.shape[0] < X.shape[1] and X.shape[1] == y.shape[0]:
        X = X.T

    if X.shape[0] != y.shape[0]:
        n = min(X.shape[0], y.shape[0])
        X, y = X[:n], y[:n]

    if X.ndim != 2:
        raise ValueError(f"X must be 2D; got shape {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"y must be 1D; got shape {y.shape}")

    return X, y


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

# ----------------------- Shuffle baseline -----------------------

def shuffle_labels_return_scores(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    n_shuffles: int = 200,
    *,
    train_frac: float = 0.8,
    scale: str = "standard",
    features: str = "none",           # 'none' | 'pca'
    pca_n: Optional[int] = None,
    pca_var: float = 0.95,
    pca_whiten: bool = False,
    metric: str = "accuracy",
    rng: Union[None, int, np.random.Generator, RngManager] = None,
) -> np.ndarray:
    """
    Permutation baseline: shuffle trial labels, refit, and collect scores.
    Uses a fresh clone(model) per shuffle.

    RNG policy:
    - If rng is an int/None: construct a local RngManager from it.
    - If rng is a Generator: also construct a local RngManager seeded by drawing a child.
    - If rng is an RngManager: use it directly to derive per-iteration streams.
    """
    if n_shuffles < 1:
        raise ValueError("n_shuffles must be >= 1")
    # gen = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)

        # Normalize to an RngManager
    if isinstance(rng, RngManager):
        rngm = rng
    elif isinstance(rng, np.random.Generator):
        # derive a stable int seed from the provided generator
        tmp_seed = int(rng.integers(1 << 32))
        rngm = RngManager(tmp_seed)
    else:
        # rng is int|None
        rngm = RngManager(None if rng is None else int(rng))

    
    scores = np.empty(n_shuffles, dtype=float)
    for i in range(n_shuffles):
        lbl_gen = rngm.child_generator(f"shuffle_{i}/labels")
        pipe_seed = rngm.child_seed(f"shuffle_{i}/pipeline")

        y_shuf = shuffle_simple_vector(y, rng=lbl_gen)

        scores[i], _ = train_and_score_classifier(
            clone(model),
            X, y_shuf,
            train_frac=train_frac,
            scale=scale,
            feature_scaling=features,
            pca_n_components=pca_n,
            pca_variance_threshold=pca_var,
            pca_whiten=pca_whiten,
            rng=pipe_seed,
            metric=metric,
            debug=False,
            return_details=True,
        )
    return scores
# ----------------------- Orchestrator -----------------------

def run_logreg_decoding(
    x_path: Optional[str] = None,
    y_path: Optional[str] = None,
    *,
    npz_path: Optional[str] = None,
    x_key: str = "X",
    y_key: str = "y",
    n_shuffles: int = 200,
    train_frac: float = 0.8,
    scale: str = "standard",          # 'standard'|'robust'|'minmax'|'maxabs'|'quantile'|'none'
    features: str = "none",           # 'none'|'pca'
    pca_n: Optional[int] = None,
    pca_var: float = 0.95,
    pca_whiten: bool = False,
    C: float = 1.0,
    penalty: str = "l2",
    solver: str = "lbfgs",
    max_iter: int = 1000,
    metric: str = "accuracy",
    rng: Union[None, int, np.random.Generator, RngManager] = None,
):
    """
    Load X, y → declare model → fit/score (with optional scaling & PCA) → confusion
    → shuffle-label baseline, with centralized RNG control.
    """
    # 1) Load data
    X, y = _load_dataset(x_path, y_path, npz_path, x_key=x_key, y_key=y_key)
    X, y = _coerce_shapes(X, y)

    # 2) Quick sanity on labels
    n_classes = np.unique(y).size
    if n_classes < 2:
        raise ValueError("y must contain at least two classes.")
    if X.shape[0] < 2 * n_classes:
        print("[WARN] Few trials per class; results may be unstable.")

    # 3) Central RNG manager
    if isinstance(rng, RngManager):
        rngm = rng
    elif isinstance(rng, np.random.Generator):
        rngm = RngManager(int(rng.integers(1 << 32)))
    else:
        rngm = RngManager(None if rng is None else int(rng))

    # 4) Declare model (change freely)
    model = LogisticRegression(
        C=C, penalty=penalty, solver=solver, max_iter=max_iter, multi_class="auto"
    )

    # 5) Real fit/score (+ confusion)
    real_seed = (int(rng) if rng is not None else None)
    real_score, details = train_and_score_classifier(
        model,
        X, y,
        train_frac=train_frac,
        scale=scale,
        feature_scaling=features,
        pca_n_components=pca_n,
        pca_variance_threshold=pca_var,
        pca_whiten=pca_whiten,
        rng=real_seed,
        metric=metric,
        debug=True,
        return_details=True,
    )
    _plot_confusion(details["y_test"], details["y_pred"],
                    title=f"Confusion ({metric}={real_score:.3f})")
    plot_decision_regions(details["X_train"], details["y_train"],
                          classifier=details["model"], resolution=0.02,
                          title="train (space used by the model)")

    # 6) Shuffle-label baseline (independent per-iteration streams)
    shuffled_scores = shuffle_labels_return_scores(
        model,
        X, y,
        n_shuffles=n_shuffles,
        train_frac=train_frac,
        scale=scale,
        features=features,
        pca_n=pca_n,
        pca_var=pca_var,
        pca_whiten=pca_whiten,
        metric=metric,
        rng=rngm,  # pass the manager; function derives child streams
    )

    print(f"Real {metric}: {real_score:.3f}")
    print(f"Shuffled mean: {np.mean(shuffled_scores):.3f} ± {np.std(shuffled_scores):.3f}")
    plot_shuffle_results(real_score, shuffled_scores, metric=metric)

    return real_score, shuffled_scores


# ----------------------- CLI -----------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Trial-level decoding via Logistic Regression with label-shuffle baseline."
    )
    # Inputs: .mat pair or a single .npz
    parser.add_argument("--x_path", type=str, default=None,
                        help="Path to .mat with features (trials x features), or .npz if --y_path omitted.")
    parser.add_argument("--y_path", type=str, default=None,
                        help="Path to .mat with labels (trials,).")
    parser.add_argument("--npz_path", type=str, default=None,
                        help="Path to .npz containing X and y.")
    parser.add_argument("--x_key", type=str, default="X", help="Key for X inside .npz.")
    parser.add_argument("--y_key", type=str, default="y", help="Key for y inside .npz.")

    parser.add_argument("--n_shuffles", type=int, default=200, help="Number of label shuffles.")
    parser.add_argument("--train_frac", type=float, default=0.8, help="Fraction of trials for training.")
    parser.add_argument("--scale", type=str, default="standard",
                        choices=["standard", "robust", "minmax", "maxabs", "quantile", "none"],
                        help="Feature scaling method.")
    parser.add_argument("--features", type=str, default="none", choices=["none", "pca"],
                        help="Feature extraction step.")
    parser.add_argument("--pca_n", type=int, default=None, help="PCA components (None → auto by variance).")
    parser.add_argument("--pca_var", type=float, default=0.95, help="PCA variance threshold for auto selection.")
    parser.add_argument("--pca_whiten", action="store_true", help="Whiten PCA components.")

    parser.add_argument("--C", type=float, default=1.0, help="Inverse regularization strength (logreg).")
    parser.add_argument("--penalty", type=str, default="l2", choices=["l2", "none"], help="Penalty (logreg).")
    parser.add_argument("--solver", type=str, default="lbfgs", help="Solver (logreg).")
    parser.add_argument("--max_iter", type=int, default=1000, help="Max iterations (logreg).")

    parser.add_argument("--metric", type=str, default="accuracy",
                        choices=["accuracy", "balanced_accuracy", "f1_macro"],
                        help="Primary metric.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")

    args = parser.parse_args()

    run_logreg_decoding(
        x_path=args.x_path,
        y_path=args.y_path,
        npz_path=args.npz_path,
        x_key=args.x_key,
        y_key=args.y_key,
        n_shuffles=args.n_shuffles,
        train_frac=args.train_frac,
        scale=args.scale,
        features=args.features,
        pca_n=args.pca_n,
        pca_var=args.pca_var,
        pca_whiten=args.pca_whiten,
        C=args.C,
        penalty=args.penalty,
        solver=args.solver,
        max_iter=args.max_iter,
        metric=args.metric,
        rng=args.seed,
    )