from __future__ import annotations

import warnings
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from utils.parse.data_read import load_mat_variable
from utils.permutations.shuffle import shuffle_simple_vector

warnings.filterwarnings("ignore", category=FutureWarning)
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


def _split_trials(X, y, train_frac: float, rng) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified split at the trial level (random, not time-contiguous)."""
    if not (0.0 < train_frac < 1.0):
        raise ValueError("train_frac must be in (0,1).")
    sss = StratifiedShuffleSplit(n_splits=1, train_size=train_frac, random_state=rng)
    (train_idx, test_idx), = sss.split(X, y)
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def _metric(y_true, y_pred, metric: str = "accuracy") -> float:
    if metric == "accuracy":
        return accuracy_score(y_true, y_pred)
    elif metric == "balanced_accuracy":
        return balanced_accuracy_score(y_true, y_pred)
    elif metric == "f1_macro":
        return f1_score(y_true, y_pred, average="macro")
    else:
        raise ValueError(f"Unknown metric '{metric}'")


def train_and_score_classifier(
    X: np.ndarray,
    y: np.ndarray,
    *,
    train_frac: float = 0.8,
    standardize: bool = True,
    C: float = 1.0,
    penalty: str = "l2",
    solver: str = "lbfgs",
    max_iter: int = 1000,
    metric: str = "accuracy",
    rng: Union[None, int, np.random.Generator] = None,
    debug: bool = True,
) -> float:
    """
    Stratified trial split -> (optional) StandardScaler -> LogisticRegression -> score.
    """
    gen = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)

    X_train, X_test, y_train, y_test = _split_trials(X, y, train_frac=train_frac, rng=gen.integers(1<<32))

    scaler = None
    if standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    clf = LogisticRegression(
        C=C,
        penalty=penalty,
        solver=solver,   # 'lbfgs' works well for multinomial when penalty is L2
        max_iter=max_iter,
        multi_class="auto",
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    score = _metric(y_test, y_pred, metric=metric)

    if debug:
        print(f"[INFO] {metric} = {score:.3f}  (train_frac={train_frac}, C={C}, penalty={penalty}, solver={solver})")
        print(classification_report(y_test, y_pred, digits=3))
        _plot_confusion(y_test, y_pred, title=f"Confusion matrix ({metric}={score:.3f})")

    return score


def shuffle_labels_return_scores(
    X: np.ndarray,
    y: np.ndarray,
    n_shuffles: int = 200,
    *,
    train_frac: float = 0.8,
    standardize: bool = True,
    C: float = 1.0,
    penalty: str = "l2",
    solver: str = "lbfgs",
    max_iter: int = 1000,
    metric: str = "accuracy",
    rng: Union[None, int, np.random.Generator] = None,
) -> np.ndarray:
    """
    Permutation baseline: shuffle trial labels, refit, and collect scores.
    """
    if n_shuffles < 1:
        raise ValueError("n_shuffles must be >= 1")
    gen = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)

    scores = np.empty(n_shuffles, dtype=float)
    for i in range(n_shuffles):
        y_shuf = shuffle_simple_vector(y, rng=gen)  # preserves label histogram in expectation
        scores[i] = train_and_score_classifier(
            X, y_shuf,
            train_frac=train_frac,
            standardize=standardize,
            C=C, penalty=penalty, solver=solver, max_iter=max_iter,
            metric=metric,
            rng=gen.integers(1<<32),
            debug=False,
        )
    return scores


def _plot_confusion(y_true, y_pred, title="Confusion matrix"):
    cm = confusion_matrix(y_true, y_pred)
    with np.errstate(invalid="ignore"):
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    # tick labels
    classes = np.unique(np.concatenate([y_true, y_pred]))
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)

    # annotate cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            ax.text(j, i, str(val), ha="center", va="center", color="w" if cm_norm[i, j] > 0.5 else "k")

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

def run_logreg_decoding(
    x_path: str,
    y_path: str,
    *,
    n_shuffles: int = 200,
    train_frac: float = 0.8,
    standardize: bool = True,
    C: float = 1.0,
    penalty: str = "l2",
    solver: str = "lbfgs",
    max_iter: int = 1000,
    metric: str = "accuracy",
    rng: Union[None, int, np.random.Generator] = None,
):
    """
    Load X, y (.mat); coerce shapes; logistic regression with stratified split; shuffle baseline.
    """
    X = np.asarray(load_mat_variable(x_path))
    y = np.asarray(load_mat_variable(y_path)).ravel()
    X, y = _coerce_shapes(X, y)

    # sanity: label cardinality vs samples
    n_classes = np.unique(y).size
    if n_classes < 2:
        raise ValueError("y must contain at least two classes.")
    if X.shape[0] < 2 * n_classes:
        print("[WARN] Few trials per class; results may be unstable.")

    real_score = train_and_score_classifier(
        X, y,
        train_frac=train_frac,
        standardize=standardize,
        C=C, penalty=penalty, solver=solver, max_iter=max_iter,
        metric=metric,
        rng=rng,
        debug=True,
    )
    shuffled_scores = shuffle_labels_return_scores(
        X, y, n_shuffles=n_shuffles,
        train_frac=train_frac,
        standardize=standardize,
        C=C, penalty=penalty, solver=solver, max_iter=max_iter,
        metric=metric,
        rng=rng,
    )

    print(f"Real {metric}: {real_score:.3f}")
    print(f"Shuffled mean: {np.mean(shuffled_scores):.3f} ± {np.std(shuffled_scores):.3f}")
    plot_shuffle_results(real_score, shuffled_scores, metric=metric)

    return real_score, shuffled_scores


# ----------------------- CLI -----------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Trial-level decoding of stimulus contrast via Logistic Regression "
                    "with label-shuffle baseline."
    )
    parser.add_argument("--x_path", type=str, required=True,
                        help="Path to .mat with features (trials x features).")
    parser.add_argument("--y_path", type=str, required=True,
                        help="Path to .mat with labels (trials,).")
    parser.add_argument("--n_shuffles", type=int, default=200,
                        help="Number of label shuffles for permutation baseline.")
    parser.add_argument("--train_frac", type=float, default=0.8,
                        help="Fraction of trials used for training (stratified).")
    parser.add_argument("--standardize", action="store_true",
                        help="Standardize features (fit on train, apply to test).")
    parser.add_argument("--C", type=float, default=1.0,
                        help="Inverse of regularization strength (larger = less regularization).")
    parser.add_argument("--penalty", type=str, default="l2", choices=["l2", "none"],
                        help="Penalty type for LogisticRegression.")
    parser.add_argument("--solver", type=str, default="lbfgs",
                        help="Solver (e.g., 'lbfgs', 'saga' for large sparse).")
    parser.add_argument("--max_iter", type=int, default=1000,
                        help="Max iterations for solver.")
    parser.add_argument("--metric", type=str, default="accuracy",
                        choices=["accuracy", "balanced_accuracy", "f1_macro"],
                        help="Primary metric to score.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for splits and shuffles.")

    args = parser.parse_args()

    run_logreg_decoding(
        x_path=args.x_path,
        y_path=args.y_path,
        n_shuffles=args.n_shuffles,
        train_frac=args.train_frac,
        standardize=args.standardize,
        C=args.C,
        penalty=args.penalty,
        solver=args.solver,
        max_iter=args.max_iter,
        metric=args.metric,
        rng=args.seed,
    )
