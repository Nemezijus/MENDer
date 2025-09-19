from __future__ import annotations

from typing import Union
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error
from utils.parse.data_read import load_mat_variable

import matplotlib.pyplot as plt

from utils.permutations.shuffle import shuffle_simple_vector
from utils.filters.filter_out_uncorrelating_features import filter_out_uncorrelating_features

"""
Ridge regression decoding with label-shuffle validation (permutation baseline).

Overview
--------
This module trains a Ridge regression model to decode a continuous behavioral variable
(e.g., velocity) from neural population activity (df/f), using a *contiguous* time-based
train/test split (no shuffling of samples). To assess whether the observed performance
exceeds chance, it computes a permutation baseline by repeatedly shuffling the labels (y)
while keeping X fixed, refitting on each shuffle, and collecting R² scores.

Assumptions
-----------
- Single continuous recording: no repeated trials.
- X: time x neurons (if provided as neurons x time, the code transposes it).
- y: vector with one value per time point; aligned to X (same length after trimming).
- Split: train uses the *start* of the session; test uses the *end* (optionally with a gap).

Key Functions
-------------
- train_and_score(X, y, train_frac=0.8, gap=0, alpha=1.0) -> float
    Fits Ridge on the head (train) and evaluates R² on the tail (test).
    Scaling is fit on train only (StandardScaler), then applied to test.

- shuffle_labels_return_scores(X, y, n_shuffles=100, train_frac=0.8, gap=0, alpha=1.0, rng=None) -> np.ndarray
    Runs a label permutation test: shuffles y `n_shuffles` times, refits the same pipeline,
    and returns an array of R² scores for the shuffled labels.

- run_decoding_with_shuffle(x_path, y_path, n_shuffles=200, **kwargs) -> (float, np.ndarray)
    Convenience orchestrator that:
      1) loads .mat variables via `load_mat_variable`,
      2) coerces shapes (transposes X if needed, trims lengths),
      3) computes the real R²,
      4) computes shuffled-label R² scores,
      5) plots a histogram of shuffled scores with the real R² overlaid.

Parameters (common)
-------------------
- train_frac : float (0 < train_frac < 1)
    Fraction of timepoints used for training (contiguous from the start).
- gap : int (≥ 0)
    Number of samples skipped between train and test to reduce temporal leakage.
- alpha : float
    Ridge regularization strength.
- rng : int | numpy.random.Generator | None
    Seed or Generator for reproducible shuffles.

I/O Expectations
----------------
- X .mat file contains a single numeric array of df/f (either TxN or NxT).
- y .mat file contains a single numeric vector (T,).
- Use MATLAB v7 or v5 if possible; for v7.3 (HDF5) ensure `h5py` is installed.

Outputs
-------
- Printed metrics:
    Real R², shuffled-label mean±std.
- Figure:
    Histogram of shuffled R² scores with a vertical line for real R².
- Return value:
    (real_score: float, shuffled_scores: np.ndarray)

Quickstart
----------
# Run with your defaults (expects test .mat files under data/test/)
python -m instances.ridge_regression_with_shuffle_validation

# Or call programmatically:
from instances.ridge_regression_with_shuffle_validation import run_decoding_with_shuffle
real_r2, perm_r2 = run_decoding_with_shuffle(
    x_path="data/Fdff.mat",
    y_path="data/velocity.mat",
    n_shuffles=500,
    train_frac=0.8,
    gap=0,
    alpha=1.0,
)

Notes
-----
- Negative R² generally indicates poor generalization relative to a constant baseline.
- Don’t interpret per-neuron coefficients unless held-out R² is meaningfully > 0.
- If alignment/latency is a concern, consider adding lagged features or sweeping X↔y lags.
"""

def train_and_score(X, y, train_frac=0.8, gap=0, alpha=1.0, debug_plot=False):
    """Train Ridge regression on contiguous split and return R^2 score."""
    n = len(y)
    split_idx = int(n * train_frac)
    test_start = min(split_idx + gap, n)

    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[test_start:], y[test_start:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # --- Diagnostics ---
    if debug_plot:
        plt.figure(figsize=(10, 4))
        t = np.arange(len(y_test))
        plt.plot(t, y_test, label="True y_test", color="k")
        plt.plot(t, y_pred, label="Predicted y_pred", color="red", alpha=0.7)
        plt.title("Ridge Regression: True vs Predicted on Test Set")
        plt.xlabel("Time (frames in test set)")
        plt.ylabel("Velocity")
        plt.legend()
        plt.tight_layout()
        plt.show()

        print("[DEBUG] Mean of y_test:", np.mean(y_test))
        print("[DEBUG] Mean of y_pred:", np.mean(y_pred))
        print("[DEBUG] Var(y_test):", np.var(y_test))
        print("[DEBUG] Var(y_pred):", np.var(y_pred))
        print("[DEBUG] MSE:", mean_squared_error(y_test, y_pred))

    return r2_score(y_test, y_pred)

def shuffle_labels_return_scores(
    X: np.ndarray,
    y: np.ndarray,
    n_shuffles: int = 100,
    *,
    train_frac: float = 0.8,
    gap: int = 0,
    alpha: float = 1.0,
    rng: Union[None, int, np.random.Generator] = None,
) -> np.ndarray:
    """
    Compute permutation (label-shuffle) scores by shuffling y `n_shuffles` times.

    This is a simple Monte Carlo baseline: destroy any real X–y relationship by
    shuffling y, train/evaluate the same pipeline, and collect the scores.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix. Must be 2D.
    y : np.ndarray of shape (n_samples,)
        Target vector. Must be 1D and match X along n_samples.
    n_shuffles : int, default=100
        Number of independent shuffles to run.
    train_frac : float, default=0.8
        Fraction of samples used for the training split (contiguous, time-aware
        inside your `train_and_score` function).
    gap : int, default=0
        Optional gap between train and test (passed through to `train_and_score`).
    alpha : float, default=1.0
        Regularization strength for the Ridge model (passed through).
    rng : None | int | np.random.Generator, default=None
        Random seed or Generator for reproducible shuffles.

    Returns
    -------
    np.ndarray of shape (n_shuffles,)
        The score from each shuffled run (e.g., R^2 if `train_and_score` returns R^2).

    Raises
    ------
    ValueError
        If inputs have the wrong shape or arguments are out of range.
    """
    # --- basic validation ---
    X = np.asarray(X)
    y = np.asarray(y).ravel()

    if X.ndim != 2:
        raise ValueError(f"`X` must be 2D (n_samples, n_features). Got shape {X.shape}.")
    if y.ndim != 1:
        raise ValueError(f"`y` must be 1D (n_samples,). Got shape {y.shape}.")
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y must have the same number of samples. "
            f"Got X: {X.shape[0]} vs y: {y.shape[0]}."
        )
    if not isinstance(n_shuffles, int) or n_shuffles < 1:
        raise ValueError("`n_shuffles` must be a positive integer.")
    if not (0.0 < train_frac < 1.0):
        raise ValueError("`train_frac` must be in the open interval (0, 1).")
    if not isinstance(gap, int) or gap < 0:
        raise ValueError("`gap` must be a non-negative integer.")

    # --- RNG setup ---
    gen = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)

    scores = np.empty(n_shuffles, dtype=float)

    for i in range(n_shuffles):
        # shuffle y (copy, original y untouched)
        y_shuf = shuffle_simple_vector(y, rng=gen)
        # train & score with the shuffled labels
        scores[i] = train_and_score(
            X, y_shuf, train_frac=train_frac, gap=gap, alpha=alpha, debug_plot=False
        )

    return scores

def plot_shuffle_results(real_score, shuffled_scores):
    """Plot histogram of shuffled scores with real score overlay."""
    plt.figure(figsize=(8, 5))
    plt.hist(shuffled_scores, bins=20, alpha=0.7, color="gray", edgecolor="black")
    plt.axvline(real_score, color="red", linewidth=2, label=f"Real R^2 = {real_score:.3f}")
    plt.xlabel("R^2 Score")
    plt.ylabel("Count")
    plt.title("Shuffle Test for Decoding")
    plt.legend()
    plt.show()

def apply_feature_filtering(X, y, corr_method="pearson", thr=0.0):
    """
    Apply correlation-based feature filtering to X given y.
    """
    return filter_out_uncorrelating_features(
        feature_matrix=X,
        label_vector=y,
        corr_method=corr_method,
        thr=thr,
    )

def run_ridge_decoding(x_path="data/test/Fdff.mat",
                       y_path="data/test/rotationalVelocity.mat",
                       n_shuffles=200,
                       rng=None,
                       use_filter=True,
                       corr_method="pearson",
                       thr=0.0,
                       **kwargs):
    X = np.asarray(load_mat_variable(x_path))
    y = np.asarray(load_mat_variable(y_path)).ravel()
    real_score, shuffled_scores = decode_with_shuffle(
        X, y, n_shuffles=n_shuffles, rng=rng,
        use_filter=use_filter, corr_method=corr_method, thr=thr,
        **kwargs
    )
    return real_score, shuffled_scores

def decode_with_shuffle(X,
                        y,
                        n_shuffles=200,
                        rng=None,
                        use_filter=True,
                        corr_method="pearson",
                        thr=0.0,
                        **kwargs):
    """
    Decode with Ridge regression + optional feature filtering.
    """
    # --- Shape handling
    if X.ndim == 1:
        X = X[:, None]
    if X.shape[0] < X.shape[1]:
        X = X.T

    # Align lengths conservatively
    n = min(X.shape[0], y.shape[0])
    X, y = X[:n], y[:n]

    # --- Optional feature filtering
    if use_filter:
        X = apply_feature_filtering(X, y, corr_method=corr_method, thr=thr)
        print(f"Filtered down to {X.shape[1]} features using {corr_method} with thr={thr}")

    # print(f"[DEBUG] Final X shape: {X.shape}, y shape: {y.shape}")
    # print(f"[DEBUG] X sample (first row, 10 values): {X[0, :10]}")
    # print(f"[DEBUG] y sample (first 10): {y[:10]}")

    # # --- Quick plot of traces
    # plt.figure(figsize=(10, 5))
    # n_neurons_to_plot = min(40, X.shape[1])
    # t = np.arange(X.shape[0])

    # for i in range(n_neurons_to_plot):
    #     plt.plot(t, X[:, i] + i*5, lw=0.8, label=f"Neuron {i}")  # offset for clarity

    # plt.plot(t, (y - np.mean(y)) / np.std(y) * 5 - 10, 'k', lw=1.2, label="Velocity (scaled)")  
    # plt.title("Calcium traces (subset) + velocity")
    # plt.xlabel("Time (frames)")
    # plt.ylabel("df/f (offset per neuron)")
    # plt.legend(loc="upper right")
    # plt.tight_layout()
    # plt.show()

    # --- Train real + shuffled
    real_score = train_and_score(X, y, **kwargs)
    shuffled_scores = shuffle_labels_return_scores(
        X, y, n_shuffles=n_shuffles, rng=rng, **kwargs
    )

    print("Real R^2:", real_score)
    print("Shuffled mean:", np.mean(shuffled_scores), "±", np.std(shuffled_scores))
    plot_shuffle_results(real_score, shuffled_scores)

    return real_score, shuffled_scores


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Decode behavior from calcium activity using Ridge regression "
                    "with shuffled-label validation (permutation baseline)."
    )
    parser.add_argument("--x_path", type=str, default="data/test/Fdff.mat",
                        help="Path to .mat file with calcium df/f.")
    parser.add_argument("--y_path", type=str, default="data/test/rotationalVelocity.mat",
                        help="Path to .mat file with behavioral variable.")
    parser.add_argument("--n_shuffles", type=int, default=200,
                        help="Number of label shuffles.")
    parser.add_argument("--train_frac", type=float, default=0.8,
                        help="Fraction of samples for training (contiguous).")
    parser.add_argument("--gap", type=int, default=0,
                        help="Samples skipped between train and test.")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Ridge regularization strength.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for shuffling.")
    # NEW:
    parser.add_argument("--use_filter", action="store_true",
                        help="Apply correlation-based feature filtering.")
    parser.add_argument("--corr_method", type=str, default="pearson",
                        choices=["pearson"],
                        help="Correlation method for filtering.")
    parser.add_argument("--thr", type=float, default=0.0,
                        help="Absolute correlation threshold |r| to keep a feature.")

    args = parser.parse_args()

    run_ridge_decoding(
        x_path=args.x_path,
        y_path=args.y_path,
        n_shuffles=args.n_shuffles,
        train_frac=args.train_frac,
        gap=args.gap,
        alpha=args.alpha,
        rng=args.seed,
        use_filter=args.use_filter,
        corr_method=args.corr_method,
        thr=args.thr,
    )
