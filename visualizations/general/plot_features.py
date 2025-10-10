from __future__ import annotations

from typing import Optional, Sequence
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def plot_features(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    *,
    feature_names: Optional[Sequence[str]] = None,
    max_features: int = 5,
    figsize: tuple = (12, 9),
    alpha: float = 0.8,
):
    """
    Pairwise scatter plots for up to `max_features` features (default 5 ⇒ C(5,2)=10 panels).
    Plots the first `max_features` columns if X has more.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    y : array-like, shape (n_samples,), optional
        If provided, color by class label.
    feature_names : list[str], optional
        Names for the axes; defaults to f"f{i}".
    max_features : int, default=5
        Cap on how many features to visualize (pairwise combinations).
    figsize : tuple, default=(12, 9)
        Figure size for the grid of subplots.
    alpha : float, default=0.8
        Marker opacity.

    Returns
    -------
    None (shows a figure)
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be 2D.")
    n, d = X.shape
    d_plot = min(d, max_features)

    cols = list(range(d_plot))
    if feature_names is None:
        feature_names = [f"f{i}" for i in cols]
    else:
        feature_names = list(feature_names)[:d_plot]
        if len(feature_names) < d_plot:
            feature_names += [f"f{i}" for i in range(len(feature_names), d_plot)]

    pairs = list(itertools.combinations(range(d_plot), 2))  # all pairs
    n_panels = len(pairs)
    n_cols = 4
    n_rows = int(np.ceil(n_panels / n_cols))

    plt.figure(figsize=figsize)
    cmap = None
    if y is not None:
        classes = np.unique(y)
        base_colors = ("red", "blue", "lightgreen", "gray", "cyan", "orange", "purple")
        cmap = ListedColormap(base_colors[: len(classes)])

    for k, (i, j) in enumerate(pairs, start=1):
        ax = plt.subplot(n_rows, n_cols, k)
        if y is None:
            ax.scatter(X[:, i], X[:, j], s=20, alpha=alpha, edgecolors="none")
        else:
            classes = np.unique(y)
            for idx, cl in enumerate(classes):
                sel = (y == cl)
                ax.scatter(
                    X[sel, i], X[sel, j],
                    s=20, alpha=alpha,
                    c=[cmap(idx)],
                    label=str(cl) if k == 1 else None,
                    edgecolors="none",
                )
        ax.set_xlabel(feature_names[i])
        ax.set_ylabel(feature_names[j])

    if y is not None:
        plt.legend(loc="best", frameon=True, title="Class")

    plt.tight_layout()
    plt.show()
