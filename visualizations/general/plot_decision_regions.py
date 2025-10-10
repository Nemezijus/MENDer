from __future__ import annotations

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from visualizations.general.plot_features import plot_features


def plot_decision_regions(
    X: np.ndarray,
    y: np.ndarray,
    *,
    classifier,
    resolution: float = 0.02,
    title: Optional[str] = None,
):
    """
    If X has exactly 2 features: plot classic decision regions.
    Otherwise: fall back to pairwise feature plots (no decision surface).

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    y : array-like, shape (n_samples,)
    classifier : fitted estimator with .predict()
        REQUIRED (for 2D decision regions).
    resolution : float, default=0.02
        Grid step in the 2D case.
    title : str or None
        Optional title.

    Returns
    -------
    None
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim != 2:
        raise ValueError("X must be 2D.")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError("y must be 1D and aligned with X.")

    if X.shape[1] != 2:
        # Not 2D: show pairwise features instead (no surface)
        print("[INFO] plot_decision_regions: X has != 2 features; showing pairwise feature plots.")
        plot_features(X, y)
        return

    # --- 2D decision regions ---
    markers = ("o", "s", "^", "v", "<")
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    classes = np.unique(y)
    cmap = ListedColormap(colors[: len(classes)])

    x1_min, x1_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    x2_min, x2_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution),
    )

    grid = np.c_[xx1.ravel(), xx2.ravel()]
    lab = classifier.predict(grid).reshape(xx1.shape)

    plt.figure(figsize=(6.5, 5.5))
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(classes):
        plt.scatter(
            X[y == cl, 0],
            X[y == cl, 1],
            alpha=0.8,
            c=colors[idx],
            marker=markers[idx % len(markers)],
            label=f"{cl}",
            edgecolor="black",
        )

    if title:
        plt.title(title)
    plt.xlabel("PC 1" if title is None else "")
    plt.ylabel("PC 2" if title is None else "")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()