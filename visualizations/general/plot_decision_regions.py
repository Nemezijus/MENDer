from __future__ import annotations

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm
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
    If X has exactly 2 features: plot decision regions (multi-class OK).
    Otherwise: fall back to pairwise feature plots.

    - Handles arbitrary number of classes (dynamic color palette).
    - Works with string or numeric labels (maps to class indices for contour).
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim != 2:
        raise ValueError("X must be 2D.")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError("y must be 1D and aligned with X.")

    if X.shape[1] != 2:
        print("[INFO] plot_decision_regions: X has != 2 features; showing pairwise feature plots.")
        plot_features(X, y)
        return

    # Classes and palette
    classes = np.unique(y)
    n_classes = len(classes)

    # use a large qualitative palette (tab20) sufficient for many classes
    base_cmap = cm.get_cmap("tab20")
    colors = [base_cmap(i / max(1, n_classes - 1)) for i in range(n_classes)]
    cmap = ListedColormap(colors)

    # Mesh grid
    x1_min, x1_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    x2_min, x2_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution),
    )

    # Predict on grid
    grid = np.c_[xx1.ravel(), xx2.ravel()]
    lab_pred = classifier.predict(grid)

    # Map labels (possibly strings) to integer indices for contourf
    class_to_idx = {c: i for i, c in enumerate(classes)}
    Z = np.vectorize(class_to_idx.get)(lab_pred).reshape(xx1.shape)

    # Plot decision surface
    plt.figure(figsize=(6.8, 5.6))
    # levels so each class is a distinct region
    levels = np.arange(-0.5, n_classes + 0.5, 1.0)
    plt.contourf(xx1, xx2, Z, levels=levels, alpha=0.30, cmap=cmap, antialiased=True)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # Scatter training points
    markers = ("o", "s", "^", "v", "<", ">", "P", "X", "D", "h")
    for idx, cl in enumerate(classes):
        mask = (y == cl)
        plt.scatter(
            X[mask, 0],
            X[mask, 1],
            alpha=0.85,
            c=[colors[idx]],             # single color per class
            marker=markers[idx % len(markers)],
            label=str(cl),
            edgecolor="black",
            linewidths=0.5,
        )

    if title:
        plt.title(title)
    else:
        plt.xlabel("PC 1")
        plt.ylabel("PC 2")
    plt.legend(loc="best", fontsize=9, framealpha=0.9)
    plt.tight_layout()
    plt.show()
