from __future__ import annotations

from typing import Optional, Sequence
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm


def _class_palette(y: np.ndarray) -> tuple[np.ndarray, ListedColormap]:
    classes = np.unique(y)
    n_classes = len(classes)
    base_cmap = cm.get_cmap("tab20")
    colors = [base_cmap(i / max(1, n_classes - 1)) for i in range(n_classes)]
    return classes, ListedColormap(colors)


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

    If y is given, classes are colored consistently across panels.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be 2D.")
    n, d = X.shape

    d_plot = min(d, max_features)
    if d_plot < 2:
        print("[INFO] plot_features: Need at least 2 features to plot; skipping.")
        return

    cols = list(range(d_plot))
    if feature_names is None:
        feat_names = [f"f{i}" for i in cols]
    else:
        feat_names = list(feature_names)[:d_plot]
        if len(feat_names) < d_plot:
            feat_names += [f"f{i}" for i in range(len(feat_names), d_plot)]

    pairs = list(itertools.combinations(range(d_plot), 2))
    n_panels = len(pairs)
    n_cols = 4
    n_rows = int(np.ceil(n_panels / n_cols))

    plt.figure(figsize=figsize)

    if y is not None:
        y = np.asarray(y).ravel()
        classes, cmap = _class_palette(y)
        colors = cmap
        markers = ("o", "s", "^", "v", "<", ">", "P", "X", "D", "h")
    else:
        classes, colors, markers = None, None, None

    for k, (i, j) in enumerate(pairs, start=1):
        ax = plt.subplot(n_rows, n_cols, k)
        if y is None:
            ax.scatter(X[:, i], X[:, j], s=20, alpha=alpha, edgecolors="none")
        else:
            for idx, cl in enumerate(classes):
                sel = (y == cl)
                ax.scatter(
                    X[sel, i], X[sel, j],
                    s=20, alpha=alpha,
                    c=[colors(idx)],
                    label=str(cl) if k == 1 else None,
                    edgecolors="none",
                    marker=markers[idx % len(markers)],
                )
        ax.set_xlabel(feat_names[i])
        ax.set_ylabel(feat_names[j])

    if y is not None:
        handles, labels = plt.gca().get_legend_handles_labels()
        if handles and labels:
            plt.figlegend(handles, labels, loc="upper right", frameon=True, title="Class")

    plt.tight_layout(rect=[0, 0, 0.92, 1])
    plt.show()
