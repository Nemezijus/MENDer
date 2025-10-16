from __future__ import annotations

from typing import Optional, Sequence, Tuple
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm


def _class_palette(y: np.ndarray) -> Tuple[np.ndarray, ListedColormap]:
    classes = np.unique(y)
    n_classes = len(classes)
    base_cmap = cm.get_cmap("tab20")
    colors = np.array([base_cmap(i / max(1, n_classes - 1)) for i in range(n_classes)])
    return classes, ListedColormap(colors)


def _markers():
    return ("o", "s", "^", "v", "<", ">", "P", "X", "D", "h")


def _scatter_points(ax, X, y, classes, colors, label_once=False):
    mks = _markers()
    for idx, cl in enumerate(classes):
        sel = (y == cl)
        ax.scatter(
            X[sel, 0], X[sel, 1],
            alpha=0.85,
            c=[colors(idx)],
            marker=mks[idx % len(mks)],
            label=str(cl) if label_once else None,
            edgecolor="black",
            linewidths=0.5,
            s=26,
        )


def _plot_2d_decision(ax, clf, X2d_bounds, other_fixed, i, j, classes, cmap, resolution=0.02):
    """Plot decision surface in (i,j) plane while fixing other dims to their means."""
    (x1_min, x1_max), (x2_min, x2_max) = X2d_bounds
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution),
    )
    # Build grid in full-D space: vary i, j; others fixed
    n_pts = xx1.size
    D = other_fixed.shape[0]
    grid = np.tile(other_fixed, (n_pts, 1))
    grid[:, i] = xx1.ravel()
    grid[:, j] = xx2.ravel()

    lab_pred = clf.predict(grid)
    class_to_idx = {c: k for k, c in enumerate(classes)}
    Z = np.vectorize(class_to_idx.get)(lab_pred).reshape(xx1.shape)

    levels = np.arange(-0.5, len(classes) + 0.5, 1.0)
    ax.contourf(xx1, xx2, Z, levels=levels, alpha=0.30, cmap=cmap, antialiased=True)
    ax.set_xlim(x1_min, x1_max)
    ax.set_ylim(x2_min, x2_max)


def plot_decision_regions(
    X: np.ndarray,
    y: np.ndarray,
    *,
    classifier,
    resolution: float = 0.02,
    title: Optional[str] = None,
    feature_names: Optional[Sequence[str]] = None,
    max_features_for_pairwise: int = 5,
):
    """
    Decision-region visualization with smart fallbacks:

    - If X has 2 features: classic decision region plot with training points.
    - If 3–5 features: pairwise (C(d,2)) panels; for each pair (i,j), we draw the
      decision surface while fixing all other features at their mean.
    - If >5 features: skip plotting (print info). Intended when no dimensionality reduction was used.

    Parameters
    ----------
    X : (n_samples, n_features)
    y : (n_samples,)
    classifier : fitted model with .predict
    resolution : grid resolution for decision surface
    title : optional fig title (used for the 2D case)
    feature_names : optional names for axes (used in pairwise)
    max_features_for_pairwise : cap for pairwise mode (default 5)
    """
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    if X.ndim != 2:
        raise ValueError("X must be 2D.")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError("y must be 1D and aligned with X.")

    n, d = X.shape
    classes, cmap = _class_palette(y)
    colors = cmap
    mks = _markers()

    # === Case 1: exactly 2 features -> standard single-panel decision region ===
    if d == 2:
        x1_min, x1_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
        x2_min, x2_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
        xx1, xx2 = np.meshgrid(
            np.arange(x1_min, x1_max, resolution),
            np.arange(x2_min, x2_max, resolution),
        )
        grid = np.c_[xx1.ravel(), xx2.ravel()]
        lab_pred = classifier.predict(grid)
        class_to_idx = {c: i for i, c in enumerate(classes)}
        Z = np.vectorize(class_to_idx.get)(lab_pred).reshape(xx1.shape)

        plt.figure(figsize=(6.8, 5.6))
        levels = np.arange(-0.5, len(classes) + 0.5, 1.0)
        plt.contourf(xx1, xx2, Z, levels=levels, alpha=0.30, cmap=cmap, antialiased=True)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())

        for idx, cl in enumerate(classes):
            sel = (y == cl)
            plt.scatter(
                X[sel, 0], X[sel, 1],
                alpha=0.85,
                c=[colors(idx)],
                marker=mks[idx % len(mks)],
                label=str(cl),
                edgecolor="black",
                linewidths=0.5,
                s=26,
            )

        if title:
            plt.title(title)
        else:
            plt.xlabel("Feature 1")
            plt.ylabel("Feature 2")
        plt.legend(loc="best", fontsize=9, framealpha=0.9)
        plt.tight_layout()
        plt.show()
        return

    # === Case 2: 3–5 features -> pairwise panels with decision surface ===
    if 3 <= d <= max_features_for_pairwise:
        cols = list(range(d))
        if feature_names is None:
            feat_names = [f"f{i}" for i in cols]
        else:
            feat_names = list(feature_names)[:d]
            if len(feat_names) < d:
                feat_names += [f"f{i}" for i in range(len(feat_names), d)]

        pairs = list(itertools.combinations(range(d), 2))
        n_panels = len(pairs)
        n_cols = 4
        n_rows = int(np.ceil(n_panels / n_cols))

        # Fix other dims at their mean over the provided X (train data)
        mu = X.mean(axis=0)

        plt.figure(figsize=(12, 9))
        for k, (i, j) in enumerate(pairs, start=1):
            ax = plt.subplot(n_rows, n_cols, k)

            # Determine per-plot bounds with small padding
            pad_i = 0.05 * (X[:, i].max() - X[:, i].min() + 1e-12)
            pad_j = 0.05 * (X[:, j].max() - X[:, j].min() + 1e-12)
            x1_min, x1_max = X[:, i].min() - pad_i, X[:, i].max() + pad_i
            x2_min, x2_max = X[:, j].min() - pad_j, X[:, j].max() + pad_j

            _plot_2d_decision(
                ax, classifier,
                X2d_bounds=((x1_min, x1_max), (x2_min, x2_max)),
                other_fixed=mu.copy(),
                i=i, j=j,
                classes=classes,
                cmap=cmap,
                resolution=resolution,
            )

            # scatter the projected points (i,j)
            # label only once so legend is neat
            first_panel = (k == 1)
            XY = X[:, [i, j]]
            _scatter_points(ax, XY, y, classes, colors, label_once=first_panel)

            ax.set_xlabel(feat_names[i])
            ax.set_ylabel(feat_names[j])

        if title:
            plt.suptitle(title, y=0.995)

        # Build a single legend from the first panel's labels
        handles, labels = plt.gca().get_legend_handles_labels()
        if handles and labels:
            plt.figlegend(handles, labels, loc="upper right", frameon=True)

        plt.tight_layout(rect=[0, 0, 0.92, 1])
        plt.show()
        return
    # === Case 3: > 5 features -> skip (assumed no dimensionality reduction) ===
    if d > max_features_for_pairwise:
        print("[INFO] plot_decision_regions: X has > 5 features and no reduction was applied; skipping plots.")
        return

    # === Case 4: < 2 features -> cannot draw decision regions ===
    if d < 2:
        print("[INFO] plot_decision_regions: X has < 2 features; cannot draw decision regions.")
        # optional: fall back to a simple 1D scatter/histogram or do nothing.
        return
