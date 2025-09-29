from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple

def _group_by_label(X: np.ndarray, y: np.ndarray):
    """Return unique labels, indices per label, and counts (sorted by label)."""
    labels, inv = np.unique(y, return_inverse=True)
    buckets = [np.where(inv == i)[0] for i in range(len(labels))]
    counts = np.array([len(ix) for ix in buckets], dtype=int)
    return labels, buckets, counts

def plot_orientation_distribution(out: Dict[str, object], title: str = "Trials per orientation"):
    y = np.asarray(out["y"])
    labels, _, counts = _group_by_label(None, y)
    order = np.argsort(labels)  # sorts numeric labels (e.g., 0,45,90,135)
    plt.figure(figsize=(6,3.5))
    plt.bar(range(len(order)), counts[order])
    plt.xticks(range(len(order)), [str(labels[i]) for i in order], rotation=0)
    plt.xlabel("Orientation (deg)")
    plt.ylabel("# Trials")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def compute_orientation_means(out: Dict[str, object]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      labels: (K,) unique class labels (sorted)
      mean_resp: (K, N) mean feature per orientation (rows) and neuron (cols)
    """
    X = np.asarray(out["X"])        # (trials, neurons)
    y = np.asarray(out["y"])
    labels, buckets, _ = _group_by_label(X, y)
    mean_resp = np.vstack([X[ix].mean(axis=0) for ix in buckets])  # (K, N)
    return labels, mean_resp

def plot_population_means(out: Dict[str, object], topk:int = 12):
    """
    Show:
      1) Heatmap of mean responses per orientation (rows) × neuron (cols), neurons sorted by selectivity.
      2) Population-average tuning curve (mean across neurons) with SEM.
    """
    labels, mean_resp = compute_orientation_means(out)   # (K,), (K,N)
    K, N = mean_resp.shape

    # Simple selectivity score per neuron: max(mean) - mean(others)
    max_per_neuron = mean_resp.max(axis=0)
    mean_others = (mean_resp.sum(axis=0) - max_per_neuron) / np.maximum(K-1, 1)
    sel = max_per_neuron - mean_others
    order_neurons = np.argsort(-sel)  # descending selectivity

    # 1) Heatmap for top-k most selective neurons
    k = min(topk, N)
    plt.figure(figsize=(8, 3.5))
    plt.imshow(mean_resp[:, order_neurons[:k]], aspect="auto", interpolation="nearest")
    plt.colorbar(label="Mean response (feature units)")
    plt.yticks(range(K), [str(l) for l in labels])
    plt.xlabel(f"Neuron (top {k} by selectivity)")
    plt.ylabel("Orientation")
    plt.title("Mean response per orientation × neuron (baseline-subtracted feature)")
    plt.tight_layout()
    plt.show()

    # 2) Population-average tuning with SEM
    pop_mean = mean_resp.mean(axis=1)              # (K,)
    pop_sem  = mean_resp.std(axis=1, ddof=1) / np.sqrt(N)
    sort_or = np.argsort(labels)
    xs = labels[sort_or]
    mu = pop_mean[sort_or]
    se = pop_sem[sort_or]

    plt.figure(figsize=(6,3.5))
    plt.plot(xs, mu, marker="o")
    plt.fill_between(xs, mu - se, mu + se, alpha=0.3)
    plt.xlabel("Orientation (deg)")
    plt.ylabel("Population mean response")
    plt.title("Population tuning (feature-level)")
    plt.tight_layout()
    plt.show()

def plot_example_tuning_curves(out: Dict[str, object], n_examples: int = 6):
    """
    Plot tuning curves (mean response vs orientation) for a few example neurons.
    Picks the most selective neurons by the same simple score.
    """
    labels, mean_resp = compute_orientation_means(out)
    K, N = mean_resp.shape
    # selectivity
    max_per_neuron = mean_resp.max(axis=0)
    mean_others = (mean_resp.sum(axis=0) - max_per_neuron) / np.maximum(K-1, 1)
    sel = max_per_neuron - mean_others
    order = np.argsort(-sel)

    m = min(n_examples, N)
    cols = min(3, m); rows = int(np.ceil(m/cols))
    xs = labels[np.argsort(labels)]
    plt.figure(figsize=(4*cols, 3*rows))
    for i in range(m):
        idx = order[i]
        y = mean_resp[:, idx]
        y = y[np.argsort(labels)]
        ax = plt.subplot(rows, cols, i+1)
        ax.plot(xs, y, marker="o")
        ax.set_title(f"Neuron {idx} (sel={sel[idx]:.3f})")
        ax.set_xlabel("Orientation (deg)")
        ax.set_ylabel("Mean response")
    plt.tight_layout()
    plt.show()
