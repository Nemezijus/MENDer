# instances/cv_classify_with_shuffle.py
from __future__ import annotations

import warnings
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, KFold

from shared_schemas.run_config import RunConfig
from utils.permutations.rng import RngManager

from utils.factories.data_loading_factory import make_data_loader
from utils.factories.sanity_factory import make_sanity_checker
from utils.factories.pipeline_factory import make_pipeline, make_preproc_pipeline
from utils.factories.eval_factory import make_evaluator

warnings.filterwarnings("ignore", category=FutureWarning)


def _plot_shuffle_hist(real_score: float, shuffled_scores: np.ndarray, metric: str):
    plt.figure(figsize=(8, 5))
    plt.hist(shuffled_scores, bins=20, alpha=0.8, color="gray", edgecolor="black")
    plt.axvline(real_score, color="red", linewidth=2, label=f"Real {metric} = {real_score:.3f}")
    plt.xlabel(metric.capitalize())
    plt.ylabel("Count")
    plt.title(f"K-fold shuffle baseline ({metric})")
    plt.legend()
    plt.tight_layout()
    plt.show()


def _make_cv_indices(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_splits: int,
    stratified: bool,
    shuffle: bool,
    seed: int | None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Produce a fixed list of (train_idx, test_idx) folds ONCE from the ORIGINAL y.
    We reuse these same indices for all shuffles so folds are comparable.
    """
    splitter_cls = StratifiedKFold if stratified else KFold
    splitter = splitter_cls(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=(seed if shuffle else None),
    )
    folds = [(tr, te) for tr, te in splitter.split(X, y)]
    return folds


def _cv_mean_score_for_labels(
    X: np.ndarray,
    y_labels: np.ndarray,
    *,
    cfg: RunConfig,
    rngm: RngManager,
    folds: List[Tuple[np.ndarray, np.ndarray]],
) -> float:
    evaluator = make_evaluator(cfg.eval, kind="classification")
    fold_scores: List[float] = []

    # Iterate a fixed set of folds (indices), build a fresh pipeline each time
    for i, (tr_idx, te_idx) in enumerate(folds, start=1):
        Xtr, Xte = X[tr_idx], X[te_idx]
        ytr, yte = y_labels[tr_idx], y_labels[te_idx]

        # independent seed stream per fold
        _ = rngm.child_seed(f"cv/fold{i}")
        pipe = make_pipeline(cfg, rngm, stream=f"cv/fold{i}")
        pipe.fit(Xtr, ytr)

        ypred = pipe.predict(Xte)
        score = evaluator.score(yte, ypred)
        fold_scores.append(float(score))

    return float(np.mean(fold_scores))


def run_cv_decoding(cfg: RunConfig):
    """
    Orchestrates: load → sanity → fixed CV folds → pipeline fit/score per fold → mean score →
                  label-shuffle baseline (CV mean each time) → histogram.
    Reuses your existing factories; does not modify core strategies.
    """

    # 1) Load data
    loader = make_data_loader(cfg.data)
    X, y = loader.load()

    # 2) Sanity checks
    sanity = make_sanity_checker()
    sanity.check(X, y)

    # 3) Central RNG
    rngm = RngManager(None if cfg.eval.seed is None else int(cfg.eval.seed))

    # 4) Build fixed CV folds ONCE from original labels
    n_splits = getattr(cfg.split, "n_splits", 5)
    stratified = getattr(cfg.split, "stratified", True)
    shuffle = getattr(cfg.split, "shuffle", True)
    split_seed = rngm.child_seed("cv/split")

    folds = _make_cv_indices(
        X, y,
        n_splits=int(n_splits),
        stratified=bool(stratified),
        shuffle=bool(shuffle),
        seed=split_seed,
    )
    print(f"[INFO] CV setup: n_splits={n_splits}, stratified={stratified}, shuffle={shuffle}")

    # (Optional) Preproc diagnostic plot space (fit on whole training of each fold when needed)
    # keep it minimal here; full plots add a lot of noise in CV

    # 5) Real (unshuffled) CV mean score
    real_cv_mean = _cv_mean_score_for_labels(X, y, cfg=cfg, rngm=rngm, folds=folds)
    print(f"[INFO] Real {cfg.eval.metric} (CV mean over {n_splits} folds): {real_cv_mean:.3f}")

    # 6) Shuffle baseline: for each shuffle, permute y and compute CV mean score over SAME folds
    n_shuffles = int(getattr(cfg.eval, "n_shuffles", 100) or 0)
    shuffled_means = np.empty(n_shuffles, dtype=float)

    for i in range(n_shuffles):
        # independent permutation seed
        seed_i = rngm.child_seed(f"cv/shuffle/{i}")
        y_perm = np.random.default_rng(seed_i).permutation(y)
        shuffled_means[i] = _cv_mean_score_for_labels(X, y_perm, cfg=cfg, rngm=rngm, folds=folds)

        if (i + 1) % max(1, n_shuffles // 10) == 0:
            print(f"[INFO] Shuffle {i + 1}/{n_shuffles}: mean={shuffled_means[i]:.3f}")

    if n_shuffles > 0:
        print(f"[INFO] Shuffled CV mean: {np.mean(shuffled_means):.3f} ± {np.std(shuffled_means):.3f}")

    # 7) Plot histogram
    if n_shuffles > 0:
        _plot_shuffle_hist(real_cv_mean, shuffled_means, metric=cfg.eval.metric)

    return real_cv_mean, shuffled_means
