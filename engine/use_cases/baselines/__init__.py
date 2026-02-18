"""Baseline use-cases.

Baselines are *workflows* that repeatedly execute a training/evaluation loop
under some perturbation (e.g., label shuffling) to produce a reference
distribution of scores.

They are intentionally kept in ``use_cases`` because they orchestrate splitters,
pipelines, and evaluators. Components under ``engine/components`` should remain
compute-oriented and dependency-injected.
"""

from .label_shuffle import make_label_shuffle_baseline, run_label_shuffle_baseline

__all__ = [
    "make_label_shuffle_baseline",
    "run_label_shuffle_baseline",
]
