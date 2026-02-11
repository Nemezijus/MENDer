from __future__ import annotations

"""Helpers that are particularly useful for binary classification.

Currently these helpers are generic enough to work for multiclass too, but
they are organized here to keep decoder_summaries.py small and to make it easy
to extend binary-only diagnostics later.
"""

from typing import Any, Dict, List, Sequence

import numpy as np

from .common import as_1d, basic_stats, safe_float


def summarize_margin(
    *,
    margin: Any,
    quantiles: Sequence[float],
    low_margin_thresholds: Sequence[float],
) -> Dict[str, float]:
    m = as_1d(margin).astype(float)
    out: Dict[str, float] = {}
    out.update(basic_stats(m, "margin", quantiles))
    for t in low_margin_thresholds:
        try:
            frac = float(np.mean(m < float(t)))
            out[f"margin_frac_lt_{str(t).replace('.', '_')}"] = safe_float(frac)
        except Exception:
            continue
    return out


def summarize_max_proba(
    *,
    proba: np.ndarray,
    quantiles: Sequence[float],
    high_conf_thresholds: Sequence[float],
) -> Dict[str, float]:
    maxp = np.max(proba, axis=1)
    out: Dict[str, float] = {}
    out.update(basic_stats(maxp, "max_proba", quantiles))
    for t in high_conf_thresholds:
        try:
            frac = float(np.mean(maxp >= float(t)))
            out[f"max_proba_frac_ge_{str(t).replace('.', '_')}"] = safe_float(frac)
        except Exception:
            continue
    return out


def summarize_decision_scores(
    *,
    decision_scores: Any,
    quantiles: Sequence[float],
) -> Dict[str, float]:
    """Summarize decision scores for binary (1D) or multiclass (2D) scores."""
    out: Dict[str, float] = {}
    ds = np.asarray(decision_scores)
    if ds.ndim == 1:
        out.update(basic_stats(ds.astype(float), "decoder_score", quantiles))
    elif ds.ndim == 2 and ds.shape[1] > 0:
        top = np.max(ds.astype(float), axis=1)
        out.update(basic_stats(top, "top_score", quantiles))
    return out
