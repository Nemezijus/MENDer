from __future__ import annotations

"""Compute global summaries from per-sample decoder outputs.

Business-logic only (no backend/frontend dependencies).

The Decoder Outputs table stores per-sample scores/probabilities/margins.
This module turns those into compact global diagnostics (e.g. log loss,
Brier score, margin statistics) that can surface subtle confidence shifts.

Implementation details are split for SRP:
- common array/stats utilities: :mod:`engine.reporting.decoder.common`
- binary-ish summaries (margin/max_proba/decision scores): :mod:`engine.reporting.decoder.binary`
- multiclass losses: :mod:`engine.reporting.decoder.multiclass`
- calibration metrics (ECE/MCE): :mod:`engine.reporting.decoder.calibration`
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from sklearn.metrics import log_loss as _sk_log_loss  # type: ignore
except Exception:  # pragma: no cover
    _sk_log_loss = None

from .binary import summarize_decision_scores, summarize_margin, summarize_max_proba
from .calibration import calibration_ece_top1
from .common import as_1d, as_2d, normalize_proba, safe_float
from .multiclass import log_loss_fallback, multiclass_brier


def compute_decoder_summaries(
    *,
    y_true: Optional[Any],
    classes: Optional[Any] = None,
    proba: Optional[Any] = None,
    proba_source: Optional[str] = None,
    allow_vote_share_losses: bool = False,
    decision_scores: Optional[Any] = None,
    margin: Optional[Any] = None,
    low_margin_thresholds: Sequence[float] = (0.05, 0.1),
    high_conf_thresholds: Sequence[float] = (0.8, 0.9),
    quantiles: Sequence[float] = (0.1, 0.25, 0.75, 0.9),
    eps: float = 1e-15,
    # Calibration diagnostics (ECE)
    ece_n_bins: int = 10,
    include_reliability_bins: bool = True,
) -> Tuple[Dict[str, Any], List[str]]:
    """Compute compact global diagnostics from decoder outputs.

    Returns
    -------
    (summary, notes)
        summary is JSON-friendly.
    """

    summary: Dict[str, Any] = {}
    notes: List[str] = []

    if y_true is None:
        notes.append("Decoder summary: y_true missing; skipping label-dependent metrics")
        y_true_arr = None
    else:
        y_true_arr = as_1d(y_true)
        summary["n_samples"] = int(y_true_arr.size)

    classes_arr = np.asarray(classes) if classes is not None else None
    if classes_arr is not None:
        summary["n_classes"] = int(classes_arr.size)

    if proba_source is not None:
        summary["proba_source"] = str(proba_source)

    # Margin summaries
    if margin is not None:
        summary.update(
            summarize_margin(
                margin=margin,
                quantiles=quantiles,
                low_margin_thresholds=low_margin_thresholds,
            )
        )

    # Probability-based summaries
    if proba is not None:
        p = as_2d(proba)
        if p.ndim == 2 and p.shape[0] > 0 and p.shape[1] > 0:
            p = normalize_proba(p, eps)
            summary.update(
                summarize_max_proba(
                    proba=p,
                    quantiles=quantiles,
                    high_conf_thresholds=high_conf_thresholds,
                )
            )

            # Log loss + Brier (only for real model probabilities by default)
            is_vote_share = (proba_source or "").lower() == "vote_share"
            if is_vote_share and not allow_vote_share_losses:
                if y_true_arr is not None:
                    notes.append(
                        "Probabilities are vote shares (hard voting); skipping log_loss and brier by default."
                    )
            else:
                # Log loss
                if y_true_arr is not None:
                    try:
                        if _sk_log_loss is not None:
                            ll = _sk_log_loss(
                                y_true_arr,
                                p,
                                labels=classes_arr.tolist() if classes_arr is not None else None,
                            )
                            summary["log_loss"] = safe_float(ll)
                        else:
                            if classes_arr is None:
                                raise ValueError("log_loss fallback requires classes")
                            ll = log_loss_fallback(
                                y_true=y_true_arr,
                                proba=p,
                                classes=classes_arr,
                                notes=notes,
                            )
                            if ll is not None:
                                summary["log_loss"] = safe_float(ll)
                    except Exception as e:
                        notes.append(f"Log loss: failed ({type(e).__name__}: {e})")

                # Brier score (multiclass-safe)
                if y_true_arr is not None:
                    try:
                        br = multiclass_brier(y_true_arr, p, classes_arr, notes)
                        if br is not None:
                            summary["brier"] = safe_float(br)
                    except Exception as e:
                        notes.append(f"Brier: failed ({type(e).__name__}: {e})")

            # ECE + reliability bins (top-1 confidence)
            if y_true_arr is not None:
                try:
                    ece, mce, bins = calibration_ece_top1(
                        y_true=y_true_arr,
                        proba=p,
                        classes=classes_arr,
                        n_bins=int(ece_n_bins),
                        notes=notes,
                    )
                    if ece is not None:
                        summary["ece"] = safe_float(ece)
                        summary["ece_n_bins"] = int(ece_n_bins)
                    if mce is not None:
                        summary["mce"] = safe_float(mce)
                    if include_reliability_bins and bins is not None:
                        summary["reliability_bins"] = bins
                except Exception as e:
                    notes.append(f"ECE: failed ({type(e).__name__}: {e})")
        else:
            notes.append(f"Decoder summary: proba has invalid shape {p.shape}")

    # Decision-score summaries (optional)
    if decision_scores is not None:
        try:
            summary.update(summarize_decision_scores(decision_scores=decision_scores, quantiles=quantiles))
        except Exception as e:
            notes.append(f"Decision score summaries: failed ({type(e).__name__}: {e})")

    return summary, notes
