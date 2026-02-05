from __future__ import annotations

"""Compute global summaries from per-sample decoder outputs.

Business-logic only (no backend/frontend dependencies).

The Decoder Outputs table stores per-sample scores/probabilities/margins.
This module turns those into compact global diagnostics (e.g. log loss,
Brier score, margin statistics) that can surface subtle confidence shifts.

Includes:
- ECE (Expected Calibration Error) + reliability bins (top-1 confidence)

Notes on probability sources
----------------------------
Some ensembles (e.g., VotingClassifier with voting="hard") do not provide
predict_proba. In that case, MENDer may compute vote-share "probabilities"
(fraction of estimators voting for each class). These are *not calibrated*
probabilities. For vote-share probabilities, we compute confidence/margin/ECE
diagnostics, but skip log loss and Brier by default (unless explicitly enabled).
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from sklearn.metrics import log_loss as _sk_log_loss  # type: ignore
except Exception:  # pragma: no cover
    _sk_log_loss = None


def _as_1d(x: Any) -> np.ndarray:
    a = np.asarray(x)
    if a.ndim == 0:
        return a.reshape(1)
    return a.reshape(-1)


def _as_2d(x: Any) -> np.ndarray:
    a = np.asarray(x)
    if a.ndim == 0:
        return a.reshape(1, 1)
    if a.ndim == 1:
        return a.reshape(-1, 1)
    if a.ndim != 2:
        raise ValueError(f"Expected 2D array; got shape {a.shape}.")
    return a


def _safe_float(x: Any) -> float:
    """Convert to a finite float (nan/inf become finite)."""
    try:
        v = float(x)
    except Exception:
        return 0.0
    if not np.isfinite(v):
        v = float(np.nan_to_num(v, nan=0.0, posinf=1.0, neginf=0.0))
    return float(v)


def _quantile_dict(a: np.ndarray, qs: Sequence[float], prefix: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if a.size == 0:
        return out
    for q in qs:
        try:
            val = float(np.quantile(a, q))
            out[f"{prefix}_q{int(round(q * 100)):02d}"] = _safe_float(val)
        except Exception:
            continue
    return out


def _basic_stats(a: np.ndarray, prefix: str, qs: Sequence[float]) -> Dict[str, float]:
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {}

    out: Dict[str, float] = {
        f"{prefix}_mean": _safe_float(np.mean(a)),
        f"{prefix}_median": _safe_float(np.median(a)),
        f"{prefix}_std": _safe_float(np.std(a)),
        f"{prefix}_min": _safe_float(np.min(a)),
        f"{prefix}_max": _safe_float(np.max(a)),
    }
    out.update(_quantile_dict(a, qs, prefix))
    return out


def _normalize_proba(p: np.ndarray, eps: float) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1.0)
    row_sums = p.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums <= 0, 1.0, row_sums)
    return p / row_sums


def _label_to_index_map(classes: np.ndarray) -> Dict[Any, int]:
    mp: Dict[Any, int] = {}
    for i, c in enumerate(list(classes)):
        try:
            if isinstance(c, np.generic):
                c = c.item()
        except Exception:
            pass
        mp[c] = int(i)
    return mp


def _multiclass_brier(
    y_true: np.ndarray,
    proba: np.ndarray,
    classes: Optional[np.ndarray],
    notes: List[str],
) -> Optional[float]:
    """Compute multiclass Brier score: mean ||p - onehot(y)||^2."""
    if proba is None:
        return None
    p = np.asarray(proba, dtype=float)
    if p.ndim != 2:
        notes.append(f"Brier: expected proba 2D, got shape {p.shape}")
        return None

    n, k = p.shape
    if n == 0 or k == 0:
        return None

    if classes is None:
        # best-effort infer: assume y_true already 0..k-1
        try:
            y_idx = _as_1d(y_true).astype(int)
            if y_idx.size != n:
                notes.append("Brier: y_true length mismatch")
                return None
            if np.any((y_idx < 0) | (y_idx >= k)):
                notes.append("Brier: cannot infer classes; y_true indices out of range")
                return None
        except Exception:
            notes.append("Brier: cannot infer classes; provide classes")
            return None
    else:
        cls = np.asarray(classes)
        mp = _label_to_index_map(cls)
        y_raw = _as_1d(y_true)
        if y_raw.size != n:
            notes.append("Brier: y_true length mismatch")
            return None

        y_idx = np.full((n,), -1, dtype=int)
        for i, lab in enumerate(list(y_raw)):
            try:
                if isinstance(lab, np.generic):
                    lab = lab.item()
            except Exception:
                pass
            if lab in mp:
                y_idx[i] = mp[lab]

        ok = y_idx >= 0
        if not np.all(ok):
            notes.append(f"Brier: dropped {int(np.sum(~ok))} rows with unknown labels")
        if int(np.sum(ok)) == 0:
            return None

        y_idx = y_idx[ok]
        p = p[ok]
        n = int(p.shape[0])

    onehot = np.zeros((n, k), dtype=float)
    onehot[np.arange(n), y_idx] = 1.0

    dif = p - onehot
    per_row = np.sum(dif * dif, axis=1)
    return float(np.mean(per_row))


def _calibration_ece_top1(
    *,
    y_true: np.ndarray,
    proba: np.ndarray,
    classes: Optional[np.ndarray],
    n_bins: int,
    notes: List[str],
) -> Tuple[Optional[float], Optional[float], Optional[List[Dict[str, Any]]]]:
    """Compute ECE/MCE and reliability bins using top-1 confidence."""
    if proba is None:
        return None, None, None

    p = np.asarray(proba, dtype=float)
    if p.ndim != 2:
        notes.append(f"ECE: expected proba 2D, got shape {p.shape}")
        return None, None, None
    n, k = p.shape
    if n == 0 or k == 0:
        return None, None, None
    if n_bins <= 0:
        notes.append("ECE: n_bins must be > 0")
        return None, None, None

    y_raw = _as_1d(y_true)
    if y_raw.size != n:
        notes.append("ECE: y_true length mismatch")
        return None, None, None

    # Map y_true labels to indices (drop unknown labels if mapping provided)
    if classes is None:
        try:
            y_idx = y_raw.astype(int)
            if np.any((y_idx < 0) | (y_idx >= k)):
                notes.append("ECE: cannot infer classes; y_true indices out of range")
                return None, None, None
            ok = np.ones((n,), dtype=bool)
        except Exception:
            notes.append("ECE: cannot infer classes; provide classes")
            return None, None, None
    else:
        cls = np.asarray(classes)
        mp = _label_to_index_map(cls)
        y_idx = np.full((n,), -1, dtype=int)
        for i, lab in enumerate(list(y_raw)):
            try:
                if isinstance(lab, np.generic):
                    lab = lab.item()
            except Exception:
                pass
            if lab in mp:
                y_idx[i] = mp[lab]
        ok = y_idx >= 0
        dropped = int(np.sum(~ok))
        if dropped > 0:
            notes.append(f"ECE: dropped {dropped} rows with unknown labels")
        if int(np.sum(ok)) == 0:
            return None, None, None

        y_idx = y_idx[ok]
        p = p[ok]
        n = int(p.shape[0])

    pred_idx = np.argmax(p, axis=1)
    conf = np.max(p, axis=1)
    correct = (pred_idx == y_idx).astype(float)

    # Assign bin index; conf==1 goes to last bin
    bin_idx = np.minimum((conf * n_bins).astype(int), n_bins - 1)

    edges = np.linspace(0.0, 1.0, n_bins + 1)

    bins: List[Dict[str, Any]] = []
    ece = 0.0
    mce = 0.0

    for b in range(n_bins):
        mask = bin_idx == b
        cnt = int(np.sum(mask))
        lo = float(edges[b])
        hi = float(edges[b + 1])

        if cnt == 0:
            bins.append(
                {
                    "bin": b,
                    "bin_lo": lo,
                    "bin_hi": hi,
                    "count": 0,
                    "avg_confidence": None,
                    "accuracy": None,
                    "gap": None,
                }
            )
            continue

        avg_conf = float(np.mean(conf[mask]))
        acc = float(np.mean(correct[mask]))
        gap = abs(acc - avg_conf)

        weight = cnt / float(n)
        ece += weight * gap
        mce = max(mce, gap)

        bins.append(
            {
                "bin": b,
                "bin_lo": lo,
                "bin_hi": hi,
                "count": cnt,
                "avg_confidence": _safe_float(avg_conf),
                "accuracy": _safe_float(acc),
                "gap": _safe_float(gap),
            }
        )

    return _safe_float(ece), _safe_float(mce), bins


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
        y_true_arr = _as_1d(y_true)
        summary["n_samples"] = int(y_true_arr.size)

    classes_arr = np.asarray(classes) if classes is not None else None
    if classes_arr is not None:
        summary["n_classes"] = int(classes_arr.size)

    if proba_source is not None:
        summary["proba_source"] = str(proba_source)

    # Margin summaries
    if margin is not None:
        m = _as_1d(margin).astype(float)
        summary.update(_basic_stats(m, "margin", quantiles))
        for t in low_margin_thresholds:
            try:
                frac = float(np.mean(m < float(t)))
                summary[f"margin_frac_lt_{str(t).replace('.', '_')}"] = _safe_float(frac)
            except Exception:
                continue

    # Probability-based summaries
    if proba is not None:
        p = _as_2d(proba)
        if p.ndim == 2 and p.shape[0] > 0 and p.shape[1] > 0:
            p = _normalize_proba(p, eps)
            maxp = np.max(p, axis=1)
            summary.update(_basic_stats(maxp, "max_proba", quantiles))
            for t in high_conf_thresholds:
                try:
                    frac = float(np.mean(maxp >= float(t)))
                    summary[f"max_proba_frac_ge_{str(t).replace('.', '_')}"] = _safe_float(frac)
                except Exception:
                    continue

            # Log loss + Brier (only for real model probabilities by default)
            is_vote_share = (proba_source or "").lower() == "vote_share"
            if is_vote_share and not allow_vote_share_losses:
                if y_true_arr is not None:
                    notes.append("Probabilities are vote shares (hard voting); skipping log_loss and brier by default.")
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
                            summary["log_loss"] = _safe_float(ll)
                        else:
                            # manual fallback
                            if classes_arr is None:
                                raise ValueError("log_loss fallback requires classes")
                            mp = _label_to_index_map(classes_arr)
                            y_idx = np.array([mp.get(l, -1) for l in list(y_true_arr)], dtype=int)
                            ok = y_idx >= 0
                            if not np.all(ok):
                                notes.append(f"Log loss: dropped {int(np.sum(~ok))} rows with unknown labels")
                            if int(np.sum(ok)) > 0:
                                y_idx = y_idx[ok]
                                pp = p[ok]
                                ll = float(np.mean(-np.log(pp[np.arange(pp.shape[0]), y_idx])))
                                summary["log_loss"] = _safe_float(ll)
                    except Exception as e:
                        notes.append(f"Log loss: failed ({type(e).__name__}: {e})")

                # Brier score (multiclass-safe)
                if y_true_arr is not None:
                    try:
                        br = _multiclass_brier(y_true_arr, p, classes_arr, notes)
                        if br is not None:
                            summary["brier"] = _safe_float(br)
                    except Exception as e:
                        notes.append(f"Brier: failed ({type(e).__name__}: {e})")

            # ECE + reliability bins (top-1 confidence)
            if y_true_arr is not None:
                try:
                    ece, mce, bins = _calibration_ece_top1(
                        y_true=y_true_arr,
                        proba=p,
                        classes=classes_arr,
                        n_bins=int(ece_n_bins),
                        notes=notes,
                    )
                    if ece is not None:
                        summary["ece"] = _safe_float(ece)
                        summary["ece_n_bins"] = int(ece_n_bins)
                    if mce is not None:
                        summary["mce"] = _safe_float(mce)
                    if include_reliability_bins and bins is not None:
                        summary["reliability_bins"] = bins
                except Exception as e:
                    notes.append(f"ECE: failed ({type(e).__name__}: {e})")
        else:
            notes.append(f"Decoder summary: proba has invalid shape {p.shape}")

    # Decision-score summaries (optional)
    if decision_scores is not None:
        try:
            ds = np.asarray(decision_scores)
            if ds.ndim == 1:
                summary.update(_basic_stats(ds.astype(float), "decoder_score", quantiles))
            elif ds.ndim == 2 and ds.shape[1] > 0:
                top = np.max(ds.astype(float), axis=1)
                summary.update(_basic_stats(top, "top_score", quantiles))
        except Exception as e:
            notes.append(f"Decision score summaries: failed ({type(e).__name__}: {e})")

    return summary, notes
