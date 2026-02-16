from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .common import as_1d, label_to_index_map


def calibration_ece_top1(
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

    y_raw = as_1d(y_true)
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
        mp = label_to_index_map(cls)
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

    # Top-1 confidence and correctness
    y_hat = np.argmax(p, axis=1)
    conf = np.max(p, axis=1)
    correct = (y_hat == y_idx).astype(float)

    # Bin edges in confidence space
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    ece = 0.0
    mce = 0.0
    bins: List[Dict[str, Any]] = []

    for b in range(int(n_bins)):
        lo = float(edges[b])
        hi = float(edges[b + 1])
        if b == int(n_bins) - 1:
            m = (conf >= lo) & (conf <= hi)
        else:
            m = (conf >= lo) & (conf < hi)

        nb = int(np.sum(m))
        if nb == 0:
            # Keep legacy (pre-refactor) keys so the existing frontend can render.
            bins.append(
                {
                    "bin": int(b),
                    "bin_lo": lo,
                    "bin_hi": hi,
                    "count": 0,
                    "avg_confidence": None,
                    "accuracy": None,
                    "gap": None,
                }
            )
            continue

        acc = float(np.mean(correct[m]))
        avg_conf = float(np.mean(conf[m]))
        gap = abs(acc - avg_conf)
        w = nb / float(n)
        ece += w * gap
        mce = max(mce, gap)

        bins.append(
            {
                "bin": int(b),
                "bin_lo": lo,
                "bin_hi": hi,
                "count": nb,
                "avg_confidence": avg_conf,
                "accuracy": acc,
                "gap": gap,
            }
        )

    return float(ece), float(mce), bins
