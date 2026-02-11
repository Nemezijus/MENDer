from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .common import as_1d, label_to_index_map


def multiclass_brier(
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
            y_idx = as_1d(y_true).astype(int)
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
        mp = label_to_index_map(cls)
        y_raw = as_1d(y_true)
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


def log_loss_fallback(
    *,
    y_true: np.ndarray,
    proba: np.ndarray,
    classes: np.ndarray,
    notes: List[str],
) -> Optional[float]:
    """Manual log loss fallback used when sklearn is unavailable."""
    mp = label_to_index_map(classes)
    y_idx = np.array([mp.get(l, -1) for l in list(y_true)], dtype=int)
    ok = y_idx >= 0
    if not np.all(ok):
        notes.append(f"Log loss: dropped {int(np.sum(~ok))} rows with unknown labels")
    if int(np.sum(ok)) == 0:
        return None

    y_idx = y_idx[ok]
    pp = proba[ok]
    ll = float(np.mean(-np.log(pp[np.arange(pp.shape[0]), y_idx])))
    return ll
