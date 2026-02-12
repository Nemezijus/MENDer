"""Runtime-only in-memory cache for evaluation outputs keyed by artifact uid.

Used for export/post-hoc UI actions without rerunning inference.

Notes
-----
- Process-local and ephemeral.
- For multi-worker deployments, replace with shared store or persist alongside artifacts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, Optional, Sequence, Tuple, TypeAlias, Union

import numpy as np


# Keep numpy typing simple and non-invasive.
NDArray: TypeAlias = np.ndarray


@dataclass
class EvalOutputs:
    task: Optional[str] = None
    # Row indices corresponding to y_true/y_pred order (if available)
    indices: Optional[NDArray] = None
    # Fold id per row (k-fold/OOF), aligned with indices/y_true/y_pred (if available)
    fold_ids: Optional[NDArray] = None

    # Core outputs
    y_true: Optional[NDArray] = None
    y_pred: Optional[NDArray] = None

    # Classification extras (optional)
    proba: Optional[NDArray] = None
    decision_scores: Optional[NDArray] = None
    margin: Optional[NDArray] = None
    classes: Optional[Union[Sequence[Any], NDArray]] = None

    # Unsupervised extras (optional)
    cluster_id: Optional[NDArray] = None

    # Per-sample decoder / UI payloads (best-effort)
    per_sample: Optional[Dict[str, Any]] = None
    extra: Optional[Dict[str, Any]] = None


class EvalOutputsCache:
    def __init__(self):
        self._lock = Lock()
        self._store: Dict[str, Tuple[float, EvalOutputs]] = {}

    def put(self, uid: str, data: EvalOutputs, ttl_seconds: int = 60 * 60) -> None:
        expires_at = time.time() + ttl_seconds
        with self._lock:
            self._store[uid] = (expires_at, data)
            self._cleanup_locked()

    def get(self, uid: str) -> Optional[EvalOutputs]:
        now = time.time()
        with self._lock:
            item = self._store.get(uid)
            if not item:
                return None
            exp, data = item
            if exp < now:
                del self._store[uid]
                return None
            return data

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def _cleanup_locked(self) -> None:
        now = time.time()
        expired = [k for k, (exp, _) in self._store.items() if exp < now]
        for k in expired:
            del self._store[k]


# Singleton-ish cache instance
eval_outputs_cache = EvalOutputsCache()
