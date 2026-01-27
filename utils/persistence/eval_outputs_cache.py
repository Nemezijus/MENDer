"""A tiny in-memory cache for evaluation outputs keyed by artifact uid.

This cache is intended for *export* and *post-hoc* UI actions.
During training, the backend can store evaluation predictions (e.g., OOF pooled
for k-fold CV or held-out test predictions) so the frontend can later request a
full CSV export without re-running inference.

Notes
-----
- Process-local and ephemeral (similar to :mod:`utils.persistence.artifact_cache`).
- For multi-worker deployments, replace with a shared store (e.g., Redis) or
  persist alongside the artifact on disk.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, Optional, Tuple


@dataclass
class EvalOutputs:
    """Cached evaluation outputs.

    All fields are optional to keep the cache flexible.
    Typical use:
      - task: "classification" | "regression" | "unsupervised"
      - indices: row indices in the evaluation set
      - fold_ids: optional fold id per sample (for CV)
      - y_true: ground-truth labels/targets
      - y_pred: predicted labels/targets
      - proba / decision_scores / margin: classification extras (optional)
      - cluster_id / per_sample: unsupervised outputs (optional)
    """

    task: Optional[str] = None
    indices: Optional[Any] = None
    fold_ids: Optional[Any] = None
    y_true: Optional[Any] = None
    y_pred: Optional[Any] = None
    proba: Optional[Any] = None
    decision_scores: Optional[Any] = None
    margin: Optional[Any] = None
    classes: Optional[Any] = None
    cluster_id: Optional[Any] = None
    per_sample: Optional[Dict[str, Any]] = None
    extra: Optional[Dict[str, Any]] = None


class _EvalOutputsCache:
    def __init__(self):
        self._lock = Lock()
        self._store: Dict[str, Tuple[float, EvalOutputs]] = {}  # uid -> (expires_at, data)

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
eval_outputs_cache = _EvalOutputsCache()
