"""
A tiny in-memory cache for fitted pipelines keyed by artifact uid.
Used to hold the last-trained model(s) short-term so Save/Apply can retrieve them
without shipping heavy objects over HTTP.

NOTE: This is process-local and ephemeral. For multi-worker deployments, replace
with a shared store (e.g., Redis) or adapt the service layer accordingly.
"""

from __future__ import annotations

import time
from threading import Lock
from typing import Any, Dict, Optional, Tuple


class _ArtifactCache:
    def __init__(self):
        self._lock = Lock()
        self._store: Dict[str, Tuple[float, Any]] = {}  # uid -> (expires_at, pipeline)

    def put(self, uid: str, pipeline: Any, ttl_seconds: int = 60 * 60) -> None:
        expires_at = time.time() + ttl_seconds
        with self._lock:
            self._store[uid] = (expires_at, pipeline)
            self._cleanup_locked()

    def get(self, uid: str) -> Optional[Any]:
        now = time.time()
        with self._lock:
            item = self._store.get(uid)
            if not item:
                return None
            exp, pipeline = item
            if exp < now:
                # expired
                del self._store[uid]
                return None
            return pipeline

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def _cleanup_locked(self) -> None:
        now = time.time()
        expired = [k for k, (exp, _) in self._store.items() if exp < now]
        for k in expired:
            del self._store[k]


# Singleton-ish cache instance
artifact_cache = _ArtifactCache()
