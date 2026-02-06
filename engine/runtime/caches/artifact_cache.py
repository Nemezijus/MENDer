"""Runtime-only in-memory cache for fitted pipelines.

This is intentionally placed under :mod:`engine.runtime` to keep core compute
modules cache-free and script-friendly.

Notes
-----
- Process-local and ephemeral.
- For multi-worker deployments, replace with shared storage (Redis, DB, etc.).
"""

from __future__ import annotations

import time
from threading import Lock
from typing import Any, Dict, Optional, Tuple


class ArtifactCache:
    def __init__(self):
        self._lock = Lock()
        self._store: Dict[str, Tuple[float, Any]] = {}

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
artifact_cache = ArtifactCache()
