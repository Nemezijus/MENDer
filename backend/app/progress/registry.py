from __future__ import annotations
from typing import Dict, Optional
from threading import Lock
import time

class _ProgressStore:
    def __init__(self) -> None:
        self._store: Dict[str, Dict] = {}
        self._lock = Lock()

    def init(self, pid: str, total: int = 0, label: str = "Starting…") -> None:
        with self._lock:
            self._store[pid] = {
                "progress_id": pid,
                "total": int(total),
                "current": 0,
                "percent": 0.0,
                "label": label,
                "done": False,
                "error": None,
                "updated_at": time.time(),
            }

    def update(self, pid: str, current: int, total: Optional[int] = None, label: Optional[str] = None) -> None:
        with self._lock:
            if pid not in self._store:
                return
            rec = self._store[pid]
            if total is not None:
                rec["total"] = int(total)
            rec["current"] = int(current)
            tot = max(1, int(rec["total"]))
            rec["percent"] = max(0.0, min(100.0, 100.0 * rec["current"] / tot))
            if label is not None:
                rec["label"] = label
            rec["updated_at"] = time.time()

    def finalize(self, pid: str, label: str = "Done") -> None:
        with self._lock:
            if pid not in self._store:
                return
            rec = self._store[pid]
            rec["current"] = rec["total"] or rec["current"]
            rec["percent"] = 100.0
            rec["label"] = label
            rec["done"] = True
            rec["updated_at"] = time.time()

    def fail(self, pid: str, message: str) -> None:
        with self._lock:
            if pid not in self._store:
                self._store[pid] = {
                    "progress_id": pid, "total": 1, "current": 0,
                    "percent": 0.0, "label": "Failed", "done": True,
                    "error": message, "updated_at": time.time()
                }
                return
            rec = self._store[pid]
            rec["label"] = "Failed"
            rec["done"] = True
            rec["error"] = message
            rec["updated_at"] = time.time()

    def get(self, pid: str) -> Optional[Dict]:
        with self._lock:
            return None if pid not in self._store else dict(self._store[pid])

PROGRESS = _ProgressStore()
