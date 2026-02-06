from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from .store import ArtifactStore, StoredArtifact


def _json_default(obj: Any):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def default_artifacts_dir() -> Path:
    # Use env var when running backend; keep script-friendly default otherwise.
    raw = os.getenv("MENDER_ARTIFACTS_DIR", ".mender/artifacts")
    return Path(raw)


class FileSystemArtifactStore:
    """Filesystem-based ArtifactStore.

    Layout:
      <base_dir>/
        <uid>.artifact.joblib        # payload bytes
        <uid>.artifact.meta.json     # optional meta

    Notes
    -----
    - This is safe for single-process usage and script runs.
    - For multi-process / concurrent writers, use atomic writes (future) or a DB-backed store.
    """

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = (base_dir or default_artifacts_dir()).expanduser()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _payload_path(self, uid: str) -> Path:
        return self.base_dir / f"{uid}.artifact.joblib"

    def _meta_path(self, uid: str) -> Path:
        return self.base_dir / f"{uid}.artifact.meta.json"

    def save(
        self,
        uid: str,
        payload: bytes,
        *,
        meta: Optional[Dict[str, Any]] = None,
    ) -> StoredArtifact:
        payload_path = self._payload_path(uid)
        payload_path.write_bytes(payload)

        meta_path: Optional[Path] = None
        if meta is not None:
            meta_path = self._meta_path(uid)
            with meta_path.open("w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2, default=_json_default)

        return StoredArtifact(
            uid=uid,
            payload_path=str(payload_path),
            meta_path=str(meta_path) if meta_path is not None else None,
        )

    def load(self, uid: str) -> Tuple[bytes, Optional[Dict[str, Any]]]:
        payload_path = self._payload_path(uid)
        if not payload_path.exists():
            raise FileNotFoundError(f"Artifact payload not found for uid={uid}")

        payload = payload_path.read_bytes()

        meta_path = self._meta_path(uid)
        meta: Optional[Dict[str, Any]] = None
        if meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)

        return payload, meta

    def exists(self, uid: str) -> bool:
        return self._payload_path(uid).exists()

    def delete(self, uid: str) -> None:
        p = self._payload_path(uid)
        if p.exists():
            p.unlink()
        m = self._meta_path(uid)
        if m.exists():
            m.unlink()

    def list_uids(self) -> Iterable[str]:
        for p in self.base_dir.glob("*.artifact.joblib"):
            name = p.name
            uid = name.split(".artifact.joblib")[0]
            if uid:
                yield uid


def get_default_filesystem_store() -> FileSystemArtifactStore:
    return FileSystemArtifactStore()

