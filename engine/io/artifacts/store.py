from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Protocol, Tuple


@dataclass(frozen=True)
class StoredArtifact:
    """A stable reference to a stored artifact."""

    uid: str
    payload_path: str
    meta_path: Optional[str] = None


class ArtifactStore(Protocol):
    """ArtifactStore abstracts where artifact bytes and metadata are stored.

    The canonical serialization format for the payload bytes is defined in
    :mod:`engine.io.artifacts.serialization`.

    Implementations must be deterministic and side-effect free beyond the
    storage operations themselves.
    """

    def save(
        self,
        uid: str,
        payload: bytes,
        *,
        meta: Optional[Dict[str, Any]] = None,
    ) -> StoredArtifact:
        """Persist artifact payload (and optional meta)."""

    def load(self, uid: str) -> Tuple[bytes, Optional[Dict[str, Any]]]:
        """Load artifact payload (and optional meta)."""

    def exists(self, uid: str) -> bool:
        """Return True if an artifact payload exists."""

    def delete(self, uid: str) -> None:
        """Delete payload and meta (if any)."""

    def list_uids(self) -> Iterable[str]:
        """List all artifact UIDs available in this store."""
