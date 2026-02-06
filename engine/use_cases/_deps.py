"""Dependency helpers for Engine use-cases.

Segment 12 introduces a BL façade under :mod:`engine.use_cases`.
Use-cases should avoid any backend-confirmed globals and instead accept
dependencies (store, RNG/seed) explicitly.

This module keeps *small* helpers only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from engine.io.artifacts.filesystem_store import FileSystemArtifactStore
from engine.io.artifacts.store import ArtifactStore


def default_store() -> ArtifactStore:
    """Return the default ArtifactStore.

    Uses :class:`~engine.io.artifacts.filesystem_store.FileSystemArtifactStore`
    and respects the ``MENDER_ARTIFACTS_DIR`` environment variable.
    """

    return FileSystemArtifactStore()


def resolve_store(store: Optional[ArtifactStore]) -> ArtifactStore:
    """Return ``store`` if provided, otherwise :func:`default_store`."""

    return store if store is not None else default_store()


def resolve_seed(seed: Optional[int], *, fallback: int = 0) -> int:
    """Return a deterministic seed.

    Many configs have an optional seed; when absent we still want repeatable
    behavior, hence a stable fallback.
    """

    return int(seed) if seed is not None else int(fallback)


@dataclass(frozen=True)
class UseCaseDeps:
    """A small bundle of resolved dependencies for internal orchestration."""

    store: ArtifactStore
    seed: int
