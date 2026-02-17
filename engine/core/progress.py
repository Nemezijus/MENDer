from __future__ import annotations

"""Progress reporting primitives for the Business Layer.

The engine must remain runnable without any specific backend/UI.
Use-cases may optionally accept a progress callback to report long-running
operations (e.g., shuffle baselines, searches).

Backends/UIs can adapt their own progress registries to this protocol.
"""

from typing import Protocol, Optional


class ProgressCallback(Protocol):
    """A minimal progress reporting interface."""

    def init(self, *, total: int, label: Optional[str] = None) -> None:  # pragma: no cover
        ...

    def update(self, *, current: int, label: Optional[str] = None) -> None:  # pragma: no cover
        ...

    def finalize(self, *, label: Optional[str] = None) -> None:  # pragma: no cover
        ...
