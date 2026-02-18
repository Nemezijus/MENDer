from __future__ import annotations

"""Backend progress adapter for Engine progress callbacks.

The Engine (BL) defines a small :class:`engine.api.ProgressCallback`
protocol to allow optional progress reporting during long-running tasks.

The backend tracks progress records in :mod:`backend.app.progress.registry` and
exposes them via the /progress API for the frontend to poll.

This adapter bridges the two without introducing backend dependencies into the BL.
"""

from dataclasses import dataclass
from typing import Optional

from engine.api import ProgressCallback

from .registry import PROGRESS


@dataclass(frozen=True)
class RegistryProgressCallback(ProgressCallback):
    """Bind a progress_id to the backend registry and expose the BL protocol."""

    progress_id: str

    def init(self, total: int, label: str = "Starting…") -> None:  # type: ignore[override]
        PROGRESS.init(self.progress_id, total=int(total), label=label)

    def update(
        self,
        current: int,
        *,
        total: Optional[int] = None,
        label: Optional[str] = None,
    ) -> None:  # type: ignore[override]
        PROGRESS.update(self.progress_id, current=int(current), total=total, label=label)

    def finalize(self, label: str = "Done") -> None:  # type: ignore[override]
        PROGRESS.finalize(self.progress_id, label=label)

    def fail(self, message: str) -> None:  # type: ignore[override]
        PROGRESS.fail(self.progress_id, message=message)
