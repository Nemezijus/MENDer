from __future__ import annotations

"""Auto-dispatching reader + DataModel convenience loader.

Public entrypoint for:
- extension-based dispatch (:func:`read_array_auto`)
- convenience loading from :class:`engine.contracts.run_config.DataModel` (:func:`load_from_data_model`)

Implementation details live in :mod:`engine.io.readers.auto`.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Union

from .base import LoadedArray
from .auto.dispatch import read_array_auto
from .auto.data_model import load_from_data_model


@dataclass
class AutoReader:
    """Stateful wrapper mainly for dependency injection."""

    def read(self, path: Union[str, Path], **kwargs) -> LoadedArray:
        return read_array_auto(path, **kwargs)


__all__ = ["AutoReader", "read_array_auto", "load_from_data_model"]
