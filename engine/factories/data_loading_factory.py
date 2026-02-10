from __future__ import annotations

from engine.contracts.run_config import DataModel
from engine.components.interfaces import DataLoader
from engine.components.data_loaders.data_loaders import AutoLoader


def make_data_loader(cfg: DataModel) -> DataLoader:
    """Return the default auto-detecting data loader."""
    return AutoLoader(cfg)
