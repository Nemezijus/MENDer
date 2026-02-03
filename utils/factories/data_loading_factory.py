from __future__ import annotations

from shared_schemas.run_config import DataModel
from utils.strategies.interfaces import DataLoader
from utils.strategies.data_loaders import AutoLoader


def make_data_loader(cfg: DataModel) -> DataLoader:
    """Return the default auto-detecting data loader."""
    return AutoLoader(cfg)
