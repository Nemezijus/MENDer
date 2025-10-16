from __future__ import annotations
from utils.configs.configs import DataConfig
from utils.strategies.interfaces import DataLoader
from utils.strategies.data_loaders import AutoLoader, NPZLoader, MatPairLoader

def make_data_loader(cfg: DataConfig) -> DataLoader:
    # You can get fancier (explicit type flag) later; auto is fine for now.
    return AutoLoader(cfg)