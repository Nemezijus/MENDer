from __future__ import annotations
from utils.configs.configs import ScaleConfig
from utils.strategies.interfaces import Scaler
from utils.strategies.scalers import PairScaler

def make_scaler(cfg: ScaleConfig) -> Scaler:
    """
    Create a scaler strategy from config.
    Currently: single implementation that wraps your existing scaler.
    """
    return PairScaler(cfg=cfg)
