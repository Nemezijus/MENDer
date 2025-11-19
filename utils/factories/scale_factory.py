from __future__ import annotations
from shared_schemas.scale_configs import ScaleModel
from utils.strategies.interfaces import Scaler
from utils.strategies.scalers import PairScaler

def make_scaler(cfg: ScaleModel) -> Scaler:
    """
    Create a scaler strategy from config.
    Currently: single implementation that wraps your existing scaler.
    """
    return PairScaler(cfg=cfg)
