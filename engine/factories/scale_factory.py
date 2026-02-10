from __future__ import annotations
from engine.contracts.scale_configs import ScaleModel
from engine.components.interfaces import Scaler
from engine.components.scalers.scalers import PairScaler

def make_scaler(cfg: ScaleModel) -> Scaler:
    """
    Create a scaler strategy from config.
    Currently: single implementation that wraps your existing scaler.
    """
    return PairScaler(cfg=cfg)
