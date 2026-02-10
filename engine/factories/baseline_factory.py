from __future__ import annotations
from engine.contracts.run_config import RunConfig
from engine.runtime.random.rng import RngManager
from engine.components.interfaces import BaselineRunner
from engine.components.baselines.baselines import LabelShuffleBaseline

def make_baseline(cfg: RunConfig, rngm: RngManager) -> BaselineRunner:
    return LabelShuffleBaseline(cfg=cfg, rngm=rngm)
