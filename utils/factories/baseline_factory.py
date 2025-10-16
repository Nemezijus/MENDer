# utils/factories/baseline_factory.py
from __future__ import annotations
from utils.configs.configs import RunConfig
from utils.permutations.rng import RngManager
from utils.strategies.interfaces import BaselineRunner
from utils.strategies.baselines import LabelShuffleBaseline

def make_baseline(cfg: RunConfig, rngm: RngManager) -> BaselineRunner:
    return LabelShuffleBaseline(cfg=cfg, rngm=rngm)
