from __future__ import annotations
from engine.contracts.run_config import RunConfig
from engine.runtime.random.rng import RngManager
from engine.components.interfaces import BaselineRunner
from engine.use_cases.baselines.label_shuffle import make_label_shuffle_baseline

def make_baseline(cfg: RunConfig, rngm: RngManager) -> BaselineRunner:
    """Backward-compatible baseline factory.

    Note: baseline orchestration now lives in ``engine.use_cases.baselines``.
    This factory remains as a convenience wrapper for callers that still build
    baselines via factories.
    """

    return make_label_shuffle_baseline(cfg, rngm)
