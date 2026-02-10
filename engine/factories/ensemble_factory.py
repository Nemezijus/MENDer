from __future__ import annotations

from typing import Any

from engine.contracts.ensemble_run_config import EnsembleRunConfig

from engine.registries.ensembles import make_ensemble_strategy as _make_strategy

from engine.components.interfaces import EnsembleBuilder


def make_ensemble_strategy(cfg: EnsembleRunConfig) -> EnsembleBuilder:
    """Backwards-compatible wrapper around engine registries."""
    return _make_strategy(cfg)


def make_ensemble_estimator(cfg: EnsembleRunConfig, **kwargs: Any) -> Any:
    """Convenience helper: build an unfitted ensemble estimator."""
    return make_ensemble_strategy(cfg).make_estimator(**kwargs)
