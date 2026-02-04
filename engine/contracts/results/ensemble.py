"""Compatibility module for ensemble result contracts.

Segment 4 introduced result contracts under ``engine.contracts.results``.
The backend patch imports ``engine.contracts.results.ensemble.EnsembleResult``.

For now, ``EnsembleResult`` is defined in ``training.py`` alongside other
supervised evaluation result contracts; this module re-exports it to provide a
stable import path.
"""

from .training import EnsembleResult

__all__ = ["EnsembleResult"]
