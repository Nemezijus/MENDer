"""Prediction use-case package.

This package replaces the former single-file module ``engine.use_cases.prediction``.

Public API
----------
- :func:`apply_model_to_arrays`
"""

from __future__ import annotations

from engine.use_cases.prediction.api import apply_model_to_arrays

__all__ = ["apply_model_to_arrays"]
