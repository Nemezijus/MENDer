from __future__ import annotations

"""Prediction/decoder per-sample report builders.

This module is the stable public surface. Implementation is split across:

- :mod:`engine.reporting.prediction.prediction_table`
- :mod:`engine.reporting.prediction.decoder_table`
- :mod:`engine.reporting.prediction.coercion`
- :mod:`engine.reporting.prediction.merge`
"""

from .prediction_table import build_prediction_table
from .decoder_table import build_decoder_output_table
from .merge import merge_prediction_and_decoder_tables

__all__ = [
    "build_prediction_table",
    "build_decoder_output_table",
    "merge_prediction_and_decoder_tables",
]
