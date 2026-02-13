"""Decoder outputs public API.

This subpackage is the canonical home for decoder-output generation.
"""

from .api import (
    build_decoder_outputs_from_arrays,
    build_decoder_outputs_from_parts,
    predict_decoder_outputs,
)

__all__ = [
    "predict_decoder_outputs",
    "build_decoder_outputs_from_arrays",
    "build_decoder_outputs_from_parts",
]
