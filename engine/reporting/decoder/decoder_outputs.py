from __future__ import annotations

"""Compatibility shim for decoder extraction.

Segment 6 moves *compute* functionality out of `engine.reporting` and into
`engine.components`.

Canonical compute location:
    engine.components.prediction.decoder_extraction

Canonical end-to-end decoder API (returns result contracts):
    engine.components.prediction.decoder_api.predict_decoder_outputs

This module remains as a thin re-export to avoid import drift during the
refactor.
"""

from engine.components.prediction.decoder_extraction import (  # noqa: F401
    RawDecoderOutputs,
    RawDecoderOutputs as DecoderOutputs,
    compute_decoder_outputs_raw,
    compute_decoder_outputs,
)

__all__ = [
    "RawDecoderOutputs",
    "DecoderOutputs",
    "compute_decoder_outputs_raw",
    "compute_decoder_outputs",
]
