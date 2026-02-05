"""Prediction components (compute layer).

Public API (Segment 6):
- predict_labels
- predict_scores
- predict_decoder_outputs

Lower-level helpers are available under submodules.
"""

from .predicting import predict_labels, predict_scores, predict_decoder_outputs

__all__ = [
    "predict_labels",
    "predict_scores",
    "predict_decoder_outputs",
]
