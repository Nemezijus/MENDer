"""Training-time reporting helpers.

This subpackage hosts small helpers that translate core evaluation outputs
into JSON-friendly payloads and/or typed result contracts.

The backend should not own these normalizations; use-cases and scripts
should be able to build the same outputs without importing backend code.
"""

from .decoder_payloads import build_decoder_outputs_payload
from .metrics_payloads import normalize_confusion, normalize_roc
from .regression_payloads import (
    build_regression_decoder_outputs_payload,
    build_regression_diagnostics_payload,
)

__all__ = [
    "build_decoder_outputs_payload",
    "normalize_confusion",
    "normalize_roc",
    "build_regression_diagnostics_payload",
    "build_regression_decoder_outputs_payload",
]
