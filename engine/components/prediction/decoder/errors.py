"""Decoder-specific exceptions.

These are intentionally lightweight so they can be raised from compute paths
without importing reporting modules.
"""


class DecoderOutputsError(RuntimeError):
    """Raised when decoder outputs cannot be constructed."""


class DecoderOutputsConcatError(DecoderOutputsError):
    """Raised when fold parts cannot be concatenated into full arrays."""
