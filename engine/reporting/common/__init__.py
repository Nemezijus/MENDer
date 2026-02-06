"""Shared helpers for reporting.

Reporting is the layer that produces *derived products* from core compute:
decoder outputs, diagnostics, JSON-friendly payload normalization, etc.

This subpackage must remain backend-independent.
"""

from .json_safety import dedupe_preserve_order, safe_float_list, safe_float_optional, safe_float_scalar

__all__ = [
    "safe_float_list",
    "safe_float_scalar",
    "safe_float_optional",
    "dedupe_preserve_order",
]
