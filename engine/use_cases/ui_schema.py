from __future__ import annotations

"""UI schema bundle use-case.

This use-case exists to provide a stable, backend-friendly entry point for
retrieving JSON schemas, defaults, and enum option lists without hardcoding
inventories in the backend.
"""

from typing import Any, Dict

from engine.reporting.ui.schema_bundle import build_ui_schema_bundle


def get_ui_schema_bundle(*, schema_version: int = 1) -> Dict[str, Any]:
    """Return a consolidated UI schema bundle."""

    return build_ui_schema_bundle(schema_version=schema_version)
