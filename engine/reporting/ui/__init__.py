"""UI-facing schema and defaults bundles.

These helpers generate JSON-serializable payloads intended for the frontend
(schemas, defaults, enums). They are part of the Engine reporting layer and
must remain backend/HTTP agnostic.
"""

from .schema_bundle import build_ui_schema_bundle

__all__ = ["build_ui_schema_bundle"]
