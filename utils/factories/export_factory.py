from __future__ import annotations

from typing import Literal

from engine.registries.exporters import make_exporter as _make_exporter

from utils.strategies.data_export import DataExporter


def make_exporter(
    fmt: Literal["csv"] = "csv",
) -> DataExporter:
    """Backwards-compatible wrapper around engine registries."""
    return _make_exporter(fmt)
