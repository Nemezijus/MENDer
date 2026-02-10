"""Built-in data exporter registrations."""

from __future__ import annotations

from engine.registries.exporters import register_export_format

from engine.components.exporters.data_export import CSVDataExporter


@register_export_format("csv")
def _csv():
    return CSVDataExporter()
