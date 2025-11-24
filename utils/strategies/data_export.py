from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from utils.io.export.result_export import ExportResult, export_to_csv


class DataExporter:
    """
    Minimal interface for data export strategies.

    For now, only CSV is supported, but this abstraction allows plugging in
    different formats (MAT, NPY, etc.) later.
    """

    def export(
        self,
        data: Any,
        *,
        dest: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> ExportResult:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class CSVDataExporter(DataExporter):
    """
    Export strategy for CSV files.

    Uses the generic helper `export_to_csv` under the hood.
    """

    def export(
        self,
        data: Any,
        *,
        dest: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> ExportResult:
        return export_to_csv(data, dest=dest, filename=filename)
