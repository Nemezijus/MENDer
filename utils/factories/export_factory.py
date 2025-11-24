from __future__ import annotations

from typing import Literal

from utils.strategies.data_export import CSVDataExporter, DataExporter


def make_exporter(
    fmt: Literal["csv"] = "csv",
) -> DataExporter:
    """
    Factory for data export strategies.

    Parameters
    ----------
    fmt:
        Output format. Currently only "csv" is supported.
        In the future, this can be extended to "mat", "npy", etc.

    Returns
    -------
    DataExporter
        An instance of the corresponding export strategy.
    """
    fmt_norm = fmt.lower()
    if fmt_norm == "csv":
        return CSVDataExporter()
    raise ValueError(f"Unsupported export format: {fmt}")
