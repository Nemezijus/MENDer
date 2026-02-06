from __future__ import annotations

"""Excel .xlsx reader."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .base import LoadedArray, coerce_numeric_matrix


def load_xlsx_table(
    file_path: Union[str, Path],
    *,
    sheet_name: Optional[Union[str, int]] = None,
    has_header: Optional[bool] = None,
) -> LoadedArray:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"XLSX file not found: {path}")

    try:
        import pandas as pd  # type: ignore
    except Exception as e:
        raise RuntimeError("Loading XLSX requires pandas. Install it (pip install pandas).") from e

    hdr = 0 if (has_header is True) else None
    df = pd.read_excel(path.as_posix(), sheet_name=sheet_name or 0, header=hdr)

    feature_names: Optional[list[str]] = None
    if has_header is True:
        feature_names = [str(c) for c in df.columns.tolist()]

    df_num = df.apply(pd.to_numeric, errors="coerce")
    if df_num.isna().to_numpy().any():
        n_bad = int(df_num.isna().to_numpy().sum())
        raise ValueError(
            f"{path.name}: found {n_bad} non-numeric/missing cells after parsing. "
            "Clean the sheet or export as numeric-only values."
        )

    arr = coerce_numeric_matrix(df_num.to_numpy(), context=f"XLSX '{path.name}'")
    return LoadedArray(arr, feature_names)


@dataclass
class XlsxReader:
    sheet_name: Optional[Union[str, int]] = None
    has_header: Optional[bool] = None

    def read(self, path: Union[str, Path], **kwargs) -> LoadedArray:
        return load_xlsx_table(
            path,
            sheet_name=kwargs.get("sheet_name", self.sheet_name),
            has_header=kwargs.get("has_header", self.has_header),
        )
