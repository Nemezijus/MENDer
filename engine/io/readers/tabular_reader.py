from __future__ import annotations

"""Delimited text table reader (CSV/TSV/TXT)."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import numpy as np

from .base import LoadedArray, coerce_numeric_matrix


def _read_text_head(path: Path, n_lines: int = 5, encoding: Optional[str] = None) -> list[str]:
    enc = encoding or "utf-8"
    lines: list[str] = []
    with path.open("r", encoding=enc, errors="replace") as f:
        for _ in range(n_lines):
            line = f.readline()
            if not line:
                break
            lines.append(line.strip("\n"))
    return lines


def _infer_delimiter(sample_line: str) -> str:
    if "\t" in sample_line:
        return "\t"
    if "," in sample_line:
        return ","
    if ";" in sample_line:
        return ";"
    return "whitespace"


def _split_line(line: str, delimiter: str) -> list[str]:
    if delimiter == "whitespace":
        return [t for t in line.strip().split() if t != ""]
    return [t.strip() for t in line.split(delimiter)]


def _row_is_numeric(tokens: Sequence[str]) -> bool:
    if len(tokens) == 0:
        return False
    try:
        for t in tokens:
            if t == "":
                return False
            float(t)
        return True
    except Exception:
        return False


def load_delimited_table(
    file_path: Union[str, Path],
    *,
    delimiter: Optional[str] = None,
    has_header: Optional[bool] = None,
    encoding: Optional[str] = None,
) -> LoadedArray:
    """Load a numeric table from CSV/TSV/TXT.

    - If has_header is None, header is inferred by checking whether the first row is numeric.
    - If delimiter is None, it is inferred from the first line.
    """

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Table file not found: {path}")

    head_lines = _read_text_head(path, n_lines=2, encoding=encoding)
    first = head_lines[0] if head_lines else ""

    delim = delimiter
    if delim is None:
        delim = _infer_delimiter(first)
    if delim == "\\t":
        delim = "\t"

    tokens = _split_line(first, delim)
    inferred_header = not _row_is_numeric(tokens)
    use_header = inferred_header if has_header is None else bool(has_header)

    try:
        import pandas as pd  # type: ignore
    except Exception as e:
        raise RuntimeError("Loading CSV/TSV/TXT requires pandas. Install it (pip install pandas).") from e

    read_kwargs = dict(
        encoding=encoding or "utf-8",
        engine="python",
    )

    if delim == "whitespace":
        sep = r"\s+"
    else:
        sep = delim

    header_arg = 0 if use_header else None
    df = pd.read_csv(path.as_posix(), sep=sep, header=header_arg, **read_kwargs)

    feature_names: Optional[list[str]] = None
    if use_header:
        feature_names = [str(c) for c in df.columns.tolist()]

    df_num = df.apply(pd.to_numeric, errors="coerce")
    if df_num.isna().to_numpy().any():
        n_bad = int(df_num.isna().to_numpy().sum())
        raise ValueError(
            f"{path.name}: found {n_bad} non-numeric/missing cells after parsing. "
            "Clean the file or export as purely numeric values."
        )

    arr = coerce_numeric_matrix(df_num.to_numpy(), context=f"Table '{path.name}'")
    return LoadedArray(arr, feature_names)


@dataclass
class TabularReader:
    delimiter: Optional[str] = None
    has_header: Optional[bool] = None
    encoding: Optional[str] = None

    def read(self, path: Union[str, Path], **kwargs) -> LoadedArray:
        return load_delimited_table(
            path,
            delimiter=kwargs.get("delimiter", self.delimiter),
            has_header=kwargs.get("has_header", self.has_header),
            encoding=kwargs.get("encoding", self.encoding),
        )
