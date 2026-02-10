from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, TypedDict, Union
from uuid import uuid4

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - numpy may not be installed in all environments
    np = None  # type: ignore[assignment]


PathLike = Union[str, Path]


class ExportResult(TypedDict):
    """
    Generic description of an exported file.

    - content:  binary payload (e.g. for StreamingResponse)
    - filename: suggested filename (with extension)
    - size:     length of content in bytes
    - mime_type: MIME type string (for HTTP headers); csv by default
    - path:     optional filesystem path if the export was written to disk
    """
    content: bytes
    filename: str
    size: int
    mime_type: str
    path: Optional[Path]


def _ensure_extension(name: str, ext: str) -> str:
    ext = ext.lstrip(".").lower()
    if not name.lower().endswith("." + ext):
        return f"{name}.{ext}"
    return name


def _make_filename(filename: Optional[str], ext: str) -> str:
    """
    Decide on a filename.

    - If `filename` is provided → ensure it has the given extension.
    - Otherwise → generate a UUID-based filename with that extension.
    """
    if filename:
        return _ensure_extension(filename, ext)
    return f"{uuid4().hex}.{ext.lstrip('.')}"


def _normalize_dest_path(dest: Optional[PathLike], filename: str) -> Optional[Path]:
    """
    Normalize destination path.

    - If dest is None: no on-disk write, caller just gets bytes.
    - If dest is a directory: append filename.
    - If dest is a file path: use that path as-is (ignore filename).
    """
    if dest is None:
        return None

    p = Path(dest)
    if p.is_dir() or (not p.exists() and str(p).endswith(("/", "\\"))):
        return p / filename
    return p


def _to_rows_and_header(
    data: Any,
) -> tuple[Optional[Sequence[str]], Iterable[Sequence[Any]]]:
    """
    Convert `data` into (header, rows) suitable for csv.writer.

    Supported shapes:

    1. pandas.DataFrame → header = df.columns, rows = df.itertuples(index=False)
    2. Sequence[Mapping] (e.g. list of dicts) → header = union of keys, rows = values per header
    3. 2D numpy array → header = None, rows = array rows
    4. Sequence[Sequence] → header = None, rows = sequences

    If nothing matches, we try best-effort: treat as single row.

    """
    # 1) Pandas DataFrame (duck-typed, no hard dependency)
    if hasattr(data, "to_csv") and hasattr(data, "columns"):
        # Assuming it's a pandas-like DataFrame
        try:
            cols = list(data.columns)  # type: ignore[attr-defined]

            def gen_rows():
                for row in data.itertuples(index=False):  # type: ignore[attr-defined]
                    yield list(row)

            return [str(c) for c in cols], gen_rows()
        except Exception:
            # Fall through to other strategies if DataFrame-like path fails
            pass

    # 2) Sequence of mappings (e.g. list of dicts)
    if isinstance(data, Sequence) and data and isinstance(data[0], Mapping):  # type: ignore[index]
        # Collect headers from union of keys (keep deterministic order)
        headers: list[str] = []
        for row in data:  # type: ignore[assignment]
            for key in row.keys():
                s = str(key)
                if s not in headers:
                    headers.append(s)

        def gen_rows_from_dicts():
            for row in data:  # type: ignore[assignment]
                yield [row.get(h, "") for h in headers]

        return headers, gen_rows_from_dicts()

    # 3) 2D numpy array
    if np is not None and isinstance(data, np.ndarray):
        arr = data
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim == 2:
            return None, (arr[i, :].tolist() for i in range(arr.shape[0]))

    # 4) Generic sequence-of-sequences
    if isinstance(data, Sequence) and data and isinstance(data[0], Sequence):  # type: ignore[index]
        return None, (list(row) for row in data)  # type: ignore[assignment]

    # Fallback: treat as a single row
    return None, ([data],)


def export_to_csv(
    data: Any,
    *,
    dest: Optional[PathLike] = None,
    filename: Optional[str] = None,
    encoding: str = "utf-8",
) -> ExportResult:
    """
    Export arbitrary tabular-like `data` to CSV.

    Parameters
    ----------
    data:
        Data to export. Supported shapes are described in `_to_rows_and_header`.
        In particular, this accepts the list-of-dicts produced by
        `engine.reporting.prediction.prediction_results.build_prediction_table`.
    dest:
        Optional destination path or directory.

        - None: do not write to disk; just return `ExportResult` with bytes.
        - Directory path: write file into that directory using `filename` (or a generated one).
        - File path: write exactly to that path (overwriting if exists).
    filename:
        Optional base filename. If omitted, a UUID-based name is generated.
        The `.csv` extension is enforced.

    Returns
    -------
    ExportResult:
        Contains file content (bytes), filename, size, MIME type and optional path.
    """
    header, rows = _to_rows_and_header(data)

    buf = io.StringIO()
    writer = csv.writer(buf)
    if header is not None:
        writer.writerow(header)
    for row in rows:
        writer.writerow(row)

    text = buf.getvalue()
    content = text.encode(encoding)
    size = len(content)
    final_name = _make_filename(filename, "csv")
    dest_path = _normalize_dest_path(dest, final_name)

    if dest_path is not None:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "wb") as f:
            f.write(content)

    return ExportResult(
        content=content,
        filename=final_name,
        size=size,
        mime_type="text/csv",
        path=dest_path,
    )
