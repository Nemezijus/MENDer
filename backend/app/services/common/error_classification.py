"""Backend-local helpers for classifying exceptions.

The backend is an HTTP boundary layer and should avoid embedding business logic.
However, we still need small boundary-specific utilities, such as mapping common
data-loading failures (file missing, wrong key, unreadable format, etc.) to a
backend ``LoadError`` that the global exception handlers can translate into a
400 response.
"""

from __future__ import annotations


def is_probable_load_error(exc: Exception) -> bool:
    """Best-effort classification of "data load" failures.

    The Engine BL does not depend on backend exception types, so data-loading
    errors (FileNotFoundError, missing keys, etc.) can otherwise surface as
    generic 500s. This heuristic keeps backward-compatible 400 responses for
    common load failures while not masking genuine training errors.
    """

    if isinstance(exc, (FileNotFoundError, OSError, IOError)):
        return True

    msg = str(exc).lower()
    needles = (
        "npz",
        "npy",
        "mat",
        "h5",
        "hdf5",
        "csv",
        "file",
        "path",
        "no such file",
        "not found",
        "missing",
        "x_key",
        "y_key",
        "load",
        "dataset",
        "could not read",
        "cannot read",
    )
    return any(n in msg for n in needles)
