"""Backend-local I/O adapter errors.

These errors represent boundary failures (bad/unsafe paths, missing files,
unsupported formats). They are mapped to HTTP responses by global exception
handlers in ``backend/app/main.py``.
"""


class LoadError(Exception):
    """Raised when the backend cannot resolve/load a requested dataset."""
