"""I/O adapter package.

This package contains backend boundary code for:
  - resolving user-provided file paths safely (dev vs docker)
  - constructing Engine DataModel configs
  - loading arrays via the Engine public API

Keep this package free of business logic; it should remain an interface layer.
"""
