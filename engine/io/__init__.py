"""I/O utilities (Business Layer).

This package is intended to be backend-independent.

Subpackages
-----------
- :mod:`engine.io.artifacts`: persistence for trained models and derived outputs
- :mod:`engine.io.readers`: parsing adapters for input data formats
"""

from .artifacts import *  # noqa: F401,F403
from .readers import *  # noqa: F401,F403
