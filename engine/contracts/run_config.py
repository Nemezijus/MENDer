"""Temporary contract shim.

Segment 1 of the BL refactor: we expose contracts under ``engine.contracts``
while the underlying implementations still live in ``shared_schemas``.

Later segments will migrate contract definitions into ``engine/contracts``
directly and this shim will be removed.
"""

from shared_schemas.run_config import *  # noqa: F401,F403
