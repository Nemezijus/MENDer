"""Compatibility re-export for engine contract choice types.

Refactor note (Segment 2): ``shared_schemas/choices.py`` is the canonical home
for Literal-based choice sets during the transition. This module re-exports
those names under the stable ``engine.contracts`` namespace.
"""

from shared_schemas.choices import *  # noqa: F401,F403
