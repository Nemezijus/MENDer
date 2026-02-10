from __future__ import annotations

"""Model configuration contracts (facade).

Segment 7 split: the actual per-family config classes live in
`engine.contracts.model_families.*`. This module remains the stable import path
for the rest of the codebase.
"""

from .model_families import *  # noqa: F403
from .model_families import __all__  # noqa: F401
