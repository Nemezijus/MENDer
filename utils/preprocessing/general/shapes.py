from __future__ import annotations

"""Legacy shape utilities.

NOTE
----
The canonical shape/orientation utilities now live under :mod:`engine.core.shapes`.
This module is kept as a thin compatibility shim.
"""

from engine.core.shapes import ensure_xy_aligned

__all__ = ["ensure_xy_aligned"]
