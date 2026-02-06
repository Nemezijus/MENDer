from __future__ import annotations

"""Compatibility shim.

Allen Institute fetchers have moved to :mod:`engine.extras.allen_institute.datafetch`.
This module remains to preserve legacy import paths.
"""

from engine.extras.allen_institute.datafetch.visual_behavior import *  # noqa: F401,F403
