from __future__ import annotations

"""Legacy alias module.

Historically, MENDer stored most Literal-based choice types in
``shared_schemas/types.py``. During the BL refactor, those choice sets are
centralized in ``shared_schemas/choices.py``.

This module is intentionally kept as a thin compatibility layer to avoid
churn in existing import sites. New code should prefer:

    from shared_schemas.choices import ...
"""

from .choices import *  # noqa: F401,F403
from .choices import __all__ as __all__
