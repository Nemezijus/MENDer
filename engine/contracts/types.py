from __future__ import annotations

"""Legacy alias module.


This module is intentionally kept as a thin compatibility layer to avoid
churn in existing import sites. New code should prefer:

    from engine.contracts.choices import ...
"""

from .choices import *  # noqa: F401,F403
from .choices import __all__ as __all__
