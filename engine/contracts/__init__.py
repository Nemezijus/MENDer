
"""Engine contracts.

This package provides a stable import path for configuration/result schemas.

Refactor note (Segment 1): these modules temporarily re-export the existing
schemas from ``shared_schemas``. New code should prefer importing contracts
via ``engine.contracts`` so later segments can migrate the actual
implementations without changing call sites.
"""

