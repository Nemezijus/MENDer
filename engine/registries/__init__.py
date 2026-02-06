"""Engine registries.

These registries replace factory if/else sprawl. The core idea is:
- add a new implementation
- register it
- the rest of the system stays closed for modification

During refactor, some implementations still live under utils, but the
registry API surface is stable and will remain in engine.
"""

from .models import make_model_builder, register_model_builder, list_model_algos
from .features import make_feature_extractor, register_feature_extractor, list_feature_methods
from .splitters import make_splitter, register_splitter, list_split_modes
from .ensembles import make_ensemble_strategy, register_ensemble_kind, list_ensemble_kinds
from .exporters import make_exporter, register_export_format, list_export_formats
from .metrics import get_scorer, list_metrics, is_proba_metric
