"""Auto-loading helpers.

- dispatch: choose format reader by extension
- data_model: interpret DataModel (X/y paths & keys) and load arrays

"""

from .dispatch import read_array_auto
from .data_model import load_from_data_model

__all__ = ["read_array_auto", "load_from_data_model"]
