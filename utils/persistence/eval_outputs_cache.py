"""Compatibility shim.

The canonical implementation moved to :mod:`engine.runtime.caches.eval_outputs_cache`.
"""

from engine.runtime.caches.eval_outputs_cache import (
    EvalOutputs,
    EvalOutputsCache,
    eval_outputs_cache,
)

__all__ = [
    "EvalOutputs",
    "EvalOutputsCache",
    "eval_outputs_cache",
]
