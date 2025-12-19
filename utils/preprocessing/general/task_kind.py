from __future__ import annotations

from typing import Literal, Sequence, cast

import numpy as np

from shared_schemas.model_configs import ModelConfig, get_model_task


EvalKind = Literal["classification", "regression"]


def infer_kind_from_y(y: np.ndarray) -> EvalKind:
    """Infer task kind from target values.

    Heuristic rules:
      - Non-numeric / object-like -> classification
      - Small number of unique values -> classification
      - Otherwise -> regression

    This is intended as a *sanity check* for script-mode usage.
    In MENDer, the authoritative task type should come from model configs.
    """
    y = np.asarray(y).ravel()
    if y.size == 0:
        return "classification"

    # Non-numeric targets imply classification
    if y.dtype.kind in ("U", "S", "O", "b"):
        return "classification"

    uniq = np.unique(y)

    # Many classification datasets use integer-coded classes.
    # A small number of unique values is a strong signal.
    if uniq.size <= 20:
        return "classification"

    return "regression"


def ensure_uniform_model_task(models: Sequence[ModelConfig]) -> EvalKind:
    """Ensure all model configs agree on task type.

    Returns:
      "classification" or "regression"

    Raises:
      ValueError if mixed task types are present.
    """
    tasks = [get_model_task(m) for m in models]
    unique = sorted(set(tasks))
    if len(unique) != 1:
        raise ValueError(
            "Cannot mix classification and regression estimators in the same ensemble. "
            f"Found tasks: {unique}"
        )

    return cast(EvalKind, "regression" if unique[0] == "regression" else "classification")


__all__ = ["EvalKind", "infer_kind_from_y", "ensure_uniform_model_task"]
