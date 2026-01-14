from __future__ import annotations

"""Decoder outputs for classification models.

This module is *business-logic only* (no backend/frontend dependencies).

It extracts per-sample decoder outputs from a fitted scikit-learn estimator
(often a Pipeline), namely:

  - hard predictions (predict)
  - decision scores (decision_function), when available
  - class probabilities (predict_proba), when available
  - classes ordering (estimator.classes_ when available)

It is designed for neural decoding / behavior classification use-cases where
you want per-trial values such as the decision axis score (w^T x + b) and/or
P(go).
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Optional, Sequence, Tuple

import numpy as np


def _as_2d(X: Any) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D; got shape {X.shape}.")
    return X


def _final_estimator(model: Any) -> Any:
    """Return the final estimator for a Pipeline-like model, else the model itself."""
    # sklearn.pipeline.Pipeline
    if hasattr(model, "steps") and isinstance(getattr(model, "steps"), list):
        try:
            return model.steps[-1][1]
        except Exception:
            return model
    return model


def _get_classes(model: Any) -> Optional[np.ndarray]:
    """Best-effort retrieval of classes ordering."""
    est = _final_estimator(model)
    classes = getattr(est, "classes_", None)
    if classes is None:
        return None
    return np.asarray(classes)


def _safe_call(method_name: str, model: Any, X: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Call model.<method_name>(X) if available; return (array, None) else (None, reason)."""
    if not hasattr(model, method_name):
        return None, f"model has no {method_name}"
    try:
        out = getattr(model, method_name)(X)
        return np.asarray(out), None
    except Exception as e:
        return None, f"{method_name} failed: {type(e).__name__}: {e}"


def _top1_top2_margin(values_2d: np.ndarray) -> np.ndarray:
    """Compute top1-top2 margin along axis=1 for 2D array."""
    # Use partition for efficiency and stability
    part = np.partition(values_2d, kth=-2, axis=1)
    top2 = part[:, -2]
    top1 = part[:, -1]
    return top1 - top2


def _binary_margin_from_proba(proba_2d: np.ndarray) -> np.ndarray:
    """Binary confidence margin from probabilities."""
    if proba_2d.shape[1] != 2:
        return _top1_top2_margin(proba_2d)
    return np.abs(proba_2d[:, 1] - proba_2d[:, 0])


@dataclass
class DecoderOutputs:
    """Per-sample decoder outputs extracted from a fitted classifier."""

    y_pred: np.ndarray
    classes: Optional[np.ndarray] = None

    # Raw decoder outputs
    decision_scores: Optional[np.ndarray] = None
    proba: Optional[np.ndarray] = None

    # Optional convenience fields
    positive_class_label: Optional[Any] = None
    positive_class_index: Optional[int] = None
    positive_proba: Optional[np.ndarray] = None
    positive_score: Optional[np.ndarray] = None

    # Confidence proxy
    margin: Optional[np.ndarray] = None

    # Diagnostics
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to a JSON-friendly dict (numpy arrays -> lists)."""
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()
        return d


def compute_decoder_outputs(
    model: Any,
    X: Any,
    *,
    positive_class_label: Optional[Any] = None,
    include_decision_scores: bool = True,
    include_probabilities: bool = True,
    calibrate_probabilities: bool = False,
) -> DecoderOutputs:
    """Compute per-sample decoder outputs for a fitted classifier.

    Parameters
    ----------
    model:
        Fitted scikit-learn estimator or Pipeline.
    X:
        Feature matrix, shape (n_samples, n_features).
    positive_class_label:
        For binary classification, which class label should be treated as the
        "positive" class (e.g. go=1). If provided and `proba`/`decision_scores`
        exist, `positive_proba` / `positive_score` are returned.
    include_decision_scores:
        Attempt to extract `decision_function(X)`.
    include_probabilities:
        Attempt to extract `predict_proba(X)`.
    calibrate_probabilities:
        Reserved for later (CalibratedClassifierCV). If True and predict_proba
        is missing, this currently records a note and returns `proba=None`.

    Returns
    -------
    DecoderOutputs
        Dataclass containing y_pred, optional decision_scores/proba, classes,
        and derived fields.
    """

    X2 = _as_2d(X)
    notes: list[str] = []

    # Predict hard labels
    y_pred, err = _safe_call("predict", model, X2)
    if err is not None or y_pred is None:
        raise AttributeError(f"Unable to compute predictions: {err}")

    classes = _get_classes(model)

    decision_scores: Optional[np.ndarray] = None
    proba: Optional[np.ndarray] = None

    if include_decision_scores:
        decision_scores, derr = _safe_call("decision_function", model, X2)
        if derr is not None:
            notes.append(derr)

    if include_probabilities:
        proba, perr = _safe_call("predict_proba", model, X2)
        if perr is not None:
            if calibrate_probabilities:
                notes.append(
                    "predict_proba unavailable; calibration requested but not implemented yet "
                    "(proba will be omitted)."
                )
            else:
                notes.append(perr)

    # Compute positive class mapping / convenience vectors
    pos_idx: Optional[int] = None
    pos_proba: Optional[np.ndarray] = None
    pos_score: Optional[np.ndarray] = None

    if positive_class_label is not None and classes is not None:
        matches = np.where(classes == positive_class_label)[0]
        if matches.size == 0:
            notes.append(
                f"positive_class_label={positive_class_label!r} not found in classes_={classes.tolist()}"
            )
        else:
            pos_idx = int(matches[0])

    if pos_idx is not None and proba is not None and proba.ndim == 2:
        if pos_idx < proba.shape[1]:
            pos_proba = proba[:, pos_idx]
        else:
            notes.append(
                f"positive_class_index={pos_idx} out of range for proba shape {proba.shape}"
            )

    if pos_idx is not None and decision_scores is not None:
        # decision_function binary often returns (n_samples,) where positive class is implicitly the second class
        if decision_scores.ndim == 1:
            pos_score = decision_scores
        elif decision_scores.ndim == 2:
            if pos_idx < decision_scores.shape[1]:
                pos_score = decision_scores[:, pos_idx]
            else:
                notes.append(
                    f"positive_class_index={pos_idx} out of range for decision_scores shape {decision_scores.shape}"
                )

    # Compute a confidence proxy margin
    margin: Optional[np.ndarray] = None
    if decision_scores is not None:
        if decision_scores.ndim == 1:
            margin = np.abs(decision_scores)
        elif decision_scores.ndim == 2 and decision_scores.shape[1] >= 2:
            margin = _top1_top2_margin(decision_scores)
    elif proba is not None and proba.ndim == 2 and proba.shape[1] >= 2:
        margin = _binary_margin_from_proba(proba)

    return DecoderOutputs(
        y_pred=np.asarray(y_pred),
        classes=classes,
        decision_scores=decision_scores,
        proba=proba,
        positive_class_label=positive_class_label,
        positive_class_index=pos_idx,
        positive_proba=pos_proba,
        positive_score=pos_score,
        margin=margin,
        notes=notes,
    )
