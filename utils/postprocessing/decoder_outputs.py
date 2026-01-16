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

Special handling:
  - VotingClassifier hard voting (or any estimator lacking predict_proba):
      We *do not* attempt to invent calibrated probabilities. Instead, for
      VotingClassifier with voting="hard", we compute vote-share "probabilities"
      (fraction of estimators voting for each class). These can be used to
      compute margins and other confidence diagnostics.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Tuple

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


def _transform_through_pipeline(model: Any, X: np.ndarray) -> np.ndarray:
    """If model is a fitted sklearn Pipeline, transform X through all steps except last estimator."""
    try:
        if not (hasattr(model, "steps") and isinstance(getattr(model, "steps"), list) and len(model.steps) > 1):
            return X
        Xt = X
        for _, step in model.steps[:-1]:
            if step is None or step == "passthrough":
                continue
            if hasattr(step, "transform"):
                Xt = step.transform(Xt)
            else:
                return X
        return Xt
    except Exception:
        return X


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
    part = np.partition(values_2d, kth=-2, axis=1)
    top2 = part[:, -2]
    top1 = part[:, -1]
    return top1 - top2


def _binary_margin_from_proba(proba_2d: np.ndarray) -> np.ndarray:
    """Binary confidence margin from probabilities (or vote shares)."""
    if proba_2d.shape[1] != 2:
        return _top1_top2_margin(proba_2d)
    return np.abs(proba_2d[:, 1] - proba_2d[:, 0])


def _is_voting_classifier(est: Any) -> bool:
    """Duck-typed check for sklearn VotingClassifier-like object."""
    return (
        hasattr(est, "estimators_")
        and hasattr(est, "voting")
        and hasattr(est, "le_")  # LabelEncoder used internally by VotingClassifier
    )


def _vote_share_proba_from_voting_classifier(vote_clf: Any, X_for_vote: np.ndarray, classes: Optional[np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Compute vote-share probabilities for a fitted VotingClassifier (hard voting).

    Returns (proba, note). proba shape: (n_samples, n_classes). Values sum to 1.
    """
    try:
        if getattr(vote_clf, "voting", None) != "hard":
            return None, None

        ests = getattr(vote_clf, "estimators_", None)
        if ests is None or len(ests) == 0:
            return None, "VotingClassifier has no fitted estimators_"

        cls = classes
        if cls is None:
            cls = getattr(vote_clf, "classes_", None)
            cls = np.asarray(cls) if cls is not None else None
        if cls is None or cls.size == 0:
            return None, "VotingClassifier vote-share unavailable: classes_ missing"

        n_samples = int(np.asarray(X_for_vote).shape[0])
        k = int(cls.size)

        # Determine weights
        w = getattr(vote_clf, "weights", None)
        if w is None:
            weights = np.ones((len(ests),), dtype=float)
        else:
            weights = np.asarray(w, dtype=float)
            if weights.size != len(ests):
                # sklearn tolerates None for dropped estimators; handle mismatch defensively
                weights = np.ones((len(ests),), dtype=float)

        # Collect base predictions (VotingClassifier trains base estimators on encoded labels)
        le = getattr(vote_clf, "le_", None)
        preds = []
        for est in ests:
            if est is None:
                continue
            yp_enc = np.asarray(est.predict(X_for_vote))
            # Convert encoded preds to original labels if possible
            if le is not None:
                try:
                    yp = le.inverse_transform(yp_enc.astype(int, copy=False))
                except Exception:
                    yp = yp_enc
            else:
                yp = yp_enc
            preds.append(np.asarray(yp))

        if len(preds) == 0:
            return None, "VotingClassifier vote-share unavailable: no base predictions"

        P = np.vstack([p.reshape(1, -1) for p in preds])  # (n_estimators_eff, n_samples)

        # Map class label -> column
        cls_list = cls.tolist()
        idx_map = {c: i for i, c in enumerate(cls_list)}

        vote_mass = np.zeros((n_samples, k), dtype=float)
        # Align weights to the kept estimators (we dropped None)
        kept_weights = weights[: P.shape[0]] if weights.size >= P.shape[0] else np.ones((P.shape[0],), dtype=float)
        wsum = float(np.sum(kept_weights)) if float(np.sum(kept_weights)) > 0 else float(P.shape[0])

        for ei in range(P.shape[0]):
            wi = float(kept_weights[ei])
            for si in range(n_samples):
                lab = P[ei, si]
                if lab in idx_map:
                    vote_mass[si, idx_map[lab]] += wi

        proba = vote_mass / wsum
        return proba, "predict_proba unavailable; using vote shares from hard voting (not calibrated probabilities)"
    except Exception as e:
        return None, f"vote-share proba failed: {type(e).__name__}: {e}"


@dataclass
class DecoderOutputs:
    """Per-sample decoder outputs extracted from a fitted classifier."""

    y_pred: np.ndarray
    classes: Optional[np.ndarray] = None

    # Raw decoder outputs
    decision_scores: Optional[np.ndarray] = None
    proba: Optional[np.ndarray] = None

    # How proba was obtained: "model_proba" (predict_proba) or "vote_share" (hard voting)
    proba_source: Optional[str] = None

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

    Notes on VotingClassifier hard voting:
      - If predict_proba is missing/unavailable, we compute vote-share "probabilities"
        for VotingClassifier with voting="hard". These are useful confidence proxies,
        but are not calibrated probabilities.
    """

    X2 = _as_2d(X)
    notes: list[str] = []

    # Predict hard labels (from the full model/pipeline)
    y_pred, err = _safe_call("predict", model, X2)
    if err is not None or y_pred is None:
        raise AttributeError(f"Unable to compute predictions: {err}")

    classes = _get_classes(model)

    decision_scores: Optional[np.ndarray] = None
    proba: Optional[np.ndarray] = None
    proba_source: Optional[str] = None

    if include_decision_scores:
        decision_scores, derr = _safe_call("decision_function", model, X2)
        if derr is not None:
            notes.append(derr)

    perr_msg: Optional[str] = None
    if include_probabilities:
        proba, perr_msg = _safe_call("predict_proba", model, X2)
        if proba is not None:
            proba_source = "model_proba"

    # VotingClassifier hard-voting fallback for "proba"
    if include_probabilities and proba is None:
        est = _final_estimator(model)
        if _is_voting_classifier(est):
            # VoteClassifier expects X in its feature space (after preprocessing). If the outer model
            # is a Pipeline, transform X through non-final steps.
            X_vote = _transform_through_pipeline(model, X2) if est is not model else X2
            vote_proba, vote_note = _vote_share_proba_from_voting_classifier(est, X_vote, classes)
            if vote_proba is not None:
                proba = np.asarray(vote_proba)
                proba_source = "vote_share"
                if vote_note:
                    notes.append(vote_note)
            else:
                # If vote-share could not be computed, keep the original predict_proba note.
                if vote_note:
                    notes.append(vote_note)

        # If still no proba, fall back to original error message / calibration note.
        if proba is None:
            if perr_msg is not None:
                if calibrate_probabilities:
                    notes.append(
                        "predict_proba unavailable; calibration requested but not implemented yet "
                        "(proba will be omitted)."
                    )
                else:
                    notes.append(perr_msg)

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
        proba_source=proba_source,
        positive_class_label=positive_class_label,
        positive_class_index=pos_idx,
        positive_proba=pos_proba,
        positive_score=pos_score,
        margin=margin,
        notes=notes,
    )
