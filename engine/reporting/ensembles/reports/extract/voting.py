from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


def collect_voting_base_preds_and_scores(
    *,
    voting_estimator: Any,
    X: Any,
    y_true: Any,
    evaluator: Any,
    is_classification: bool,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Collect base predictions and per-estimator scores for Voting ensembles."""

    base_preds: Dict[str, np.ndarray] = {}
    base_scores: Dict[str, float] = {}

    if is_classification:
        y_true_arr = np.asarray(y_true)
        y_true_enc = y_true_arr

        # VotingClassifier label-encodes y during fit().
        le = getattr(voting_estimator, "le_", None)
        if le is not None:
            try:
                y_true_enc = le.transform(y_true_arr)
            except Exception:
                y_true_enc = y_true_arr

        est_pairs = list(zip(getattr(voting_estimator, "estimators", []), getattr(voting_estimator, "estimators_", [])))
        for (name, _unfitted), fitted in est_pairs:
            yp_enc = fitted.predict(X)
            yp_report = yp_enc
            if le is not None:
                try:
                    yp_report = le.inverse_transform(np.asarray(yp_enc))
                except Exception:
                    yp_report = yp_enc

            base_preds[name] = np.asarray(yp_report)

            y_proba_i = None
            y_score_i = None
            if hasattr(fitted, "predict_proba"):
                try:
                    y_proba_i = fitted.predict_proba(X)
                except Exception:
                    y_proba_i = None
            if y_proba_i is None and hasattr(fitted, "decision_function"):
                try:
                    y_score_i = fitted.decision_function(X)
                except Exception:
                    y_score_i = None

            base_scores[name] = float(
                evaluator.score(
                    y_true_enc,
                    y_pred=yp_enc,
                    y_proba=y_proba_i,
                    y_score=y_score_i,
                )
            )

        return base_preds, base_scores

    # Regression
    y_true_arr = np.asarray(y_true)
    est_pairs = list(zip(getattr(voting_estimator, "estimators", []), getattr(voting_estimator, "estimators_", [])))
    for (name, _unfitted), fitted in est_pairs:
        yp = np.asarray(fitted.predict(X))
        base_preds[name] = yp
        base_scores[name] = float(evaluator.score(y_true_arr, y_pred=yp, y_proba=None, y_score=None))

    return base_preds, base_scores
