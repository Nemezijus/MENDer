from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Dict

import numpy as np

from engine.contracts.metrics_configs import MetricsModel
from engine.components.interfaces import MetricsComputer
from engine.components.evaluation.types import MetricsPayload
from engine.components.evaluation.scoring import (
    confusion_matrix_metrics,
    binary_roc_curve_from_scores,
    multiclass_roc_curves_from_scores,
)


@dataclass
class SklearnMetrics(MetricsComputer):
    """
    Strategy for computing structured evaluation metrics (confusion-matrix-based
    metrics and ROC curves) from true labels and predictions.

    This is intentionally focused on *post*-training metrics. It does not fit
    or access models; it only consumes arrays produced elsewhere.

    Configuration is provided via `MetricsModel` (kind / which metrics to
    compute) so it plugs into the same config/factory pattern as your other
    strategies.
    """
    cfg: MetricsModel

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        *,
        y_proba: Optional[np.ndarray] = None,
        y_score: Optional[np.ndarray] = None,
        labels: Optional[Sequence] = None,
    ) -> MetricsPayload:
        """
        Parameters
        ----------
        y_true : array-like, shape (n_samples,)
            Ground-truth labels.
        y_pred : array-like, shape (n_samples,)
            Predicted hard labels.
        y_proba : array-like, optional
            Class probabilities, shape (n_samples, n_classes) or (n_samples,)
            for the positive class in binary tasks.
        y_score : array-like, optional
            Continuous scores / decision function values, used if `y_proba`
            is not provided.
        labels : sequence, optional
            Explicit label ordering to use for confusion matrices and to align
            ROC curves (especially in multi-class).

        Returns
        -------
        dict
            {
                "confusion": dict | None,
                "roc": dict | None,
            }
        """
        y_true_arr = np.asarray(y_true).ravel()
        y_pred_arr = np.asarray(y_pred).ravel()

        # Currently we only define these metrics for classification.
        if self.cfg.kind != "classification":
            return {"confusion": None, "roc": None}

        # ---------------- Confusion-matrix-based metrics -----------------
        confusion = None
        if self.cfg.compute_confusion:
            confusion = confusion_matrix_metrics(
                y_true_arr,
                y_pred_arr,
                labels=labels,
            )

        # ------------------------- ROC curves ----------------------------
        roc: Optional[Dict[str, Any]] = None

        if self.cfg.compute_roc:
            Z: Optional[np.ndarray] = None
            if y_proba is not None:
                Z = np.asarray(y_proba)
            elif y_score is not None:
                Z = np.asarray(y_score)

            if Z is not None:
                if Z.ndim == 1:
                    # Binary case with a single score/probability per sample
                    roc = binary_roc_curve_from_scores(y_true_arr, Z)
                elif Z.ndim == 2:
                    # Multi-output scores/probabilities.
                    # Decide between binary and multiclass by number of distinct labels.
                    uniq = np.unique(y_true_arr)
                    n_classes = uniq.size

                    if n_classes == 2:
                        # Binary case represented with 2 columns.
                        # We treat the "positive" label as the second in sorted order.
                        sorted_labels = np.sort(uniq)
                        pos_label = sorted_labels[1]

                        # If an explicit label ordering is provided, try to align columns to it.
                        if labels is not None:
                            try:
                                pos_index = list(labels).index(pos_label)
                            except ValueError:
                                pos_index = 1
                        else:
                            # Fallback: assume the second column corresponds to the positive label.
                            pos_index = 1

                        scores_1d = Z[:, pos_index]
                        roc = binary_roc_curve_from_scores(
                            y_true_arr,
                            scores_1d,
                            pos_label=pos_label,
                        )
                    else:
                        # Multi-class one-vs-rest ROC curves.
                        roc = multiclass_roc_curves_from_scores(
                            y_true_arr,
                            Z,
                            labels=labels,
                        )

        return {"confusion": confusion, "roc": roc}