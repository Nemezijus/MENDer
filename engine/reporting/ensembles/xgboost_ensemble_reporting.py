# utils/postprocessing/ensembles/xgboost_ensemble_reporting.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .common import _safe_float, _safe_int, _mean_std


def _to_float_list(xs: Any) -> Optional[List[float]]:
    try:
        a = np.asarray(xs, dtype=float).ravel()
        a = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=0.0)
        return [float(x) for x in a.tolist()]
    except Exception:
        return None


def _merge_feature_importances(
    store: Dict[str, List[float]],
    importances: Sequence[float],
    feature_names: Optional[Sequence[str]] = None,
) -> None:
    imp = np.asarray(importances, dtype=float).ravel()
    if imp.size == 0:
        return

    if feature_names is None:
        names = [f"f{i}" for i in range(imp.size)]
    else:
        names = [str(n) for n in feature_names]
        if len(names) != imp.size:
            # fallback to indices if mismatch
            names = [f"f{i}" for i in range(imp.size)]

    for n, v in zip(names, imp.tolist()):
        if not np.isfinite(v):
            continue
        store.setdefault(n, []).append(float(v))


def _aggregate_curve_mean(curves: List[List[float]]) -> Optional[Dict[str, Any]]:
    """Align curves by shortest length and return mean + std arrays."""
    if not curves:
        return None
    lens = [len(c) for c in curves if c]
    if not lens:
        return None
    L = min(lens)
    if L <= 0:
        return None

    M = np.asarray([c[:L] for c in curves], dtype=float)
    mean = np.mean(M, axis=0)
    std = np.std(M, axis=0)
    return {
        "mean": [float(x) for x in mean.tolist()],
        "std": [float(x) for x in std.tolist()],
        "n_rounds": int(L),
    }

def _finite_float_or_none(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        fx = float(x)
        if not np.isfinite(fx):
            return None
        return fx
    except Exception:
        return None
    
@dataclass
class XGBoostEnsembleReportAccumulator:
    metric_name: str
        # store params (opaque dict is fine; service populates)
    params: Dict[str, Any]
    train_eval_metric: Optional[str]
    task: Optional[str] = None

    _n_folds: int = 0

    _best_iterations: List[int] = None
    _best_scores: List[float] = None

    # eval curves: key like "validation_0:mlogloss" -> list of curve lists (per fold)
    _curves: Dict[str, List[List[float]]] = None

    # feature importances across folds
    _feat_imps: Dict[str, List[float]] = None

    @classmethod
    def create(
        cls,
        *,
        metric_name: str,
        task: Optional[str] = None,
        train_eval_metric: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> "XGBoostEnsembleReportAccumulator":
        tem = str(train_eval_metric) if train_eval_metric is not None else None
        return cls(
            metric_name=str(metric_name),
            task=(str(task) if task is not None else None),
            train_eval_metric=tem,
            params=dict(params or {}),
            _best_iterations=[],
            _best_scores=[],
            _curves={},
            _feat_imps={},
        )

    def add_fold(
        self,
        *,
        best_iteration: Optional[int] = None,
        best_score: Optional[float] = None,
        evals_result: Optional[Dict[str, Dict[str, Sequence[float]]]] = None,
        feature_importances: Optional[Sequence[float]] = None,
        feature_names: Optional[Sequence[str]] = None,
    ) -> None:
        """Add fold information.

        evals_result (sklearn wrapper style) typically looks like:
          {"validation_0": {"mlogloss": [...], "auc": [...]}, "validation_1": {...}}
        We'll flatten into keys: "validation_0:mlogloss".
        """
        self._n_folds += 1

        if best_iteration is not None:
            self._best_iterations.append(int(best_iteration))
        if best_score is not None:
            self._best_scores.append(float(best_score))

        if evals_result:
            for ds_name, metrics in evals_result.items():
                if not isinstance(metrics, dict):
                    continue
                for m_name, curve in metrics.items():
                    cl = _to_float_list(curve)
                    if not cl:
                        continue
                    key = f"{ds_name}:{m_name}"
                    self._curves.setdefault(key, []).append(cl)

        if feature_importances is not None:
            _merge_feature_importances(self._feat_imps, feature_importances, feature_names)

    def finalize(self, *, top_k_features: int = 12) -> Dict[str, Any]:
        bi_mean, bi_std = (None, None)
        bs_mean, bs_std = (None, None)

        if self._best_iterations:
            m, s = _mean_std(self._best_iterations)
            bi_mean, bi_std = _finite_float_or_none(m), _finite_float_or_none(s)

        if self._best_scores:
            m, s = _mean_std(self._best_scores)
            bs_mean, bs_std = _finite_float_or_none(m), _finite_float_or_none(s)

        curves_out: Dict[str, Any] = {}
        for key, curves in (self._curves or {}).items():
            agg = _aggregate_curve_mean(curves)
            if agg is not None:
                curves_out[key] = agg

        # feature importances: average across folds, normalize to sum=1 for readability
        feat_means: List[Tuple[str, float]] = []
        for name, vals in (self._feat_imps or {}).items():
            if not vals:
                continue
            feat_means.append((name, float(np.mean(np.asarray(vals, dtype=float)))))

        feat_means.sort(key=lambda x: x[1], reverse=True)

        if feat_means:
            total = sum(v for _, v in feat_means)
            if total > 0:
                feat_means_norm = [(n, v / total) for n, v in feat_means]
            else:
                feat_means_norm = feat_means
        else:
            feat_means_norm = []

        top_feats = [
            {"name": n, "importance": _safe_float(v)}
            for n, v in feat_means_norm[: max(0, int(top_k_features))]
        ]

        return {
            "kind": "xgboost",
            "task": self.task,
            "metric_name": self.metric_name,
            "train_eval_metric": self.train_eval_metric,
            "xgboost": {
                "params": self.params,
                "best_iteration_mean": bi_mean,
                "best_iteration_std": bi_std,
                "best_score_mean": bs_mean,
                "best_score_std": bs_std,
            },
            "learning_curves": curves_out if curves_out else None,
            "feature_importance": {
                "top_features": top_feats,
                "n_features_seen": _safe_int(len(feat_means_norm)),
            }
            if feat_means_norm
            else None,
            "meta": {
                "n_folds": _safe_int(self._n_folds),
            },
        }


__all__ = ["XGBoostEnsembleReportAccumulator"]
