from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

from engine.reporting.common.json_safety import ReportError, add_report_error
from engine.core.sklearn_utils import transform_through_pipeline as _transform_through_pipeline_core
from engine.core.sklearn_utils import unwrap_final_estimator

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]


def _as_2d(X: Any) -> Any:
    if np is None:
        return X
    a = np.asarray(X)
    if a.ndim != 2:
        raise ValueError(f"X must be 2D; got shape {a.shape}.")
    return a


_final_estimator = unwrap_final_estimator


def _transform_through_pipeline(model: Any, X: Any) -> Any:
    if np is None:
        return X
    return _transform_through_pipeline_core(model, X)


def cluster_summary(labels: Any) -> Mapping[str, Any]:
    errors: List[ReportError] = []
    if np is None:
        return {
            "n_clusters": None,
            "n_noise": None,
            "noise_ratio": None,
            "cluster_sizes": None,
        }
    try:
        y = np.asarray(labels).reshape(-1)
        if y.size == 0:
            return {
                "n_clusters": 0,
                "n_noise": 0,
                "noise_ratio": 0.0,
                "cluster_sizes": {},
            }

        unique, counts = np.unique(y, return_counts=True)
        sizes = {int(k): int(v) for k, v in zip(unique.tolist(), counts.tolist())}
        n_noise = int(sizes.get(-1, 0))
        n_clusters = int(sum(1 for k in sizes.keys() if int(k) != -1))
        noise_ratio = float(n_noise / y.size) if int(y.size) > 0 else 0.0
        return {
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "noise_ratio": noise_ratio,
            "cluster_sizes": sizes,
        }
    except Exception as e:
        add_report_error(errors, where="reporting.clustering.cluster_summary", exc=e)
        return {
            "n_clusters": None,
            "n_noise": None,
            "noise_ratio": None,
            "cluster_sizes": None,
            "errors": errors,
        }


def model_diagnostics(model: Any, X: Any, labels: Any) -> Tuple[Mapping[str, Any], List[str]]:
    warnings: List[str] = []
    out: Dict[str, Any] = {}

    if np is None:
        return out, ["numpy unavailable; cannot compute model diagnostics."]

    est = _final_estimator(model)
    Xa = _transform_through_pipeline(model, X)
    try:
        Xa = _as_2d(Xa)
    except Exception:
        try:
            Xa = _as_2d(X)
        except Exception:
            Xa = None

    try:
        inertia = getattr(est, "inertia_", None)
        if inertia is not None:
            out["inertia"] = float(inertia)
        n_iter = getattr(est, "n_iter_", None)
        if n_iter is not None:
            out["n_iter"] = int(n_iter)
    except Exception:
        pass

    if Xa is not None:
        try:
            if hasattr(est, "aic"):
                out["aic"] = float(est.aic(Xa))
            if hasattr(est, "bic"):
                out["bic"] = float(est.bic(Xa))
        except Exception as e:
            warnings.append(f"Failed to compute AIC/BIC: {type(e).__name__}: {e}")

        try:
            if hasattr(est, "score_samples"):
                ll = np.asarray(est.score_samples(Xa)).reshape(-1)
                if ll.size:
                    out["mean_log_likelihood"] = float(np.mean(ll))
                    out["std_log_likelihood"] = float(np.std(ll))
        except Exception:
            pass

    try:
        conv = getattr(est, "converged_", None)
        if conv is not None:
            out["converged"] = bool(conv)
        nit = getattr(est, "n_iter_", None)
        if nit is not None and "n_iter" not in out:
            out["n_iter"] = int(nit)
        lb = getattr(est, "lower_bound_", None)
        if lb is not None:
            out["lower_bound"] = float(lb)
    except Exception:
        pass

    try:
        out["label_summary"] = dict(cluster_summary(labels))
    except Exception:
        pass

    return out, warnings


def per_sample_outputs(
    model: Any,
    X: Any,
    labels: Any,
    *,
    include_cluster_probabilities: bool = False,
) -> Tuple[Mapping[str, Any], List[str]]:
    warnings: List[str] = []
    out: Dict[str, Any] = {}

    if np is None:
        return out, ["numpy unavailable; cannot extract per-sample outputs."]

    y = np.asarray(labels).reshape(-1)
    out["cluster_id"] = [int(v) for v in y.tolist()]
    out["is_noise"] = [bool(int(v) == -1) for v in y.tolist()]

    est = _final_estimator(model)
    Xa = _transform_through_pipeline(model, X)
    try:
        Xa = _as_2d(Xa)
    except Exception:
        try:
            Xa = _as_2d(X)
        except Exception as e:
            warnings.append(f"Invalid X for per_sample_outputs: {e}")
            return out, warnings

    try:
        if hasattr(est, "transform"):
            d = np.asarray(est.transform(Xa))
            if d.ndim == 2 and d.shape[0] == Xa.shape[0] and d.shape[1] >= 1:
                out["distance_to_center"] = [
                    float(v) for v in np.min(d, axis=1).tolist()
                ]
    except Exception as e:
        warnings.append(
            f"Failed to compute distance_to_center: {type(e).__name__}: {e}"
        )

    try:
        if hasattr(est, "predict_proba"):
            P = np.asarray(est.predict_proba(Xa))
            if P.ndim == 2 and P.shape[0] == Xa.shape[0]:
                out["max_membership_prob"] = [
                    float(v) for v in np.max(P, axis=1).tolist()
                ]
                if include_cluster_probabilities:
                    out["cluster_probabilities"] = [
                        [float(v) for v in row.tolist()] for row in P
                    ]
    except Exception as e:
        warnings.append(
            f"Failed to compute membership probabilities: {type(e).__name__}: {e}"
        )

    try:
        if hasattr(est, "score_samples"):
            ll = np.asarray(est.score_samples(Xa)).reshape(-1)
            if ll.size == Xa.shape[0]:
                out["log_likelihood"] = [float(v) for v in ll.tolist()]
    except Exception as e:
        warnings.append(f"Failed to compute log_likelihood: {type(e).__name__}: {e}")

    try:
        core_idx = getattr(est, "core_sample_indices_", None)
        if core_idx is not None:
            core = np.zeros((Xa.shape[0],), dtype=bool)
            core[np.asarray(core_idx, dtype=int)] = True
            out["is_core"] = [bool(v) for v in core.tolist()]
    except Exception:
        pass

    return out, warnings


@dataclass
class UnsupervisedDiagnostics:
    """Convenience container for unsupervised diagnostics."""

    cluster_summary: Mapping[str, Any]
    model_diagnostics: Mapping[str, Any]
    per_sample: Mapping[str, Any]
    embedding_2d: Optional[Mapping[str, List[float]]] = None
    warnings: List[str] = None  # type: ignore[assignment]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_summary": dict(self.cluster_summary) if self.cluster_summary is not None else None,
            "model_diagnostics": dict(self.model_diagnostics) if self.model_diagnostics is not None else None,
            "per_sample": dict(self.per_sample) if self.per_sample is not None else None,
            "embedding_2d": dict(self.embedding_2d) if self.embedding_2d is not None else None,
            "warnings": list(self.warnings) if self.warnings is not None else [],
        }
