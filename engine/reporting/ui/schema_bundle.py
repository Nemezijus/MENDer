from __future__ import annotations

"""UI schema bundle generator.

The backend must not hardcode inventories of Engine contracts (models, enums,
defaults). Instead, the Engine BL produces a single stable bundle that the UI
can consume.

This module intentionally:
- uses only Engine contracts + Pydantic schema generation
- derives per-algo defaults/meta by introspecting the discriminated unions
"""

from typing import Any, Annotated, Dict, List, Literal, Optional, Type, Union, get_args, get_origin

from pydantic import TypeAdapter

from engine.contracts import types as T
from engine.contracts.ensemble_configs import EnsembleConfig
from engine.contracts.eval_configs import EvalModel
from engine.contracts.feature_configs import FeaturesModel
from engine.contracts.model_configs import ModelConfig
from engine.contracts.run_config import DataModel
from engine.contracts.scale_configs import ScaleModel
from engine.contracts.split_configs import SplitCVModel, SplitHoldoutModel
from engine.contracts.unsupervised_configs import UnsupervisedEvalModel, UnsupervisedRunConfig


# ---------------------------------------------------------------------------
# Introspection helpers
# ---------------------------------------------------------------------------


def _iter_discriminated_union_members(tp: Any) -> List[Type[Any]]:
    """Return concrete member types of an Annotated[Union[...], Field(...)] contract."""

    # Unwrap Annotated[...] if present
    if get_origin(tp) is Annotated:
        tp = get_args(tp)[0]

    origin = get_origin(tp)
    if origin is Union:
        return [t for t in get_args(tp) if isinstance(t, type)]

    # Not a Union; nothing to enumerate.
    return []


def _defaults_for_union(tp: Any, *, key_field: str) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {}
    for cls in _iter_discriminated_union_members(tp):
        try:
            inst = cls()  # type: ignore[call-arg]
            key = getattr(inst, key_field)
            defaults[str(key)] = inst.model_dump()
        except Exception:
            # Don't allow a single config class to break schema generation.
            continue
    return defaults


def _model_defaults_and_meta() -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Defaults + meta for ModelConfig, derived from union members."""

    defaults: Dict[str, Any] = {}
    meta: Dict[str, Any] = {}

    for cls in _iter_discriminated_union_members(ModelConfig):
        try:
            inst = cls()  # type: ignore[call-arg]
            algo = str(getattr(inst, "algo"))
            defaults[algo] = inst.model_dump()
            meta[algo] = {
                "task": getattr(cls, "task", None),
                "family": getattr(cls, "family", None),
            }
        except Exception:
            continue

    return defaults, meta


# ---------------------------------------------------------------------------
# Enum extraction
# ---------------------------------------------------------------------------


def _flatten_literal_alias(alias: Any) -> Optional[List[Any]]:
    """Return a flat list of Literal values from a TypeAlias.

    Supports:
      - Literal[...]
      - Union[Literal[...], ...]
      - Optional[Literal[...]]

    Non-Literal branches (e.g. int/float) are ignored.
    """

    try:
        if get_origin(alias) is Literal:
            return list(get_args(alias))

        if get_origin(alias) is Union:
            vals: List[Any] = []
            for sub in get_args(alias):
                if get_origin(sub) is Literal:
                    vals.extend(list(get_args(sub)))
                elif sub is type(None):
                    vals.append(None)
            return vals if vals else None

        return None
    except Exception:
        return None


def _enums_payload() -> Dict[str, Any]:
    """Centralized enum lists for dropdowns — single source of truth."""

    def safe(alias: Any) -> Optional[List[Any]]:
        return _flatten_literal_alias(alias)

    enums: Dict[str, Any] = {
        # modeling
        "PenaltyName": safe(T.PenaltyName),
        "LogRegSolver": safe(T.LogRegSolver),
        "SVMKernel": safe(T.SVMKernel),
        "SVMDecisionShape": safe(T.SVMDecisionShape),
        "TreeCriterion": safe(T.TreeCriterion),
        "TreeSplitter": safe(T.TreeSplitter),
        "MaxFeaturesName": safe(T.MaxFeaturesName),

        # regression trees / forests
        "RegTreeCriterion": safe(T.RegTreeCriterion),

        # coordinate descent (Lasso / ElasticNet)
        "CoordinateDescentSelection": safe(T.CoordinateDescentSelection),

        # SVR / LinearSVR
        "LinearSVRLoss": safe(T.LinearSVRLoss),

        # class weights
        "ClassWeightBalanced": safe(T.ClassWeightBalanced),
        "ForestClassWeight": safe(T.ForestClassWeight),

        # KNN
        "KNNWeights": safe(T.KNNWeights),
        "KNNAlgorithm": safe(T.KNNAlgorithm),
        "KNNMetric": safe(T.KNNMetric),

        # RidgeClassifier
        "RidgeSolver": safe(T.RidgeSolver),

        # SGDClassifier
        "SGDLoss": safe(T.SGDLoss),
        "SGDPenalty": safe(T.SGDPenalty),
        "SGDLearningRate": safe(T.SGDLearningRate),

        # HistGradientBoostingClassifier
        "HGBLoss": safe(T.HGBLoss),

        # pipeline-wide
        "ScaleName": safe(T.ScaleName),
        "FeatureName": safe(T.FeatureName),
        "MetricName": safe(T.MetricName),

        # task routing / unsupervised
        "ModelTaskName": safe(T.ModelTaskName),
        "UnsupervisedMetricName": safe(T.UnsupervisedMetricName),
        "FitScopeName": safe(T.FitScopeName),

        # ensembles
        "EnsembleKind": safe(T.EnsembleKind),
    }

    cls = safe(T.ClassificationMetricName)
    reg = safe(T.RegressionMetricName)
    unsup = safe(T.UnsupervisedMetricName)
    metric_by_task: Dict[str, Any] = {}
    if cls:
        metric_by_task["classification"] = cls
    if reg:
        metric_by_task["regression"] = reg
    if unsup:
        metric_by_task["unsupervised"] = unsup
    if metric_by_task:
        enums["MetricByTask"] = metric_by_task

    return {k: v for k, v in enums.items() if v is not None}


# ---------------------------------------------------------------------------
# Public bundle
# ---------------------------------------------------------------------------


def build_ui_schema_bundle(*, schema_version: int = 1) -> Dict[str, Any]:
    """Build the consolidated UI schema/defaults/enums payload."""

    model_defaults, model_meta = _model_defaults_and_meta()
    ensemble_defaults = _defaults_for_union(EnsembleConfig, key_field="kind")

    payload: Dict[str, Any] = {
        "models": {
            "schema": TypeAdapter(ModelConfig).json_schema(),
            "defaults": model_defaults,
            "meta": model_meta,
        },
        "ensembles": {
            "schema": TypeAdapter(EnsembleConfig).json_schema(),
            "defaults": ensemble_defaults,
        },
        "scale": {
            "schema": TypeAdapter(ScaleModel).json_schema(),
            "defaults": ScaleModel().model_dump(),
        },
        "features": {
            "schema": TypeAdapter(FeaturesModel).json_schema(),
            "defaults": FeaturesModel().model_dump(),
        },
        "split": {
            "holdout": {
                "schema": TypeAdapter(SplitHoldoutModel).json_schema(),
                "defaults": SplitHoldoutModel().model_dump(),
            },
            "kfold": {
                "schema": TypeAdapter(SplitCVModel).json_schema(),
                "defaults": SplitCVModel().model_dump(),
            },
        },
        "eval": {
            "schema": TypeAdapter(EvalModel).json_schema(),
            "defaults": EvalModel().model_dump(),
        },
        "unsupervised": {
            "run": {
                "schema": TypeAdapter(UnsupervisedRunConfig).json_schema(),
                # UnsupervisedRunConfig cannot be instantiated without choosing a concrete ModelConfig.
                # Provide a partial defaults dict so the UI can construct the full payload.
                "defaults": {
                    "task": "unsupervised",
                    "data": DataModel().model_dump(),
                    "apply": None,
                    "fit_scope": "train_only",
                    "scale": ScaleModel().model_dump(),
                    "features": FeaturesModel().model_dump(),
                    "model": None,
                    "eval": UnsupervisedEvalModel().model_dump(),
                    "use_y_for_external_metrics": False,
                    "external_metrics": [],
                },
            },
            "eval": {
                "schema": TypeAdapter(UnsupervisedEvalModel).json_schema(),
                "defaults": UnsupervisedEvalModel().model_dump(),
            },
        },
        "enums": _enums_payload(),
        "schema_version": int(schema_version),
    }

    return payload
