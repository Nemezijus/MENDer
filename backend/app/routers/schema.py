from __future__ import annotations

from fastapi import APIRouter
from pydantic import TypeAdapter
from typing import Literal, Union, get_args, get_origin
from shared_schemas import types as T

from shared_schemas.model_configs import (
    ModelConfig,           # ← discriminated union
    LogRegConfig, SVMConfig, TreeConfig, ForestConfig, KNNConfig,
    GaussianNBConfig,
    RidgeClassifierConfig,
    SGDClassifierConfig,
    ExtraTreesConfig,
    HistGradientBoostingConfig,
    # ---------------- regressors ----------------
    LinearRegConfig,
    RidgeRegressorConfig,
    RidgeCVRegressorConfig,
    ElasticNetRegressorConfig,
    ElasticNetCVRegressorConfig,
    LassoRegressorConfig,
    LassoCVRegressorConfig,
    BayesianRidgeRegressorConfig,
    SVRRegressorConfig,
    LinearSVRRegressorConfig,
    KNNRegressorConfig,
    DecisionTreeRegressorConfig,
    RandomForestRegressorConfig,
)
from shared_schemas.split_configs import SplitCVModel, SplitHoldoutModel
from shared_schemas.scale_configs import ScaleModel
from shared_schemas.feature_configs import FeaturesModel
from shared_schemas.eval_configs import EvalModel

from shared_schemas.ensemble_configs import (
    EnsembleConfig,
    VotingEnsembleConfig,
    BaggingEnsembleConfig,
    AdaBoostEnsembleConfig,
    XGBoostEnsembleConfig,
)

router = APIRouter()

# ---------- helpers ----------

def _flatten_literal_alias(alias):
    """
    Return a flat list of values from a TypeAlias that may be:
      - Literal["a","b"]
      - Union[Literal["a","b"], None]
      - Optional[Literal["a","b"]]  (Union[Literal, NoneType])
    """
    try:
        # Literal[...] directly?
        if get_origin(alias) is Literal:
            return list(get_args(alias))

        # Union of Literal/None?
        if get_origin(alias) is Union:
            vals = []
            for sub in get_args(alias):
                if get_origin(sub) is Literal:
                    vals.extend(list(get_args(sub)))
                elif sub is type(None):
                    vals.append(None)
            return vals if vals else None

        # Not a Literal or Union[Literal,...] – nothing to enumerate
        return None
    except Exception as e:
        print("[/schema/enums] _flatten_literal_alias ERROR:", alias, repr(e))
        return None


def _schema_and_defaults(pyd_model):
    """
    For *concrete* Pydantic models: return json schema + defaults from instance().
    (Do NOT pass unions here.)
    """
    ta = TypeAdapter(pyd_model)
    schema = ta.json_schema()
    defaults = pyd_model().model_dump()
    return {"schema": schema, "defaults": defaults}


def _model_union_schema_and_defaults():
    """
    Special-cased for the discriminated union ModelConfig:
    - Build schema via TypeAdapter(ModelConfig)
    - Build per-algo defaults by instantiating each concrete config class
    """
    ta = TypeAdapter(ModelConfig)
    schema = ta.json_schema()

    defaults = {}
    for cls in (
        LogRegConfig,
        SVMConfig,
        TreeConfig,
        ForestConfig,
        KNNConfig,
        GaussianNBConfig,
        RidgeClassifierConfig,
        SGDClassifierConfig,
        ExtraTreesConfig,
        HistGradientBoostingConfig,
        # ---------------- regressors ----------------
        LinearRegConfig,
        RidgeRegressorConfig,
        RidgeCVRegressorConfig,
        ElasticNetRegressorConfig,
        ElasticNetCVRegressorConfig,
        LassoRegressorConfig,
        LassoCVRegressorConfig,
        BayesianRidgeRegressorConfig,
        SVRRegressorConfig,
        LinearSVRRegressorConfig,
        KNNRegressorConfig,
        DecisionTreeRegressorConfig,
        RandomForestRegressorConfig,

    ):
        try:
            inst = cls()
            defaults[inst.algo] = inst.model_dump()
        except Exception:
            # Shouldn't happen, but don't let a single class break the endpoint
            continue

    return {"schema": schema, "defaults": defaults}


def _ensemble_union_schema_and_defaults():
    """
    Special-cased for the discriminated union EnsembleConfig:
    - Build schema via TypeAdapter(EnsembleConfig)
    - Build per-kind defaults by instantiating each concrete config class
    """
    ta = TypeAdapter(EnsembleConfig)
    schema = ta.json_schema()

    defaults = {}
    for cls in (VotingEnsembleConfig, BaggingEnsembleConfig, AdaBoostEnsembleConfig, XGBoostEnsembleConfig):
        try:
            inst = cls()
            defaults[inst.kind] = inst.model_dump()
        except Exception:
            continue

    return {"schema": schema, "defaults": defaults}


def _schema_defaults(pyd_model):
    """Return defaults by instantiating a concrete Pydantic model."""
    return pyd_model().model_dump()


def _enums_payload():
    """Centralized enum lists for dropdowns — single source of truth."""
    def safe(alias): return _flatten_literal_alias(alias) if alias is not None else None
    enums = {
        # modeling
        "PenaltyName":          safe(getattr(T, "PenaltyName", None)),
        "LogRegSolver":         safe(getattr(T, "LogRegSolver", None)),
        "SVMKernel":            safe(getattr(T, "SVMKernel", None)),
        "SVMDecisionShape":     safe(getattr(T, "SVMDecisionShape", None)),
        "TreeCriterion":        safe(getattr(T, "TreeCriterion", None)),
        "TreeSplitter":         safe(getattr(T, "TreeSplitter", None)),
        "MaxFeaturesName":      safe(getattr(T, "MaxFeaturesName", None)),

        # regression trees / forests
        "RegTreeCriterion":     safe(getattr(T, "RegTreeCriterion", None)),

        # coordinate descent (Lasso / ElasticNet)
        "CoordinateDescentSelection": safe(getattr(T, "CoordinateDescentSelection", None)),

        # SVR / LinearSVR
        "LinearSVRLoss":        safe(getattr(T, "LinearSVRLoss", None)),

        # class weights
        "ClassWeightBalanced":  safe(getattr(T, "ClassWeightBalanced", None)),
        "ForestClassWeight":    safe(getattr(T, "ForestClassWeight", None)),

        # KNN
        "KNNWeights":           safe(getattr(T, "KNNWeights", None)),
        "KNNAlgorithm":         safe(getattr(T, "KNNAlgorithm", None)),
        "KNNMetric":            safe(getattr(T, "KNNMetric", None)),

        # RidgeClassifier
        "RidgeSolver":          safe(getattr(T, "RidgeSolver", None)),

        # SGDClassifier
        "SGDLoss":              safe(getattr(T, "SGDLoss", None)),
        "SGDPenalty":           safe(getattr(T, "SGDPenalty", None)),
        "SGDLearningRate":      safe(getattr(T, "SGDLearningRate", None)),

        # HistGradientBoostingClassifier
        "HGBLoss":              safe(getattr(T, "HGBLoss", None)),

        # pipeline-wide
        "ScaleName":            safe(getattr(T, "ScaleName", None)),
        "FeatureName":          safe(getattr(T, "FeatureName", None)),
        "MetricName":           safe(getattr(T, "MetricName", None)),

        # ensembles
        "EnsembleKind":         safe(getattr(T, "EnsembleKind", None)),
    }
    cls = safe(getattr(T, "ClassificationMetricName", None))
    reg = safe(getattr(T, "RegressionMetricName", None))
    metric_by_task = {}
    if cls: metric_by_task["classification"] = cls
    if reg: metric_by_task["regression"] = reg
    if metric_by_task:
        enums["MetricByTask"] = metric_by_task
    return {k: v for k, v in enums.items() if v is not None}


def _model_defaults_and_meta():
    """
    Return:
      - defaults: {algo: {...}}
      - meta: {algo: {task: 'classification'|'regression'|'nn', family: 'svm'|'tree'|...}}
    For now, categorize existing algos as classification; expand as you add regressors/NNs.
    """
    algo_classes = [
        LogRegConfig,
        SVMConfig,
        TreeConfig,
        ForestConfig,
        KNNConfig,
        GaussianNBConfig,
        RidgeClassifierConfig,
        SGDClassifierConfig,
        ExtraTreesConfig,
        HistGradientBoostingConfig,

        # ---------------- regressors ----------------
        LinearRegConfig,
        RidgeRegressorConfig,
        RidgeCVRegressorConfig,
        ElasticNetRegressorConfig,
        ElasticNetCVRegressorConfig,
        LassoRegressorConfig,
        LassoCVRegressorConfig,
        BayesianRidgeRegressorConfig,
        SVRRegressorConfig,
        LinearSVRRegressorConfig,
        KNNRegressorConfig,
        DecisionTreeRegressorConfig,
        RandomForestRegressorConfig,

    ]
    defaults = {}
    meta = {}

    for cls in algo_classes:
        inst = cls()
        algo = inst.algo
        defaults[algo] = inst.model_dump()
        meta[algo] = {
            "task": cls.task,
            "family": cls.family,
        }

    return defaults, meta


# ---------- routes ----------

@router.get("/defaults")
def get_all_defaults():
    """
    Consolidated defaults + enums for the UI.
    Use this to remove hardcoded DEFAULTS and static option lists on the frontend.
    """
    model_defaults, model_meta = _model_defaults_and_meta()
    ensemble_defaults = _ensemble_union_schema_and_defaults()

    payload = {
        "models": {
            "schema": TypeAdapter(ModelConfig).json_schema(),  # union schema (oneOf + discriminator)
            "defaults": model_defaults,
            "meta": model_meta,
        },
        "ensembles": {
            "schema": TypeAdapter(EnsembleConfig).json_schema(),
            "defaults": ensemble_defaults["defaults"],
        },
        "scale": {
            "schema": TypeAdapter(ScaleModel).json_schema(),
            "defaults": _schema_defaults(ScaleModel),
        },
        "features": {
            "schema": TypeAdapter(FeaturesModel).json_schema(),
            "defaults": _schema_defaults(FeaturesModel),
        },
        "split": {
            "holdout": {
                "schema": TypeAdapter(SplitHoldoutModel).json_schema(),
                "defaults": _schema_defaults(SplitHoldoutModel),
            },
            "kfold": {
                "schema": TypeAdapter(SplitCVModel).json_schema(),
                "defaults": _schema_defaults(SplitCVModel),
            },
        },
        "eval": {
            "schema": TypeAdapter(EvalModel).json_schema(),
            "defaults": _schema_defaults(EvalModel),
        },
        "enums": _enums_payload(),
        "schema_version": 1,
    }
    return payload


@router.get("/enums")
def get_enums():
    """Expose centralized enum values defined in shared_schemas/types.py."""
    enums = _enums_payload()
    return {"enums": enums}


@router.get("/model")
def get_model_schema():
    # ModelConfig is a union → use the special-cased helper
    return _model_union_schema_and_defaults()


@router.get("/ensemble")
def get_ensemble_schema():
    # EnsembleConfig is a union → use the special-cased helper
    return _ensemble_union_schema_and_defaults()


@router.get("/features")
def get_features_schema():
    return _schema_and_defaults(FeaturesModel)


@router.get("/split/holdout")
def get_split_holdout_schema():
    return _schema_and_defaults(SplitHoldoutModel)


@router.get("/split/kfold")
def get_split_kfold_schema():
    return _schema_and_defaults(SplitCVModel)


@router.get("/scale")
def get_scale_schema():
    return _schema_and_defaults(ScaleModel)


@router.get("/eval")
def get_eval_schema():
    return _schema_and_defaults(EvalModel)
