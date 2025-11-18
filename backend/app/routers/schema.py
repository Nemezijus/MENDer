from fastapi import APIRouter
from pydantic import TypeAdapter

from ..models.v1.shared import (
    ModelModel, FeaturesModel, SplitHoldoutModel, SplitCVModel, ScaleModel, EvalModel
)

router = APIRouter()

def _schema_and_defaults(pyd_model):
    # JSON schema (OpenAPI-compatible)
    ta = TypeAdapter(pyd_model)
    schema = ta.json_schema()
    # Defaults from an instance
    defaults = pyd_model().model_dump()
    return {"schema": schema, "defaults": defaults}

@router.get("/model")
def get_model_schema():
    return _schema_and_defaults(ModelModel)

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
