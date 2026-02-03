from __future__ import annotations

from typing import Optional, Union

from pydantic import BaseModel

from .split_configs import SplitHoldoutModel, SplitCVModel
from .scale_configs import ScaleModel
from .feature_configs import FeaturesModel
from .model_configs import ModelConfig
from .eval_configs import EvalModel


class DataModel(BaseModel):
    x_path: Optional[str] = None
    y_path: Optional[str] = None
    npz_path: Optional[str] = None
    x_key: Optional[str] = None
    y_key: Optional[str] = None

    # Optional parsing hints for tabular/container formats.
    # These are used by some loaders (csv/tsv/txt, xlsx, h5/hdf5).
    delimiter: Optional[str] = None
    has_header: Optional[bool] = None
    encoding: Optional[str] = None
    sheet_name: Optional[Union[str, int]] = None


class RunConfig(BaseModel):
    data: DataModel
    split: Union[SplitHoldoutModel, SplitCVModel]
    scale: ScaleModel
    features: FeaturesModel
    model: ModelConfig
    eval: EvalModel
