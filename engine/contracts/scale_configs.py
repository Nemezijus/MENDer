from __future__ import annotations

from pydantic import BaseModel
from .types import ScaleName

class ScaleModel(BaseModel):
    method: ScaleName = "standard"
