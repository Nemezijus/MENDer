from __future__ import annotations

from typing import ClassVar, Literal, Optional

from pydantic import BaseModel


class GaussianNBConfig(BaseModel):
    algo: Literal["gnb"] = "gnb"

    task: ClassVar[str] = "classification"
    family: ClassVar[str] = "naive_bayes"

    var_smoothing: float = 1e-9
