from __future__ import annotations

from typing import Any, ClassVar, TypedDict


class ModelMeta(TypedDict):
    task: str
    family: str


def get_model_meta(model_cfg: Any) -> ModelMeta:
    cls = model_cfg.__class__
    return {
        "task": getattr(cls, "task", "classification"),
        "family": getattr(cls, "family", "other"),
    }


def get_model_task(model_cfg: Any) -> str:
    cls = model_cfg.__class__
    return getattr(cls, "task", "classification")
