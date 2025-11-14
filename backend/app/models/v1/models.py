from typing import Optional
from pydantic import BaseModel

from .model_artifact import ModelArtifactMeta


class SaveModelRequest(BaseModel):
    artifact_uid: str
    artifact_meta: ModelArtifactMeta
    filename: Optional[str] = None  # optional client-suggested name (e.g., "my_model.mend")


class LoadModelResponse(BaseModel):
    artifact: ModelArtifactMeta
