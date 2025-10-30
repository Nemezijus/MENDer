from fastapi import APIRouter
from ..models.v1.pipeline_models import (
    PipelinePreviewRequest, PipelinePreviewResponse
)
from ..services.pipeline_service import preview_pipeline

router = APIRouter()

@router.post("/pipeline/preview", response_model=PipelinePreviewResponse)
def pipeline_preview(req: PipelinePreviewRequest):
    result = preview_pipeline(req)
    return PipelinePreviewResponse(**result)
