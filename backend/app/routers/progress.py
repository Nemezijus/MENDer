from typing import Any

from fastapi import APIRouter, HTTPException
from ..progress.registry import PROGRESS

router = APIRouter(prefix="/progress", tags=["progress"])

@router.get("/{progress_id}")
def get_progress(progress_id: str) -> dict[str, Any]:
    rec = PROGRESS.get(progress_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Unknown progress id")
    return rec
