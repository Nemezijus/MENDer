from fastapi import APIRouter

# NOTE:
# - Versioning (/api/v1) is applied in backend/app/main.py when the router is mounted.
# - This router itself should remain prefix-free to avoid double-prefix issues.
router = APIRouter()

@router.get("/ping")
def ping():
    return {"ok": True}
