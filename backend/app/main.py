from fastapi import FastAPI
from .routers.data import router as data_router
from .routers.pipeline import router as pipeline_router
from .routers.train import router as train_router
from .routers.cv import router as cv_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="MENDer Local API",
    version="1.0.0",
    description="Local-first API exposing MENDer pipeline logic"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Route registration ---
app.include_router(data_router, prefix="/api/v1", tags=["data"])
app.include_router(pipeline_router, prefix="/api/v1", tags=["pipeline"])
app.include_router(train_router, prefix="/api/v1", tags=["train"])
app.include_router(cv_router, prefix="/api/v1", tags=["cv"])

# --- Health check ---
@app.get("/healthz")
def healthz():
    """Simple liveness check"""
    return {"ok": True}
