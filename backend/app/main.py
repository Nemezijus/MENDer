from fastapi import FastAPI, Request
import time, os
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Routers
from .routers.data import router as data_router
from .routers.pipeline import router as pipeline_router
from .routers.train import router as train_router
from .routers.cv import router as cv_router
from .routers.health import router as health_router            # NEW
from .routers.files import router as files_router              # NEW

app = FastAPI(
    title="MENDer Local API",
    version="1.0.0",
    description="Local-first API exposing MENDer pipeline logic",
)

# CORS for dev (Vite @ 5173) + optional env override
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
extra = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
if extra:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=extra,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Robust request logging (won't crash on exceptions)
@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.time()
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        dt = (time.time() - t0) * 1000
        print(f"{request.method} {request.url.path} -> ERR in {dt:.1f}ms: {type(e).__name__}: {e}")
        raise
    finally:
        # Only logs successful responses here (exceptions already printed above)
        pass

# Routers
app.include_router(health_router, prefix="/api/v1", tags=["health"])   # NEW
app.include_router(files_router,  prefix="/api/v1", tags=["files"])    # NEW
app.include_router(data_router,   prefix="/api/v1", tags=["data"])
app.include_router(pipeline_router, prefix="/api/v1", tags=["pipeline"])
app.include_router(train_router,  prefix="/api/v1", tags=["train"])
app.include_router(cv_router,     prefix="/api/v1", tags=["cv"])

if os.path.isdir("frontend_dist"):
    app.mount("/", StaticFiles(directory="frontend_dist", html=True), name="static")
else:
    # Dev mode: no static mount; the frontend runs on Vite.
    pass
@app.get("/healthz")
def healthz():
    return {"ok": True}
