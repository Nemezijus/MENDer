from fastapi import FastAPI, Request
import time, os
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# NEW: selective access-log filter for uvicorn
import logging

class _SkipProgressAccessLogs(logging.Filter):
    """Hide uvicorn access logs for /api/v1/progress/* to prevent console spam."""
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        # Keep all logs except progress endpoint polls
        return "/api/v1/progress/" not in msg

# Attach the filter once
_access_logger = logging.getLogger("uvicorn.access")
# Avoid duplicate filters on reload
if not any(isinstance(f, _SkipProgressAccessLogs) for f in getattr(_access_logger, "filters", [])):
    _access_logger.addFilter(_SkipProgressAccessLogs())

# Routers
from .routers.data import router as data_router
from .routers.pipeline import router as pipeline_router
from .routers.train import router as train_router
from .routers.health import router as health_router
from .routers.files import router as files_router
from .routers.learning_curve import router as learning_curve_router
from .routers.progress import router as progress_router
from .routers.models import router as models_router
from .routers.schema import router as schema_router
from .routers.predict import router as predict_router  # <-- NEW

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
        # Successful responses are logged by uvicorn.access; we filtered progress calls above.
        pass

# Routers
app.include_router(health_router,        prefix="/api/v1",        tags=["health"])
app.include_router(files_router,         prefix="/api/v1",        tags=["files"])
app.include_router(data_router,          prefix="/api/v1",        tags=["data"])
app.include_router(pipeline_router,      prefix="/api/v1",        tags=["pipeline"])
app.include_router(train_router,         prefix="/api/v1",        tags=["train"])
app.include_router(learning_curve_router, prefix="/api/v1",       tags=["learning-curve"])
app.include_router(progress_router,      prefix="/api/v1",        tags=["progress"])
app.include_router(models_router,        prefix="/api/v1",        tags=["models"])
app.include_router(predict_router,       prefix="/api/v1",        tags=["predict"])
app.include_router(schema_router,        prefix="/api/v1/schema", tags=["schema"])

if os.path.isdir("frontend_dist"):
    app.mount("/", StaticFiles(directory="frontend_dist", html=True), name="static")
else:
    # Dev mode: no static mount; the frontend runs on Vite.
    pass

@app.get("/healthz")
def healthz():
    return {"ok": True}
